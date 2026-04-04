"""Autonomous walk execution logic extracted from orchestrator."""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from geopy.distance import geodesic

from src.core import settings


def _closest_link(links: List[Dict], target_heading: float) -> Dict:
    return min(
        links,
        key=lambda l: abs(
            ((l.get("heading") or l.get("yawDeg") or 0) - target_heading + 180) % 360 - 180
        ),
    )


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference from a to b, range [-180, 180]."""
    return ((b - a + 180) % 360) - 180


class AutonomousWalkRunner:
    """Encapsulates branch exploration and autonomous walk execution."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def __getattr__(self, name: str) -> Any:
        # Delegate agent dependencies (logger, planner, persona_reasoner, etc.)
        return getattr(self._agent, name)

    async def _resolve_pano_chain(
        self,
        start_pano_id: str,
        initial_heading: float,
        direction_label: str,
        lookahead_depth: int,
    ) -> List[Dict[str, Any]]:
        """Follow pano links for lookahead_depth steps and return (pano_id, heading) pairs.

        Sequential because each step's pano_id depends on the previous step's metadata.
        Returns a list of dicts: [{"pano_id": ..., "heading": ..., "label": ...}, ...]
        Each entry is a future pano (not start_pano_id itself), so chain[0] is the 1st step ahead.
        """
        import httpx
        from demo.server import MapTilesSession

        nav = MapTilesSession(settings.google_maps_api_key)
        chain: List[Dict[str, Any]] = []
        current_pano = start_pano_id
        current_heading = initial_heading

        for depth in range(lookahead_depth):
            # Advance first: follow the link closest to current_heading
            try:
                meta = await nav.get_metadata(current_pano)
                links = meta.get("links", [])
                if not links:
                    break
                next_link = min(
                    links,
                    key=lambda l: abs(
                        ((l.get("heading") or l.get("yawDeg") or 0)
                         - current_heading + 180) % 360 - 180
                    ),
                )
                current_pano = next_link.get("panoId") or next_link.get("id")
                current_heading = float(
                    next_link.get("heading")
                    or next_link.get("yawDeg")
                    or current_heading
                )
            except Exception as e:
                self.logger.warning(
                    f"Pano chain advance failed at depth={depth} "
                    f"direction={direction_label}: {e}"
                )
                break

            chain.append({
                "pano_id": current_pano,
                "heading": current_heading,
                "direction_label": direction_label,
                "depth": depth,
            })

        return chain

    @staticmethod
    async def _fetch_sv_image(pano_id: str, heading: float) -> Optional[bytes]:
        """Download a single Street View Static image. Returns None on failure."""
        import httpx

        url = (
            f"https://maps.googleapis.com/maps/api/streetview"
            f"?pano={pano_id}&size=640x640"
            f"&heading={heading:.0f}&fov=90&pitch=0"
            f"&key={settings.google_maps_api_key}"
        )
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.content
        except Exception:
            return None

    def _run_vlm(
        self,
        image_bytes: bytes,
        heading: float,
        waypoint_id: int,
    ) -> Optional[Any]:
        """Write image to a temp file and run analyze_waypoint() (sync, CPU-bound)."""
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            tmp.write(image_bytes)
            tmp.flush()
            tmp.close()
            image_path = Path(tmp.name)
            try:
                return self.continuous_analyzer.analyze_waypoint(
                    waypoint_id=waypoint_id,
                    image_path=image_path,
                    metadata={"heading": heading, "lat": 0, "lon": 0},
                    visual_change_detected=False,
                    phash_distance=None,
                    branch_exploration=True,
                )
            finally:
                image_path.unlink(missing_ok=True)
        except Exception:
            tmp_path = Path(tmp.name)
            tmp_path.unlink(missing_ok=True)
            return None

    async def branch_decision(
        self,
        branch_pano_id: str,
        candidate_headings: List[float],
        memory_manager: Any,
        destination_context: Optional[str] = None,
        visit_counts: Optional[Dict[str, int]] = None,
        output_dir: Optional[Path] = None,
        lookahead_depth: int = 1,
        progress_callback=None,
        wp_bearing: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate candidate directions at a branch point and choose the best.

        Args:
            branch_pano_id: pano_id of the branch point (current position)
            candidate_headings: List of compass headings to evaluate (2–5 directions)
            memory_manager: Current route MemoryManager (provides LTM context)
            output_dir: Optional directory to save branch decision log
            lookahead_depth: Waypoints to explore per direction (1 = current behaviour)

        Returns:
            {
                "chosen_direction": str,
                "chosen_heading": float,
                "chosen_pano_id": str,
                "reason": str,
                "confidence": float,
                "ranking": List[str],
                "candidates": List[Dict],  # full per-direction analysis
            }
        """
        import asyncio
        import string

        if not (2 <= len(candidate_headings) <= 5):
            raise ValueError(
                f"candidate_headings must have 2–5 entries, got {len(candidate_headings)}"
            )

        direction_labels = list(string.ascii_uppercase[: len(candidate_headings)])

        self.logger.info(
            f"Branch decision: {len(candidate_headings)} candidates "
            f"× depth={lookahead_depth} at pano={branch_pano_id[:12]}"
        )

        # ── Step 1: resolve pano chains (sequential per direction, parallel across directions)
        chains_per_dir: List[List[Dict]] = list(
            await asyncio.gather(
                *[
                    self._resolve_pano_chain(
                        start_pano_id=branch_pano_id,
                        initial_heading=heading,
                        direction_label=label,
                        lookahead_depth=lookahead_depth,
                    )
                    for label, heading in zip(direction_labels, candidate_headings)
                ]
            )
        )

        # Flatten all (direction, depth) slots into a single list for bulk download
        all_slots: List[Dict] = [
            slot for chain in chains_per_dir for slot in chain
        ]

        # ── Step 2: download all images in parallel (gather — no VLM involved)
        image_bytes_list: List[Optional[bytes]] = list(
            await asyncio.gather(
                *[self._fetch_sv_image(s["pano_id"], s["heading"]) for s in all_slots]
            )
        )
        self.logger.debug(
            f"Branch look-ahead: downloaded {sum(1 for b in image_bytes_list if b)} "
            f"/ {len(all_slots)} images"
        )

        # ── Step 3: VLM calls — Semaphore(48) + retry(2)
        _branch_semaphore = asyncio.Semaphore(48)
        loop = asyncio.get_event_loop()

        async def _vlm_with_retry(image_bytes: Optional[bytes], slot: Dict, idx: int, max_retry: int = 2):
            if not image_bytes:
                return None
            for attempt in range(max_retry + 1):
                async with _branch_semaphore:
                    try:
                        result = await loop.run_in_executor(
                            None,
                            lambda b=image_bytes, h=slot["heading"], i=idx: self._run_vlm(b, h, i),
                        )
                        if result is not None and result.persona_scores:
                            if progress_callback:
                                try:
                                    await progress_callback({
                                        "direction": slot["direction_label"],
                                        "depth": slot["depth"],
                                        "pano_id": slot["pano_id"],
                                        "heading": slot["heading"],
                                        "scores": dict(result.persona_scores),
                                    })
                                except Exception:
                                    pass
                            return result
                    except Exception as e:
                        if attempt == max_retry:
                            self.logger.warning(
                                f"Branch VLM failed after {max_retry + 1} attempts "
                                f"(slot={idx}): {e}"
                            )
                            return None
                if attempt < max_retry:
                    await asyncio.sleep(0.5 * (attempt + 1))
            return None

        analyses = list(
            await asyncio.gather(
                *[_vlm_with_retry(img, slot, i) for i, (img, slot) in enumerate(zip(image_bytes_list, all_slots))]
            )
        )

        # ── Step 4: reassemble per-direction results
        idx = 0
        direction_results: List[Dict[str, Any]] = []
        for chain, (label, heading) in zip(chains_per_dir, zip(direction_labels, candidate_headings)):
            last_pano_id = chain[-1]["pano_id"] if chain else branch_pano_id
            waypoint_scores: List[Dict[str, float]] = []
            waypoint_summaries: List[str] = []

            for _ in chain:
                analysis = analyses[idx]
                idx += 1
                if analysis is not None:
                    waypoint_scores.append(dict(analysis.persona_scores))
                    reasoning = analysis.persona_reasoning or {}
                    summary = next(iter(reasoning.values()), "").split(".")[0][:80]
                    waypoint_summaries.append(summary)

            if not waypoint_scores:
                direction_results.append({
                    "direction": label,
                    "heading": heading,
                    "pano_id": branch_pano_id,
                    "last_pano_id": last_pano_id,
                    "scores": {},
                    "system1_reasoning": {},
                    "waypoint_scores": [],
                    "waypoint_summaries": [],
                    "score_trend": "unknown",
                    "avg_scores": {},
                    "visit_count": (visit_counts or {}).get(branch_pano_id, 0),
                })
                continue

            all_dims = set(k for s in waypoint_scores for k in s)
            avg_scores = {
                dim: sum(s.get(dim, 0) for s in waypoint_scores) / len(waypoint_scores)
                for dim in all_dims
            }

            if len(waypoint_scores) >= 2:
                first_avg = sum(waypoint_scores[0].values()) / len(waypoint_scores[0])
                last_avg = sum(waypoint_scores[-1].values()) / len(waypoint_scores[-1])
                delta = last_avg - first_avg
                trend = "improving" if delta > 0.5 else "declining" if delta < -0.5 else "stable"
            else:
                trend = "stable"

            direction_results.append({
                "direction": label,
                "heading": heading,
                "pano_id": branch_pano_id,
                "last_pano_id": last_pano_id,
                "scores": waypoint_scores[0],
                "system1_reasoning": {},
                "waypoint_scores": waypoint_scores,
                "waypoint_summaries": waypoint_summaries,
                "score_trend": trend,
                "avg_scores": avg_scores,
                "visit_count": (visit_counts or {}).get(branch_pano_id, 0),
            })

        direction_results.sort(key=lambda x: direction_labels.index(x["direction"]))

        ltm_patterns = {
            "snapshots": list(memory_manager._route_snapshots),
            "reasoning_episodes": list(memory_manager._route_reasoning_log),
        }
        stm_ctx = memory_manager.stm.get_context()

        def run_interpreter(dr: Dict) -> Dict:
            interp = self.persona_reasoner.interpreter.interpret_waypoint(
                waypoint_id=0,
                system1_scores=dr["avg_scores"] or dr["scores"],
                system1_reasoning=dr["system1_reasoning"],
                stm_context=stm_ctx,
                trigger_reason=None,
                personality=self.personality,
                dimension_ids=self.persona_reasoner.dimension_ids,
                dimensions=self.persona_reasoner.dimensions,
                waypoints_since_trigger=memory_manager._waypoints_since_trigger,
                ltm_patterns=ltm_patterns,
            )
            return {
                **dr,
                "interpretation": interp.get("text", ""),
                "key_concern": interp.get("key_concern"),
            }

        # Interpreters in parallel (sync → run_in_executor)
        loop = asyncio.get_event_loop()
        candidates = list(
            await asyncio.gather(
                *[loop.run_in_executor(None, run_interpreter, dr) for dr in direction_results]
            )
        )

        # Tag each candidate with angular deviation from wp_bearing
        if wp_bearing is not None:
            for c in candidates:
                c["route_deviation"] = round(abs(_angle_diff(wp_bearing, c["heading"])), 1)
        else:
            for c in candidates:
                c["route_deviation"] = None

        decision = self.persona_reasoner.decider.decide_branch(
            candidates=candidates,
            ltm_patterns=ltm_patterns,
            personality=self.personality,
            dimension_ids=self.persona_reasoner.dimension_ids,
            dimensions=self.persona_reasoner.dimensions,
            destination_context=destination_context,
            wp_bearing=wp_bearing,
        )

        self.logger.info(
            f"Branch decision: chosen={decision['chosen_direction']} "
            f"heading={decision['chosen_heading']} confidence={decision['confidence']}"
        )

        result = {**decision, "candidates": candidates}

        if output_dir:
            branch_log = {
                "branch_pano_id": branch_pano_id,
                "candidate_headings": candidate_headings,
                "lookahead_depth": lookahead_depth,
                "candidates": [
                    {
                        "direction": c["direction"],
                        "heading": c["heading"],
                        "scores": c["scores"],
                        "avg_scores": c.get("avg_scores", {}),
                        "score_trend": c.get("score_trend", "stable"),
                        "waypoint_scores": c.get("waypoint_scores", []),
                        "interpretation": c["interpretation"],
                        "key_concern": c["key_concern"],
                    }
                    for c in candidates
                ],
                "decision": decision,
            }
            log_path = Path(output_dir) / "branch_decisions.json"
            existing = []
            if log_path.exists():
                with open(log_path) as f:
                    existing = json.load(f)
            existing.append(branch_log)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)

        return result

    async def autonomous_walk(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float,
        max_steps: int = 60,
        arrival_threshold_m: float = 50.0,
        output_dir: Optional[Path] = None,
        step_callback=None,
        save_images: bool = False,
        lookahead_depth: int = 1,
    ) -> Dict[str, Any]:
        """Walk autonomously from start to destination.

        Each step:
        1. Fetch pano metadata → links
        2. Analyze current position (System 1)
        3. Check arrival
        4. Filter candidate headings toward destination
        5. branch_decision() if 2+ candidates, else take single heading
        6. Move to next pano
        """
        import asyncio
        import tempfile

        import httpx
        import imagehash
        from PIL import Image

        from src.agent.memory.memory_manager import MemoryManager
        from src.agent.system2.persona_reasoner import TriggerReason, ReasoningResult
        from demo.server import MapTilesSession

        loop = asyncio.get_event_loop()
        nav = MapTilesSession(settings.google_maps_api_key)

        memory_manager = MemoryManager(agent_id=f"walk_{int(time.time())}")
        if self.personality:
            memory_manager.set_agent_attributes(
                personality=self.personality,
                profile={"name": "Autonomous Walker"},
                status={"mode": "autonomous"},
            )
        self.continuous_analyzer.memory_manager = memory_manager

        walk_start_ts = time.time()
        walk_start_time = datetime.now().isoformat()

        visit_counts: Dict[str, int] = {}
        route_taken: List[Dict] = []
        arrived = False
        current_lat, current_lng = start_lat, start_lng
        prev_avg_score: Optional[float] = None
        distance_from_last_trigger: float = 0.0
        trigger_reason = None
        current_intended_heading: Optional[float] = None
        prev_image_hash = None  # imagehash.ImageHash | None

        images_dir: Optional[Path] = None
        if save_images and output_dir:
            images_dir = Path(output_dir) / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            self.logger.image_saving_enabled(images_dir)

        pano_id = await nav.coord_to_pano_id(start_lat, start_lng)
        if not pano_id:
            raise ValueError(f"No Street View pano at ({start_lat}, {start_lng})")

        self.logger.autonomous_walk_start(
            pano_id=pano_id,
            dest_lat=dest_lat,
            dest_lng=dest_lng,
            max_steps=max_steps,
        )

        # Directions API — get walking route waypoints for bearing guidance
        self.planner.reset()
        await self.planner.init_route(
            (start_lat, start_lng), (dest_lat, dest_lng),
            settings.google_maps_api_key,
        )
        if step_callback and self.planner.route_waypoints:
            await step_callback({
                "__event__": "planned_route",
                "waypoints": [[lat, lng] for lat, lng in self.planner.route_waypoints],
            })

        # Gating state — track last analyzed position for 3-signal gating
        last_analyzed_lat: Optional[float] = None
        last_analyzed_lng: Optional[float] = None
        last_analyzed_heading: Optional[float] = None
        last_analysis_scores: Dict = {}
        last_analysis_reasoning: Dict = {}
        analyzed_step_count: int = 0

        for step in range(max_steps):
            # Hard stop: same pano visited 5+ times
            if visit_counts.get(pano_id, 0) >= 5:
                self.logger.step_stuck(step=step, pano_id=pano_id)
                break

            # 1. Metadata
            try:
                metadata = await nav.get_metadata(pano_id)
            except Exception as e:
                self.logger.step_metadata_error(step=step, error=e)
                break

            current_lat = metadata.get("lat", current_lat)
            current_lng = metadata.get("lng", current_lng)
            links = metadata.get("links", [])

            if not links:
                self.logger.step_no_links(step=step)
                break

            # 2. Arrival check
            dist_m = geodesic(
                (current_lat, current_lng), (dest_lat, dest_lng)
            ).meters

            if dist_m <= arrival_threshold_m:
                arrived = True
                self.logger.step_arrived(step=step, dist_m=dist_m)
                break

            # 3. Navigation context
            nav = self.planner.get_navigation_context(
                current_lat, current_lng, dest_lat, dest_lng,
            )
            dest_bearing = nav["dest_bearing"]
            cardinal = nav["dest_cardinal"]
            dest_context = nav["dest_context"]

            # 4. Gating decision — should we analyze this pano?
            is_intersection = len(links) >= 3
            current_primary_heading = float(links[0].get("heading") or links[0].get("yawDeg") or 0)

            should_analyze = False
            heading_delta: float = 0.0
            dist_from_last: float = 0.0

            if step == 0 or is_intersection:
                should_analyze = True
            else:
                if last_analyzed_heading is not None:
                    heading_delta = abs(
                        ((current_primary_heading - last_analyzed_heading) + 180) % 360 - 180
                    )
                    if heading_delta > 30:
                        should_analyze = True
                if not should_analyze and last_analyzed_lat is not None:
                    dist_from_last = geodesic(
                        (last_analyzed_lat, last_analyzed_lng), (current_lat, current_lng)
                    ).meters
                    if dist_from_last > 30:
                        should_analyze = True

            if not should_analyze:
                self.logger.step_skip(
                    step=step,
                    heading_delta=heading_delta,
                    dist_from_last=dist_from_last,
                )
                # Skip: no image download, no VLM, no S2, no callback, no route_taken
                # Simple navigation: follow dest_bearing closest link
                visit_counts[pano_id] = visit_counts.get(pano_id, 0) + 1
                ref = current_intended_heading if current_intended_heading is not None else dest_bearing
                unvisited = [
                    l for l in links
                    if visit_counts.get(l.get("panoId") or l.get("id"), 0) == 0
                ]
                target = unvisited if unvisited else links
                next_link = _closest_link(target, ref)
                next_pano_id = next_link.get("panoId") or next_link.get("id")
                if not next_pano_id:
                    self.logger.step_no_next_pano(step=step)
                    break
                pano_id = next_pano_id
                continue

            # 5. Analyze current pano (System 1)
            analysis = None
            image_path = None
            saved_image_path = None
            phash_distance = None
            visual_change = True  # fallback default
            img_bytes = None
            try:
                sv_url = (
                    f"https://maps.googleapis.com/maps/api/streetview"
                    f"?pano={pano_id}&size=640x640"
                    f"&heading={links[0].get('heading', 0):.0f}"
                    f"&fov=90&pitch=0&key={settings.google_maps_api_key}"
                )
                async with httpx.AsyncClient(timeout=15.0) as client:
                    img_resp = await client.get(sv_url)
                    img_resp.raise_for_status()

                img_bytes = img_resp.content
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tmp.write(img_resp.content)
                tmp.flush()
                tmp.close()
                image_path = Path(tmp.name)

                try:
                    # pHash — only updated on analyzed steps
                    current_hash = imagehash.phash(Image.open(image_path))
                    phash_distance = (
                        float(current_hash - prev_image_hash)
                        if prev_image_hash is not None
                        else None
                    )
                    prev_image_hash = current_hash
                    visual_change = phash_distance is None or phash_distance > 30
                    self.logger.step_phash(
                        step=step,
                        phash_distance=phash_distance,
                        visual_change=visual_change,
                    )

                    # 이미지 저장 (save_images=True일 때만)
                    if images_dir is not None:
                        saved_image_path = images_dir / f"step_{step:03d}_pano_{pano_id[:8]}.jpg"
                        saved_image_path.write_bytes(img_resp.content)
                        self.logger.step_image_saved(
                            step=step, image_name=saved_image_path.name
                        )

                    # System 1 분석
                    analysis = await loop.run_in_executor(
                        None,
                        lambda: self.continuous_analyzer.analyze_waypoint(
                            waypoint_id=step,
                            image_path=image_path,
                            metadata={
                                "heading": links[0].get("heading", 0),
                                "lat": current_lat,
                                "lon": current_lng,
                            },
                            visual_change_detected=visual_change,
                            phash_distance=phash_distance,
                        ),
                    )
                finally:
                    image_path.unlink(missing_ok=True)

            except Exception as e:
                self.logger.step_image_analysis_error(step=step, error=e)
                analysis = None
                phash_distance = None
                visual_change = True

            # 6. Update memory (S2 trigger logic)
            trigger_reason = None
            if analysis:
                current_avg = (
                    sum(analysis.persona_scores.values()) / len(analysis.persona_scores)
                    if analysis.persona_scores else 0.0
                )
                score_delta = abs(current_avg - prev_avg_score) if prev_avg_score is not None else 0.0

                trigger_reason = self.persona_reasoner.should_trigger(
                    waypoint_id=step,
                    visual_change=False,
                    score_delta=score_delta,
                    distance_from_last=distance_from_last_trigger,
                )

                if is_intersection:
                    if trigger_reason is None:
                        trigger_reason = TriggerReason.INTERSECTION
                    self.logger.step_intersection_trigger(step=step)

                memory_manager.process_waypoint(
                    analysis,
                    triggered=trigger_reason is not None,
                    trigger_reason=trigger_reason,
                )

                if trigger_reason is not None:
                    distance_from_last_trigger = 0.0

                prev_avg_score = current_avg

            # 7. All links as candidates — exclude the direction we came from
            came_from_heading = (
                (current_intended_heading + 180) % 360
                if current_intended_heading is not None else None
            )
            candidate_headings = [
                float(l.get("heading") or l.get("yawDeg") or 0)
                for l in links
                if came_from_heading is None or
                   abs(((float(l.get("heading") or l.get("yawDeg") or 0)) - came_from_heading + 180) % 360 - 180) > 45
            ]
            if not candidate_headings:
                candidate_headings = [
                    float(l.get("heading") or l.get("yawDeg") or 0)
                    for l in links
                ]

            # 8. Choose direction
            chosen_heading = candidate_headings[0]
            recommendation = None
            step_confidence: Optional[float] = None
            branch_result: Optional[Dict[str, Any]] = None
            wp_bearing: Optional[float] = None
            nearest_wp_dist: Optional[float] = None
            urgency_tier: Optional[str] = None

            if (is_intersection or trigger_reason is not None) and analysis:
                try:
                    # Re-route if deviation exceeds threshold
                    urgency_tier = nav["urgency_tier"]
                    wp_bearing = nav["wp_bearing"]
                    nearest_wp_dist = nav["nearest_wp_dist_m"]

                    if nearest_wp_dist is not None and nearest_wp_dist >= self.planner.reroute_threshold_m:
                        self.logger.step_rerouting(
                            step=step, nearest_wp_dist=nearest_wp_dist
                        )
                        rerouted = await self.planner.check_and_reroute(
                            current_lat, current_lng, dest_lat, dest_lng,
                            settings.google_maps_api_key,
                        )
                        if rerouted:
                            nav = self.planner.get_navigation_context(
                                current_lat, current_lng, dest_lat, dest_lng,
                            )
                            wp_bearing = nav["wp_bearing"]
                            nearest_wp_dist = nav["nearest_wp_dist_m"]
                            dest_context = nav["dest_context"]
                            if step_callback and self.planner.route_waypoints:
                                await step_callback({
                                    "__event__": "planned_route",
                                    "waypoints": [[lat, lng] for lat, lng in self.planner.route_waypoints],
                                })

                    if step_callback:
                        await step_callback({
                            "__event__": "branch_start",
                            "lat": current_lat,
                            "lng": current_lng,
                            "pano_id": pano_id,
                            "candidates": [
                                {"direction": lbl, "heading": round(hdg, 1)}
                                for lbl, hdg in zip(
                                    list(__import__("string").ascii_uppercase[:len(candidate_headings)]),
                                    candidate_headings,
                                )
                            ],
                            "lookahead_depth": lookahead_depth,
                        })

                    async def _branch_progress(data: dict):
                        if step_callback:
                            await step_callback({"__event__": "branch_progress", **data})

                    branch_result = await self.branch_decision(
                        branch_pano_id=pano_id,
                        candidate_headings=candidate_headings,
                        memory_manager=memory_manager,
                        destination_context=dest_context,
                        visit_counts=visit_counts,
                        lookahead_depth=lookahead_depth,
                        progress_callback=_branch_progress,
                        wp_bearing=wp_bearing,
                    )
                    chosen_heading = branch_result["chosen_heading"]
                    current_intended_heading = chosen_heading
                    recommendation = branch_result.get("reason")
                    step_confidence = branch_result.get("confidence", 0.5)
                    self.logger.step_branch_choice(
                        step=step,
                        is_intersection=is_intersection,
                        chosen_direction=branch_result["chosen_direction"],
                        chosen_heading=chosen_heading,
                        confidence=float(step_confidence),
                    )
                    if analysis:
                        branch_reasoning_result = ReasoningResult(
                            waypoint_id=step,
                            trigger_reason=trigger_reason or TriggerReason.VISUAL_CHANGE,
                            interpretation=branch_result.get("reason", ""),
                            score_change_reason=None,
                            persona_divergence=None,
                            key_concern=None,
                            significance="high" if is_intersection else "medium",
                            avoid_recommendation=False,
                            decision_reason=branch_result.get("reason"),
                            prediction=None,
                            alternative_suggestion=None,
                            recommendation=branch_result.get("reason"),
                            confidence=branch_result.get("confidence", 0.5),
                            system1_scores=analysis.persona_scores,
                        )
                        memory_manager.update_with_system2_result(
                            step, branch_reasoning_result, episode=None,
                        )
                        self.logger.step_branch_logged(
                            step=step,
                            is_intersection=is_intersection,
                            episodes=len(memory_manager._route_reasoning_log),
                        )
                except Exception as e:
                    self.logger.step_branch_error(step=step, error=e)
                    ref = current_intended_heading if current_intended_heading is not None else dest_bearing
                    chosen_heading = float(
                        _closest_link(links, ref).get("heading")
                        or _closest_link(links, ref).get("yawDeg")
                        or 0
                    )

            else:
                # Straight + no S2 — maintain heading
                if current_intended_heading is not None:
                    chosen_heading = float(
                        _closest_link(links, current_intended_heading).get("heading")
                        or _closest_link(links, current_intended_heading).get("yawDeg")
                        or 0
                    )
                else:
                    # First analyzed step: initialise toward destination
                    chosen_heading = float(
                        _closest_link(links, dest_bearing).get("heading")
                        or _closest_link(links, dest_bearing).get("yawDeg")
                        or 0
                    )
                    current_intended_heading = chosen_heading
                    self.logger.step_initial_heading(step=step, chosen_heading=chosen_heading)

            # Update gating state after successful analysis
            last_analyzed_lat = current_lat
            last_analyzed_lng = current_lng
            last_analyzed_heading = chosen_heading
            last_analysis_scores = analysis.persona_scores if analysis else last_analysis_scores
            last_analysis_reasoning = analysis.persona_reasoning if analysis else last_analysis_reasoning
            analyzed_step_count += 1

            # 9. Find next pano — if branch explored lookahead, jump to its last pano
            branch_last_pano: Optional[str] = None
            if branch_result:
                chosen_candidate = next(
                    (c for c in branch_result.get("candidates", [])
                     if c["direction"] == branch_result["chosen_direction"]),
                    None,
                )
                if chosen_candidate:
                    branch_last_pano = chosen_candidate.get("last_pano_id")

            if branch_last_pano and branch_last_pano != pano_id:
                next_pano_id = branch_last_pano
                self.logger.step_lookahead_jump(
                    step=step,
                    next_pano_id=next_pano_id,
                    lookahead_depth=lookahead_depth,
                )
            else:
                if branch_result:
                    self.logger.step_lookahead_jump_skipped(
                        step=step,
                        branch_last_pano=branch_last_pano,
                        current_pano_id=pano_id,
                    )
                unvisited_links = [
                    l for l in links
                    if visit_counts.get(l.get("panoId") or l.get("id"), 0) == 0
                ]
                target_links = unvisited_links if unvisited_links else links
                next_link = _closest_link(target_links, chosen_heading)
                next_pano_id = next_link.get("panoId") or next_link.get("id")

            if not next_pano_id:
                self.logger.step_no_next_pano(step=step)
                break

            # 10. Record step
            step_result = {
                "step": step,
                "pano_id": pano_id,
                "lat": current_lat,
                "lng": current_lng,
                "heading": round(chosen_heading, 1),
                "dist_to_dest_m": round(dist_m, 1),
                "dest_bearing": round(dest_bearing, 1),
                "dest_cardinal": cardinal,
                "scores": analysis.persona_scores if analysis else {},
                "reasoning": analysis.persona_reasoning if analysis else {},
                "image_base64": base64.b64encode(img_bytes).decode("utf-8") if img_bytes else None,
                "recommendation": recommendation,
                "is_intersection": is_intersection,
                "branch_triggered": is_intersection or trigger_reason is not None,
                "intended_heading": round(current_intended_heading or chosen_heading, 1),
                "candidate_count": len(candidate_headings),
                "visit_count": visit_counts.get(pano_id, 0),
                "trigger_reason": trigger_reason.value if trigger_reason else None,
                "phash_distance": phash_distance,
                "visual_change": visual_change,
                "image_path": str(saved_image_path) if saved_image_path else None,
                "confidence": step_confidence,
                "wp_bearing": round(wp_bearing, 1) if (self.planner.route_waypoints and wp_bearing is not None) else None,
                "urgency_tier": urgency_tier if self.planner.route_waypoints else None,
                "route_deviation_m": round(nearest_wp_dist, 1) if (self.planner.route_waypoints and nearest_wp_dist is not None) else None,
                "branch_candidates": [
                    {
                        "direction": c["direction"],
                        "heading": round(c["heading"], 1),
                        "pano_id": c.get("pano_id"),
                        "scores": c.get("avg_scores") or c.get("scores", {}),
                        "avg_score": round(
                            sum(c["scores"].values()) / len(c["scores"])
                            if c.get("scores") else 0,
                            2,
                        ),
                        "trend": c.get("score_trend", "stable"),
                        "interpretation": c.get("interpretation", ""),
                        "key_concern": c.get("key_concern"),
                        "chosen": c["direction"] == (branch_result or {}).get("chosen_direction"),
                        "lookahead_depth": len(c.get("waypoint_scores") or []) or 1,
                    }
                    for c in ((branch_result or {}).get("candidates") or [])
                ],
            }
            route_taken.append(step_result)
            visit_counts[pano_id] = visit_counts.get(pano_id, 0) + 1
            distance_from_last_trigger += 20.0

            self.logger.step_progress(
                step=step,
                dist_m=dist_m,
                cardinal=cardinal,
                dest_bearing=dest_bearing,
                chosen_heading=chosen_heading,
                next_pano_id=next_pano_id,
            )

            if step_callback:
                await step_callback(step_result)

            pano_id = next_pano_id

        final_dist = geodesic(
            (current_lat, current_lng), (dest_lat, dest_lng)
        ).meters

        # Generate route-level report
        route_report = None
        try:
            all_scores = [s["scores"] for s in route_taken if s.get("scores")]
            dim_avgs: Dict[str, float] = {}
            if all_scores:
                for dim in list(all_scores[0].keys()):
                    vals = [s[dim] for s in all_scores if dim in s]
                    dim_avgs[dim] = round(sum(vals) / len(vals), 2) if vals else 0.0

            route_report = self.reporter.report_route(
                snapshots=memory_manager._route_snapshots,
                episodes=memory_manager._route_reasoning_log,
                route_stats={
                    "steps": len(route_taken),
                    "analyzed_steps": analyzed_step_count,
                    "skip_rate": round(1 - analyzed_step_count / max(step + 1, 1), 2),
                    "arrived": arrived,
                    "final_distance_m": round(final_dist, 1),
                    "duration_seconds": round(time.time() - walk_start_ts, 1),
                    "dimension_avgs": dim_avgs,
                },
                planner_summary=self.planner.get_summary(),
                personality=self.personality,
            )
            self.logger.route_report_generated(
                recommendation=route_report.get("recommendation", "?")
            )
        except Exception as e:
            self.logger.route_report_failed(error=e)

        # Send route_report via SSE before "complete"
        if step_callback and route_report:
            await step_callback({"__event__": "route_report", **route_report})

        result = {
            "arrived": arrived,
            "steps": len(route_taken),
            "final_distance_m": round(final_dist, 1),
            "route_taken": route_taken,
            "persona": getattr(self.personality, "name", "objective"),
            "analyzed_steps": analyzed_step_count,
            "total_steps": step + 1,
            "skip_rate": round(1 - analyzed_step_count / max(step + 1, 1), 2),
            "planner_summary": self.planner.get_summary(),
            "route_report": route_report,
            "memory_debug": {
                "snapshots": memory_manager._route_snapshots,
                "episodes": memory_manager._route_reasoning_log,
            },
        }

        if output_dir:
            from src.utils.walk_output_writer import save_walk_outputs

            run_params = {
                "start_lat": start_lat,
                "start_lng": start_lng,
                "dest_lat": dest_lat,
                "dest_lng": dest_lng,
                "persona": getattr(self.personality, "name", "objective"),
                "max_steps": max_steps,
                "lookahead_depth": lookahead_depth,
                "save_images": save_images,
                "start_time": walk_start_time,
                "duration_seconds": round(time.time() - walk_start_ts, 1),
            }

            saved = save_walk_outputs(
                output_dir=Path(output_dir),
                walk_result=result,
                run_params=run_params,
                framework_id=self.framework_id,
            )
            self.logger.walk_outputs_saved(
                output_dir=output_dir,
                files=list(saved.keys()),
            )

        self.logger.walk_complete(
            arrived=arrived,
            steps=len(route_taken),
            analyzed_step_count=analyzed_step_count,
            total_steps=step + 1,
            final_dist=final_dist,
        )
        return result
