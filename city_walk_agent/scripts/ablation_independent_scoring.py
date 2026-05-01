#!/usr/bin/env python3
"""
ablation_independent_scoring.py — Independent (context-free) VLM scoring baseline

Re-scores saved Street View images from existing experiment runs WITHOUT any context:
no STM, no LTM, no previous_context. Each image is evaluated independently,
simulating how traditional walkability studies (Walk Score, Place Pulse) assess locations.

Usage:
    python scripts/ablation_independent_scoring.py \\
        --input-dir outputs/test_runs/20260427_120000/route_homebuyer_20260427/ \\
        --persona homebuyer

    # Score all experiments under a run dir:
    python scripts/ablation_independent_scoring.py \\
        --input-dir outputs/test_runs/20260427_120000/

    # Override framework and concurrency:
    python scripts/ablation_independent_scoring.py \\
        --input-dir outputs/test_runs/20260427_120000/ \\
        --framework-id place_pulse_2.0 \\
        --max-concurrent 4 \\
        --output-suffix independent

Output (per experiment dir):
    <experiment_dir>/ablation_independent/
        independent_scores.json
        independent_score_timeline.csv
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import csv
import json
import sys
import time
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.core import settings, load_framework
from src.core.evaluation.evaluator import Evaluator
from src.core.evaluation.vlm_client import VLMClient, VLMConfig
from src.agent.config.personalities import get_enhanced_personality

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_FRAMEWORK_ID = "place_pulse_2.0"
DEFAULT_MAX_CONCURRENT = 8
DEFAULT_OUTPUT_SUFFIX = "independent"

# Map display names → personality IDs (run_metadata stores display names)
_PERSONA_NAME_TO_ID: Dict[str, str] = {
    "prospective homebuyer": "homebuyer",
    "urban runner": "runner",
    "parent with young children": "parent_with_kids",
    "street photographer": "photographer",
    "tourist": "tourist",
    "elderly walker": "elderly_walker",
    "elderly": "elderly",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_experiment_dir(path: Path) -> bool:
    """Return True if path looks like a single experiment output directory."""
    return (path / "walk_log.json").exists()


def _find_experiment_dirs(input_dir: Path) -> List[Path]:
    """Return all experiment directories under input_dir (or input_dir itself)."""
    if _is_experiment_dir(input_dir):
        return [input_dir]
    # Look one level deep (run_dir / experiment_dir)
    dirs = [d for d in sorted(input_dir.iterdir()) if d.is_dir() and _is_experiment_dir(d)]
    return dirs


def _load_walk_log(exp_dir: Path) -> Optional[List[Dict[str, Any]]]:
    walk_log_path = exp_dir / "walk_log.json"
    if not walk_log_path.exists():
        return None
    with walk_log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # walk_log.json can be a list or dict with "route_taken" key
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("route_taken", data.get("steps", []))
    return None


def _load_run_metadata(exp_dir: Path) -> Dict[str, Any]:
    meta_path = exp_dir / "run_metadata.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_persona_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    """Extract persona ID from run_metadata.json (handles both ID and display name)."""
    params = metadata.get("parameters", metadata)
    raw = params.get("persona") or params.get("personality_id")
    if raw is None:
        return None
    # If it's already a known ID, return as-is
    from src.agent.config.personalities import ENHANCED_PERSONALITIES
    if raw in ENHANCED_PERSONALITIES:
        return raw
    # Try to map display name → ID
    return _PERSONA_NAME_TO_ID.get(raw.lower(), raw)


def _resolve_image_path(
    step: Dict[str, Any],
    step_idx: int,
    exp_dir: Path,
) -> Optional[Path]:
    """Find image path for a walk step via multiple fallback strategies."""
    # 1. Direct image_path field in step
    raw = step.get("image_path")
    if raw:
        candidate = Path(raw)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        # relative to exp_dir
        candidate = exp_dir / raw
        if candidate.exists():
            return candidate

    # 2. Glob by step index in images/ subdir
    images_dir = exp_dir / "images"
    if images_dir.exists():
        pattern = str(images_dir / f"step_{step_idx:03d}_pano_*.jpg")
        matches = glob(pattern)
        if matches:
            return Path(matches[0])

    return None


def _step_has_scores(step: Dict[str, Any]) -> bool:
    """Return True if original step was analyzed and has non-empty scores."""
    if not step.get("analyzed", False):
        return False
    scores = step.get("scores", {})
    return bool(scores)


def _pano_id_from_path(image_path: Path) -> str:
    """Extract pano_id from filename like step_000_pano_ABCDEF.jpg."""
    stem = image_path.stem  # e.g. step_000_pano_ABCDEF
    parts = stem.split("_pano_")
    return parts[1] if len(parts) == 2 else stem


def _make_evaluator(framework_id: str) -> Evaluator:
    framework = load_framework(framework_id)
    vlm_config = VLMConfig(
        api_key=settings.vlm_api_key,
        model=settings.vlm_model,
        api_url=settings.vlm_api_url,
    )
    return Evaluator(vlm_config, framework, show_progress=False)


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring logic
# ─────────────────────────────────────────────────────────────────────────────

def _score_image_sync(
    evaluator: Evaluator,
    persona_config,
    image_path: Path,
) -> Optional[Dict[str, Any]]:
    """Score one image with no context. Returns raw results list or None."""
    try:
        results = evaluator.evaluate_image(
            image_path=str(image_path),
            previous_context=None,        # ← THE KEY: no context
            persona=persona_config,
            evaluation_mode="persona",
        )
        return results
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠  VLM call failed for {image_path.name}: {exc}")
        return None


async def _score_all_images(
    tasks: List[Tuple[int, str, Path]],  # (step_idx, pano_id, image_path)
    evaluator: Evaluator,
    persona_config,
    max_concurrent: int,
) -> List[Optional[List[Dict[str, Any]]]]:
    """Run all image scoring tasks with bounded concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)

    results: List[Optional[List[Dict[str, Any]]]] = [None] * len(tasks)
    completed = [0]
    total = len(tasks)
    start_times: List[float] = []

    async def score_one(idx: int, task_idx: int, step_idx: int, pano_id: str, image_path: Path):
        async with semaphore:
            t0 = time.monotonic()
            result = await loop.run_in_executor(
                executor,
                _score_image_sync,
                evaluator,
                persona_config,
                image_path,
            )
            elapsed = time.monotonic() - t0
            start_times.append(elapsed)
            results[task_idx] = result
            completed[0] += 1
            avg = sum(start_times) / len(start_times) if start_times else 0
            bar_len = 28
            filled = int(bar_len * completed[0] / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"\r   Scoring... [{bar}] {completed[0]}/{total} (avg {avg:.1f}s/image)",
                end="",
                flush=True,
            )

    coros = [
        score_one(i, i, step_idx, pano_id, image_path)
        for i, (step_idx, pano_id, image_path) in enumerate(tasks)
    ]
    await asyncio.gather(*coros)
    executor.shutdown(wait=False)
    print()  # newline after progress bar
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-experiment processing
# ─────────────────────────────────────────────────────────────────────────────

def process_experiment(
    exp_dir: Path,
    persona_name: Optional[str],
    framework_id: str,
    max_concurrent: int,
    output_suffix: str,
) -> bool:
    """Score images for one experiment directory. Returns True on success."""
    print(f"\n🔬 Independent Scoring Run")
    print(f"   Experiment: {exp_dir.name}")

    # --- Load walk log ---
    walk_log = _load_walk_log(exp_dir)
    if walk_log is None:
        print(f"   ❌ walk_log.json not found in {exp_dir}")
        return False

    # --- Resolve persona ---
    if not persona_name:
        metadata = _load_run_metadata(exp_dir)
        persona_name = _get_persona_from_metadata(metadata)
    if not persona_name:
        print(f"   ❌ Could not determine persona. Pass --persona explicitly.")
        return False
    print(f"   Persona: {persona_name}")

    # --- Load persona config ---
    try:
        persona_config = get_enhanced_personality(persona_name)
    except ValueError as exc:
        print(f"   ❌ {exc}")
        return False

    # --- Check images dir ---
    images_dir = exp_dir / "images"
    if not images_dir.exists():
        print(
            f"   ❌ No images/ directory found.\n"
            f"      Re-run experiments with --save-images flag first."
        )
        return False

    # --- Gather steps to score ---
    tasks: List[Tuple[int, str, Path]] = []
    skipped_no_image = 0
    skipped_no_scores = 0

    for step in walk_log:
        step_idx = step.get("step", step.get("waypoint_id", 0))
        if not _step_has_scores(step):
            skipped_no_scores += 1
            continue
        image_path = _resolve_image_path(step, step_idx, exp_dir)
        if image_path is None:
            skipped_no_image += 1
            continue
        pano_id = _pano_id_from_path(image_path)
        tasks.append((step_idx, pano_id, image_path))

    print(f"   Images found: {len(tasks)}")
    if skipped_no_scores:
        print(f"   Skipped (no original scores): {skipped_no_scores}")
    if skipped_no_image:
        print(f"   Skipped (image file missing): {skipped_no_image}")

    if not tasks:
        print(f"   ❌ No scoreable images found.")
        return False

    # --- Initialize evaluator ---
    try:
        evaluator = _make_evaluator(framework_id)
    except Exception as exc:
        print(f"   ❌ Failed to initialize evaluator: {exc}")
        return False

    # --- Score all images ---
    print()
    raw_results = asyncio.run(
        _score_all_images(tasks, evaluator, persona_config, max_concurrent)
    )

    # --- Build output records ---
    scored_records: List[Dict[str, Any]] = []
    n_success = 0
    n_failed = 0

    for (step_idx, pano_id, image_path), result_list in zip(tasks, raw_results):
        if result_list is None:
            n_failed += 1
            continue
        n_success += 1
        scores: Dict[str, float] = {}
        reasoning: Dict[str, str] = {}
        for r in result_list:
            dim = r.get("dimension_id", "")
            if dim:
                scores[dim] = r.get("score", 0.0)
                reasoning[dim] = r.get("reasoning", "")
        scored_records.append({
            "step": step_idx,
            "pano_id": pano_id,
            "image_path": str(image_path.relative_to(exp_dir)),
            "scores": scores,
            "reasoning": reasoning,
        })

    # Sort by step
    scored_records.sort(key=lambda r: r["step"])

    print(f"   ✅ Complete: {n_success}/{len(tasks)} successful, {n_failed} failed")

    # --- Write output ---
    out_dir = exp_dir / f"ablation_{output_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. independent_scores.json
    experiment_id = exp_dir.name
    output_json = {
        "experiment_id": experiment_id,
        "persona": persona_name,
        "framework_id": framework_id,
        "scoring_mode": "independent",
        "n_images_scored": n_success,
        "scores": scored_records,
    }
    json_path = out_dir / "independent_scores.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    # 2. independent_score_timeline.csv
    csv_path = out_dir / "independent_score_timeline.csv"
    if scored_records:
        all_dims = sorted({d for r in scored_records for d in r["scores"]})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "pano_id"] + all_dims)
            for rec in scored_records:
                row = [rec["step"], rec["pano_id"]]
                for dim in all_dims:
                    row.append(rec["scores"].get(dim, ""))
                writer.writerow(row)

    print(f"   📄 Saved to: {out_dir}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-score saved Street View images without context (ablation baseline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Experiment dir or run dir containing experiment subdirs",
    )
    parser.add_argument(
        "--persona", type=str, default=None,
        help="Persona name (auto-detected from run_metadata.json if omitted)",
    )
    parser.add_argument(
        "--framework-id", type=str, default=DEFAULT_FRAMEWORK_ID,
        dest="framework_id",
        help=f"Evaluation framework ID (default: {DEFAULT_FRAMEWORK_ID})",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
        dest="max_concurrent",
        help=f"Max concurrent VLM calls (default: {DEFAULT_MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--output-suffix", type=str, default=DEFAULT_OUTPUT_SUFFIX,
        dest="output_suffix",
        help=f"Output subdirectory suffix: ablation_<suffix> (default: {DEFAULT_OUTPUT_SUFFIX})",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1

    exp_dirs = _find_experiment_dirs(input_dir)
    if not exp_dirs:
        print(f"❌ No experiment directories found under {input_dir}")
        return 1

    print(f"Found {len(exp_dirs)} experiment(s) to process.")

    n_ok = 0
    n_fail = 0
    for exp_dir in exp_dirs:
        ok = process_experiment(
            exp_dir=exp_dir,
            persona_name=args.persona,
            framework_id=args.framework_id,
            max_concurrent=args.max_concurrent,
            output_suffix=args.output_suffix,
        )
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n{'='*60}")
    print(f"Done. {n_ok} succeeded, {n_fail} failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
