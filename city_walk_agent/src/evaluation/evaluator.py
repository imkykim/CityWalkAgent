"""VLM evaluation orchestrator for CityWalkAgent."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.config import DEFAULT_MAX_CONCURRENT
from src.utils.logging import get_logger

from .batch_processor import BatchProcessor, EvaluationTask
from .prompt_builder import PromptBuilder
from .response_parser import ResponseParser
from .vlm_client import VLMClient, VLMConfig

if TYPE_CHECKING:
    from src.utils.data_models import Route


class Evaluator:
    """
    Main evaluation orchestrator

    High-level interface for evaluating walking routes using VLM frameworks.
    Coordinates VLM calls, prompt generation, and result aggregation.

    Usage:
        ```python
        # Initialize evaluator
        config = VLMConfig(
            api_key="your-qwen-key",
            model="Qwen3-VL-30B-A3B-Instruct-FP8",
            api_url="your-qwen-api-url"
        )
        evaluator = Evaluator(config, framework)

        # Evaluate images
        results = evaluator.evaluate_images(image_paths)
        ```
    """

    def __init__(
        self,
        vlm_config: VLMConfig,
        framework: Dict[str, Any],
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        show_progress: bool = True
    ) -> None:
        """
        Initialize evaluator

        Args:
            vlm_config: VLM client configuration
            framework: Framework configuration dict (from JSON)
            max_concurrent: Maximum concurrent API calls
            show_progress: Show progress bar
        """
        self.framework = framework
        self.framework_id = framework["framework_id"]
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components
        self.vlm_client = VLMClient(vlm_config)
        self.prompt_builder = PromptBuilder(framework)
        self.response_parser = ResponseParser()
        self.batch_processor = BatchProcessor(max_concurrent, show_progress)

        # Build prompts for all dimensions
        self.prompts = self.prompt_builder.build_all_prompts(language="cn")
        self.logger.info(
            "Evaluator initialized",
            framework_id=self.framework_id,
            dimensions=len(self.prompts),
            max_concurrent=max_concurrent
        )

    def evaluate_image(
        self,
        image_path: str,
        dimensions: Optional[List[str]] = None,
        previous_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate single image across specified dimensions

        Args:
            image_path: Path to street view image
            dimensions: List of dimension IDs to evaluate (None = all)
            previous_context: Optional dict containing previous waypoint info:
                - waypoint_id: int
                - scores: Dict[str, float]
                - reasoning: Dict[str, str]
                - phash_distance: float
                - visual_change_detected: bool

        Returns:
            List of evaluation results (one per dimension)
        """
        if dimensions is None:
            dimensions = self.prompt_builder.get_all_dimension_ids()

        results = []
        image_id = Path(image_path).stem

        for dimension_id in dimensions:
            if dimension_id not in self.prompts:
                self.logger.warning(
                    "Dimension not in framework",
                    framework_id=self.framework_id,
                    dimension_id=dimension_id
                )
                continue

            base_prompt = self.prompts[dimension_id]

            # Prepend context if available
            if previous_context:
                context_text = self._format_context_for_prompt(previous_context)
                prompt = context_text + base_prompt
            else:
                prompt = base_prompt

            # Call VLM
            response = self.vlm_client.call_vlm(prompt, image_path)

            if response and "content" in response:
                parsed = self.response_parser.parse_response(
                    response["content"],
                    dimension_id
                )

                if parsed:
                    dimension_meta = self.prompt_builder.get_dimension_metadata(dimension_id)

                    result = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "framework_id": self.framework_id,
                        "dimension_id": dimension_id,
                        "dimension_name": dimension_meta.get("name_cn", dimension_id),
                        "score": parsed["score"],
                        "reasoning": parsed["reasoning"],
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)

        return results

    def evaluate_images(
        self,
        image_paths: List[str],
        dimensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple images in batch

        Args:
            image_paths: List of image file paths
            dimensions: List of dimension IDs to evaluate (None = all)

        Returns:
            List of all evaluation results
        """
        if dimensions is None:
            dimensions = self.prompt_builder.get_all_dimension_ids()

        # Filter prompts to requested dimensions
        filtered_prompts = {
            dim_id: self.prompts[dim_id]
            for dim_id in dimensions
            if dim_id in self.prompts
        }

        if not filtered_prompts:
            self.logger.warning(
                "No prompts available for requested dimensions",
                requested_dimensions=dimensions
            )
            return []

        # Create evaluation tasks
        tasks = BatchProcessor.create_tasks_for_images(
            image_paths,
            self.framework,
            filtered_prompts
        )

        self.logger.info(
            "Evaluating image batch",
            images=len(image_paths),
            dimensions=len(filtered_prompts),
            total_tasks=len(tasks)
        )

        batch_result = self.batch_processor.process_batch(
            tasks,
            self.vlm_client.call_vlm_async,
            self.response_parser
        )

        self.logger.info(
            "Batch complete",
            successful=batch_result.successful,
            failed=batch_result.failed,
            total_tasks=batch_result.total_tasks,
            total_time=batch_result.total_time,
            average_task_time=batch_result.avg_time_per_task
        )

        # Enrich results with dimension metadata
        enriched_results = []
        for result in batch_result.results:
            dimension_meta = self.prompt_builder.get_dimension_metadata(
                result["dimension_id"]
            )

            enriched = {
                **result,
                "dimension_name": dimension_meta.get("name_cn", result["dimension_id"]),
                "timestamp": datetime.now().isoformat()
            }
            enriched_results.append(enriched)

        return enriched_results

    def evaluate_route(
        self,
        route: "Route",
        dimensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate complete route

        Args:
            route: Route object with waypoints
            dimensions: List of dimension IDs to evaluate

        Returns:
            List of evaluation results for all waypoints
        """
        # Collect image paths from waypoints
        image_paths = []
        for waypoint in route.waypoints:
            if waypoint.image_path and Path(waypoint.image_path).exists():
                image_paths.append(waypoint.image_path)

        if not image_paths:
            self.logger.warning(
                "No images available for route",
                route_id=route.route_id,
                total_waypoints=len(route.waypoints)
            )
            return []

        self.logger.info(
            "Evaluating route images",
            route_id=route.route_id,
            images=len(image_paths),
            waypoints=len(route.waypoints)
        )

        return self.evaluate_images(image_paths, dimensions)

    def evaluate_with_comparison(
        self,
        previous_image_path: Path,
        current_image_path: Path,
        previous_context: Dict[str, Any],
        dimensions: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate current waypoint by comparing with previous waypoint visually.

        Uses multi-image VLM call to directly compare two consecutive waypoints.
        More expensive but provides better insight on transitions.

        Args:
            previous_image_path: Path to previous waypoint image
            current_image_path: Path to current waypoint image
            previous_context: Previous waypoint metadata
            dimensions: Dimensions to evaluate (default: all)

        Returns:
            Dict mapping dimension_id to {score, reasoning, comparison_notes}
        """
        if dimensions is None:
            dimensions = self.prompt_builder.get_all_dimension_ids()

        results = {}

        for dimension_id in dimensions:
            if dimension_id not in self.prompts:
                continue

            # Build comparison prompt
            base_prompt = self.prompts[dimension_id]
            dimension_meta = self.prompt_builder.get_dimension_metadata(dimension_id)
            dimension_name = dimension_meta.get("name_cn", dimension_id)

            prev_score = previous_context.get('scores', {}).get(dimension_id, 'N/A')
            prev_score_str = f"{prev_score:.1f}" if isinstance(prev_score, (int, float)) else str(prev_score)

            comparison_prompt = f"""
# Sequential Waypoint Comparison - {dimension_name}

You are evaluating two consecutive waypoints along a walking route.

**Previous Waypoint (ID: {previous_context.get('waypoint_id', 'N/A')})**
- Previous {dimension_name} score: {prev_score_str}/10
- Visual change detected: {'Yes' if previous_context.get('visual_change_detected') else 'No'}
- pHash distance: {previous_context.get('phash_distance', 'N/A'):.1f}

**Images:**
Image 1 (Previous): First image below
Image 2 (Current): Second image below

**Evaluation Task:**
1. Compare the two images directly
2. Identify what changed between previous and current location
3. Determine if the change improves or worsens the {dimension_name}
4. Assign a score to the CURRENT waypoint (Image 2)

{base_prompt}

**Additional Focus:**
- Did a sudden barrier appear?
- Is this a gradual transition or abrupt change?
- How does the current location compare to what came before?
"""

            # Multi-image VLM call
            response = self.vlm_client.call_vlm_multi_image(
                prompt=comparison_prompt,
                image_paths=[str(previous_image_path), str(current_image_path)]
            )

            if response and "content" in response:
                parsed = self.response_parser.parse_response(
                    response["content"],
                    dimension_id
                )

                if parsed:
                    results[dimension_id] = {
                        "score": parsed["score"],
                        "reasoning": parsed["reasoning"],
                        "comparison_mode": True,
                        "previous_score": previous_context.get('scores', {}).get(dimension_id)
                    }

        return results

    def _format_context_for_prompt(
        self,
        context: Dict[str, Any]
    ) -> str:
        """
        Format previous waypoint context for prompt inclusion.

        Args:
            context: Previous waypoint information

        Returns:
            Formatted string to prepend to evaluation prompt
        """
        if not context:
            return ""

        scores = context.get("scores", {})
        if scores:
            scores_str = ", ".join([
                f"{dim}: {score:.1f}/10"
                for dim, score in scores.items()
            ])
        else:
            scores_str = "N/A"

        phash_info = ""
        if context.get("phash_distance") is not None:
            phash_info = f"\n- Visual change: {context['phash_distance']:.1f} (perceptual hash distance)"

        # Get a summary from reasoning if available
        reasoning = context.get("reasoning", {})
        if reasoning:
            # Use the first reasoning entry as summary
            summary = next(iter(reasoning.values()), "N/A")
        else:
            summary = "N/A"

        context_text = f"""
## Previous Waypoint Context (Waypoint {context.get('waypoint_id', 'N/A')})

This evaluation is part of a sequential route analysis. Consider the previous location:

- Previous scores: {scores_str}
- Summary: {summary}{phash_info}

**Evaluation Instructions:**
- Compare the current waypoint to the previous one
- Identify significant changes (improvements or deteriorations)
- Consider how the walking experience has evolved
- Note any sudden transitions or barriers

---

"""
        return context_text

    def get_statistics(self) -> Dict[str, Any]:
        """Return evaluation and framework statistics."""
        return {
            "framework": self.prompt_builder.get_framework_info(),
            "vlm_stats": self.vlm_client.get_stats(),
            "dimensions_available": len(self.prompts)
        }
