"""
Main evaluator orchestrating VLM-based walkability assessment

This is the high-level interface for evaluating routes using VLM frameworks
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .vlm_client import VLMClient, VLMConfig
from .prompt_builder import PromptBuilder
from .response_parser import ResponseParser
from .batch_processor import BatchProcessor, EvaluationTask


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
        max_concurrent: int = 5,
        show_progress: bool = True
    ):
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

        # Initialize components
        self.vlm_client = VLMClient(vlm_config)
        self.prompt_builder = PromptBuilder(framework)
        self.response_parser = ResponseParser()
        self.batch_processor = BatchProcessor(max_concurrent, show_progress)

        # Build prompts for all dimensions
        self.prompts = self.prompt_builder.build_all_prompts(language="cn")

    def evaluate_image(
        self,
        image_path: str,
        dimensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate single image across specified dimensions

        Args:
            image_path: Path to street view image
            dimensions: List of dimension IDs to evaluate (None = all)

        Returns:
            List of evaluation results (one per dimension)
        """
        if dimensions is None:
            dimensions = self.prompt_builder.get_all_dimension_ids()

        results = []
        image_id = Path(image_path).stem

        for dimension_id in dimensions:
            if dimension_id not in self.prompts:
                print(f"Warning: Dimension '{dimension_id}' not found in framework")
                continue

            prompt = self.prompts[dimension_id]

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

        # Create evaluation tasks
        tasks = BatchProcessor.create_tasks_for_images(
            image_paths,
            self.framework,
            filtered_prompts
        )

        print(f"\n🎯 Evaluating {len(image_paths)} images with {self.framework_id}")
        print(f"   Dimensions: {len(filtered_prompts)}")
        print(f"   Total evaluations: {len(tasks)}")

        # Process batch
        batch_result = self.batch_processor.process_batch(
            tasks,
            self.vlm_client.call_vlm_async,
            self.response_parser
        )

        print(f"\n✅ Batch complete:")
        print(f"   Successful: {batch_result.successful}/{batch_result.total_tasks}")
        print(f"   Failed: {batch_result.failed}")
        print(f"   Total time: {batch_result.total_time:.1f}s")
        print(f"   Avg per task: {batch_result.avg_time_per_task:.2f}s")

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
        route: Any,  # Route object from data_models
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
            print(f"Warning: No images found for route {route.route_id}")
            return []

        print(f"\n📍 Evaluating route: {route.route_id}")
        print(f"   Waypoints with images: {len(image_paths)}/{len(route.waypoints)}")

        return self.evaluate_images(image_paths, dimensions)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get evaluation statistics

        Returns:
            Dict with VLM client stats and framework info
        """
        return {
            "framework": self.prompt_builder.get_framework_info(),
            "vlm_stats": self.vlm_client.get_stats(),
            "dimensions_available": len(self.prompts)
        }

    @staticmethod
    def create_from_framework_file(
        framework_path: str,
        vlm_config: VLMConfig,
        max_concurrent: int = 5
    ) -> "Evaluator":
        """
        Create evaluator from framework JSON file

        Args:
            framework_path: Path to framework JSON file
            vlm_config: VLM client configuration
            max_concurrent: Maximum concurrent API calls

        Returns:
            Initialized Evaluator instance
        """
        import json

        with open(framework_path, 'r', encoding='utf-8') as f:
            framework = json.load(f)

        return Evaluator(vlm_config, framework, max_concurrent)

    @staticmethod
    def create_from_framework_id(
        framework_id: str,
        vlm_config: VLMConfig,
        frameworks_dir: Optional[Path] = None,
        max_concurrent: int = 5
    ) -> "Evaluator":
        """
        Create evaluator by framework ID

        Args:
            framework_id: Framework identifier (e.g., "sagai_2025")
            vlm_config: VLM client configuration
            frameworks_dir: Directory containing framework JSON files
            max_concurrent: Maximum concurrent API calls

        Returns:
            Initialized Evaluator instance

        Raises:
            FileNotFoundError: If framework file not found
        """
        import json

        if frameworks_dir is None:
            # Default to project's frameworks directory
            frameworks_dir = Path(__file__).parent.parent.parent / "experiments" / "configs" / "frameworks"

        framework_path = frameworks_dir / f"{framework_id}.json"

        if not framework_path.exists():
            raise FileNotFoundError(
                f"Framework '{framework_id}' not found at {framework_path}"
            )

        with open(framework_path, 'r', encoding='utf-8') as f:
            framework = json.load(f)

        return Evaluator(vlm_config, framework, max_concurrent)
