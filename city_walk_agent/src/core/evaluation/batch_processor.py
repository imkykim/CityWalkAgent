"""
Batch processor for efficient VLM evaluation

Handles parallel and batched evaluation of multiple images
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.utils.logging import get_logger


@dataclass
class EvaluationTask:
    """Single evaluation task"""

    image_path: str
    image_id: str
    dimension_id: str
    prompt: str
    framework_id: str


@dataclass
class BatchResult:
    """Results from batch processing"""

    results: List[Dict[str, Any]]
    total_tasks: int
    successful: int
    failed: int
    total_time: float
    avg_time_per_task: float


class BatchProcessor:
    """
    Efficient batch processing for VLM evaluations

    Features:
    - Parallel async processing
    - Progress tracking
    - Error recovery
    - Rate limiting
    - Result aggregation
    """

    def __init__(self, max_concurrent: int = 10, show_progress: bool = True) -> None:
        """
        Initialize batch processor

        Args:
            max_concurrent: Maximum concurrent API calls
            show_progress: Show progress bar (requires tqdm)
        """
        self.max_concurrent = max_concurrent
        self.show_progress = show_progress and HAS_TQDM
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    async def process_batch_async(
        self, tasks: List[EvaluationTask], vlm_call_func: Callable, response_parser: Any
    ) -> BatchResult:
        """
        Process batch of evaluation tasks asynchronously

        Args:
            tasks: List of evaluation tasks
            vlm_call_func: Async function to call VLM (e.g., vlm_client.call_vlm_async)
            response_parser: ResponseParser instance

        Returns:
            BatchResult with all results
        """
        start_time = datetime.now()

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Prepare task coroutines
        async def process_task(task: EvaluationTask) -> Dict[str, Any]:
            async with semaphore:
                error_message: Optional[str] = None
                try:
                    # Call VLM
                    response = await vlm_call_func(task.prompt, task.image_path)

                    if response and "content" in response:
                        # Parse response
                        parsed = response_parser.parse_response(
                            response["content"], task.dimension_id
                        )

                        if parsed:
                            return {
                                "image_id": task.image_id,
                                "framework_id": task.framework_id,
                                "dimension_id": task.dimension_id,
                                "score": parsed["score"],
                                "reasoning": parsed["reasoning"],
                                "success": True,
                                "usage": response.get("usage", {}),
                            }

                except Exception as error:
                    self.logger.error(
                        "Batch task failed",
                        image_id=task.image_id,
                        dimension_id=task.dimension_id,
                        error=str(error),
                    )
                    error_message = str(error)

                # Fallback result on failure
                return {
                    "image_id": task.image_id,
                    "framework_id": task.framework_id,
                    "dimension_id": task.dimension_id,
                    "score": 5.0,
                    "reasoning": "Evaluation failed",
                    "success": False,
                    "error": error_message or "Unknown error",
                }

        # Execute all tasks
        if self.show_progress:
            # With progress bar
            results = []
            with tqdm(total=len(tasks), desc="Evaluating") as pbar:
                for coro in asyncio.as_completed([process_task(t) for t in tasks]):
                    result = await coro
                    results.append(result)
                    pbar.update(1)
        else:
            # Without progress bar
            results = await asyncio.gather(*[process_task(t) for t in tasks])

        # Calculate statistics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful

        return BatchResult(
            results=results,
            total_tasks=len(tasks),
            successful=successful,
            failed=failed,
            total_time=total_time,
            avg_time_per_task=total_time / len(tasks) if tasks else 0,
        )

    def process_batch(
        self, tasks: List[EvaluationTask], vlm_call_func: Callable, response_parser: Any
    ) -> BatchResult:
        """
        Synchronous wrapper for batch processing

        Args:
            tasks: List of evaluation tasks
            vlm_call_func: Async function to call VLM
            response_parser: ResponseParser instance

        Returns:
            BatchResult with all results
        """
        return asyncio.run(
            self.process_batch_async(tasks, vlm_call_func, response_parser)
        )

    @staticmethod
    def create_tasks_for_images(
        image_paths: List[str], framework: Dict[str, Any], prompts: Dict[str, str]
    ) -> List[EvaluationTask]:
        """
        Create evaluation tasks for multiple images

        Args:
            image_paths: List of image file paths
            framework: Framework configuration dict
            prompts: Dict mapping dimension_id to prompt text

        Returns:
            List of EvaluationTask objects
        """
        tasks = []
        framework_id = framework["framework_id"]

        for image_path in image_paths:
            image_id = Path(image_path).stem

            for dimension_id, prompt in prompts.items():
                task = EvaluationTask(
                    image_path=image_path,
                    image_id=image_id,
                    dimension_id=dimension_id,
                    prompt=prompt,
                    framework_id=framework_id,
                )
                tasks.append(task)

        return tasks

    @staticmethod
    def group_results_by_image(
        results: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group results by image ID

        Args:
            results: List of evaluation results

        Returns:
            Dict mapping image_id to list of dimension results
        """
        grouped = {}

        for result in results:
            image_id = result["image_id"]
            if image_id not in grouped:
                grouped[image_id] = []
            grouped[image_id].append(result)

        return grouped

    @staticmethod
    def group_results_by_framework(
        results: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group results by framework ID

        Args:
            results: List of evaluation results

        Returns:
            Dict mapping framework_id to list of results
        """
        grouped = {}

        for result in results:
            framework_id = result["framework_id"]
            if framework_id not in grouped:
                grouped[framework_id] = []
            grouped[framework_id].append(result)

        return grouped

    @staticmethod
    def calculate_overall_scores(
        grouped_results: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, float]:
        """
        Calculate overall scores for grouped results

        Args:
            grouped_results: Results grouped by image_id or framework_id

        Returns:
            Dict mapping group key to overall score
        """
        overall_scores = {}

        for key, results in grouped_results.items():
            scores = [r["score"] for r in results if "score" in r]
            if scores:
                overall_scores[key] = sum(scores) / len(scores)
            else:
                overall_scores[key] = 0.0

        return overall_scores
