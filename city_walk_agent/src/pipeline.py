"""
Main pipeline orchestrator for CityWalkAgent

This is the high-level interface that coordinates all components:
- Route generation
- Image collection
- VLM evaluation (Qwen VLM)
- Sequential analysis
- Comparison and reporting

Usage:
    ```python
    from src.pipeline import WalkingAgentPipeline

    # Initialize pipeline
    pipeline = WalkingAgentPipeline(
        framework_id="sagai_2025"
    )

    # Run full analysis
    result = pipeline.analyze_route(
        start=(40.7589, -73.9851),
        end=(40.7614, -73.9776),
        interval_meters=10
    )
    ```
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# Handle both package and direct imports
try:
    from .config import settings, load_framework
    from .data_collection.route_generator import RouteGenerator
    from .data_collection.image_collector import ImageCollector
    from .evaluation import Evaluator
    from .evaluation.vlm_client import VLMConfig
    from .analysis import SequentialAnalyzer, MethodComparator
    from .utils import get_logger, get_cost_tracker
except ImportError:
    # Direct import when run as script
    from config import settings, load_framework
    from data_collection.route_generator import RouteGenerator
    from data_collection.image_collector import ImageCollector
    from evaluation import Evaluator
    from evaluation.vlm_client import VLMConfig
    from analysis import SequentialAnalyzer, MethodComparator
    from utils import get_logger, get_cost_tracker


@dataclass
class PipelineResult:
    """Complete pipeline execution result"""
    route_id: str
    route_info: Dict[str, Any]
    evaluation_results: List[Dict[str, Any]]
    sequential_analysis: Dict[str, Any]
    method_comparison: Dict[str, Any]
    statistics: Dict[str, Any]
    output_files: Dict[str, Path]
    timestamp: str


class WalkingAgentPipeline:
    """
    Main pipeline orchestrator

    Coordinates the entire CityWalkAgent workflow:
    1. Route generation (or load existing)
    2. Image collection (or use existing)
    3. VLM-based evaluation
    4. Sequential analysis
    5. Method comparison
    6. Results export

    This is the primary interface for production use.
    """

    def __init__(
        self,
        framework_id: str,
        max_concurrent: int = 5,
        output_dir: Optional[Path] = None,
        enable_cost_tracking: bool = True
    ):
        """
        Initialize pipeline

        Args:
            framework_id: Framework to use for evaluation (e.g., "sagai_2025")
            max_concurrent: Maximum concurrent API calls
            output_dir: Output directory for results
            enable_cost_tracking: Track API costs
        """
        self.framework_id = framework_id
        self.logger = get_logger()

        # Load framework
        self.logger.info(f"Loading framework: {framework_id}")
        self.framework = load_framework(framework_id)

        # Initialize VLM config
        self.vlm_config = VLMConfig(
            api_key=settings.qwen_vlm_api_key,
            model=settings.qwen_vlm_model,
            api_url=settings.qwen_vlm_api_url
        )

        # Initialize components
        self.route_generator = RouteGenerator()
        self.max_concurrent = max_concurrent
        self.output_dir = output_dir or settings.results_dir / "pipeline"
        self.enable_cost_tracking = enable_cost_tracking

        if enable_cost_tracking:
            self.cost_tracker = get_cost_tracker()

        self.logger.info("Pipeline initialized", framework=framework_id)


    def analyze_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        interval_meters: int = 10,
        route_name: Optional[str] = None,
        collect_images: bool = True
    ) -> PipelineResult:
        """
        Run complete analysis on a route

        Args:
            start: (lat, lon) start coordinates
            end: (lat, lon) end coordinates
            interval_meters: Sampling interval in meters
            route_name: Optional route name
            collect_images: Whether to collect images (set False if using existing)

        Returns:
            PipelineResult with all analysis outputs
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.logger.info(
            "Starting route analysis",
            start=start,
            end=end,
            interval=interval_meters
        )

        # Step 1: Generate route
        self.logger.info("Step 1: Generating route...")
        route = self.route_generator.create_simple_route(
            start[0], start[1], end[0], end[1],
            interval_meters=interval_meters,
            route_name=route_name
        )

        # Save route
        route_file = settings.data_dir / "routes" / f"{route.route_id}.json"
        route_file.parent.mkdir(parents=True, exist_ok=True)
        self.route_generator.save_route(route, route_file)

        self.logger.info(
            "Route generated",
            route_id=route.route_id,
            waypoints=len(route.waypoints)
        )

        # Step 2: Collect images (if requested)
        if collect_images:
            self.logger.info("Step 2: Collecting images...")
            try:
                image_collector = ImageCollector(
                    api_key=settings.google_maps_api_key
                )
                image_collector.collect_google_street_view_images(route)
                self.logger.info("Images collected", count=len([w for w in route.waypoints if w.image_path]))
            except Exception as e:
                self.logger.error("Image collection failed", error=str(e))
                raise
        else:
            self.logger.info("Step 2: Skipping image collection (using existing)")

        # Step 3: VLM evaluation
        self.logger.info("Step 3: Running VLM evaluation...")
        evaluator = Evaluator(
            vlm_config=self.vlm_config,
            framework=self.framework,
            max_concurrent=self.max_concurrent
        )

        evaluation_results = evaluator.evaluate_route(route)

        self.logger.info(
            "Evaluation complete",
            evaluations=len(evaluation_results)
        )

        # Step 4: Sequential analysis
        self.logger.info("Step 4: Running sequential analysis...")
        analyzer = SequentialAnalyzer(route, evaluation_results)
        sequential_analysis = analyzer.full_analysis()

        self.logger.info(
            "Sequential analysis complete",
            pattern=sequential_analysis.pattern_type,
            barriers=len(sequential_analysis.hidden_barriers)
        )

        # Step 5: Method comparison
        self.logger.info("Step 5: Comparing methods...")
        comparator = MethodComparator(route, evaluation_results)
        comparison = comparator.compare()

        self.logger.info(
            "Method comparison complete",
            better_method=comparison.which_method_better
        )

        # Step 6: Export results
        self.logger.info("Step 6: Exporting results...")
        output_files = self._export_results(
            route,
            evaluation_results,
            sequential_analysis,
            comparison,
            timestamp
        )

        # Gather statistics
        stats = {
            "vlm_stats": evaluator.get_statistics(),
            "route_waypoints": len(route.waypoints),
            "total_evaluations": len(evaluation_results),
            "execution_time": timestamp
        }

        if self.enable_cost_tracking:
            stats["cost"] = self.cost_tracker.get_summary()

        result = PipelineResult(
            route_id=route.route_id,
            route_info={
                "start": (route.start_lat, route.start_lon),
                "end": (route.end_lat, route.end_lon),
                "interval_meters": route.interval_meters,
                "waypoints": len(route.waypoints)
            },
            evaluation_results=evaluation_results,
            sequential_analysis=asdict(sequential_analysis),
            method_comparison=asdict(comparison),
            statistics=stats,
            output_files=output_files,
            timestamp=timestamp
        )

        self.logger.info("Pipeline complete!", route_id=route.route_id)

        return result

    def analyze_existing_route(
        self,
        route_file: Path
    ) -> PipelineResult:
        """
        Analyze existing route from file

        Args:
            route_file: Path to route JSON file

        Returns:
            PipelineResult
        """
        self.logger.info("Loading existing route", file=str(route_file))

        route = self.route_generator.load_route(str(route_file))

        # Check for images
        has_images = any(w.image_path for w in route.waypoints)

        if not has_images:
            self.logger.warning("Route has no images - will attempt to collect")
            return self.analyze_route(
                start=(route.start_lat, route.start_lon),
                end=(route.end_lat, route.end_lon),
                interval_meters=route.interval_meters,
                route_name=route.route_name,
                collect_images=True
            )

        # Continue with evaluation
        return self.analyze_route(
            start=(route.start_lat, route.start_lon),
            end=(route.end_lat, route.end_lon),
            interval_meters=route.interval_meters,
            route_name=route.route_name,
            collect_images=False
        )

    def _export_results(
        self,
        route: Any,
        evaluations: List[Dict[str, Any]],
        analysis: Any,
        comparison: Any,
        timestamp: str
    ) -> Dict[str, Path]:
        """Export results to files"""
        output_dir = self.output_dir / f"{route.route_id}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        files = {}

        # 1. Evaluation results CSV
        csv_file = output_dir / "evaluations.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("image_id,dimension_id,dimension_name,score,reasoning\n")
            for r in evaluations:
                reasoning = r.get('reasoning', '').replace('"', '""').replace('\n', ' ')
                f.write(
                    f"{r['image_id']},{r['dimension_id']},{r.get('dimension_name', '')},"
                    f"{r['score']},\"{reasoning}\"\n"
                )
        files["evaluations_csv"] = csv_file

        # 2. Sequential analysis JSON
        analysis_file = output_dir / "sequential_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(analysis), f, indent=2, ensure_ascii=False)
        files["analysis_json"] = analysis_file

        # 3. Method comparison JSON
        comparison_file = output_dir / "method_comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(comparison), f, indent=2, ensure_ascii=False)
        files["comparison_json"] = comparison_file

        # 4. Summary report
        report_file = output_dir / "summary_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_summary_report(route, analysis, comparison))
        files["report_txt"] = report_file

        self.logger.info("Results exported", output_dir=str(output_dir))

        return files

    def _generate_summary_report(
        self,
        route: Any,
        analysis: Any,
        comparison: Any
    ) -> str:
        """Generate human-readable summary report"""
        report = f"""
CityWalkAgent Analysis Report
{'='*70}

Route: {route.route_id}
Framework: {self.framework_id}
Generated: {datetime.now().isoformat()}

{'─'*70}
ROUTE INFORMATION
{'─'*70}
Start: ({route.start_lat:.6f}, {route.start_lon:.6f})
End: ({route.end_lat:.6f}, {route.end_lon:.6f})
Interval: {route.interval_meters}m
Waypoints: {len(route.waypoints)}

{'─'*70}
SCORES
{'─'*70}
Aggregate Score:   {analysis.aggregate_score:.2f}/10
Sequential Score:  {analysis.sequential_score:.2f}/10
Difference:        {analysis.aggregate_score - analysis.sequential_score:+.2f}

{'─'*70}
PATTERN ANALYSIS
{'─'*70}
Pattern Type:  {analysis.pattern_type}
Volatility:    {analysis.volatility:.2f}
Barriers:      {len(analysis.hidden_barriers)}

Key Insight:
{analysis.key_insight}

Recommendation:
{analysis.recommendation}

{'─'*70}
METHOD COMPARISON
{'─'*70}
Better Method: {comparison.which_method_better}
Recommendation Change: {comparison.recommendation_change}

Key Differences:
"""

        for i, diff in enumerate(comparison.key_differences, 1):
            report += f"{i}. {diff}\n"

        return report


# Convenience function for quick analysis
def quick_analyze(
    start: Tuple[float, float],
    end: Tuple[float, float],
    framework_id: str = "sagai_2025",
    interval_meters: int = 10
) -> PipelineResult:
    """
    Quick route analysis with minimal configuration

    Args:
        start: (lat, lon) start coordinates
        end: (lat, lon) end coordinates
        framework_id: Framework to use
        interval_meters: Sampling interval

    Returns:
        PipelineResult
    """
    pipeline = WalkingAgentPipeline(framework_id=framework_id)
    return pipeline.analyze_route(start, end, interval_meters)
