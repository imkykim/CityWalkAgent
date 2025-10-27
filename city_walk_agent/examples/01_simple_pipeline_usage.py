#!/usr/bin/env python3
"""
Example 1: Simple pipeline usage

Demonstrates the basic usage of the CityWalkAgent production pipeline.

This example shows how to:
1. Initialize the pipeline
2. Analyze a route from coordinates
3. Access the results
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from pipeline import WalkingAgentPipeline

def main():
    """Run simple pipeline example"""
    print("="*70)
    print("CityWalkAgent - Simple Pipeline Usage Example")
    print("="*70)

    # Initialize pipeline with SAGAI framework
    print("\n1. Initializing pipeline...")
    pipeline = WalkingAgentPipeline(
        framework_id="sagai_2025",
        max_concurrent=5
    )

    # Define route coordinates
    # Example: Times Square to Central Park (NYC)
    start = (40.7589, -73.9851)  # Times Square
    end = (40.7614, -73.9776)    # Near Central Park

    print(f"\n2. Analyzing route...")
    print(f"   Start: {start}")
    print(f"   End:   {end}")
    print(f"   Interval: 10 meters")

    # Run analysis
    result = pipeline.analyze_route(
        start=start,
        end=end,
        interval_meters=10,
        route_name="times_square_to_central_park"
    )

    # Access results
    print(f"\n3. Results:")
    print(f"   Route ID: {result.route_id}")
    print(f"   Waypoints: {result.route_info['waypoints']}")
    print(f"   Evaluations: {len(result.evaluation_results)}")

    print(f"\n4. Sequential Analysis:")
    analysis = result.sequential_analysis
    print(f"   Aggregate Score:  {analysis['aggregate_score']:.2f}/10")
    print(f"   Sequential Score: {analysis['sequential_score']:.2f}/10")
    print(f"   Pattern:          {analysis['pattern_type']}")
    print(f"   Volatility:       {analysis['volatility']:.2f}")
    print(f"   Barriers:         {len(analysis['hidden_barriers'])}")

    print(f"\n5. Method Comparison:")
    comparison = result.method_comparison
    print(f"   Better method:    {comparison['which_method_better']}")
    print(f"   Rec. change:      {comparison['recommendation_change']}")

    print(f"\n6. Key Insights:")
    for insight in comparison['key_differences']:
        print(f"   • {insight}")

    print(f"\n7. Output Files:")
    for name, path in result.output_files.items():
        print(f"   {name:20s} {path}")

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
