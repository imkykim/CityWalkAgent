#!/usr/bin/env python3
"""
Example 3: Analysis only (no evaluation)

Demonstrates using just the analysis engine on existing evaluation results.

This example shows how to:
1. Load existing evaluation results
2. Run sequential analysis
3. Compare with aggregate methods
4. Access detailed metrics
"""

import sys
import json
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils.data_models import Route
from analysis import SequentialAnalyzer, MethodComparator


def main():
    """Run analysis-only example"""
    print("="*70)
    print("CityWalkAgent - Analysis Only Example")
    print("="*70)

    # Load route
    print("\n1. Loading route...")
    route_file = Path("data/routes/example_route_123.json")  # Replace with actual route

    with open(route_file, 'r') as f:
        route_data = json.load(f)
        route = Route.model_validate(route_data)

    print(f"   Route ID: {route.route_id}")
    print(f"   Waypoints: {len(route.waypoints)}")

    # Load evaluation results
    print("\n2. Loading evaluation results...")
    results_file = Path("experiments/results/vlm_evaluation_20250101/full_results.json")

    with open(results_file, 'r') as f:
        evaluations = json.load(f)

    print(f"   Evaluations: {len(evaluations)}")

    # Run sequential analysis
    print("\n3. Running sequential analysis...")
    analyzer = SequentialAnalyzer(route, evaluations)
    analysis = analyzer.full_analysis()

    print(f"\n4. Results:")
    print(f"   Aggregate Score:  {analysis.aggregate_score:.2f}/10")
    print(f"   Sequential Score: {analysis.sequential_score:.2f}/10")
    print(f"   Difference:       {analysis.aggregate_score - analysis.sequential_score:+.2f}")
    print(f"   Pattern:          {analysis.pattern_type}")
    print(f"   Volatility:       {analysis.volatility:.2f}")

    print(f"\n5. Hidden Barriers:")
    if analysis.hidden_barriers:
        for i, barrier in enumerate(analysis.hidden_barriers[:3], 1):
            print(f"   Barrier {i}:")
            print(f"     Waypoint:   {barrier['waypoint_sequence']}")
            print(f"     Score drop: {barrier['score_drop']:.1f} points")
            print(f"     Severity:   {barrier['severity_level']}")
            print(f"     Dimension:  {barrier['dimension']}")
    else:
        print("   No significant barriers detected")

    print(f"\n6. Transitions:")
    trans = analysis.transition_analysis
    print(f"   Smooth:   {trans['smooth_transitions']}")
    print(f"   Moderate: {trans['moderate_transitions']}")
    print(f"   Abrupt:   {trans['abrupt_transitions']}")
    print(f"   Severe:   {trans['severe_transitions']}")

    # Method comparison
    print(f"\n7. Method Comparison:")
    comparator = MethodComparator(route, evaluations)
    comparison = comparator.compare()

    print(f"   Better method: {comparison.which_method_better}")
    print(f"   Key differences:")
    for diff in comparison.key_differences:
        print(f"     • {diff}")

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
