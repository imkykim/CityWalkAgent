#!/usr/bin/env python3
"""
Quick test script to verify the production framework works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")


def test_imports():
    """Test all major imports"""
    print("=" * 70)
    print("Testing Production Framework Imports")
    print("=" * 70)

    try:
        from config import settings, load_framework, list_frameworks

        print("✓ Config imports OK")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False

    try:
        from evaluation import Evaluator, VLMClient, PromptBuilder

        print("✓ Evaluation imports OK")
    except Exception as e:
        print(f"✗ Evaluation import failed: {e}")
        return False

    try:
        from analysis import SequentialAnalyzer, MethodComparator

        print("✓ Analysis imports OK")
    except Exception as e:
        print(f"✗ Analysis import failed: {e}")
        return False

    try:
        from pipeline import WalkingAgentPipeline

        print("✓ Pipeline import OK")
    except Exception as e:
        print(f"✗ Pipeline import failed: {e}")
        return False

    try:
        from utils import get_logger, get_cost_tracker

        print("✓ Utils imports OK")
    except Exception as e:
        print(f"✗ Utils import failed: {e}")
        return False

    return True


def test_framework_loading():
    """Test framework loading"""
    print("\n" + "=" * 70)
    print("Testing Framework Loading")
    print("=" * 70)

    try:
        from config import list_frameworks, load_framework

        # List frameworks
        frameworks = list_frameworks()
        print(f"\n✓ Found {len(frameworks)} frameworks:")
        for fw in frameworks:
            print(f"  - {fw['id']:20s} {fw['name_cn']}")

        # Try loading one
        if frameworks:
            fw_id = frameworks[0]["id"]
            framework = load_framework(fw_id)
            print(f"\n✓ Successfully loaded framework: {fw_id}")
            print(f"  Dimensions: {len(framework['dimensions'])}")
            return True
        else:
            print("⚠️  No frameworks found")
            return False

    except Exception as e:
        print(f"✗ Framework loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_models():
    """Test data models"""
    print("\n" + "=" * 70)
    print("Testing Data Models")
    print("=" * 70)

    try:
        from utils.data_models import Route, Waypoint
        from datetime import datetime

        # Create test waypoint
        wp = Waypoint(
            lat=40.7589,
            lon=-73.9851,
            sequence_id=0,
            heading=90.0,
            timestamp=datetime.now(),
        )
        print("✓ Waypoint model OK")

        # Create test route
        route = Route(
            route_id="test_route",
            start_lat=40.7589,
            start_lon=-73.9851,
            end_lat=40.7614,
            end_lon=-73.9776,
            waypoints=[wp],
            interval_meters=10,
        )
        print("✓ Route model OK")

        return True

    except Exception as e:
        print(f"✗ Data models failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_analysis_algorithms():
    """Test analysis algorithms"""
    print("\n" + "=" * 70)
    print("Testing Analysis Algorithms")
    print("=" * 70)

    try:
        from analysis.metrics import (
            calculate_volatility,
            analyze_transitions,
            classify_route_pattern,
        )

        # Test volatility
        scores = [7.0, 7.5, 7.2, 6.8, 7.1]
        volatility = calculate_volatility(scores)
        print(f"✓ Volatility calculation OK: {volatility:.2f}")

        # Test transitions
        transitions = analyze_transitions(scores)
        print(f"✓ Transition analysis OK: {transitions.smooth_transitions} smooth")

        # Test pattern classification
        pattern = classify_route_pattern(
            avg_score=7.0, volatility=volatility, num_barriers=0, score_trend=0.1
        )
        print(f"✓ Pattern classification OK: {pattern}")

        return True

    except Exception as e:
        print(f"✗ Analysis algorithms failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_prompt_builder():
    """Test prompt builder"""
    print("\n" + "=" * 70)
    print("Testing Prompt Builder")
    print("=" * 70)

    try:
        from config import load_framework
        from evaluation.prompt_builder import PromptBuilder

        # Load a framework
        framework = load_framework("sagai_2025")

        # Create prompt builder
        builder = PromptBuilder(framework)

        # Get all prompts
        prompts = builder.build_all_prompts(language="cn")
        print(f"✓ Built {len(prompts)} prompts")

        # Show sample
        if prompts:
            first_dim = list(prompts.keys())[0]
            prompt = prompts[first_dim]
            print(f"\n  Sample prompt for '{first_dim}':")
            print(f"  {prompt[:150]}...")

        return True

    except Exception as e:
        print(f"✗ Prompt builder failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("CityWalkAgent Production Framework Test Suite")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Framework Loading", test_framework_loading()))
    results.append(("Data Models", test_data_models()))
    results.append(("Analysis Algorithms", test_analysis_algorithms()))
    results.append(("Prompt Builder", test_prompt_builder()))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Framework is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
