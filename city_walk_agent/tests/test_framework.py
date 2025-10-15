#!/usr/bin/env python3
"""
CityWalkAgent Framework Test Suite

Unit and integration tests to validate framework functionality.
Run this to ensure all components work correctly before development.

Usage: python tests/test_framework.py
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path.resolve()))

def test_imports():
    """Test that all modules can be imported"""
    print("🔧 Testing imports...")

    try:
        from config import settings
        print("✅ Config module imported successfully")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

    try:
        from data_collection import RouteGenerator, ImageCollector
        print("✅ Data collection modules imported successfully")
    except Exception as e:
        print(f"❌ Data collection import failed: {e}")
        return False

    try:
        from utils.data_models import Route, Waypoint, WalkabilityScore
        print("✅ Data models imported successfully")
    except Exception as e:
        print(f"❌ Data models import failed: {e}")
        return False

    return True

def test_route_generation():
    """Test basic route generation functionality"""
    print("\n🗺️  Testing route generation...")

    try:
        from data_collection import RouteGenerator

        # Test without API key (offline functionality)
        print("Creating route generator (no API key required for simple routes)...")

        # Create simple route between two points
        route_gen = RouteGenerator(api_key="test_key")  # Dummy key for testing

        # Generate a simple test route
        start_lat, start_lon = 40.7589, -73.9851  # Times Square
        end_lat, end_lon = 40.7614, -73.9776       # Near Central Park

        route = route_gen.create_simple_route(
            start_lat=start_lat,
            start_lon=start_lon,
            end_lat=end_lat,
            end_lon=end_lon,
            interval_meters=50,  # Larger interval for testing
            route_name="test_route"
        )

        print(f"✅ Route generated successfully!")
        print(f"   Route ID: {route.route_id}")
        print(f"   Number of waypoints: {len(route.waypoints)}")
        print(f"   Interval: {route.interval_meters} meters")

        # Test route saving/loading
        route_file = route_gen.save_route(route)
        print(f"✅ Route saved to: {route_file}")

        loaded_route = RouteGenerator.load_route(route_file)
        print(f"✅ Route loaded successfully: {loaded_route.route_id}")

        return True

    except Exception as e:
        print(f"❌ Route generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_models():
    """Test data model validation"""
    print("\n📊 Testing data models...")

    try:
        from utils.data_models import Waypoint, WalkabilityScore, Route
        from datetime import datetime

        # Test waypoint creation
        waypoint = Waypoint(
            lat=40.7589,
            lon=-73.9851,
            sequence_id=0,
            timestamp=datetime.now()
        )
        print("✅ Waypoint model validation successful")

        # Test walkability score
        score = WalkabilityScore(
            dimension="safety",
            score=7.5,
            confidence=0.8,
            reasoning="Well-lit area with good visibility"
        )
        print("✅ WalkabilityScore model validation successful")

        return True

    except Exception as e:
        print(f"❌ Data model validation failed: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print("\n⚙️  Testing configuration...")

    try:
        from config import settings

        print(f"✅ Settings loaded successfully")
        print(f"   Project root: {settings.project_root}")
        print(f"   Data directory: {settings.data_dir}")
        print(f"   Default dimensions: {settings.default_dimensions}")
        print(f"   Default sampling interval: {settings.default_sampling_interval}m")

        # Check if directories exist
        if settings.data_dir.exists():
            print("✅ Data directory exists")
        else:
            print("ℹ️  Data directory will be created when needed")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 CityWalkAgent Framework Test Suite")
    print("=" * 50)

    tests = [
        test_imports,
        test_configuration,
        test_data_models,
        test_route_generation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n❌ Test failed: {test.__name__}")

    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Framework is ready for use.")
        print("\nNext steps:")
        print("1. Set up API keys in .env file")
        print("2. Run experiments/notebooks/01_basic_route_test.ipynb")
        print("3. Start building VLM evaluation engine")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)