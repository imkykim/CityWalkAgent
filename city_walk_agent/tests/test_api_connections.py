#!/usr/bin/env python3
"""
API Connection Test Suite

Tests connections to all external APIs required for CityWalkAgent:
- Google Maps API (Directions, Street View)
- Mapillary API (Street View images)
- Anthropic API (Claude for evaluation)
- ZenSVI integration

Usage: python tests/test_api_connections.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables from .env file before importing other modules
load_dotenv()

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import settings


class APIConnectionTester:
    """Test suite for external API connections"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {"passed": 0, "failed": 0, "skipped": 0},
        }

    def print_test_header(self, test_name):
        """Print formatted test header"""
        print(f"\n{'=' * 60}")
        print(f"🔌 Testing: {test_name}")
        print(f"{'=' * 60}")

    def print_result(self, test_name, success, message, details=None):
        """Print and record test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {message}")

        self.results["tests"][test_name] = {
            "success": success,
            "message": message,
            "details": details or {},
        }

        if success:
            self.results["summary"]["passed"] += 1
        else:
            self.results["summary"]["failed"] += 1

    def print_skip(self, test_name, reason):
        """Print and record skipped test"""
        print(f"⏭️  SKIP: {reason}")
        self.results["tests"][test_name] = {
            "success": None,
            "message": f"Skipped: {reason}",
            "details": {},
        }
        self.results["summary"]["skipped"] += 1

    def test_google_maps_api(self):
        """Test Google Maps API connection and functionality"""
        self.print_test_header("Google Maps API")

        api_key = settings.google_maps_api_key

        if not api_key:
            self.print_skip("google_maps", "No API key provided in settings")
            return

        try:
            import googlemaps

            # Test client initialization
            client = googlemaps.Client(key=api_key)
            self.print_result(
                "google_maps_init", True, "Google Maps client initialized successfully"
            )

            # Test Directions API
            try:
                directions = client.directions(
                    origin="Times Square, New York, NY",
                    destination="Central Park, New York, NY",
                    mode="walking",
                )

                if directions:
                    route = directions[0]
                    duration = route["legs"][0]["duration"]["text"]
                    distance = route["legs"][0]["distance"]["text"]

                    self.print_result(
                        "google_directions",
                        True,
                        f"Directions API working - {distance}, {duration}",
                        {"distance": distance, "duration": duration},
                    )
                else:
                    self.print_result(
                        "google_directions", False, "Directions API returned no results"
                    )

            except Exception as e:
                self.print_result(
                    "google_directions", False, f"Directions API failed: {str(e)}"
                )

            # Test Geocoding API
            try:
                geocode_result = client.geocode("Times Square, New York, NY")
                if geocode_result:
                    location = geocode_result[0]["geometry"]["location"]
                    self.print_result(
                        "google_geocoding",
                        True,
                        f"Geocoding API working - lat: {location['lat']:.4f}, lng: {location['lng']:.4f}",
                        {"lat": location["lat"], "lng": location["lng"]},
                    )
                else:
                    self.print_result(
                        "google_geocoding", False, "Geocoding API returned no results"
                    )
            except Exception as e:
                self.print_result(
                    "google_geocoding", False, f"Geocoding API failed: {str(e)}"
                )

            # Test Street View Static API (basic check)
            try:
                # This doesn't actually make a request, just validates the API key format
                street_view_url = f"https://maps.googleapis.com/maps/api/streetview?size=640x640&location=40.7580,-73.9855&key={api_key}"

                import requests

                response = requests.head(street_view_url, timeout=10)

                if response.status_code == 200:
                    self.print_result(
                        "google_street_view", True, "Street View Static API accessible"
                    )
                elif response.status_code == 403:
                    self.print_result(
                        "google_street_view",
                        False,
                        "Street View Static API access denied - check API key permissions",
                    )
                else:
                    self.print_result(
                        "google_street_view",
                        False,
                        f"Street View Static API returned status: {response.status_code}",
                    )

            except Exception as e:
                self.print_result(
                    "google_street_view",
                    False,
                    f"Street View Static API test failed: {str(e)}",
                )

        except ValueError as e:
            if "Invalid API key" in str(e):
                self.print_result(
                    "google_maps_init", False, "Invalid Google Maps API key"
                )
            else:
                self.print_result(
                    "google_maps_init", False, f"Google Maps API error: {str(e)}"
                )
        except Exception as e:
            self.print_result(
                "google_maps_init",
                False,
                f"Failed to initialize Google Maps client: {str(e)}",
            )

    def test_mapillary_api(self):
        """Test Mapillary API connection"""
        self.print_test_header("Mapillary API")

        api_key = settings.mapillary_api_key

        if not api_key:
            self.print_skip("mapillary", "No API key provided in settings")
            return

        try:
            import requests

            # Test basic API access
            headers = {"Authorization": f"OAuth {api_key}"}

            # Test with a simple API call to get images near Times Square
            url = "https://graph.mapillary.com/images"
            params = {
                "access_token": api_key,
                "bbox": "-73.9851,40.7589,-73.9776,40.7614",  # Times Square area
                "limit": 1,
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                image_count = len(data.get("data", []))
                self.print_result(
                    "mapillary_api",
                    True,
                    f"Mapillary API working - found {image_count} images in test area",
                    {"image_count": image_count},
                )
            elif response.status_code == 401:
                self.print_result(
                    "mapillary_api",
                    False,
                    "Mapillary API authentication failed - check API key",
                )
            else:
                self.print_result(
                    "mapillary_api",
                    False,
                    f"Mapillary API returned status: {response.status_code}",
                )

        except Exception as e:
            self.print_result(
                "mapillary_api", False, f"Mapillary API test failed: {str(e)}"
            )

    def test_anthropic_api(self):
        """Test Anthropic API connection"""
        self.print_test_header("Anthropic API (Claude)")

        api_key = settings.anthropic_api_key

        if not api_key:
            self.print_skip("anthropic", "No API key provided in settings")
            return

        try:
            import anthropic

            # Test client initialization
            client = anthropic.Anthropic(api_key=api_key)

            # Test a simple message
            try:
                message = client.messages.create(
                    model="claude-3-haiku-20240307",  # Use fastest model for testing
                    max_tokens=100,
                    messages=[
                        {
                            "role": "user",
                            "content": "Respond with exactly: 'API connection test successful'",
                        }
                    ],
                )

                response_text = message.content[0].text.strip()

                if "API connection test successful" in response_text:
                    self.print_result(
                        "anthropic_api",
                        True,
                        f"Anthropic API working - Model: {message.model}",
                        {"model": message.model, "response": response_text},
                    )
                else:
                    self.print_result(
                        "anthropic_api",
                        True,
                        f"Anthropic API connected but unexpected response: {response_text}",
                        {"model": message.model, "response": response_text},
                    )

            except Exception as e:
                error_str = str(e)
                if (
                    "authentication" in error_str.lower()
                    or "api_key" in error_str.lower()
                ):
                    self.print_result(
                        "anthropic_api",
                        False,
                        "Anthropic API authentication failed - check API key",
                    )
                elif "rate_limit" in error_str.lower():
                    self.print_result(
                        "anthropic_api", False, "Anthropic API rate limit exceeded"
                    )
                else:
                    self.print_result(
                        "anthropic_api",
                        False,
                        f"Anthropic API request failed: {error_str}",
                    )

        except Exception as e:
            self.print_result(
                "anthropic_api",
                False,
                f"Failed to initialize Anthropic client: {str(e)}",
            )

    def test_zensvi_integration(self):
        """Test ZenSVI library integration"""
        self.print_test_header("ZenSVI Integration")

        zensvi_path = settings.zensvi_root

        if not zensvi_path.exists():
            self.print_skip("zensvi", f"ZenSVI directory not found at {zensvi_path}")
            return

        try:
            # Add ZenSVI to path
            import sys

            if str(zensvi_path / "src") not in sys.path:
                sys.path.insert(0, str(zensvi_path / "src"))

            # Test ZenSVI imports
            try:
                from zensvi.download import MLYDownloader, GSVDownloader

                self.print_result(
                    "zensvi_imports", True, "ZenSVI modules imported successfully"
                )

                # Test downloader initialization (without API keys)
                try:
                    # This should work even without API keys
                    mly_downloader = MLYDownloader(key="test_key", max_workers=1)
                    self.print_result(
                        "zensvi_mly_init", True, "Mapillary downloader initialized"
                    )
                except Exception as e:
                    self.print_result(
                        "zensvi_mly_init",
                        False,
                        f"Mapillary downloader init failed: {str(e)}",
                    )

                try:
                    gsv_downloader = GSVDownloader(key="test_key")
                    self.print_result(
                        "zensvi_gsv_init",
                        True,
                        "Google Street View downloader initialized",
                    )
                except Exception as e:
                    self.print_result(
                        "zensvi_gsv_init",
                        False,
                        f"GSV downloader init failed: {str(e)}",
                    )

            except ImportError as e:
                self.print_result(
                    "zensvi_imports", False, f"ZenSVI import failed: {str(e)}"
                )

        except Exception as e:
            self.print_result(
                "zensvi_integration", False, f"ZenSVI integration test failed: {str(e)}"
            )

    def test_virl_integration(self):
        """Test VIRL library integration"""
        self.print_test_header("VIRL Integration")

        virl_path = settings.virl_root

        if not virl_path.exists():
            self.print_skip("virl", f"VIRL directory not found at {virl_path}")
            return

        try:
            # Test if VIRL is accessible
            virl_config = virl_path / "virl" / "config.py"
            if virl_config.exists():
                self.print_result(
                    "virl_structure",
                    True,
                    f"VIRL project structure found at {virl_path}",
                )
            else:
                self.print_result(
                    "virl_structure",
                    False,
                    f"VIRL config not found at expected location",
                )

            # Test basic VIRL imports (if possible without breaking the environment)
            try:
                original_path = sys.path.copy()
                sys.path.insert(0, str(virl_path))

                # Try importing VIRL config
                import virl.config as virl_config

                self.print_result(
                    "virl_imports", True, "VIRL config module imported successfully"
                )

                # Restore path
                sys.path = original_path

            except ImportError as e:
                self.print_result(
                    "virl_imports", False, f"VIRL import failed: {str(e)}"
                )
            except Exception as e:
                self.print_result(
                    "virl_imports", False, f"VIRL integration error: {str(e)}"
                )

        except Exception as e:
            self.print_result(
                "virl_integration", False, f"VIRL integration test failed: {str(e)}"
            )

    def save_results(self):
        """Save test results to file"""
        results_dir = Path(__file__).parent.parent / "data" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"api_connection_test_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        return results_file

    def print_summary(self):
        """Print test summary"""
        print(f"\n{'=' * 60}")
        print("📊 API Connection Test Summary")
        print(f"{'=' * 60}")

        total = sum(self.results["summary"].values())
        passed = self.results["summary"]["passed"]
        failed = self.results["summary"]["failed"]
        skipped = self.results["summary"]["skipped"]

        print(f"Total tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")

        if failed == 0 and passed > 0:
            print("\n🎉 All API tests passed! Ready for full integration.")
        elif failed > 0:
            print(f"\n⚠️  {failed} test(s) failed. Check API keys and permissions.")
        else:
            print("\n⚠️  No tests could be run. Check API key configuration.")

        # Save results
        results_file = self.save_results()
        print(f"\n💾 Results saved to: {results_file}")

        return failed == 0 and passed > 0


def main():
    """Run all API connection tests"""
    print("🔌 CityWalkAgent API Connection Test Suite")
    print(f"Testing connections to external services...")

    tester = APIConnectionTester()

    # Run all tests
    tester.test_google_maps_api()
    tester.test_mapillary_api()
    tester.test_anthropic_api()
    tester.test_zensvi_integration()
    tester.test_virl_integration()

    # Print summary
    success = tester.print_summary()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
