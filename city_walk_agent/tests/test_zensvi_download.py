#!/usr/bin/env python3
"""
ZenSVI Image Download Integration Test

Tests actual image downloading using ZenSVI for CityWalkAgent.
Based on working patterns from ZenSVI/my_test/ files.

Platforms tested:
- Mapillary (requires API key)
- Google Street View (requires API key)
- KartaView (free, no API key required)

Usage:
    python tests/test_zensvi_download.py
"""

import sys
from pathlib import Path
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import settings


class ZenSVIDownloadTester:
    """Test ZenSVI image downloading for all platforms"""

    def __init__(self):
        self.output_dir = Path(__file__).parent.parent / "data" / "test_downloads"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test coordinates (verified working locations from ZenSVI tests)
        self.test_locations = {
            "nigeria": {
                "name": "Maiduguri, Nigeria",
                "lat": 11.827575599999989,
                "lon": 13.146558000000027,
                "description": "Verified Mapillary data location"
            },
            "nyc": {
                "name": "Times Square, NYC",
                "lat": 40.7580,
                "lon": -73.9855,
                "description": "High coverage location"
            },
            "sf": {
                "name": "San Francisco",
                "lat": 37.7749,
                "lon": -122.4194,
                "description": "Good KartaView coverage"
            }
        }

    def print_header(self, title):
        """Print formatted section header"""
        print(f"\n{'=' * 60}")
        print(f"{title}")
        print(f"{'=' * 60}")

    def test_mapillary_metadata(self, location_key="nigeria"):
        """
        Test Mapillary metadata download
        Based on ZenSVI/my_test/test_mapillary.py
        """
        self.print_header("🗺️  Mapillary Metadata Test")

        if not settings.mapillary_api_key:
            print("⏭️  Skipped: No Mapillary API key found")
            print("   Set MAPILLARY_API_KEY in .env file")
            return False

        location = self.test_locations[location_key]
        print(f"📍 Location: {location['name']}")
        print(f"   Coordinates: ({location['lat']}, {location['lon']})")
        print(f"   {location['description']}")

        try:
            # Import ZenSVI components
            from zensvi.download import MLYDownloader
            from zensvi.download.mapillary import interface

            # Set access token (from test_mapillary.py pattern)
            interface.set_access_token(settings.mapillary_api_key)

            # Initialize downloader
            output_path = self.output_dir / "mapillary" / location_key
            output_path.mkdir(parents=True, exist_ok=True)

            downloader = MLYDownloader(
                mly_api_key=settings.mapillary_api_key,
                log_path=str(output_path / "log.log"),
                max_workers=1,
                verbosity=2
            )

            print(f"\n📥 Downloading metadata (buffer: 100m)...")
            start_time = time.time()

            # Download metadata only
            downloader.download_svi(
                str(output_path),
                lat=location["lat"],
                lon=location["lon"],
                buffer=100,
                metadata_only=True
            )

            elapsed = time.time() - start_time

            # Check results - look for mly_pids.csv
            pid_file = output_path / "mly_pids.csv"
            if pid_file.exists() and pid_file.stat().st_size > 0:
                import pandas as pd
                df = pd.read_csv(pid_file)
                print(f"✅ Success: {len(df)} records found ({elapsed:.1f}s)")
                print(f"   Output: {pid_file}")
                return True
            else:
                print(f"⚠️  No data found at this location ({elapsed:.1f}s)")
                return False

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_mapillary_images(self, location_key="nigeria", max_images=5):
        """
        Test Mapillary actual image download
        Based on ZenSVI/my_test/test_mapillary.py

        IMPORTANT: Must download metadata first, then images
        """
        self.print_header("📸 Mapillary Image Download Test")

        if not settings.mapillary_api_key:
            print("⏭️  Skipped: No Mapillary API key found")
            return False

        location = self.test_locations[location_key]
        print(f"📍 Location: {location['name']}")
        print(f"   Limit: Small buffer (50m)")

        try:
            from zensvi.download import MLYDownloader
            from zensvi.download.mapillary import interface

            interface.set_access_token(settings.mapillary_api_key)

            output_path = self.output_dir / "mapillary_images" / location_key
            output_path.mkdir(parents=True, exist_ok=True)

            downloader = MLYDownloader(
                mly_api_key=settings.mapillary_api_key,
                log_path=str(output_path / "log.log"),
                max_workers=1,
                verbosity=2
            )

            # Step 1: Download metadata first (required for image download)
            print(f"\n📥 Step 1/2: Downloading metadata...")
            start_time = time.time()

            downloader.download_svi(
                str(output_path),
                lat=location["lat"],
                lon=location["lon"],
                buffer=50,
                metadata_only=True
            )

            metadata_time = time.time() - start_time

            # Check if metadata was downloaded successfully
            pid_file = output_path / "mly_pids.csv"
            if not pid_file.exists():
                print(f"❌ Metadata download failed - no pids file created")
                return False

            import pandas as pd
            df = pd.read_csv(pid_file)
            print(f"✅ Metadata downloaded: {len(df)} records ({metadata_time:.1f}s)")

            # Step 2: Now download actual images using the metadata
            print(f"\n📥 Step 2/2: Downloading images based on metadata...")
            start_time = time.time()

            # Re-run with metadata_only=False to download images
            downloader.download_svi(
                str(output_path),
                lat=location["lat"],
                lon=location["lon"],
                buffer=50,
                metadata_only=False
            )

            image_time = time.time() - start_time
            total_time = metadata_time + image_time

            # Check for downloaded images in mly_svi directory
            svi_dir = output_path / "mly_svi"
            if svi_dir.exists():
                image_files = list(svi_dir.rglob("*.jpg")) + list(svi_dir.rglob("*.png"))
                if image_files:
                    total_size = sum(f.stat().st_size for f in image_files) / 1024 / 1024
                    print(f"✅ Success: {len(image_files)} images, {total_size:.2f}MB")
                    print(f"   Total time: {total_time:.1f}s (metadata: {metadata_time:.1f}s, images: {image_time:.1f}s)")
                    print(f"   Output: {svi_dir}")
                    return True

            print(f"⚠️  No images downloaded ({total_time:.1f}s)")
            print(f"   Metadata was found, but image download failed")
            return False

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_gsv_metadata(self, location_key="nyc"):
        """
        Test Google Street View metadata download
        """
        self.print_header("🗺️  Google Street View Metadata Test")

        if not settings.google_maps_api_key:
            print("⏭️  Skipped: No Google Maps API key found")
            print("   Set GOOGLE_MAPS_API_KEY in .env file")
            return False

        location = self.test_locations[location_key]
        print(f"📍 Location: {location['name']}")

        try:
            from zensvi.download import GSVDownloader

            output_path = self.output_dir / "gsv" / location_key
            output_path.mkdir(parents=True, exist_ok=True)

            downloader = GSVDownloader(
                gsv_api_key=settings.google_maps_api_key,
                log_path=str(output_path / "log.log"),
                max_workers=1,
                verbosity=2
            )

            print(f"\n📥 Downloading metadata...")
            start_time = time.time()

            downloader.download_svi(
                str(output_path),
                lat=location["lat"],
                lon=location["lon"],
                buffer=100,
                metadata_only=True
            )

            elapsed = time.time() - start_time

            # Check for GSV metadata files
            csv_files = list(output_path.glob("*.csv"))
            if csv_files:
                import pandas as pd
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        if len(df) > 0:
                            print(f"✅ Success: {len(df)} records in {csv_file.name} ({elapsed:.1f}s)")
                            return True
                    except:
                        pass

            print(f"⚠️  No metadata found ({elapsed:.1f}s)")
            return False

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_kartaview_metadata(self, location_key="sf"):
        """
        Test KartaView metadata download (no API key required!)
        Based on ZenSVI/my_test/test_kartaview.py
        """
        self.print_header("🚗 KartaView Metadata Test (Free!)")

        location = self.test_locations[location_key]
        print(f"📍 Location: {location['name']}")
        print(f"   {location['description']}")
        print("   💡 No API key required!")

        try:
            from zensvi.download import KVDownloader

            output_path = self.output_dir / "kartaview" / location_key
            output_path.mkdir(parents=True, exist_ok=True)

            # KartaView doesn't require API key
            downloader = KVDownloader(
                log_path=str(output_path / "log.log"),
                max_workers=1,
                verbosity=2
            )

            print(f"\n📥 Downloading metadata (buffer: 100m)...")
            start_time = time.time()

            downloader.download_svi(
                str(output_path),
                lat=location["lat"],
                lon=location["lon"],
                buffer=100,
                metadata_only=True
            )

            elapsed = time.time() - start_time

            # Check for CSV files
            csv_files = list(output_path.rglob("*.csv"))
            if csv_files:
                import pandas as pd
                total_records = 0
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        total_records += len(df)
                    except:
                        pass

                if total_records > 0:
                    print(f"✅ Success: {total_records} records found ({elapsed:.1f}s)")
                    print(f"   Output: {output_path}")
                    return True

            print(f"⚠️  No data found at this location ({elapsed:.1f}s)")
            print(f"   💡 Try a location with better KartaView coverage (Europe recommended)")
            return False

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_kartaview_images(self, location_key="sf"):
        """
        Test KartaView actual image download
        Based on ZenSVI/my_test/test_kartaview.py
        """
        self.print_header("📸 KartaView Image Download Test (Free!)")

        location = self.test_locations[location_key]
        print(f"📍 Location: {location['name']}")
        print("   💡 Small test area (50m buffer)")

        try:
            from zensvi.download import KVDownloader

            output_path = self.output_dir / "kartaview_images" / location_key
            output_path.mkdir(parents=True, exist_ok=True)

            downloader = KVDownloader(
                log_path=str(output_path / "log.log"),
                max_workers=1,
                verbosity=2
            )

            print(f"\n📥 Downloading actual images...")
            start_time = time.time()

            downloader.download_svi(
                str(output_path),
                lat=location["lat"],
                lon=location["lon"],
                buffer=50,  # Small buffer for testing
                metadata_only=False
            )

            elapsed = time.time() - start_time

            # Check for downloaded images
            image_files = list(output_path.rglob("*.jpg")) + list(output_path.rglob("*.png"))
            if image_files:
                total_size = sum(f.stat().st_size for f in image_files if f.exists()) / 1024 / 1024
                print(f"✅ Success: {len(image_files)} images, {total_size:.2f}MB ({elapsed:.1f}s)")
                print(f"   Output: {output_path}")
                return True

            print(f"⚠️  No images found at this location ({elapsed:.1f}s)")
            return False

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run ZenSVI download tests"""
    print("🚀 ZenSVI Image Download Integration Test")
    print("=" * 60)
    print(f"Output directory: {Path(__file__).parent.parent / 'data' / 'test_downloads'}")

    tester = ZenSVIDownloadTester()

    # Check ZenSVI installation
    try:
        import zensvi
        print(f"✅ ZenSVI version: {zensvi.__version__}")
    except ImportError:
        print("❌ ZenSVI not installed!")
        print("   Run: cd ../ZenSVI && pip install -e .")
        return

    print("\n📋 Test Options:")
    print("1. Quick test (metadata only, all platforms)")
    print("2. Full test (metadata + images)")
    print("3. Mapillary only")
    print("4. KartaView only (free!)")
    print("5. Google Street View only")

    choice = input("\nSelect test (1-5, default=1): ").strip() or "1"

    results = {}

    if choice == "1":
        # Quick metadata tests for all platforms
        print("\n⚡ Running quick metadata tests...")
        results["Mapillary"] = tester.test_mapillary_metadata()
        results["KartaView"] = tester.test_kartaview_metadata()
        results["Google SV"] = tester.test_gsv_metadata()

    elif choice == "2":
        # Full tests with images
        print("\n🔄 Running full tests (this may take a while)...")
        results["Mapillary Metadata"] = tester.test_mapillary_metadata()
        results["Mapillary Images"] = tester.test_mapillary_images()
        results["KartaView Metadata"] = tester.test_kartaview_metadata()
        results["KartaView Images"] = tester.test_kartaview_images()
        results["GSV Metadata"] = tester.test_gsv_metadata()

    elif choice == "3":
        # Mapillary only
        results["Mapillary Metadata"] = tester.test_mapillary_metadata()
        if results["Mapillary Metadata"]:
            if input("\nDownload images too? (y/N): ").strip().lower() in ["y", "yes"]:
                results["Mapillary Images"] = tester.test_mapillary_images()

    elif choice == "4":
        # KartaView only
        results["KartaView Metadata"] = tester.test_kartaview_metadata()
        if results["KartaView Metadata"]:
            if input("\nDownload images too? (y/N): ").strip().lower() in ["y", "yes"]:
                results["KartaView Images"] = tester.test_kartaview_images()

    elif choice == "5":
        # Google Street View only
        results["GSV Metadata"] = tester.test_gsv_metadata()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL" if result is False else "⏭️  SKIP"
        print(f"{status}: {test_name}")

    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])

    print(f"\n📈 Success Rate: {passed}/{total} tests passed")

    print("\n💡 Next Steps:")
    print("- Check output in data/test_downloads/")
    print("- Use successful patterns in image_collector.py")
    print("- KartaView is free - great for testing!")
    print("- Mapillary has good global coverage")

    return results


if __name__ == "__main__":
    main()
