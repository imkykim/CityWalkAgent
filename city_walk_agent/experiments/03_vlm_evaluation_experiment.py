#!/usr/bin/env python3
"""
Experiment 03: VLM Framework Comparison (PoC)

Tests 4 evaluation frameworks on the same images to compare effectiveness:
1. StreetAgent 5D (spatial_sequence, visual_coherence, sensory_complexity, spatial_legibility, functional_quality)
2. Ewing & Handy 5D (imageability, enclosure, human_scale, transparency, complexity)
3. Kaplan 4D (coherence, legibility, complexity_mystery, affordance)
4. Phenomenology 3D (material_atmosphere, spatial_emotion, sensory_experience)

PoC Goal: Determine which framework best captures walkability patterns.
"""

import sys
import json
import base64
import time
import re
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional

# Simple dependency handling
try:
    import requests
except ImportError:
    print("❌ Missing required package: requests")
    print("   Install with: pip install requests")
    sys.exit(1)

# Optional dependencies
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not available, progress bars disabled")

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import settings
from utils.data_models import Route, Waypoint

# ============================================================================
# CONFIGURATION
# ============================================================================

RATE_LIMIT_DELAY = 1.0  # Qwen API: 1 request per second
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0

# Test configuration
NUM_TEST_IMAGES = 5  # Start small for PoC
FRAMEWORKS_DIR = Path(__file__).parent / "configs" / "frameworks"
RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# FRAMEWORK LOADER
# ============================================================================

def load_frameworks() -> List[Dict]:
    """Load all framework JSON configs"""
    print("\n📦 Loading evaluation frameworks...")

    if not FRAMEWORKS_DIR.exists():
        print(f"❌ Frameworks directory not found: {FRAMEWORKS_DIR}")
        print(f"   Expected location: {FRAMEWORKS_DIR.absolute()}")
        return []

    frameworks = []
    framework_files = sorted(FRAMEWORKS_DIR.glob("*.json"))

    if not framework_files:
        print(f"❌ No framework files found in {FRAMEWORKS_DIR}")
        return []

    for file_path in framework_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                framework = json.load(f)
                frameworks.append(framework)
                print(f"   ✓ {framework['framework_id']}: {framework['framework_name_cn']} ({framework['num_dimensions']} dimensions)")
        except Exception as e:
            print(f"   ✗ Failed to load {file_path.name}: {e}")

    print(f"\n✅ Loaded {len(frameworks)} frameworks")
    return frameworks


# ============================================================================
# IMAGE MANAGEMENT
# ============================================================================

def link_images_to_route(route: Route) -> Route:
    """Link existing images to route waypoints"""
    images_dir = settings.images_dir / "gsv" / route.route_id

    if not images_dir.exists():
        return route

    # Find all image files
    image_files = list(images_dir.glob("waypoint_*.jpg"))

    # Create mapping from sequence_id to image path
    image_map = {}
    for img_file in image_files:
        match = re.search(r"waypoint_(\d+)\.jpg", img_file.name)
        if match:
            seq_id = int(match.group(1))
            image_map[seq_id] = str(img_file.absolute())

    # Update waypoints with image paths
    updated_waypoints = []
    for wp in route.waypoints:
        if wp.sequence_id in image_map:
            updated_wp = Waypoint(
                lat=wp.lat,
                lon=wp.lon,
                sequence_id=wp.sequence_id,
                heading=wp.heading,
                timestamp=wp.timestamp,
                image_path=image_map[wp.sequence_id]
            )
            updated_waypoints.append(updated_wp)
        else:
            updated_waypoints.append(wp)

    # Create new route with updated waypoints
    updated_route = Route(
        route_id=route.route_id,
        start_lat=route.start_lat,
        start_lon=route.start_lon,
        end_lat=route.end_lat,
        end_lon=route.end_lon,
        waypoints=updated_waypoints,
        route_name=route.route_name,
        description=route.description,
        created_at=route.created_at,
        interval_meters=route.interval_meters
    )

    return updated_route


def load_test_images() -> List[str]:
    """Load test images from most recent route"""
    print("\n📸 Loading test images...")

    routes_dir = settings.data_dir / "routes"
    if not routes_dir.exists():
        print(f"❌ Routes directory not found: {routes_dir}")
        return []

    route_files = sorted(routes_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

    if not route_files:
        print("❌ No route files found")
        return []

    # Try to find a route with images
    for route_file in route_files[:5]:
        try:
            with open(route_file, 'r') as f:
                route_data = json.load(f)
                try:
                    route = Route.model_validate(route_data)
                except AttributeError:
                    route = Route.parse_obj(route_data)

            # Link images
            route = link_images_to_route(route)

            # Get images
            waypoints_with_images = [wp for wp in route.waypoints if wp.image_path and Path(wp.image_path).exists()]

            if waypoints_with_images:
                print(f"   ✓ Found route: {route.route_id}")
                print(f"   ✓ Available images: {len(waypoints_with_images)}")

                # Take first N images for testing
                test_images = [wp.image_path for wp in waypoints_with_images[:NUM_TEST_IMAGES]]
                print(f"   ✓ Using {len(test_images)} images for testing\n")
                return test_images

        except Exception as e:
            continue

    print("❌ No routes with images found")
    print("   Run experiment 02 first to collect images")
    return []


# ============================================================================
# VLM API CALLER
# ============================================================================

class SimpleVLMCaller:
    """Simple VLM API caller for PoC"""

    def __init__(self):
        self.api_url = settings.qwen_vlm_api_url
        self.api_key = settings.qwen_vlm_api_key
        self.model = settings.qwen_vlm_model

        if not self.api_url or not self.api_key:
            raise ValueError("Qwen VLM API not configured. Set QWEN_VLM_API_URL and QWEN_VLM_API_KEY in .env")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        self.total_calls = 0
        self.total_time = 0.0
        self.errors = 0
        self.last_call_time = 0.0

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def call_vlm(self, prompt: str, image_path: str) -> Optional[Dict]:
        """
        Call VLM API with rate limiting and retry logic

        Returns: {"score": float, "reasoning": str} or None on failure
        """
        # Rate limiting
        if self.last_call_time > 0:
            elapsed = time.time() - self.last_call_time
            if elapsed < RATE_LIMIT_DELAY:
                time.sleep(RATE_LIMIT_DELAY - elapsed)

        # Encode image
        try:
            image_base64 = self.encode_image(image_path)
        except Exception as e:
            print(f"      ✗ Failed to encode image: {e}")
            return None

        # Prepare payload
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }],
            "max_tokens": 300,
            "temperature": 0.7
        }

        # Retry loop
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                start_time = time.time()

                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                elapsed = time.time() - start_time
                self.last_call_time = time.time()
                self.total_calls += 1
                self.total_time += elapsed

                if response.status_code == 429:  # Rate limit
                    wait_time = RETRY_DELAY * (attempt + 1)
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                result = response.json()

                # Parse response
                response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return self._parse_response(response_text)

            except Exception as e:
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    self.errors += 1
                    print(f"      ✗ API call failed: {e}")
                    return None
                time.sleep(RETRY_DELAY * (attempt + 1))

        return None

    def _parse_response(self, response_text: str) -> Optional[Dict]:
        """Parse VLM response to extract score and reasoning"""
        # Clean response (remove code blocks)
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1]
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
        if response_text.endswith("```"):
            response_text = response_text.rsplit("```", 1)[0]

        response_text = response_text.strip()

        # Try to parse JSON
        try:
            data = json.loads(response_text)
            score = float(data.get("score", 5.0))
            reasoning = data.get("reasoning", response_text)

            # Validate score range
            if not (1.0 <= score <= 10.0):
                score = 5.0

            return {"score": score, "reasoning": reasoning}

        except json.JSONDecodeError:
            # Fallback: try to extract score from text
            score_patterns = [
                r"(?:score|评分)[:：]\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*[/／]\s*10",
                r"(\d+(?:\.\d+)?)\s*分"
            ]

            for pattern in score_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        if 1.0 <= score <= 10.0:
                            return {"score": score, "reasoning": response_text}
                    except ValueError:
                        continue

            # Last resort: default score
            return {"score": 5.0, "reasoning": response_text}

    def get_stats(self) -> Dict:
        """Get API usage statistics"""
        return {
            "total_calls": self.total_calls,
            "total_time": self.total_time,
            "avg_time": self.total_time / self.total_calls if self.total_calls > 0 else 0,
            "errors": self.errors,
            "error_rate": self.errors / self.total_calls if self.total_calls > 0 else 0
        }


# ============================================================================
# EVALUATION ENGINE
# ============================================================================

def evaluate_image_with_framework(
    vlm: SimpleVLMCaller,
    image_path: str,
    image_id: str,
    framework: Dict
) -> List[Dict]:
    """
    Evaluate one image with one framework

    Returns: List of dimension results
    """
    results = []

    for dimension in framework["dimensions"]:
        dim_id = dimension["id"]
        prompt = dimension["vlm_prompt"]

        # Call VLM
        response = vlm.call_vlm(prompt, image_path)

        if response:
            result = {
                "image_id": image_id,
                "framework_id": framework["framework_id"],
                "dimension_id": dim_id,
                "dimension_name": dimension["name_cn"],
                "score": response["score"],
                "reasoning": response["reasoning"]
            }
            results.append(result)
            print(f"      ✓ {dimension['name_cn']}: {response['score']:.1f}/10")
        else:
            # Failed - use default score
            result = {
                "image_id": image_id,
                "framework_id": framework["framework_id"],
                "dimension_id": dim_id,
                "dimension_name": dimension["name_cn"],
                "score": 5.0,
                "reasoning": "API调用失败，使用默认分数"
            }
            results.append(result)
            print(f"      ✗ {dimension['name_cn']}: 失败 (默认 5.0)")

    return results


def run_evaluation_experiment(frameworks: List[Dict], test_images: List[str]) -> List[Dict]:
    """
    Main evaluation loop

    Returns: List of all evaluation results
    """
    print("\n" + "="*70)
    print("🚀 Starting VLM Evaluation Experiment")
    print("="*70)

    # Initialize VLM caller
    try:
        vlm = SimpleVLMCaller()
        print(f"✅ VLM API initialized: {vlm.model}")
    except ValueError as e:
        print(f"❌ {e}")
        return []

    all_results = []

    # Evaluate each image
    for img_idx, image_path in enumerate(test_images, 1):
        image_id = Path(image_path).stem

        print(f"\n{'─'*70}")
        print(f"📷 [{img_idx}/{len(test_images)}] Evaluating: {image_id}")
        print(f"{'─'*70}")

        # Evaluate with each framework
        for framework in frameworks:
            print(f"\n  🎯 Framework: {framework['framework_name_cn']} ({framework['num_dimensions']} dimensions)")

            results = evaluate_image_with_framework(vlm, image_path, image_id, framework)
            all_results.extend(results)

            # Calculate overall score
            scores = [r["score"] for r in results]
            overall = sum(scores) / len(scores) if scores else 0
            print(f"  → Overall: {overall:.2f}/10")

    print("\n" + "="*70)
    print("✅ Evaluation Complete!")
    print("="*70)

    # Print statistics
    stats = vlm.get_stats()
    print(f"\n📊 API Statistics:")
    print(f"   Total API calls: {stats['total_calls']}")
    print(f"   Total time: {stats['total_time']:.1f}s")
    print(f"   Avg time per call: {stats['avg_time']:.2f}s")
    print(f"   Errors: {stats['errors']} ({stats['error_rate']:.1%})")

    return all_results


# ============================================================================
# RESULTS ANALYSIS & EXPORT
# ============================================================================

def analyze_framework_performance(results: List[Dict]) -> Dict:
    """Analyze and compare framework performance"""
    print("\n" + "="*70)
    print("📈 Framework Performance Analysis")
    print("="*70)

    # Group by framework
    by_framework = defaultdict(list)
    for r in results:
        by_framework[r["framework_id"]].append(r["score"])

    # Calculate statistics
    framework_stats = {}
    for framework_id, scores in by_framework.items():
        if scores:
            framework_stats[framework_id] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
                "std": (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores)) ** 0.5
            }

    # Print comparison table
    print("\n Framework           | Mean Score | Std Dev | Range      | Evaluations")
    print("-" * 70)
    for framework_id, stats in sorted(framework_stats.items()):
        print(f" {framework_id:18} | {stats['mean']:10.2f} | {stats['std']:7.2f} | {stats['min']:.1f} - {stats['max']:.1f} | {stats['count']:11}")

    return framework_stats


def export_results(results: List[Dict], frameworks: List[Dict], timestamp: str) -> Dict[str, Path]:
    """Export results to CSV and JSON"""
    print("\n💾 Exporting results...")

    # Create results directory
    results_dir = RESULTS_DIR / f"vlm_evaluation_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # 1. Full results CSV (manual CSV writing)
    csv_file = results_dir / "evaluation_results.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("image_id,framework_id,dimension_id,dimension_name,score,reasoning\n")
        # Write rows
        for r in results:
            # Escape reasoning text (remove quotes and newlines)
            reasoning = r['reasoning'].replace('"', '""').replace('\n', ' ')
            f.write(f'{r["image_id"]},{r["framework_id"]},{r["dimension_id"]},"{r["dimension_name"]}",{r["score"]},"{reasoning}"\n')

    output_files["csv"] = csv_file
    print(f"   ✓ CSV: {csv_file.name}")

    # 2. Framework comparison JSON
    framework_stats = analyze_framework_performance(results)

    summary = {
        "timestamp": timestamp,
        "num_images": len(set(r["image_id"] for r in results)),
        "num_frameworks": len(frameworks),
        "total_evaluations": len(results),
        "framework_performance": framework_stats,
        "frameworks": [
            {
                "id": f["framework_id"],
                "name": f["framework_name_cn"],
                "dimensions": f["num_dimensions"]
            }
            for f in frameworks
        ]
    }

    summary_file = results_dir / "experiment_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    output_files["summary"] = summary_file
    print(f"   ✓ Summary: {summary_file.name}")

    # 3. Full results JSON
    full_results_file = results_dir / "full_results.json"
    with open(full_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    output_files["full"] = full_results_file
    print(f"   ✓ Full results: {full_results_file.name}")

    print(f"\n📁 Results saved to: {results_dir}")

    return output_files


# ============================================================================
# MAIN
# ============================================================================

def select_framework(frameworks: List[Dict]) -> Optional[Dict]:
    """
    Interactive framework selection menu

    Returns: Selected framework or None if cancelled
    """
    print("\n📋 Available Frameworks:")
    print("="*70)
    for idx, framework in enumerate(frameworks, 1):
        print(f"  [{idx}] {framework['framework_name_cn']}")
        print(f"      ID: {framework['framework_id']}")
        print(f"      Dimensions: {framework['num_dimensions']}")
        print(f"      Description: {framework.get('description_cn', framework.get('description', 'N/A'))}")
        print()

    print("  [0] Run ALL frameworks (sequential)")
    print("  [q] Quit")
    print("="*70)

    while True:
        try:
            choice = input("\n🎯 Select framework to test (number or 'q' to quit): ").strip().lower()

            if choice == 'q':
                print("\n❌ Cancelled by user")
                return None

            choice_num = int(choice)

            if choice_num == 0:
                return "ALL"  # Special marker for all frameworks

            if 1 <= choice_num <= len(frameworks):
                selected = frameworks[choice_num - 1]
                print(f"\n✅ Selected: {selected['framework_name_cn']}")
                return selected
            else:
                print(f"⚠️  Please enter a number between 0 and {len(frameworks)}")

        except ValueError:
            print("⚠️  Please enter a valid number or 'q'")
        except KeyboardInterrupt:
            print("\n\n❌ Cancelled by user")
            return None


def validate_evaluation_quality(evaluation_results: List[Dict]) -> Dict:
    """
    Validate quality of evaluation results

    Args:
        evaluation_results: List of evaluation result dicts

    Returns:
        Quality validation report
    """
    print("\n" + "="*70)
    print("🔍 Validating Evaluation Quality")
    print("="*70)

    if not evaluation_results:
        return {"valid": False, "error": "No evaluation results to validate"}

    # Collect all scores by dimension
    dimension_scores = defaultdict(list)
    all_scores = []
    reasoning_lengths = []

    for result in evaluation_results:
        score = result["score"]
        dimension_scores[result["dimension_id"]].append(score)
        all_scores.append(score)

        if result.get("reasoning"):
            reasoning_lengths.append(len(result["reasoning"]))

    # Check 1: Score distribution (avoid all same scores)
    unique_scores = len(set(all_scores))

    # Calculate entropy
    if unique_scores > 1:
        score_counts = {}
        for s in all_scores:
            score_counts[s] = score_counts.get(s, 0) + 1
        score_entropy = -sum(
            (count / len(all_scores)) * math.log2(count / len(all_scores))
            for count in score_counts.values()
        )
    else:
        score_entropy = 0

    # Simple std calculation
    mean_score = sum(all_scores) / len(all_scores)
    variance = sum((x - mean_score) ** 2 for x in all_scores) / len(all_scores)
    std_score = variance ** 0.5

    print(f"\n  Score Distribution:")
    print(f"    Unique scores: {unique_scores}")
    print(f"    Mean: {mean_score:.2f}")
    print(f"    Std: {std_score:.2f}")
    print(f"    Range: {min(all_scores):.1f} - {max(all_scores):.1f}")

    # Check 2: Dimension variation
    print(f"\n  Dimension Variation:")
    for dimension, scores in dimension_scores.items():
        dim_mean = sum(scores) / len(scores)
        dim_var = sum((x - dim_mean) ** 2 for x in scores) / len(scores)
        dim_std = dim_var ** 0.5
        print(f"    {dimension}: μ={dim_mean:.2f}, σ={dim_std:.2f}")

    # Check 3: Reasoning quality
    if reasoning_lengths:
        mean_length = sum(reasoning_lengths) / len(reasoning_lengths)
        print(f"\n  Reasoning Quality:")
        print(f"    Mean length: {mean_length:.0f} chars")
        print(f"    Min/Max: {min(reasoning_lengths)}/{max(reasoning_lengths)}")
        print(f"    Provided: {len(reasoning_lengths)}/{len(evaluation_results)} results")

    # Quality flags
    flags = []
    if unique_scores < 5:
        flags.append("Low score diversity - possible bias")
    if std_score < 1.0:
        flags.append("Low variance - model may not be discriminating")
    if reasoning_lengths and mean_length < 50:
        flags.append("Short reasoning - may lack detail")

    if flags:
        print(f"\n  ⚠️  Quality Flags:")
        for flag in flags:
            print(f"    - {flag}")
    else:
        print(f"\n  ✅ No quality issues detected")

    return {
        "valid": len(flags) == 0,
        "total_evaluations": len(evaluation_results),
        "unique_scores": unique_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "dimension_stats": {
            dim: {
                "mean": sum(scores) / len(scores),
                "std": (sum((x - sum(scores)/len(scores)) ** 2 for x in scores) / len(scores)) ** 0.5
            }
            for dim, scores in dimension_scores.items()
        },
        "reasoning_stats": {
            "mean_length": mean_length if reasoning_lengths else 0,
            "provision_rate": len(reasoning_lengths) / len(evaluation_results) if evaluation_results else 0,
        },
        "quality_flags": flags,
    }


def main():
    """Run the PoC experiment"""
    print("🧪 CityWalkAgent VLM Framework Comparison (PoC)")
    print("="*70)
    print("Test one framework at a time with quality validation")
    print("="*70)

    # Load frameworks
    frameworks = load_frameworks()
    if not frameworks:
        print("\n❌ No frameworks loaded. Cannot proceed.")
        return

    # Interactive selection
    selected = select_framework(frameworks)
    if selected is None:
        return

    # Determine which frameworks to test
    if selected == "ALL":
        frameworks_to_test = frameworks
        print(f"\n✅ Will test ALL {len(frameworks)} frameworks sequentially")
    else:
        frameworks_to_test = [selected]
        print(f"\n✅ Will test: {selected['framework_name_cn']}")

    # Load test images
    test_images = load_test_images()
    if not test_images:
        print("\n❌ No test images found. Cannot proceed.")
        print("   Run: python experiments/02_image_collection_experiment.py")
        return

    print(f"\n🎯 Experiment Configuration:")
    print(f"   Frameworks to test: {len(frameworks_to_test)}")
    print(f"   Test images: {len(test_images)}")
    total_evals = sum(f['num_dimensions'] for f in frameworks_to_test) * len(test_images)
    print(f"   Total evaluations: {total_evals}")
    print(f"   Estimated time: ~{total_evals * 1.2 / 60:.1f} minutes")

    # Confirm
    print("\n⏸  Press Enter to start, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        return

    # Run evaluation for each framework
    all_framework_results = {}

    for framework in frameworks_to_test:
        print("\n" + "="*70)
        print(f"🎯 Testing Framework: {framework['framework_name_cn']}")
        print("="*70)

        results = run_evaluation_experiment([framework], test_images)

        if not results:
            print(f"\n❌ No results generated for {framework['framework_id']}")
            continue

        # Validate quality for this framework
        quality_report = validate_evaluation_quality(results)

        # Store results
        all_framework_results[framework['framework_id']] = {
            "results": results,
            "quality": quality_report
        }

        # Export individual framework results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        framework_name = framework['framework_id']

        # Create framework-specific output directory
        framework_dir = RESULTS_DIR / f"vlm_{framework_name}_{timestamp}"
        framework_dir.mkdir(parents=True, exist_ok=True)

        # Export results
        output_files = {}

        # 1. CSV results
        csv_file = framework_dir / "evaluation_results.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("image_id,framework_id,dimension_id,dimension_name,score,reasoning\n")
            for r in results:
                reasoning = r['reasoning'].replace('"', '""').replace('\n', ' ')
                f.write(f'{r["image_id"]},{r["framework_id"]},{r["dimension_id"]},"{r["dimension_name"]}",{r["score"]},"{reasoning}"\n')
        output_files["csv"] = csv_file
        print(f"\n   ✓ CSV: {csv_file.name}")

        # 2. Quality report JSON
        quality_file = framework_dir / "quality_report.json"
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
        output_files["quality"] = quality_file
        print(f"   ✓ Quality: {quality_file.name}")

        # 3. Full results JSON
        full_results_file = framework_dir / "full_results.json"
        with open(full_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        output_files["full"] = full_results_file
        print(f"   ✓ Full results: {full_results_file.name}")

        # 4. Framework summary
        summary = {
            "framework_id": framework['framework_id'],
            "framework_name": framework['framework_name_cn'],
            "timestamp": timestamp,
            "num_images": len(test_images),
            "num_dimensions": framework['num_dimensions'],
            "total_evaluations": len(results),
            "quality_report": quality_report
        }

        summary_file = framework_dir / "framework_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        output_files["summary"] = summary_file
        print(f"   ✓ Summary: {summary_file.name}")

        print(f"\n📁 Results saved to: {framework_dir}")

        # Print framework-specific quality summary
        print(f"\n📊 Quality Summary for {framework['framework_name_cn']}:")
        print(f"   Valid: {'✅' if quality_report['valid'] else '⚠️'}")
        print(f"   Mean Score: {quality_report['mean_score']:.2f}")
        print(f"   Std Dev: {quality_report['std_score']:.2f}")
        print(f"   Unique Scores: {quality_report['unique_scores']}")

        if len(frameworks_to_test) > 1:
            print("\n" + "─"*70)
            print("⏭️  Moving to next framework...")
            time.sleep(2)

    # Final summary
    print("\n" + "="*70)
    print("🎉 Experiment Complete!")
    print("="*70)

    if len(all_framework_results) > 1:
        print("\n📊 Framework Comparison:")
        print("-"*70)
        print(f" {'Framework':<20} | {'Mean':<6} | {'Std':<6} | {'Valid':<6} | {'Flags':<10}")
        print("-"*70)
        for fw_id, data in all_framework_results.items():
            quality = data['quality']
            valid_mark = "✅" if quality['valid'] else "⚠️"
            num_flags = len(quality['quality_flags'])
            print(f" {fw_id:<20} | {quality['mean_score']:<6.2f} | {quality['std_score']:<6.2f} | {valid_mark:<6} | {num_flags:<10}")

    print("\n📋 Next Steps:")
    if len(frameworks_to_test) == 1:
        print("   1. Review quality_report.json for validation results")
        print("   2. Check evaluation_results.csv for detailed scores")
        print("   3. Examine full_results.json for VLM reasoning")
        print("   4. If quality is good, run more images (increase NUM_TEST_IMAGES)")
        print("   5. Test other frameworks to compare")
    else:
        print("   1. Compare quality reports across all frameworks")
        print("   2. Identify which framework has best discrimination (highest std)")
        print("   3. Check which has most reasonable reasoning")
        print("   4. Select best framework for production use")

    print("\n💡 Quality Metrics to Check:")
    print("   - Unique scores: Should be > 5 (good discrimination)")
    print("   - Std dev: Higher = more discriminative (target: > 1.0)")
    print("   - Reasoning length: Should be > 50 chars (detailed explanations)")
    print("   - Quality flags: Fewer is better (target: 0 flags)")


if __name__ == "__main__":
    main()
