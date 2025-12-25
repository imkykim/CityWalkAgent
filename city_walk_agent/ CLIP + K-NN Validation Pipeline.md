### Overview Prompt

```markdown
# CLIP + K-NN Validation Pipeline for StreetAgent

You are implementing a validation pipeline that compares VLM-generated walkability
scores against human perception data from Place Pulse 2.0 using CLIP embeddings
and K-Nearest Neighbors.

## Goal

Validate that our VLM (Qwen) scores correlate with human perception by:

1. Using Place Pulse 2.0 as ground truth (110K images with crowdsourced scores)
2. Finding visually similar images via CLIP embeddings
3. Comparing VLM scores against K-NN predicted human scores

## Key Constraint

- VLM uses direct scoring (1-10), Place Pulse uses pairwise comparison (TrueSkill)
- These scales are NOT directly comparable
- Use Spearman correlation (rank-based) as primary metric
- Use Z-score normalization for secondary metrics

## Project Structure

Create all new files under: city_walk_agent/src/validation/
Experiment script: city_walk_agent/experiments/04_clip_knn_validation.py
```

---

### Step 1: Place Pulse Data Loader

````markdown
# Step 1: Create Place Pulse 2.0 Data Loader

## Task

Create `src/validation/place_pulse_loader.py` that handles Place Pulse 2.0 dataset.

## Requirements

### Data Source

- Place Pulse 2.0 from MIT Media Lab
- Contains ~110,988 street view images with crowdsourced perception scores
- 6 dimensions: safe, lively, beautiful, wealthy, boring, depressing
- We only need 4: safe, lively, beautiful, wealthy

### Class: PlacePulseLoader

```python
class PlacePulseLoader:
    def __init__(self, data_dir: Path):
        """Initialize with path to Place Pulse data directory."""

    def load_scores(self) -> pd.DataFrame:
        """
        Load and return scores DataFrame.
        Columns: image_id, safe, lively, beautiful, wealthy
        Scores should be normalized to 0-10 range.
        """

    def get_image_path(self, image_id: str) -> Path:
        """Return full path to image file."""

    def get_score_statistics(self) -> Dict[str, Dict]:
        """
        Return per-dimension statistics:
        {dimension: {mean, std, min, max, count}}
        """

    def sample_images(self, n: int, seed: int = 42) -> List[str]:
        """Return n random image_ids for testing."""
```
````

### Configuration

Add to `src/config/settings.py`:

- `place_pulse_dir: Path` - directory containing Place Pulse data
- `place_pulse_image_dir: Path` - subdirectory for images
- `place_pulse_scores_file: Path` - CSV with scores

### Validation

- Print dataset statistics on load
- Verify image files exist for sampled IDs
- Check score distributions are reasonable (no all-zeros, proper range)

### Output

When complete, running this should print:

```
Place Pulse 2.0 loaded: 110,988 images
Dimensions: safe, lively, beautiful, wealthy
Score ranges: 0.0 - 10.0
Mean scores: safe=5.2, lively=4.8, beautiful=5.1, wealthy=4.9
```

````

---

### Step 2: CLIP Feature Extractor

```markdown
# Step 2: Create CLIP Feature Extractor

## Task
Create `src/validation/clip_extractor.py` for extracting CLIP embeddings.

## Requirements

### Model Selection
- Use `openai/clip-vit-base-patch32` (512-dim) as default
- Option for `openai/clip-vit-large-patch14` (768-dim) for higher accuracy
- Use same model for both anchor set and target images

### Class: CLIPExtractor

```python
class CLIPExtractor:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",  # auto-detect GPU
        batch_size: int = 32
    ):
        """Initialize CLIP model and processor."""

    def extract_single(self, image_path: Path) -> np.ndarray:
        """Extract feature vector for single image. Returns (512,) or (768,)."""

    def extract_batch(self, image_paths: List[Path]) -> np.ndarray:
        """
        Extract features for multiple images.
        Returns (N, 512) or (N, 768) numpy array.
        Shows progress bar.
        """

    def extract_and_cache(
        self,
        image_paths: List[Path],
        cache_path: Path
    ) -> np.ndarray:
        """
        Extract features with caching.
        - If cache exists and matches image count, load from cache
        - Otherwise extract and save to cache
        """
````

### Performance Considerations

- Use GPU if available (check torch.cuda.is_available())
- Batch processing to avoid OOM
- Cache extracted features to avoid recomputation
- Place Pulse full extraction takes ~2-4 hours on GPU

### Cache Format

Save as `.npy` file with accompanying `.json` metadata:

```json
{
    "model_name": "openai/clip-vit-base-patch32",
    "feature_dim": 512,
    "num_images": 110988,
    "image_ids": ["id1", "id2", ...],
    "extraction_date": "2025-01-15"
}
```

### Validation

- Verify output dimensions match model spec
- Check features are normalized (L2 norm ≈ 1.0)
- Test on 10 random images before full extraction

````

---

### Step 3: K-NN Predictor

```markdown
# Step 3: Create K-NN Score Predictor

## Task
Create `src/validation/knn_predictor.py` for K-NN based score prediction.

## Requirements

### Class: KNNPredictor

```python
class KNNPredictor:
    def __init__(
        self,
        anchor_features: np.ndarray,  # (N, D) Place Pulse features
        anchor_scores: pd.DataFrame,   # DataFrame with dimension scores
        k: int = 10,
        use_faiss: bool = True  # Use FAISS for fast search
    ):
        """
        Initialize predictor with anchor set.
        Build FAISS index if use_faiss=True.
        """

    def predict_single(
        self,
        query_feature: np.ndarray
    ) -> Dict[str, float]:
        """
        Predict scores for single query image.
        Returns: {dimension: predicted_score}
        """

    def predict_batch(
        self,
        query_features: np.ndarray
    ) -> pd.DataFrame:
        """
        Predict scores for multiple query images.
        Returns DataFrame with columns: [safe, lively, beautiful, wealthy]
        """

    def get_neighbors(
        self,
        query_feature: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (indices, similarities) of K nearest neighbors.
        For debugging and analysis.
        """
````

### K-NN Algorithm Details

```
1. Compute cosine similarity between query and all anchors
2. Select top-K most similar anchors
3. Weight each neighbor by similarity score
4. Predicted score = weighted average of neighbor scores

Formula:
    weights = softmax(similarities)  # or just normalize
    predicted = sum(weights * neighbor_scores)
```

### FAISS Integration (Recommended)

- Use `faiss.IndexFlatIP` for inner product (cosine sim with normalized vectors)
- Much faster than numpy for large anchor sets
- Fallback to numpy if FAISS not available

### Validation

- Verify K neighbors are returned
- Check predicted scores are in valid range (0-10)
- Test retrieval quality: similar images should have similar scores

````

---

### Step 4: Score Normalizer

```markdown
# Step 4: Create Score Normalizer

## Task
Create `src/validation/score_normalizer.py` for handling score distribution differences.

## Requirements

### Class: ScoreNormalizer

```python
class ScoreNormalizer:
    def __init__(self):
        self.vlm_stats = None
        self.knn_stats = None

    def fit(
        self,
        vlm_scores: np.ndarray,
        knn_scores: np.ndarray
    ):
        """
        Compute statistics for both score distributions.
        Store mean, std for each.
        """

    def zscore_normalize(
        self,
        vlm_scores: np.ndarray
    ) -> np.ndarray:
        """
        Z-score normalize VLM scores and rescale to KNN distribution.

        z = (vlm - vlm_mean) / vlm_std
        normalized = z * knn_std + knn_mean
        """

    def rank_transform(
        self,
        scores: np.ndarray
    ) -> np.ndarray:
        """
        Convert scores to ranks (1 to N).
        Used for Spearman correlation.
        """

    def get_distribution_report(self) -> Dict:
        """
        Return comparison of VLM vs KNN distributions:
        - Mean, std, min, max for each
        - Skewness comparison
        - Score range utilization
        """
````

### Why Normalization Matters

```
VLM Direct Scoring tends to:
- Cluster around 6-8 (optimistic bias)
- Rarely use extreme values (1-3, 9-10)
- Have lower variance than human pairwise data

Place Pulse TrueSkill scores:
- More uniform distribution
- Better differentiation at extremes
- Higher variance

Without normalization:
- MAE/RMSE would be artificially high
- Pearson correlation would be misleading
```

### Output Report Format

```
Distribution Comparison:
                 VLM         K-NN Pred
Mean:           7.12         5.34
Std:            0.89         1.67
Min:            4.50         1.23
Max:            8.50         8.91
Skewness:      -0.32         0.12

Score Range Utilization:
VLM:    [4.5 ----####----  8.5] (40% of 1-10 range)
K-NN:   [1.2 --##########-- 8.9] (77% of 1-10 range)
```

````

---

### Step 5: Validation Analyzer

```markdown
# Step 5: Create Validation Analyzer

## Task
Create `src/validation/validation_analyzer.py` for statistical analysis.

## Requirements

### Class: ValidationAnalyzer

```python
class ValidationAnalyzer:
    def __init__(
        self,
        vlm_scores: pd.DataFrame,    # Columns: safe, lively, beautiful, wealthy
        knn_predictions: pd.DataFrame,  # Same structure
        image_ids: List[str]
    ):
        """Store score pairs for analysis."""

    def analyze_dimension(
        self,
        dimension: str
    ) -> Dict[str, Any]:
        """
        Full analysis for single dimension.

        Returns:
        {
            'dimension': str,
            'n_samples': int,

            # Primary metric (no normalization needed)
            'spearman_rho': float,
            'spearman_p': float,
            'spearman_significant': bool,

            # Secondary metrics (after z-score normalization)
            'pearson_r': float,
            'pearson_p': float,
            'mae': float,
            'rmse': float,
            'r2': float,

            # Distribution info
            'vlm_mean': float,
            'vlm_std': float,
            'knn_mean': float,
            'knn_std': float,
        }
        """

    def analyze_all_dimensions(self) -> pd.DataFrame:
        """
        Run analysis for all 4 dimensions.
        Returns summary DataFrame.
        """

    def generate_report(self) -> str:
        """
        Generate human-readable validation report.
        Include interpretation of results.
        """

    def plot_correlation(
        self,
        dimension: str,
        output_path: Path
    ):
        """
        Create scatter plot: VLM scores vs K-NN predictions.
        Include regression line and correlation stats.
        """

    def plot_all_dimensions(self, output_dir: Path):
        """Create 2x2 subplot with all dimension correlations."""
````

### Statistical Tests

```python
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Primary: Spearman (rank correlation, no normalization)
rho, p_value = spearmanr(vlm_scores, knn_scores)

# Secondary: Pearson (after z-score normalization)
r, p = pearsonr(vlm_normalized, knn_scores)

# Error metrics (after normalization)
mae = mean_absolute_error(knn_scores, vlm_normalized)
rmse = np.sqrt(mean_squared_error(knn_scores, vlm_normalized))
```

### Report Format

```
═══════════════════════════════════════════════════════════════
            CLIP + K-NN Validation Report
═══════════════════════════════════════════════════════════════

Dataset: 62 waypoints from Hong Kong route
Anchor Set: Place Pulse 2.0 (110,988 images)
K-NN Configuration: K=10, cosine similarity

───────────────────────────────────────────────────────────────
                    PRIMARY METRIC: Spearman Correlation
───────────────────────────────────────────────────────────────
Dimension    │  ρ (rho)  │  p-value  │  Significant
─────────────┼───────────┼───────────┼──────────────
Safety       │   0.68    │   <0.001  │     ✓
Lively       │   0.54    │   <0.001  │     ✓
Beautiful    │   0.61    │   <0.001  │     ✓
Wealthy      │   0.72    │   <0.001  │     ✓
─────────────┼───────────┼───────────┼──────────────
Average      │   0.64    │     -     │   4/4 ✓

───────────────────────────────────────────────────────────────
                    SECONDARY METRICS (Z-normalized)
───────────────────────────────────────────────────────────────
Dimension    │ Pearson r │   MAE    │   RMSE   │    R²
─────────────┼───────────┼──────────┼──────────┼─────────
Safety       │   0.65    │   1.12   │   1.38   │   0.42
Lively       │   0.51    │   1.45   │   1.72   │   0.26
Beautiful    │   0.58    │   1.28   │   1.55   │   0.34
Wealthy      │   0.69    │   1.05   │   1.31   │   0.48

───────────────────────────────────────────────────────────────
                    INTERPRETATION
───────────────────────────────────────────────────────────────
✓ All dimensions show statistically significant correlation
  with human perception (p < 0.05)

✓ Average Spearman ρ = 0.64 indicates MODERATE-STRONG alignment
  between VLM scores and crowdsourced human judgments

✓ Strongest correlation: Wealthy (ρ=0.72)
  - VLM reliably identifies affluence cues

△ Weakest correlation: Lively (ρ=0.54)
  - Temporal/social activity harder to infer from static images

Reference: Prior work (arXiv 2407.19719) achieved R²≈0.44 for
Safety using similar CLIP+K-NN approach.
═══════════════════════════════════════════════════════════════
```

````

---

### Step 6: Experiment Script

```markdown
# Step 6: Create Main Experiment Script

## Task
Create `experiments/04_clip_knn_validation.py` that orchestrates the full pipeline.

## Requirements

### Script Structure
```python
"""
CLIP + K-NN Validation Experiment

Validates VLM walkability scores against Place Pulse 2.0 human perception data.

Usage:
    python experiments/04_clip_knn_validation.py --input results/analysis_results.json
    python experiments/04_clip_knn_validation.py --input results/analysis_results.json --k 10 --use-cache
"""

def main():
    # 1. Parse arguments
    args = parse_args()

    # 2. Load VLM evaluation results
    vlm_results = load_vlm_results(args.input)
    print(f"Loaded {len(vlm_results)} waypoint evaluations")

    # 3. Load Place Pulse anchor set
    pp_loader = PlacePulseLoader(settings.place_pulse_dir)
    anchor_scores = pp_loader.load_scores()
    print(f"Loaded Place Pulse: {len(anchor_scores)} images")

    # 4. Extract/load CLIP features for anchor set
    clip_extractor = CLIPExtractor()
    anchor_features = clip_extractor.extract_and_cache(
        pp_loader.get_all_image_paths(),
        cache_path=settings.place_pulse_dir / "clip_features.npy"
    )

    # 5. Extract CLIP features for our images
    our_image_paths = [Path(r['image_path']) for r in vlm_results]
    our_features = clip_extractor.extract_batch(our_image_paths)

    # 6. Run K-NN prediction
    predictor = KNNPredictor(anchor_features, anchor_scores, k=args.k)
    knn_predictions = predictor.predict_batch(our_features)

    # 7. Extract VLM scores into DataFrame
    vlm_scores = extract_vlm_scores(vlm_results, score_type='objective')

    # 8. Run validation analysis
    analyzer = ValidationAnalyzer(vlm_scores, knn_predictions)
    results = analyzer.analyze_all_dimensions()

    # 9. Generate report and plots
    report = analyzer.generate_report()
    print(report)

    analyzer.plot_all_dimensions(output_dir=args.output_dir)

    # 10. Save results
    save_validation_results(results, report, args.output_dir)
````

### Command Line Arguments

```
--input         Path to analysis_results.json (required)
--output-dir    Output directory for results (default: results/validation/)
--k             Number of neighbors for K-NN (default: 10)
--use-cache     Use cached CLIP features if available
--score-type    Which VLM scores to use: 'objective' or 'persona' (default: objective)
--model         CLIP model to use (default: openai/clip-vit-base-patch32)
--skip-plots    Skip generating visualization plots
```

### Output Files

```
results/validation/YYYYMMDD_HHMMSS/
├── validation_report.txt      # Human-readable report
├── validation_results.json    # Machine-readable results
├── correlation_plots/
│   ├── all_dimensions.png     # 2x2 subplot
│   ├── safety.png
│   ├── lively.png
│   ├── beautiful.png
│   └── wealthy.png
├── distribution_comparison.png
└── experiment_config.json     # Reproducibility info
```

### Error Handling

- Check if Place Pulse data exists, provide download instructions if not
- Verify image paths in analysis_results.json are accessible
- Handle missing dimensions gracefully
- Provide clear error messages for common issues

````

---

### Step 7: Integration Test

```markdown
# Step 7: Create Integration Test

## Task
Create `tests/test_validation_pipeline.py` to verify the pipeline works end-to-end.

## Requirements

### Test Cases

```python
class TestValidationPipeline:
    """Integration tests for CLIP + K-NN validation."""

    def test_place_pulse_loader(self):
        """Verify Place Pulse data loads correctly."""
        loader = PlacePulseLoader(settings.place_pulse_dir)
        scores = loader.load_scores()

        assert len(scores) > 100000
        assert set(scores.columns) >= {'safe', 'lively', 'beautiful', 'wealthy'}
        assert scores['safe'].between(0, 10).all()

    def test_clip_extraction(self):
        """Verify CLIP features are extracted correctly."""
        extractor = CLIPExtractor()

        # Test on sample image
        sample_path = get_sample_image_path()
        features = extractor.extract_single(sample_path)

        assert features.shape == (512,)
        assert np.isclose(np.linalg.norm(features), 1.0, atol=0.01)

    def test_knn_prediction(self):
        """Verify K-NN predictions are reasonable."""
        # Use small subset for testing
        predictor = KNNPredictor(anchor_features[:1000], anchor_scores[:1000], k=5)

        # Query with anchor image (should get high similarity)
        query_feature = anchor_features[0]
        prediction = predictor.predict_single(query_feature)

        # Prediction should be close to actual score
        actual = anchor_scores.iloc[0]
        for dim in ['safe', 'lively', 'beautiful', 'wealthy']:
            assert abs(prediction[dim] - actual[dim]) < 2.0

    def test_spearman_correlation(self):
        """Verify Spearman correlation is computed correctly."""
        analyzer = ValidationAnalyzer(vlm_scores, knn_predictions)
        result = analyzer.analyze_dimension('safe')

        assert -1 <= result['spearman_rho'] <= 1
        assert 0 <= result['spearman_p'] <= 1

    def test_end_to_end(self):
        """Run minimal end-to-end validation."""
        # Use subset of data for speed
        result = run_validation_pipeline(
            vlm_results=sample_vlm_results,
            anchor_subset=1000,
            k=5
        )

        assert 'spearman_rho' in result
        assert result['n_samples'] > 0
````

### Mocking for CI

```python
@pytest.fixture
def mock_place_pulse_data():
    """Generate synthetic Place Pulse data for testing without real dataset."""
    n_samples = 1000
    return {
        'features': np.random.randn(n_samples, 512).astype(np.float32),
        'scores': pd.DataFrame({
            'safe': np.random.uniform(0, 10, n_samples),
            'lively': np.random.uniform(0, 10, n_samples),
            'beautiful': np.random.uniform(0, 10, n_samples),
            'wealthy': np.random.uniform(0, 10, n_samples),
        })
    }
```

```

```
