# CLIP + K-NN Validation Pipeline

This module validates VLM walkability scores against human perception data from Place Pulse 2.0 using CLIP embeddings and K-Nearest Neighbors.

## Overview

### Problem
VLM uses direct scoring (1-10), while Place Pulse 2.0 uses pairwise comparison (TrueSkill). These scales are NOT directly comparable.

### Solution
1. Use CLIP to find visually similar images from Place Pulse 2.0
2. Use K-NN to predict human perception scores based on visual similarity
3. Compare VLM scores with K-NN predictions using Spearman correlation (rank-based)

### Why Spearman?
- Primary metric: Spearman correlation (rank-based, handles different distributions)
- Secondary metrics: Pearson, MAE, RMSE, R² (after Z-score normalization)

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers Pillow scipy scikit-learn pandas numpy matplotlib tqdm faiss-cpu
```

For GPU support (recommended):
```bash
pip install faiss-gpu  # instead of faiss-cpu
```

### 2. Download Place Pulse 2.0 Dataset

1. Visit: http://pulse.media.mit.edu/data/
2. Download the dataset
3. Extract to: `data/place_pulse/`
4. Ensure structure:
   ```
   data/place_pulse/
   ├── scores.csv          # Score data
   └── images/             # Image directory
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```

### 3. Run Validation

```bash
# Basic validation
python experiments/05_clip_knn_validation.py --input results/analysis_results.json

# With caching (faster on repeated runs)
python experiments/05_clip_knn_validation.py --input results/analysis_results.json --use-cache

# Custom K value
python experiments/05_clip_knn_validation.py --input results/analysis_results.json --k 15

# Use larger CLIP model (more accurate)
python experiments/05_clip_knn_validation.py --input results/analysis_results.json --model openai/clip-vit-large-patch14
```

## Module Components

### 1. PlacePulseLoader (`place_pulse_loader.py`)
Loads and processes Place Pulse 2.0 dataset.

```python
from src.validation import PlacePulseLoader

loader = PlacePulseLoader(data_dir)
scores = loader.load_scores()  # Returns DataFrame with normalized scores (0-10)
```

### 2. CLIPExtractor (`clip_extractor.py`)
Extracts CLIP embeddings from images.

```python
from src.validation import CLIPExtractor

extractor = CLIPExtractor(model_name="openai/clip-vit-base-patch32")
features = extractor.extract_batch(image_paths)  # Returns (N, 512) array
```

**Features:**
- Auto device detection (CUDA > MPS > CPU)
- Batch processing with progress bars
- Smart caching with metadata
- L2 normalization for cosine similarity

### 3. KNNPredictor (`knn_predictor.py`)
Predicts scores using K-Nearest Neighbors.

```python
from src.validation import KNNPredictor

predictor = KNNPredictor(
    anchor_features=anchor_features,
    anchor_scores=anchor_scores,
    k=10,
    use_faiss=True
)
predictions = predictor.predict_batch(query_features)
```

**Weighting methods:**
- `uniform`: Equal weight for all neighbors
- `similarity`: Weight proportional to cosine similarity
- `softmax`: Emphasizes most similar neighbors (default)

### 4. ScoreNormalizer (`score_normalizer.py`)
Handles distribution differences between VLM and K-NN scores.

```python
from src.validation import ScoreNormalizer

normalizer = ScoreNormalizer()
normalizer.fit(vlm_scores, knn_scores)
normalized_vlm = normalizer.zscore_normalize(vlm_scores)
```

**Why needed:**
- VLM tends to cluster around 6-8 (optimistic bias)
- Place Pulse has more uniform distribution
- Z-score normalization aligns distributions for secondary metrics

### 5. ValidationAnalyzer (`validation_analyzer.py`)
Comprehensive statistical analysis and reporting.

```python
from src.validation import ValidationAnalyzer

analyzer = ValidationAnalyzer(vlm_scores, knn_predictions)
results = analyzer.analyze_all_dimensions()
report = analyzer.generate_report()
analyzer.plot_all_dimensions(output_dir)
```

**Metrics computed:**
- **Primary**: Spearman ρ (rank correlation)
- **Secondary**: Pearson r, MAE, RMSE, R²
- **Distribution**: Mean, std, skewness, range utilization

## Output Files

After running validation, you'll get:

```
results/validation/TIMESTAMP/
├── experiment_config.json          # Configuration used
├── validation_report.txt           # Human-readable report
├── validation_summary.json         # Key findings
├── validation_metrics.csv          # Per-dimension metrics
├── score_comparison.csv            # Raw VLM vs K-NN scores
└── correlation_plots/
    ├── all_dimensions.png          # 2x2 subplot
    ├── safe.png                    # Individual plots
    ├── lively.png
    ├── beautiful.png
    ├── wealthy.png
    └── *_distribution.png          # Distribution comparisons
```

## Interpreting Results

### Spearman Correlation (ρ)
- **ρ ≥ 0.7**: Strong correlation - VLM aligns well with human perception
- **0.5 ≤ ρ < 0.7**: Moderate-strong correlation - Reasonable alignment
- **0.3 ≤ ρ < 0.5**: Moderate correlation - Some alignment
- **ρ < 0.3**: Weak correlation - Poor alignment

### Statistical Significance
- **p < 0.05**: Statistically significant (not due to chance)
- **p ≥ 0.05**: Not significant (could be random)

### Example Report Interpretation

```
Dimension    │  ρ (rho)  │  p-value  │  Significant
─────────────┼───────────┼───────────┼──────────────
Safety       │   0.68    │   <0.001  │     ✓
Lively       │   0.54    │   <0.001  │     ✓
Beautiful    │   0.61    │   <0.001  │     ✓
Wealthy      │   0.72    │   <0.001  │     ✓
```

**Interpretation:**
- All dimensions are statistically significant (p < 0.001)
- Average ρ = 0.64 indicates **moderate-strong** alignment
- Wealthy shows strongest correlation (ρ=0.72) - VLM good at identifying affluence
- Lively shows weakest (ρ=0.54) - Social activity harder from static images

## Performance Tips

### 1. Use FAISS for Large Anchor Sets
```bash
# CPU version (slower)
pip install faiss-cpu

# GPU version (much faster for 110K+ anchors)
pip install faiss-gpu
```

### 2. Cache CLIP Features
```bash
# First run: Extracts and caches features (~2-4 hours for full Place Pulse)
python experiments/05_clip_knn_validation.py --input results.json --use-cache

# Subsequent runs: Loads from cache (seconds)
python experiments/05_clip_knn_validation.py --input results.json --use-cache
```

### 3. Test with Subset First
```bash
# Quick test with 1000 anchors
python experiments/05_clip_knn_validation.py --input results.json --anchor-subset 1000

# Full validation with all 110K anchors
python experiments/05_clip_knn_validation.py --input results.json
```

### 4. Choose Right CLIP Model
- **Base** (`openai/clip-vit-base-patch32`): 512-dim, faster, good accuracy
- **Large** (`openai/clip-vit-large-patch14`): 768-dim, slower, better accuracy

## Testing

### Test Individual Components

```bash
# Test Place Pulse loader
python src/validation/test_place_pulse_loader.py

# Test CLIP extractor
python src/validation/test_clip_extractor.py

# Test K-NN predictor
python src/validation/test_knn_predictor.py

# Test score normalizer
python src/validation/test_score_normalizer.py

# Test validation analyzer
python src/validation/test_validation_analyzer.py
```

### Integration Test

```bash
# Run full pipeline test
python tests/test_validation_pipeline.py
```

## Troubleshooting

### "Place Pulse 2.0 dataset not found"
Download from http://pulse.media.mit.edu/data/ and extract to `data/place_pulse/`

### "CUDA out of memory"
Reduce batch size: `--batch-size 16` or `--batch-size 8`

### "FAISS not available"
Install faiss: `pip install faiss-cpu` or `pip install faiss-gpu`

### Slow CLIP extraction
- Use GPU if available
- Increase batch size: `--batch-size 64`
- Use cached features: `--use-cache`

### Low correlation scores
- Check if VLM scores are reasonable (not all same value)
- Verify image paths are correct
- Try different K values: `--k 5` or `--k 20`
- Consider using persona scores: `--score-type persona`

## References

- Place Pulse 2.0: http://pulse.media.mit.edu/
- CLIP Paper: https://arxiv.org/abs/2103.00020
- Related work on CLIP+KNN for perception: arXiv 2407.19719

## License

This validation pipeline is part of the CityWalkAgent project.
