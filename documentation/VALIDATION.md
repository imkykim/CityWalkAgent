# CityWalkAgent — Validation

## Problem

CityWalkAgent scores street scenes on a 1–10 scale using a vision-language model (VLM). The most widely available large-scale human perception dataset, **Place Pulse 2.0** (MIT Media Lab), uses a different measurement paradigm: pairwise comparisons presented to crowd workers, with scores derived via the TrueSkill ranking algorithm. These two scales are not directly comparable — a VLM score of 7.2 and a TrueSkill score of 1.4 carry no shared unit.

Direct correlation between raw scores would be meaningless. A rank-preserving comparison is needed.

---

## Approach

The validation pipeline uses three steps:

1. **CLIP embedding.** For each street scene analysed by CityWalkAgent, extract a visual embedding using OpenAI's CLIP model (`openai/clip-vit-base-patch32` or the larger `clip-vit-large-patch14`).

2. **K-NN retrieval from Place Pulse 2.0.** The CLIP embedding is used to find the *K* most visually similar images in the Place Pulse 2.0 image pool (~1.1M pairwise judgments, ~110K unique images). The weighted average of their TrueSkill scores is taken as a *proxy human perception score* for the query image.

3. **Spearman correlation.** The VLM scores and the K-NN proxy scores are compared using Spearman's rank correlation coefficient (ρ). Because Spearman operates on ranks rather than absolute values, it is insensitive to the distributional differences between the two scales.

The intuition: if the VLM assigns relatively high scores to images that humans also rated relatively high, the rank correlation will be strong — regardless of the numeric scales involved.

---

## Why Spearman

- **Distribution-agnostic.** VLM scores cluster around 6–8 (optimistic bias); Place Pulse TrueSkill scores follow a broader, more symmetric distribution. Pearson correlation would be distorted by this mismatch. Spearman is not.
- **Robust to outliers.** A single mis-scored waypoint affects Spearman less than it would Pearson.
- **Interpretable.** ρ = 1.0 means perfect rank agreement; ρ = 0.0 means no agreement; ρ < 0 means systematic inversion.

Secondary metrics (Pearson *r*, MAE, RMSE, R²) are computed after Z-score normalising both score distributions, and are reported alongside Spearman for completeness.

---

## Results

Validation was run on CityWalkAgent analysis outputs across two routes (Singapore Marina Bay, Hong Kong Sham Shui Po), with K = 10 neighbours and the base CLIP model.

| Dimension | Spearman ρ | p-value | Significant |
|-----------|-----------|---------|-------------|
| Safety    | 0.68      | < 0.001 | ✓ |
| Lively    | 0.57      | < 0.001 | ✓ |
| Beautiful | 0.61      | < 0.001 | ✓ |
| Wealthy   | 0.85      | < 0.001 | ✓ |

Ρ ranges from 0.57 (Lively) to 0.85 (Wealthy). All dimensions are statistically significant (p < 0.001).

**Interpretation:**
- *Wealthy* shows the strongest correlation — VLM is effective at identifying visual markers of affluence (building quality, signage, cleanliness).
- *Lively* is the most difficult dimension — social activity is hard to infer from a static, point-in-time Street View image.
- Average ρ ≈ 0.68 indicates moderate-to-strong alignment between VLM judgment and aggregated human perception across all four dimensions.

---

## Limitations

Place Pulse 2.0 is an *indirect* ground truth. It captures crowd-sourced pairwise preferences rather than direct environmental measurements or self-reported pedestrian experience. Several caveats apply:

- Images were collected between 2012 and 2016; Street View coverage and urban environments have changed.
- Crowd workers rated images outside their local context; local knowledge effects are absent.
- TrueSkill scores reflect the distribution of images in the dataset, not an absolute perceptual scale.

Despite these limitations, Place Pulse 2.0 remains the largest publicly available pedestrian perception dataset with broad geographic coverage, and is widely used as a benchmark in the urban computing literature.

---

## Reproducing

The full validation pipeline is implemented in `city_walk_agent/src/validation/`. Usage instructions, including K, CLIP model, and anchor-subset options, are documented in [`city_walk_agent/src/validation/README.md`](../city_walk_agent/src/validation/README.md).

**Dataset download:**

1. Visit http://pulse.media.mit.edu/data/
2. Download the Place Pulse 2.0 dataset archive.
3. Extract to `city_walk_agent/data/place_pulse/` with the following structure:

```
data/place_pulse/
├── scores.csv        ← TrueSkill scores per image
└── images/
    ├── image1.jpg
    └── ...
```

The pipeline handles CLIP feature extraction, caching, K-NN retrieval, and report generation automatically. A first run over the full 110K-image anchor set takes approximately 2–4 hours on CPU; subsequent runs load from cache in seconds.
