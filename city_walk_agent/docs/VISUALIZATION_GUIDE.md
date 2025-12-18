# Visualization Guide

## Complete Visualization Set

The dual VLM demo generates **TWO SETS** of visualizations:

### Set 1: Persona Impact (Neutral vs Persona-Aware)

Shows how personality hints influence VLM evaluation:

| File | Description |
|------|-------------|
| `persona_neutral_overview.png` | Combined average trend + heatmap of persona adjustments |
| `persona_comparison.png` | Line plots showing neutral vs persona-aware scores with adjustment arrows |
| `persona_summary_radar.png` | Radar chart comparing average neutral vs persona scores across dimensions |
| `persona_delta_distribution.png` | Histograms + box plot showing distribution of score adjustments |
| `scores_neutral.png` | Timeline of neutral scores (no persona bias) |
| `scores_persona_aware.png` | Timeline of persona-aware scores (with personality hints) |

### Set 2: Memory/Thinking System (System 1 vs System 2)

Shows how short-term memory and thinking influence evaluation:

| File | Description |
|------|-------------|
| `system1_vs_system2_comparison.png` | 2x2 comparison showing memory influence on scores |
| `system2_adjustments_heatmap.png` | Heatmap of score adjustments from System 2 thinking |
| `system1_scores.png` | Timeline of System 1 (fast perception) scores |
| `final_scores.png` | Timeline of final scores (System 2 when triggered) |
| `narrative_timeline.png` | Score timeline with narrative chapter markers |

---

## Regenerate Missing Visualizations (single CLI)

If your current run is missing some visualizations, use the built-in CLI in `src/utils/visualization.py` (replaces `examples/regenerate_visualizations.py`):

### Option 1: Regenerate All Visualizations

```bash
python -m src.utils.visualization \
    --regenerate-output-dir outputs/hongkong_parent_30B \
    --personality parent_with_kids
```

This regenerates BOTH persona and system visualizations from your existing `analysis_results.json`.

### Option 2: Regenerate Only Persona Visualizations

```bash
python -m src.utils.visualization \
    --regenerate-output-dir outputs/hongkong_parent_30B \
    --regenerate-only persona \
    --personality parent_with_kids
```

### Option 3: Regenerate Only System Visualizations

```bash
python -m src.utils.visualization \
    --regenerate-output-dir outputs/hongkong_parent_30B \
    --regenerate-only system
```

### Option 4: Specify Personality Name for Better Titles

```bash
python -m src.utils.visualization \
    --regenerate-output-dir outputs/hongkong_parent_30B \
    --personality parent_with_kids
```

### Option 5: Include Radar Chart Sets (Slower)

```bash
python -m src.utils.visualization \
    --regenerate-output-dir outputs/hongkong_parent_30B \
    --radar-charts
```

This generates individual radar charts for System 1 and System 2 scores at each waypoint (can be slow for long routes).

---

## Quick Commands for Your Current Run

Based on your output directory `outputs/hongkong_parent_30B`:

### Generate Missing System Visualizations

```bash
python -m src.utils.visualization \
    --regenerate-output-dir outputs/hongkong_parent_30B \
    --regenerate-only system
```

This will add:
- `system1_vs_system2_comparison.png`
- `system2_adjustments_heatmap.png`
- `system1_scores.png`
- `final_scores.png`
- `narrative_timeline.png`

### Regenerate Everything with Correct Personality Name

```bash
python -m src.utils.visualization \
    --regenerate-output-dir outputs/hongkong_parent_30B \
    --personality parent_with_kids
```

---

## Understanding the Visualizations

### Persona Impact Visualizations

**Purpose:** Show how personality hints change VLM perception

**Example insights:**
- Parent personality boosts safety scores near schools
- Photographer personality boosts aesthetics in visually interesting areas
- Runner personality penalizes uneven surfaces

**Use case:** Understanding systematic bias in persona-aware evaluation

---

### Memory/Thinking System Visualizations

**Purpose:** Show how short-term memory and deep thinking revise scores

**Example insights:**
- System 2 triggers on visual changes
- Memory context causes score revisions
- Thinking adds interpretation and significance

**Use case:** Understanding how cognitive architecture influences decision-making

---

## Visualization-Only Workflow

If you want to regenerate visualizations without re-running VLM:

1. **Run demo once** to generate `analysis_results.json`:
   ```bash
   python examples/dual_system_demo.py --personality parent_with_kids
   ```

2. **Regenerate visualizations anytime**:
   ```bash
   python -m src.utils.visualization \
       --regenerate-output-dir outputs/hongkong_parent_30B \
       --personality parent_with_kids
   ```

3. **Experiment with different visualization settings** without VLM costs

---

## Troubleshooting

### Issue: "analysis_results.json not found"

**Solution:** Run the demo first to generate analysis data:
```bash
python examples/dual_system_demo.py --personality parent_with_kids
```

### Issue: Missing persona visualizations

**Cause:** No persona adjustments in data (persona hint not configured)

**Check:** Look for `persona_adjustments` and `neutral_scores` fields in `analysis_results.json`

### Issue: System visualizations look wrong

**Cause:** May need to specify correct framework ID

**Solution:**
```bash
python -m src.utils.visualization \
    --regenerate-output-dir outputs/hongkong_parent_30B \
    --framework-id streetagent_5d
```

---

## Output Structure

After running the demo or regeneration script:

```
outputs/hongkong_parent_30B/
├── analysis_results.json           # Complete data with dual VLM scores
├── analysis_results_system1.json   # System 1 only
├── analysis_results_system2.json   # System 2 only
├── thinking_results.json           # Deep thinking outputs
├── narrative_chapters.json         # Narrative chapters
└── visualizations/
    ├── persona_neutral_overview.png
    ├── persona_comparison.png
    ├── persona_summary_radar.png
    ├── persona_delta_distribution.png
    ├── scores_neutral.png
    ├── scores_persona_aware.png
    ├── system1_vs_system2_comparison.png
    ├── system2_adjustments_heatmap.png
    ├── system1_scores.png
    ├── final_scores.png
    └── narrative_timeline.png
```

---

## Quick Reference Commands

```bash
# Generate missing visualizations for your current run
python -m src.utils.visualization \
    --regenerate-output-dir outputs/hongkong_parent_30B \
    --personality parent_with_kids

# Only persona visualizations
python -m src.utils.visualization \
    --regenerate-output-dir outputs/hongkong_parent_30B \
    --personality parent_with_kids \
    --regenerate-only persona

# Only system visualizations
python -m src.utils.visualization \
    --regenerate-output-dir outputs/hongkong_parent_30B \
    --regenerate-only system

# With radar charts
python -m src.utils.visualization \
    --regenerate-output-dir outputs/hongkong_parent_30B \
    --radar-charts
```

---

Enjoy exploring your data! 📊🎨
