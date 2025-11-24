# Manual Testing Checklist

## System 2 Score Revision
- [ ] System 2 triggers correctly (visual change, score volatility, distance)
- [ ] VLM receives image + context prompt
- [ ] Revised scores are parsed correctly
- [ ] Score adjustments calculated correctly
- [ ] Fallback works when VLM fails

## Sequential Context
- [ ] STM contains System 1 scores initially
- [ ] STM updates with System 2 scores after thinking
- [ ] Next waypoint's System 2 gets previous System 2 scores in context
- [ ] System 2 cache clears at route end

## Progressive Narrative
- [ ] First chapter generated correctly
- [ ] Subsequent chapters reference previous chapters
- [ ] Narrative context includes last 3 chapters
- [ ] Complete narrative compiles all chapters
- [ ] Emotional tone reflects scores

## Visualization
- [ ] Dual-system comparison plot shows both lines
- [ ] Adjustment arrows appear on triggered waypoints
- [ ] Heatmap shows adjustments correctly
- [ ] Narrative timeline aligns with score timeline
- [ ] Statistics boxes show correct values

## Output Files
- [ ] analysis_results.json contains both score sets
- [ ] narrative.jsonl contains all chapters
- [ ] complete_narrative.md is readable
- [ ] All visualizations generate without errors
