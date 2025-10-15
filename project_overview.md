# CityWalkAgent Project

## Project Overview

CityWalkAgent is an AI-powered pedestrian experience analysis platform that evaluates continuous walking experiences along entire routes. Unlike traditional point-based approaches that aggregate individual location scores, CityWalkAgent analyzes the sequential flow of pedestrian perception to detect hidden barriers, experience volatility, and transition patterns that affect actual walking satisfaction.

## Core Differentiation

**Traditional Approach**: Point A [score] + Point B [score] + Point C [score] → Average
**CityWalkAgent**: Point A → transition → Point B → transition → Point C → Experience Analysis

This paradigm shift from discrete measurement to continuous experience evaluation is the project's primary innovation.

## Core Technology Stack

- **Google Maps API**: Route generation and waypoint creation
- **ZenSVI**: Street View image collection (open-source)
- **VLM**: Vision-Language Model for scene understanding
- **Analysis Framework**: multi-dimensional evaluation
- **VIRL Architecture**: Reference for agent pipeline design

## Project Goals

1. Analyze complete walking experiences from Point A to Point B
2. Track how pedestrian perceptions change along routes
3. Generate actionable insights for urban planning, real estate, tourism, accessibility, and research
4. Create personalized route recommendations based on individual preferences

5. **Prove Sequential Analysis Superior to Aggregate Scoring**: Demonstrate that continuous measurement reveals patterns invisible to average-based methods
6. **Detect Hidden Barriers**: Identify dangerous/unpleasant segments that high average scores conceal
7. **Quantify Experience Volatility**: Measure how perception changes affect overall walking satisfaction
8. **Validate AI-Human Alignment**: Ensure AI evaluations correlate with actual pedestrian experience

## Key Features (In Development)

- **Sequential Analysis**: Volatility detection, hidden barrier identification, transition analysis
- **Multi-dimensional Evaluation**: Testing various walkability dimensions (safety, interest, comfort, aesthetics as baseline)
- **Comparative Framework**: Side-by-side comparison of Aggregate vs Sequential methods
- **Flexible Experimentation**: Rapid iteration on prompts, metrics, and analysis methods

## Technical Pipeline

### Phase 1: Data Collection (ZenSVI)

- Route definition via Maps API
- Path sampling at regular intervals
- Parallel Street View image download
- Metadata management (GPS, timestamps, angles)

### Phase 2: AI Analysis (VIRL)

- Vision-Language Model inference
- Urban element detection (sidewalks, trees, shops, lighting, etc.)
- Batch processing for efficiency
- Score generation across set framework

### Phase 3: Insights Generation

- Timeline visualization graphs
- Interactive map overlays
- Comparative analysis reports
- Heatmap generation
- API endpoints for integration

## Future Applications

1. **Urban Planning**: Evidence-based walkway improvement prioritization, ROI prediction for infrastructure investment
2. **Real Estate**: "Walk Score" API for property valuation platforms
3. **Municipal Services**: Pre/post intervention assessment for walking environment projects
4. **Insurance**: Pedestrian accident risk prediction models
5. **Retail Analysis**: Correlation between walkability dimensions and commercial success
6. **Academic Research**: Quantitative validation of urban design theories (Lynch, Jacobs, Gehl)

## Academic Positioning

**Research Gap Addressed**:

- Existing walkability indices rely on objective metrics (sidewalk width, traffic volume) → disconnect from lived experience
- Qualitative studies (Serial Vision, Phenomenology) → not scalable
- Point-based AI evaluations (SAGAI, Walk Score) → miss continuity

**CityWalkAgent's Contribution**:

- First route-level, AI-powered, continuous pedestrian experience analysis
- Bridges quantitative scale with qualitative richness
- Provides empirical validation framework for architectural/urban theories

## When Assisting with This Project

### Code Development

- Focus on Python implementation using ZenSVI and VIRL libraries
- Modular design: Easy to swap evaluation prompts and analysis formulas
- Clear data flow: Route → Images → Scores → Analysis → Insights
- Cost-conscious: Monitor API usage, use small samples for testing
- Proper error handling: API calls may fail, handle gracefully

### Analysis & Recommendations

- **Compare explicitly**: Consider all dimensions in evaluations
- **Quantify differences**: When do methods give different recommendations?
- **Focus on insights**: Not just scores, but "what does this mean?"
- **Stay grounded**: Tie findings back to urban design theory and real pedestrian concerns

### Documentation

- Maintain distinction between hypothesis and validated findings
- Document all experimental variations (prompts, thresholds, formulas)
- Track API costs and processing times
- Note limitations and edge cases discovered

### Data Handling

- Always include metadata (coordinates, timestamps, headings) with images
- Use consistent coordinate systems (lat/lon format)
- Implement data validation before processing
- Consider storage and retrieval efficiency for large datasets

## Key Metrics & Performance (Estimated)

- 1km route ≈ 50-100 waypoints (depending on sampling interval)
- Image collection: ~2-5 minutes with ZenSVI parallel processing
- VLM analysis: ~0.1 seconds per image
- Batch size: 32-64 images for optimal GPU utilization

## Important Considerations

### Technical

- Never use localStorage/sessionStorage in artifacts (not supported)
- Use structured JSON output from Claude for reliable parsing
- Implement retry logic for API calls
- Store raw responses for reanalysis without re-running API
- Always use relative file paths and proper error handling
- Respect API rate limits and implement appropriate delays

## Project Philosophy

**"A route is not defined by its endpoints, but by the journey between them."**

The goal is not just to measure walkability, but to understand the lived experience of pedestrians and provide data-driven insights that improve urban environments for everyone. A 500m walk with identical start/end scores could be consistently pleasant [8→8→8→8] or dangerously deceptive [8→3→2→8]. Only continuous analysis reveals this critical difference.

---

# Claude Project Knowledge - CityWalkAgent

## Quick Reference

### Common Tasks

1. **Generate route coordinates**: Use ZenSVI PathGenerator with start/end points and interval
2. **Collect images**: Use GSVDownloader or MLYDownloader with multi-threading
3. **Analyze images**: Use VIRL VLMAnalyzer with appropriate prompts for each dimension
4. **Visualize results**: Create timeline graphs, interactive maps, or heatmaps
5. **Generate reports**: Synthesize data into actionable insights with specific recommendations

### Code Snippets Priority

- Always include proper imports and error handling
- Use type hints for clarity
- Implement logging for debugging
- Consider async/await for I/O operations
- Optimize for batch processing
