# Walking Agent Project - Claude Instructions

## Project Overview

Walking Agent is an AI-powered pedestrian experience analysis platform that simulates a virtual agent walking through city streets to continuously analyze and evaluate the walking environment along entire routes, not just isolated points.

## Core Technology Stack

- **ZenSVI**: Data collection engine for Street View imagery
- **VIRL**: Vision-Language Model for urban scene analysis
- **Analysis Framework**: multi-dimensional evaluation

## Project Goals

1. Analyze complete walking experiences from Point A to Point B
2. Track how pedestrian perceptions change along routes
3. Generate actionable insights for urban planning, real estate, tourism, accessibility, and research
4. Create personalized route recommendations based on individual preferences

## Key Features

- **Continuous Analysis**: 10-20 meter sampling intervals along routes
- **Multi-dimensional Evaluation**: (not yet set)
- **AI-Powered Vision**: VLM interprets street imagery like humans
- **Scalable Architecture**: City-level to nationwide analysis capability

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

## Applications

1. **Urban Planning**: Identify accident hotspots, evaluate barrier-free routes
2. **Real Estate**: Walking Environment Score as property valuation metric
3. **Tourism**: Personalized walking tour generation
4. **Accessibility**: Wheelchair-accessible route evaluation
5. **Academic Research**: Walking environment and mental health correlations
6. **Personalized Navigation**: Custom routes based on user preferences (safety/scenery/distance priority)

## When Assisting with This Project

### Code Development

- Focus on Python implementation using ZenSVI and VIRL libraries
- Prioritize parallel processing and batch operations for efficiency
- Implement proper error handling for API calls and file operations
- Use pandas/geopandas for data management
- Consider GPU optimization for VIRL analysis

### Documentation

- Maintain clear separation between data collection, analysis, and visualization phases
- Document API rate limits and cost optimization strategies
- Include both English and Chinese versions when requested
- Use minimalist design principles for presentations

### Analysis & Recommendations

- Consider all dimensions in evaluations
- Provide specific, actionable insights rather than generic observations
- Reference real-world impact examples when discussing applications
- Balance technical accuracy with accessibility for non-technical stakeholders

### Data Handling

- Always include metadata (coordinates, timestamps, headings) with images
- Use consistent coordinate systems (lat/lon format)
- Implement data validation before processing
- Consider storage and retrieval efficiency for large datasets

## Key Metrics & Performance

- 1km route ≈ 100 waypoints ≈ 400 images
- Image collection: 4 minutes (10x faster than sequential with parallel processing)
- VLM analysis: ~0.1 seconds per image
- Batch size: 32-64 images for optimal GPU utilization

## Important Considerations

- Never use localStorage/sessionStorage in artifacts (not supported in Claude.ai environment)
- Always use relative file paths and proper error handling
- Respect API rate limits and implement appropriate delays
- Consider privacy and data protection when handling street imagery
- Maintain objectivity in AI evaluations - avoid bias in scoring

## Project Philosophy

"Making cities more human-centered, one walking route at a time"

The goal is not just to measure walkability, but to understand the lived experience of pedestrians and provide data-driven insights that improve urban environments for everyone - from daily commuters to tourists, from able-bodied individuals to those with mobility challenges.

---

# Claude Project Knowledge - Walking Agent

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
