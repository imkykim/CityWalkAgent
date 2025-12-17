# Optional Cleanup Items

These are **non-critical, optional** improvements. The system works perfectly without them.

---

## 1. Remove Backward Compatibility Flag (Optional)

**File**: `src/agent/capabilities/thinking.py:128`

**Current Code**:
```python
def __init__(
    self,
    ...
    enable_vlm_deep_dive: bool = False,  # Keep for backward compatibility
    enable_score_revision: bool = True,
    ...
):
```

**Proposed Change**: Remove `enable_vlm_deep_dive` if not used externally

**Impact**: None (parameter is not actually used in current implementation)

**Justification**: The new System 2 uses `enable_score_revision` instead

**Recommendation**: Keep for now if API stability is important, remove in next major version

---

## 2. Add Public STM Update API (Enhancement)

**File**: `src/agent/capabilities/short_term_memory.py`

**Current Workaround** (in memory_manager.py:553):
```python
# Access internal deque (workaround - should use public API)
for item in self.stm._memory:
    if item.waypoint_id == waypoint_id:
        item.scores = thinking_result.revised_scores
        break
```

**Proposed Enhancement**: Add public method to ShortTermMemory

```python
def update_item_scores(
    self,
    waypoint_id: int,
    new_scores: Dict[str, float],
    new_summary: Optional[str] = None
) -> bool:
    """Update scores for a specific waypoint in memory.

    Args:
        waypoint_id: ID of waypoint to update
        new_scores: New score values
        new_summary: Optional new summary text

    Returns:
        True if item was found and updated, False otherwise
    """
    for item in self._memory:
        if item.waypoint_id == waypoint_id:
            item.scores = new_scores
            if new_summary:
                item.summary = new_summary
            return True
    return False
```

**Benefits**:
- Better encapsulation
- Cleaner API
- Easier to test

**Impact**: Very low (just improves code quality)

---

## 3. Implement GPS-Based LTM Retrieval (Future Feature)

**File**: `src/agent/capabilities/memory_manager.py:526`

**Current Code**:
```python
def _retrieve_relevant_ltm(
    self,
    current_gps: Tuple[float, float],
    radius_meters: float = 500.0
) -> List[Dict[str, Any]]:
    """Retrieve relevant experiences from LTM based on GPS proximity."""
    # For now, return empty list
    # TODO: Implement GPS-based LTM retrieval when needed
    return []
```

**Proposed Implementation**:
```python
def _retrieve_relevant_ltm(
    self,
    current_gps: Tuple[float, float],
    radius_meters: float = 500.0
) -> List[Dict[str, Any]]:
    """Retrieve relevant experiences from LTM based on GPS proximity."""
    lat, lon = current_gps

    # Query LTM for nearby experiences
    relevant_moments = []

    for moment in self.episodic_ltm.key_moments[-100:]:  # Last 100 moments
        moment_lat, moment_lon = moment.gps

        # Calculate distance (simplified haversine)
        distance = self._calculate_gps_distance(
            (lat, lon),
            (moment_lat, moment_lon)
        )

        if distance <= radius_meters:
            relevant_moments.append({
                'waypoint_id': moment.waypoint_id,
                'scores': moment.scores,
                'significance': moment.significance,
                'distance_meters': distance
            })

    # Sort by relevance (closer = more relevant)
    relevant_moments.sort(key=lambda x: x['distance_meters'])

    # Return top 5
    return relevant_moments[:5]

def _calculate_gps_distance(
    self,
    coord1: Tuple[float, float],
    coord2: Tuple[float, float]
) -> float:
    """Calculate distance between GPS coordinates in meters."""
    from math import radians, cos, sin, sqrt, atan2

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    R = 6371000  # Earth radius in meters

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)

    a = (sin(delta_lat / 2) ** 2 +
         cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
```

**Benefits**:
- Context-aware System 2 reasoning
- Better use of LTM
- Location-based learning

**When to Implement**: When route history accumulates and location context becomes valuable

---

## Summary

### Must Do: ❌ NONE
All critical issues already resolved

### Should Do: 📋 NONE
All high-priority items already complete

### Nice to Have: 💡 3 Items

1. Remove `enable_vlm_deep_dive` flag (API cleanup)
2. Add public STM update method (better encapsulation)
3. Implement GPS-based LTM retrieval (future enhancement)

**Current Status**: Production-ready without any of these changes

---

**Note**: These are enhancements, not fixes. The system is fully functional and tested as-is.
