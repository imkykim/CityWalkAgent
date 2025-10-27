"""Utility helpers for exporting CityWalkAgent artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable


def export_evaluations_csv(
    evaluations: Iterable[Dict[str, Any]],
    output_path: Path
) -> None:
    """Export evaluation results to CSV with proper escaping."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "dimension_id", "dimension_name", "score", "reasoning"])

        for record in evaluations:
            writer.writerow([
                record.get("image_id", ""),
                record.get("dimension_id", ""),
                record.get("dimension_name", ""),
                record.get("score", ""),
                str(record.get("reasoning", "")).replace("\n", " ")
            ])
