"""
API cost tracking utilities

Track and report costs for Qwen VLM API calls
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class APICall:
    """Single API call record"""
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    success: bool


@dataclass
class CostSummary:
    """Cost summary statistics"""
    total_calls: int
    successful_calls: int
    failed_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    cost_by_model: Dict[str, float] = field(default_factory=dict)


class CostTracker:
    """
    Track API costs for Qwen VLM

    Features:
    - Per-call cost tracking
    - Model-level aggregation
    - Cost estimation
    - Export to CSV/JSON
    """

    # Pricing per 1M tokens (as of 2025)
    PRICING = {
        "default": {"input": 0.00, "output": 0.00}  # Free tier or custom pricing
    }

    def __init__(self):
        """Initialize cost tracker"""
        self.calls: List[APICall] = []

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        success: bool = True
    ) -> float:
        """
        Record API call and calculate cost

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            success: Whether call succeeded

        Returns:
            Estimated cost in USD
        """
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        call = APICall(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            success=success
        )

        self.calls.append(call)
        return cost

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for API call"""
        model_pricing = self.PRICING.get(model, self.PRICING.get("default", {}))

        if not model_pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * model_pricing.get("input", 0.0)
        output_cost = (output_tokens / 1_000_000) * model_pricing.get("output", 0.0)

        return input_cost + output_cost

    def get_summary(self) -> CostSummary:
        """
        Get cost summary

        Returns:
            CostSummary object
        """
        total_calls = len(self.calls)
        successful_calls = sum(1 for c in self.calls if c.success)
        failed_calls = total_calls - successful_calls

        total_input_tokens = sum(c.input_tokens for c in self.calls)
        total_output_tokens = sum(c.output_tokens for c in self.calls)
        total_cost = sum(c.cost for c in self.calls)

        # Group by model
        cost_by_model = {}
        for call in self.calls:
            if call.model not in cost_by_model:
                cost_by_model[call.model] = 0.0
            cost_by_model[call.model] += call.cost

        return CostSummary(
            total_calls=total_calls,
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost=total_cost,
            cost_by_model=cost_by_model
        )

    def estimate_cost(
        self,
        model: str,
        num_images: int,
        num_dimensions: int,
        avg_input_tokens: int = 1500,
        avg_output_tokens: int = 150
    ) -> float:
        """
        Estimate cost for evaluation task

        Args:
            model: Model name
            num_images: Number of images to evaluate
            num_dimensions: Number of dimensions per image
            avg_input_tokens: Average input tokens per call
            avg_output_tokens: Average output tokens per call

        Returns:
            Estimated total cost in USD
        """
        total_calls = num_images * num_dimensions
        total_input = total_calls * avg_input_tokens
        total_output = total_calls * avg_output_tokens

        return self._calculate_cost(model, total_input, total_output)

    def export_to_csv(self, filepath: str) -> None:
        """
        Export call history to CSV

        Args:
            filepath: Output CSV file path
        """
        import csv

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "timestamp", "model", "input_tokens",
                "output_tokens", "cost", "success"
            ])

            # Data
            for call in self.calls:
                writer.writerow([
                    call.timestamp.isoformat(),
                    call.model,
                    call.input_tokens,
                    call.output_tokens,
                    f"{call.cost:.4f}",
                    call.success
                ])

    def print_summary(self) -> None:
        """Print cost summary to console"""
        summary = self.get_summary()

        print("\n" + "="*70)
        print("API Cost Summary")
        print("="*70)
        print(f"Total API calls:      {summary.total_calls}")
        print(f"Successful:           {summary.successful_calls}")
        print(f"Failed:               {summary.failed_calls}")
        print(f"\nTotal input tokens:   {summary.total_input_tokens:,}")
        print(f"Total output tokens:  {summary.total_output_tokens:,}")
        print(f"\nTotal cost:           ${summary.total_cost:.4f}")

        if summary.cost_by_model:
            print(f"\nCost by model:")
            for model, cost in summary.cost_by_model.items():
                print(f"  {model:30s} ${cost:.4f}")

        print("="*70)


# Global cost tracker instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """
    Get global cost tracker instance

    Returns:
        CostTracker singleton
    """
    global _cost_tracker

    if _cost_tracker is None:
        _cost_tracker = CostTracker()

    return _cost_tracker
