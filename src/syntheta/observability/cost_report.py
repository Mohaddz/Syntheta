"""Cost report: formatted post-run cost breakdown."""

from __future__ import annotations

from syntheta.llm.cost import CostTracker


def format_cost_report(cost_tracker: CostTracker, model: str, elapsed: float = 0) -> str:
    """Format a human-readable cost report."""
    cost = cost_tracker.get_cost(model)

    lines = ["\nCost summary:"]
    lines.append(
        f"  Total tokens:     {cost['total_tokens']:,} "
        f"(prompt: {cost['prompt_tokens']:,} / "
        f"completion: {cost['completion_tokens']:,})"
    )

    if cost["total_cost"] is not None:
        lines.append(f"  Estimated cost:   ${cost['total_cost']:.4f} (at {model} pricing)")
    else:
        lines.append(f"  Estimated cost:   unknown (model '{model}' not in pricing database)")

    if cost["stages"]:
        lines.append("\n  By stage:")
        total_tokens = cost["total_tokens"] or 1
        for stage, stage_data in cost["stages"].items():
            st = stage_data["prompt_tokens"] + stage_data["completion_tokens"]
            pct = st / total_tokens * 100
            cost_str = f"${stage_data['cost']:.4f}" if stage_data["cost"] is not None else "?"
            lines.append(f"    {stage + ':':<25} {st:>10,} tokens  ({cost_str})  {pct:>5.1f}%")

    return "\n".join(lines)
