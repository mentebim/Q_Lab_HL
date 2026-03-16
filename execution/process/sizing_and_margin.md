# Sizing And Margin

All wallet capital can belong to the strategy.

Execution still sizes from a margin budget, not from a fixed gross multiple.

It should size from:
- live account value
- leverage map
- target margin usage ratio
- hard margin ceiling
- minimum headroom reserve

This preserves robustness while treating the full wallet as strategy capital.
