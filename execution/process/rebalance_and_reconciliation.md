# Rebalance And Reconciliation

Execution monitors every hour, but actual trading follows the champion rebalance cadence.

Important rules:
- respect `rebalance_every_bars`
- skip trading on non-rebalance hours unless forced
- reconcile positions, fills, and open orders before acting
- keep paper and live state separate
