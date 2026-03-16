# Runtime Loop

The execution branch runs this operational loop:

1. validate the pinned champion
2. refresh or read trusted data
3. refit the pinned strategy on fresh data
4. monitor every scheduled hour
5. trade only on rebalance bars unless forced
6. reconcile venue state
7. place legal paper or live orders
8. persist logs and state
