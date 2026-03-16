# Trusted Data Input

`AgentHL` consumes the trusted market cache from `data`.

It needs:
- aligned hourly market data
- funding series
- tradability flags
- cache freshness guarantees

The research branch consumes data. It does not own ingestion or freshness enforcement.
