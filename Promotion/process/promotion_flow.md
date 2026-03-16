# Promotion Flow

The `Promotion` branch runs this gate:

1. read leaderboard and result artifacts
2. verify the result artifact exists and matches the candidate
3. verify express-filter and full-judge status
4. apply stage-specific promotion policy
5. pin the paper or live champion

It does not run research and it does not trade.
