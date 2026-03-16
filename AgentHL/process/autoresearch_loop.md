# Autoresearch Loop

The `AgentHL` branch runs this loop:

1. read the research contract
2. inspect prior results and current frontier
3. propose one bounded candidate
4. run the express filter
5. run the full judge on survivors
6. record the result artifact
7. append the leaderboard row
8. decide the next bounded mutation

This branch never promotes and never trades.
