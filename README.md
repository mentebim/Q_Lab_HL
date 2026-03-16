# Q_Lab_HL

This branch is the docs-only system view that should live on `main`.

It is intentionally not a runnable repo. It organizes the production system into four role folders:

- `data/`
- `AgentHL/`
- `Promotion/`
- `execution/`

Inside each role folder, the material is grouped into:

- `inputs/`
- `process/`
- `outputs/`

The purpose is to make the system legible by responsibility instead of by mixed implementation files.

## Root Contract

The system has one job only:

`trusted data -> bounded research -> gated promotion -> safe execution`

Each root folder represents one production branch:

- [data/README.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/data/README.md)
  Central data engine. Owns ingestion, cache publication, and data trust.
- [AgentHL/README.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/AgentHL/README.md)
  Bounded research agent. Owns candidate generation and research artifacts.
- [Promotion/README.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/Promotion/README.md)
  Promotion gate. Owns champion selection and staged promotion.
- [execution/README.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/execution/README.md)
  Runtime trader. Owns paper/live execution of pinned champions.

## Handoff Model

The branches communicate through artifacts:

1. `data` publishes a trusted market cache
2. `AgentHL` publishes research artifacts and leaderboard rows
3. `Promotion` reads those artifacts and writes pinned champion files
4. `Execution` reads pinned champions and trades them

Nothing should skip a stage.

See [SYSTEM_ARCHITECTURE.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/SYSTEM_ARCHITECTURE.md) for the detailed flow.
