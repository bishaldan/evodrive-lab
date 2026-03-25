# Contributing to EvoDrive Lab

Thanks for taking an interest in EvoDrive Lab.

This project is still in an early public stage, so small improvements are very welcome.

## Development Workflow

### 1. Start the stack

```bash
docker compose up --build
```

### 2. Run checks

```bash
make test
make lint
make typecheck
```

### 3. Main areas of the repo

- `app/web` for the Streamlit frontend
- `app/api` for the FastAPI backend
- `app/worker` for queued run execution
- `app/simulator` for track generation, sensors, and driving physics
- `app/algorithms` for GA, NEAT, and PPO runners
- `tests` for unit and integration coverage

## Contribution Priorities

Good early contributions include:

- replay and simulation UI improvements
- better benchmark summaries and charts
- more tests around algorithm behavior
- cleaner experiment configs
- README visuals and documentation polish

## Notes

- Keep generated runtime artifacts out of commits
- Prefer Docker-based verification for consistency
- If you change behavior in the simulator or algorithms, include or update tests when possible
