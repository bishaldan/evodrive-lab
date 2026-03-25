DC = docker compose

.PHONY: up down restart build test lint typecheck smoke train-ga train-neat train-ppo benchmark demo-ga

up:
	$(DC) up --build

down:
	$(DC) down

restart:
	$(DC) down
	$(DC) up -d

build:
	$(DC) build

test:
	$(DC) run --rm api pytest tests -q

lint:
	$(DC) run --rm api ruff check .

typecheck:
	$(DC) run --rm api mypy app

smoke:
	$(DC) exec api curl -sS http://localhost:8000/health

train-ga:
	$(DC) exec api curl -sS -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d '{"algorithm":"ga","mode":"train","total_iterations":8}'

train-neat:
	$(DC) exec api curl -sS -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d '{"algorithm":"neat","mode":"train","total_iterations":6}'

train-ppo:
	$(DC) exec api curl -sS -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d '{"algorithm":"ppo","mode":"train","total_iterations":1000,"algorithm_params":{"total_timesteps":1000}}'

benchmark:
	$(DC) exec api curl -sS -X POST http://localhost:8000/benchmarks -H "Content-Type: application/json" -d '{"algorithms":["ga","neat","ppo"]}'

demo-ga:
	$(DC) exec api curl -sS -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d '{"name":"github-demo","algorithm":"ga","mode":"train","seed":42,"total_iterations":8,"algorithm_params":{"population_size":8,"elite_count":2,"live_display_count":8},"env":{"physics":{"max_track_segments":52,"segment_length":6.8,"track_width":6.4}}}'
