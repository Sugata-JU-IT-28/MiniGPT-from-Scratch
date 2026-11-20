PYTHON ?= python
SEED ?= 42
EXPERIMENT_SEED ?= 0
CHECKPOINT ?= checkpoints/final.pt
PORT ?= 8000
IMAGE ?= minigpt

.PHONY: install lint test data train experiments benchmark serve smoke docker-build docker-run
install:
	$(PYTHON) -m pip install -e ".[dev]"
lint:
	$(PYTHON) -m ruff check .
test:
	$(PYTHON) -m pytest
data:
	$(PYTHON) -m scripts.prepare_data --vocab-size 512 --seed $(SEED)
	$(PYTHON) -m scripts.prepare_data --vocab-size 1024 --seed $(SEED)
	$(PYTHON) -m scripts.prepare_data --vocab-size 2048 --seed $(SEED)
train:
	$(PYTHON) -m scripts.train --config configs/cpu_tiny.yaml --preset baseline --seed $(SEED)
	$(PYTHON) -m scripts.train --config configs/cpu_tiny.yaml --preset modern --seed $(SEED)
	$(PYTHON) -m scripts.phase2_summary --seed $(SEED)
experiments:
	$(PYTHON) -m scripts.run_experiments --seed $(EXPERIMENT_SEED)
	$(PYTHON) -m scripts.aggregate_results --seed $(EXPERIMENT_SEED)
benchmark:
	$(PYTHON) -m scripts.benchmark_kv_cache --seed $(SEED)
serve:
	$(PYTHON) -m minigpt.serve --checkpoint "$(CHECKPOINT)" --port $(PORT) --seed $(SEED)
smoke:
	$(PYTHON) scripts/smoke.py --checkpoint "$(CHECKPOINT)" --seed $(SEED)
docker-build:
	docker build -t $(IMAGE) .
docker-run:
	docker run --rm -p 127.0.0.1:$(PORT):8000 -v "$(CURDIR)/checkpoints:/app/checkpoints:ro" -v "$(CURDIR)/data:/app/data:ro" -e MINIGPT_CHECKPOINT="/app/$(CHECKPOINT)" $(IMAGE)
