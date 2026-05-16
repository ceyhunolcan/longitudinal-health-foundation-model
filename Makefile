# LHFM developer Makefile.
#
# `make <target>` from the repo root. Targets that need a venv assume you've
# activated one already (or that you don't mind the system Python).

PYTHON      ?= python
PIP         ?= $(PYTHON) -m pip
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

# ---- envs --------------------------------------------------------------------

.PHONY: help install install-dev install-cpu-torch venv

help: ## list available targets
	@awk -F ':.*?## ' '/^[a-zA-Z0-9_.-]+:.*## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: install-cpu-torch ## install project + cpu torch
	$(PIP) install -r requirements.txt

install-cpu-torch: ## install torch from the cpu-only index
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url $(TORCH_INDEX)

install-dev: install ## install dev/test tooling
	$(PIP) install pytest pytest-cov ruff pre-commit

venv: ## create .venv (you still have to activate it)
	$(PYTHON) -m venv .venv
	@echo "activate with: source .venv/bin/activate"

# ---- pipeline ----------------------------------------------------------------

.PHONY: demo data train evaluate api dashboard

demo: ## 60s end-to-end demo (generate -> features -> train -> AUROC)
	$(PYTHON) scripts/demo.py

data: ## generate synthetic cohort + features (250 x 90, seed=42)
	$(PYTHON) scripts/run_pipeline.py

data-small: ## tiny cohort for quick iteration
	$(PYTHON) scripts/run_pipeline.py --participants 30 --days 45 --seed 7 --no-parquet

train: ## train SSL + downstream
	$(PYTHON) scripts/train_model.py

train-smoke: ## 1+1 epoch end-to-end smoke train
	$(PYTHON) scripts/train_model.py --ssl-epochs 1 --downstream-epochs 1 --no-baselines

train-ema-blind: ## the methodologically honest run: predict mood without EMA features
	$(PYTHON) scripts/train_model.py --exclude-ema-features --run-tag ema-blind

evaluate: ## re-evaluate the latest checkpoint without retraining
	$(PYTHON) scripts/evaluate_model.py

scale-ablation: ## sweep pretraining cohort size and produce the scale figure
	$(PYTHON) scripts/run_scale_ablation.py

climate-holdout: ## hold out heat-wave windows from training, evaluate on them
	$(PYTHON) scripts/run_climate_holdout.py

fairness-audit: ## subgroup-stratified audit on the held-out test split
	$(PYTHON) scripts/run_fairness_audit.py

release-bundle: ## package the latest checkpoint into a release/ tree
	$(PYTHON) scripts/export_release.py

api: ## run the FastAPI service
	PYTHONPATH=src uvicorn lhfm.api.main:app --reload --port 8000

dashboard: ## run the Streamlit dashboard
	$(PYTHON) scripts/launch_dashboard.py

# ---- quality -----------------------------------------------------------------

.PHONY: test test-fast test-nontorch lint format clean

test: ## full pytest suite with coverage
	pytest --cov=src --cov-report=term-missing

test-fast: ## just the non-torch tests
	pytest tests/test_synthetic_generator.py tests/test_features.py \
	       tests/test_metrics.py tests/test_regressions.py \
	       tests/test_fairness_and_regimes.py

test-nontorch: test-fast  ## alias for `test-fast`

lint: ## ruff lint
	ruff check src/lhfm tests scripts

format: ## ruff format
	ruff format src/lhfm tests scripts

clean: ## remove generated artifacts (keeps .gitkeep)
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf data/synthetic/*.csv data/processed/*.csv data/processed/*.parquet
	rm -rf results/tables/*.csv results/tables/*.json results/figures/*.png
	rm -rf checkpoints/*.pt checkpoints/*.meta.json
	rm -rf .pytest_cache .coverage coverage.xml htmlcov

# ---- docker ------------------------------------------------------------------

.PHONY: docker-build docker-up docker-down

docker-build: ## build the docker image
	docker compose build

docker-up: ## bring api+dashboard up (api healthchecked)
	docker compose up -d

docker-down: ## tear them down
	docker compose down
