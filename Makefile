.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory
export PATH := $(HOME)/.cargo/bin:$(HOME)/.local/bin:$(PATH)
SHELL := bash
PREK_VERSION ?= 0.5.4
ZENSICAL_VERSION ?= 0.0.67
COVERAGE ?= 90
CONFIG ?=
CONFIGURED = $(if $(CONFIG),--config $(CONFIG))
MODEL ?= knowledgator/gliner-bi-base-v2.0
NAME = $(notdir $(MODEL))
INTO ?=
REGISTRY = $(MODELS)/$(or $(INTO),$(NAME))
MODELS ?= data/models
IMAGE ?= gliner:latest
IMAGE_TRAIN ?= gliner-train:latest
TORCH ?= cpu
DATA ?= data/train
PORT ?= 5800
EXPORT = uv run -q --project tools/export python
MAINTAIN = uv run -q --project tools
SCRIPTS = tools/health.py tools/upstream.py tools/models.py tests/tools
ARGS = $(filter-out $(firstword $(MAKECMDGOALS)),$(MAKECMDGOALS))
.PHONY: help install toolchain tools hooks export fixture tiny serve build fmt lint bounds test backends bench smoke coverage licenses health models pytest gates check image image-train up down release docs docs-serve

help:  ## list targets
	@awk 'BEGIN{FS=":.*##"} /^[a-z][a-zA-Z0-9_-]*:.*##/{printf "  \033[32m%-10s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)

install: toolchain tools  ## everything a dev machine needs: pinned Rust, uv, prek, cargo-llvm-cov, cargo-about, git hooks

toolchain:  ## rustup + the toolchain pinned in rust-toolchain.toml, and uv
	@command -v rustup >/dev/null || curl -fsSL https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain none
	@command -v uv >/dev/null || curl -fsSL https://astral.sh/uv/install.sh | sh
	@rustup toolchain install

tools:  ## prek, cargo-llvm-cov and cargo-about, then the git hooks
	@uv tool install --quiet prek==$(PREK_VERSION)
	@cargo install --quiet --locked cargo-llvm-cov cargo-about
	@prek install --force

hooks:  ## run every prek hook on every file (what CI runs)
	@uvx prek==$(PREK_VERSION) run --all-files --show-diff-on-failure

export:  ## export a GLiNER bi-encoder: MODEL=… → a new registry data/models/<name>/1, current = 1; INTO=<registry> → its next version, current untouched
	@next=1; [ -z "$(INTO)" ] || { test -d $(REGISTRY) || { echo "no registry at $(REGISTRY)"; exit 1; }; \
	  next=$$(( $$(ls $(REGISTRY) | grep -xE '[0-9]+' | sort -n | tail -1) + 1 )); }; \
	  test ! -e $(REGISTRY)/$$next || { echo "$(REGISTRY) already holds version $$next"; exit 1; }; \
	  rm -rf $(REGISTRY)/.export-$$next && $(EXPORT) tools/export/export.py $(MODEL) --out $(REGISTRY)/.export-$$next \
	  && mv $(REGISTRY)/.export-$$next $(REGISTRY)/$$next \
	  && { [ -n "$(INTO)" ] || printf '1\n' > $(REGISTRY)/current; } && echo "$(MODEL) → $(REGISTRY)/$$next"

fixture:  ## PyTorch predictions over the parity texts → tests/resources/parity/torch.json [MODEL=…]
	@$(EXPORT) tools/export/parity.py $(MODEL) --texts tests/resources/parity/texts.json --out tests/resources/parity/torch.json

tiny:  ## regenerate the tiny artifact the model-free tests run on → tests/resources/tiny
	@$(EXPORT) tools/export/tiny.py --out tests/resources/tiny

serve:  ## run the service (release) [CONFIG=path, args forwarded]
	@cargo run -q --release -- $(CONFIGURED) serve $(ARGS)

build:  ## release build [args forwarded]
	@cargo build --release --locked $(ARGS)

fmt:  ## format the tree
	@cargo fmt --all

lint:  ## rustfmt check + clippy, warnings are errors
	@cargo fmt --all --check && cargo clippy --locked --all-targets -- -D warnings

bounds:  ## inner layers never import outer ones: schema ← config ← core ← runtime ← api
	@! grep -rnE 'crate::(config|core|runtime|api|logger)' src/schema \
	  && ! grep -rnE 'crate::(schema|core|runtime|api|logger)' src/config \
	  && ! grep -rnE 'crate::(runtime|api|logger)' src/core \
	  && ! grep -rnE 'crate::(api|logger)' src/runtime \
	  && ! grep -rnE '\b(ort|axum|tokio)::' src/core \
	  && printf '\033[32mbounds ok\033[0m\n'

test:  ## unit and integration tests, model-free (the tiny artifact) [args forwarded]
	@cargo test --locked $(ARGS)

backends:  ## model-backed tests: parity with PyTorch on the exported model (make export first) [PARITY_FIXTURE=fixture.json]
	@cargo test --locked --release --test model -- --ignored --nocapture parity $(ARGS)

bench:  ## ms per text and peak RSS on the exported model [GLINER_MODEL__THREADS=n]
	@cargo test --locked --release --test model -- --ignored --nocapture bench $(ARGS)

smoke:  ## one real epoch on CPU through the whole train job, on a scratch registry (make export first; needs uv)
	@cargo test --locked --release --test model -- --ignored --nocapture smoke $(ARGS)

coverage:  ## line coverage of the model-free tests; fails under $(COVERAGE)%
	@cargo llvm-cov --locked --quiet --ignore-filename-regex 'src/main\.rs' --fail-under-lines $(COVERAGE) --summary-only

licenses:  ## regenerate THIRD_PARTY_LICENSES.md from the dependency tree
	@cargo about generate --locked about.hbs | tr -d "\r" > THIRD_PARTY_LICENSES.md

health:  ## dependency health report (archived, vanished, stale, dormant upstreams) → stdout
	@uv run -q --no-project python -m tools.health --output /dev/stdout

models:  ## model watch: GLiNER-family models on the Hub newer than the served base (tools/models.toml) → stdout
	@uv run -q --no-project python -m tools.models --output /dev/stdout

pytest:  ## the maintenance tools' tests (offline fixtures) and ruff [args forwarded]
	@$(MAINTAIN) ruff check --config tools/pyproject.toml $(SCRIPTS) && $(MAINTAIN) ruff format --check --config tools/pyproject.toml $(SCRIPTS)
	@$(MAINTAIN) pytest -c tools/pyproject.toml $(ARGS)

gates:  ## every vars.* a workflow reads is declared inert (false or empty) in .github/ci.vars.example
	@used="$$(grep -rhoE 'vars\.[A-Z_]+' .github/workflows | sed 's/vars\.//' | sort -u)"; \
	  for gate in $$used; do grep -qE "^$$gate=(false)?( |$$)" .github/ci.vars.example || { echo "gate $$gate is undeclared or ships enabled"; exit 1; }; done; \
	  ! grep -E '^[A-Z_]+=' .github/ci.vars.example | grep -qvE '^[A-Z_]+=(false)?( |$$)' && printf '\033[32mgates ok\033[0m\n'

check: lint bounds gates test pytest  ## the local gate

image:  ## build the runtime image → $(IMAGE)
	@docker build -f docker/Dockerfile -t $(IMAGE) .

image-train:  ## build the train image (PyTorch and the tools) → $(IMAGE_TRAIN) [TORCH=cpu|cu128]
	@docker build -f docker/Dockerfile.train --build-arg TORCH=$(TORCH) -t $(IMAGE_TRAIN) .

up:  ## run the image on :$(PORT) with the registry data/models/<name> mounted read-only [MODEL=…]
	@docker run -d --rm --name gliner -p $(PORT):5800 -v $(CURDIR)/$(MODELS)/$(NAME):/models:ro $(IMAGE)

down:  ## stop the container
	@docker stop gliner

release:  ## tag v<Cargo.toml version>, open the GitHub release, publish the image if PUBLISH_IMAGE is on
	@gh workflow run release.yml

docs:  ## the documentation site into data/ops/site
	@uvx zensical==$(ZENSICAL_VERSION) build --clean

docs-serve:  ## live documentation preview on :8000
	@uvx zensical==$(ZENSICAL_VERSION) serve

%:
	@:
