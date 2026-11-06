This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Worktrees

Feature work happens in a per-issue worktree **inside this repo** at `/Users/jovan/Desktop/bmh/bmh-lerobot-compatibility/.claude/worktrees/<ISSUE-ID>` (full workflow: the workspace `/Users/jovan/Desktop/bmh/AGENTS.md`, "Working on a task"). Create it from the main checkout, `cd` into the main checkout, then enter it — a path under `.claude/worktrees/` enters without a permission prompt:

```bash
cd /Users/jovan/Desktop/bmh/bmh-lerobot-compatibility
git fetch origin
git worktree add .claude/worktrees/<ISSUE-ID> -b <ISSUE-ID> origin/bmh-lerobot-main
# then: EnterWorktree(path="/Users/jovan/Desktop/bmh/bmh-lerobot-compatibility/.claude/worktrees/<ISSUE-ID>")
```

Notes specific to this repo:

- The main branch is **`bmh-lerobot-main`**; create worktrees from `origin/bmh-lerobot-main` (not `origin/main`, which is what `EnterWorktree(name=…)` would pick), never commit on or push to it (git hooks enforce this for agent sessions).
- The main checkout's `.git/info/exclude` ignores `/.claude/worktrees/`, so worktrees never show up as untracked there. The folder above `.claude/` is Jovan's main checkout: never edit files or run git in it. Worktrees created before 2026-10-23 still live at `/Users/jovan/Desktop/bmh/.worktrees/bmh-lerobot-compatibility/<ISSUE-ID>`; keep using them where they are (entering one asks Jovan once).
- Create the venv **per worktree** (`uv sync --locked --extra test --extra dev`; uv's cache makes this mostly hardlinks). Don't run against the main checkout's `.venv`: lerobot is installed editable there, so you'd be testing `main`'s code, not yours. That `.venv` now sits in a parent folder of your worktree; `uv` still picks the worktree's own `pyproject.toml` and `.venv`, so just never activate or point at the parent one.
- The main checkout's venv is Python 3.14 and draccus parsing crashes there; see memory `fork-venv-py314-draccus-help` for the workaround before using `--help` or `draccus.parse` as a check.

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (one tier per extra, selected by pytest mark, every PR), `full_tests.yml` (all extras + E2E, GPU job runs `-m gpu`, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy**: one ruleset (`[tool.mypy]` in `pyproject.toml`) checks the paths in its `files` setting, except `lerobot.rl`, the vendored `molmoact2_hf_model` and the generated `*_pb2` modules. Run it with `pre-commit run mypy --all-files`. Add type annotations when modifying code.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
- **Test marks**: Fast Tests runs one tier per extra (`-m base|dataset|hardware|viz`) and the GPU job runs `-m gpu`. A test that needs the `dataset`, `hardware` or `viz` extra must be marked (`pytestmark = pytest.mark.dataset`), otherwise it only runs in `base` and skips there. A test that uses CUDA outside `tests/policies` and without `@require_cuda` needs `@pytest.mark.gpu`. Details in CONTRIBUTING.md, "Running Tests".
