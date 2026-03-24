<h1 align="center"><b>Causal image embedding</b><br>Embeddings and ATE estimation with image covariates</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.12-blue" alt="Python" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-526EAF.svg?logo=opensourceinitiative&logoColor=white" alt="License: MIT" /></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" /></a>
</p>

Code for a causal inference setup on **Fashion-MNIST–style** images: learn image embeddings (naive vs. debiased nets), then compare **biased**, **naive**, and **debiased** pipelines using regression, IPW, and doubly robust ATE estimators. Originally the final project for **CPSC 452/552 — Spring 2025**.

## Documentation

| Resource | Description |
|----------|-------------|
| This README | Install, run scripts, Docker, tests |
| [`Dockerfile`](Dockerfile) | Reproducible environment (default command runs `pytest`) |
| [`pyproject.toml`](pyproject.toml) | Dependencies, Ruff/Mypy/pytest settings |
| [`src/config.py`](src/config.py) | Sample sizes, dimensions, training hyperparameters |
| [`src/main_experiment.py`](src/main_experiment.py) | End-to-end experiment |
| [`src/main_analysis.py`](src/main_analysis.py) | Summarize saved results (`df_result.pkl`) |
| [`tests/`](tests/) | Pytest suite (ATE helpers, autoencoder shapes) |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | CI: Ruff, Mypy, tests |

## Installation

```bash
git clone https://github.com/tatsu432/causal-image-embedding.git
cd causal-image-embedding
curl -LsSf https://astral.sh/uv/install.sh | sh   # optional; or use pip
uv sync
```

For linting, typing, and tests:

```bash
uv sync --extra dev
```

## Run the experiment

Scripts assume imports resolve from `src` (same layout as when you `cd src` on Colab).

```bash
cd src
python main_experiment.py
```

`main_analysis.py` expects `df_result.pkl` in the current working directory (produced by the experiment flow). Run it after a successful experiment:

```bash
cd src
python main_analysis.py
```

Download **Fashion-MNIST** into `data/` (or the path set in code) if it is not already present.

## Dataset (short)

We use a modified [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): an **icon overlay** on each image encodes post-treatment information (icon type, transparency, position, size). Synthetic **covariates**, **treatment**, **post-treatment** factors, **images** \(V_i\), and **outcomes** \(Y_i\) are simulated as described in the original project notes.

## Docker

```bash
docker build -t causal-image-embedding:local .
docker run --rm causal-image-embedding:local
```

The default image command runs **`pytest`**. Override the command to run an experiment, for example:

```bash
docker run --rm -w /app/src causal-image-embedding:local uv run python main_experiment.py
```

First builds can take a while while **PyTorch** and **TensorFlow** wheels download.

## Tests & CI

From the repo root:

```bash
uv sync --extra dev
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest tests/ -q
```

Pushes and pull requests to `main` / `master` run the same checks via GitHub Actions.
