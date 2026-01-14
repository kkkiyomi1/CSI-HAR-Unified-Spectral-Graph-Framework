# CSI-HAR Unified Spectral-Graph Framework  
**A Unified Supervision Pipeline for CSI-based Human Activity Recognition (CSI-HAR)**

> **Official implementation** of our unified supervision framework for CSI-HAR, combining a **Wi-Prompt inspired TCN + Temporal Attention** backbone with **spectral graph regularization** (Cheeger-style surrogate) for robust domain generalization.

[![CI](./../../actions/workflows/ci.yml/badge.svg)](./../../actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](#installation)
[![Reproducible](https://img.shields.io/badge/Reproducible-manifest%20v2%20%2B%20config-brightgreen.svg)](#reproducibility)

---

## 🔖 Paper

- **Title**: *Structured Invariances and Spectral Graph Discriminants: A Unified Supervision Framework for CSI-HAR* *(placeholder — replace with your final title)*
- **Venue**: IEEE TMC 

> If you use this repository, please cite the paper and this code (see [Citation](#-citation)).

---

## ✨ Highlights

### Unified supervision, one codebase
This repository supports **Target-Free / Weakly Supervised / Semi-Supervised** training without changing model code—only by swapping **manifests** and **configuration knobs**.

### Wi-Prompt inspired temporal encoder
A **dilated causal TCN** is paired with **temporal attention pooling**, replacing global average pooling to focus on *active frames* in CSI sequences.

### Spectral graph regularization (Cheeger-style)
We build a **kNN graph** in representation space and apply a **Cheeger-style surrogate** on soft labels derived from logits, encouraging **compact and well-separated spectral structure**.

### Reproducible research engineering
- Config-driven CLI (`csihar-train`, `csihar-eval`, `csihar-cache`)
- **Manifest v2** (JSONL) schema for explicit splits and metadata
- Built-in caching, logging, checkpoints
- Tests + CI + citation metadata

---

## 🧠 Method in 60 seconds

We represent each CSI sample as a tensor:  
\[
x \in \mathbb{R}^{C \times T \times K}
\]
where `C` is view/feature channels (e.g., amplitude/phase or temporal transforms), `T` is time, and `K` is flattened CSI bins/subcarriers/velocity spectrum bins.

**Encoder**:
- Flatten `(C, K)` per frame → `[T, C·K]`
- `LazyLinear` projection → TCN hidden dimension
- Dilated causal TCN blocks
- **Temporal attention pooling**
- MLP head → embedding `h` and logits `p(y|x)`

**Training objective** (configurable):
- Cross-entropy (optionally label smoothing / class-balanced)
- + λₙ꜀ₑ · supervised InfoNCE (optional)
- + λ꜀ʰᵍ · Cheeger surrogate (optional)

---

## 🗂️ Supported Datasets

This repository provides a **dataset registry** with two layers:

### 1) `widar3_bvp` (native)
- Widar3.0 velocity spectrum BVP stored as `.mat` (3D array)
- Built-in preprocessing: resampling to fixed `T`, flattening, temporal feature transforms, caching

### 2) `generic_csi` (adapter for WiAR / CSI survey)
For datasets not stored as Widar BVP `.mat`, use `generic_csi` by converting raw CSI into one of:
- `.npz` *(recommended)*: contains `csi` (complex array or real/imag)
- `.npy`: contains CSI array directly
- `.pt`: torch-saved tensor or dict `{ "csi": ... }`

This adapter supports:
- amplitude/phase dual-view construction
- optional phase unwrap + sanitation
- normalization options
- preprocessing cache

> ✅ In the paper we evaluate on **Widar3.0**, **WiAR**, and **CSI survey**.  
> Public release focuses on **clean, extensible adapters**; exact dataset conversion scripts can be added per license constraints.

---

## 📦 Installation

```bash
git clone <YOUR_REPO_URL>
cd csi-har-unified

pip install -U pip
pip install -e .
```

Optional dev tools:
```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

---


---

---

## 🧱 Repository Structure

This repository is organized to keep **experiments reproducible** (configs + manifests) while keeping the **core research code modular** (data / models / losses / engine).

```text
csi-har-unified/
├─ README.md            # project overview + quickstart
├─ LICENSE              # license
├─ CITATION.cff         # citation metadata
├─ pyproject.toml       # packaging + CLI entry points
│
├─ configs/             # experiment configs (YAML)
├─ docs/                # brief documentation (data / training / API)
├─ scripts/             # reproduction helpers (bash / slurm)
├─ .github/workflows/   # CI (lint + tests)
│
├─ csihar/              # main library package
│  ├─ data/             # dataset adapters + manifest loader
│  ├─ models/           # model definitions (e.g., WiPromptTCN)
│  ├─ losses/           # losses (InfoNCE, Cheeger surrogate)
│  ├─ engine/           # training & evaluation loops
│  └─ cli/              # command-line tools (train / eval / cache)
│
└─ tests/               # minimal sanity tests for CI



## 🧾 Manifest v2 (JSONL): explicit, reproducible splits

We use JSONL manifests to make splits *explicit* and *shareable*.

Each line is one sample:
```json
{"path":"/abs/path/sample.mat","label":5,"subject":"user1","domain":"roomA","device":"d1","session":"s1"}
```

Required:
- `path`
- `label` (integer; remapped to contiguous `[0..C-1]` internally)

Recommended metadata:
- `subject`, `domain`, `device`, `session`  
These are useful when generating cross-user / cross-domain / cross-device splits.

See: `docs/data.md`

---

## 🚀 Quickstart (Widar3.0 BVP)

### Step 0: Prepare manifests
You should prepare `train.jsonl`, `val.jsonl`, `test.jsonl` for your split.

### Step 1 (optional): Build preprocessing cache
```bash
csihar-cache --config configs/widar3_bvp.yaml \
  --train-manifest /path/train.jsonl \
  --val-manifest /path/val.jsonl \
  --test-manifest /path/test.jsonl
```

### Step 2: Train
```bash
csihar-train --config configs/widar3_bvp.yaml \
  --train-manifest /path/train.jsonl \
  --val-manifest /path/val.jsonl \
  --test-manifest /path/test.jsonl
```

### Step 3: Evaluate a checkpoint
```bash
csihar-eval --ckpt runs/widar3_bvp/best.pt \
  --config configs/widar3_bvp.yaml \
  --test-manifest /path/test.jsonl
```

---

## 🌍 Running WiAR / CSI survey (via `generic_csi`)

Once you convert data to `.npz/.npy/.pt` and create manifests:

```bash
csihar-train --config configs/wiar.yaml \
  --train-manifest /path/wiar_train.jsonl \
  --val-manifest /path/wiar_val.jsonl \
  --test-manifest /path/wiar_test.jsonl
```

```bash
csihar-train --config configs/csi_survey.yaml \
  --train-manifest /path/csi_survey_train.jsonl \
  --val-manifest /path/csi_survey_val.jsonl \
  --test-manifest /path/csi_survey_test.jsonl
```

---

## 🧪 Recommended Experiment Protocol (paper-style)

For each dataset/split:
1. Fix seed(s) in config
2. Keep manifests under `splits/<dataset>/<split_name>/*.jsonl` *(not necessarily public)*
3. Run caching (optional) then training
4. Record `runs/<exp_name>/metrics.jsonl` and `best.pt`

This makes ablation studies and hyper-parameter sweeps straightforward.

---

## 📈 Outputs, Logs & Checkpoints

Each run writes to:

```text
runs/<exp_name>/
  ├─ config.resolved.yaml     # resolved config snapshot (reproducibility)
  ├─ metrics.jsonl            # JSON per epoch (train/val) + final test entry
  └─ best.pt                  # best checkpoint by val_top1
```

---

## ✅ Reproducibility

This repository is reproducible by construction:

- **Explicit split**: manifests define the split
- **Config snapshotting**: `config.resolved.yaml`
- **Seed control**: `trainer.seed` in YAML
- **Determinism mode**: available in utilities (may reduce speed)

> For reporting, always state: dataset version, split recipe, config hash (or commit hash), and seed list.

---

## 🧩 Repository Structure

```text
csihar/
  data/        dataset adapters + registry + manifest tools
  models/      encoders (WiPromptTCN)
  losses/      InfoNCE + spectral Cheeger surrogate
  engine/      training / evaluation / EMA / scheduler / early stop
  cli/         csihar-train / csihar-eval / csihar-cache

configs/       paper-friendly configs
docs/          data schema, training tips, minimal API
tests/         CI sanity checks
```

---

## ❓ FAQ

**Q1: My CSI tensor shape differs. Can I still use this repo?**  
Yes. Standardize your sample into `x ∈ R^{C×T×K}`. Use the `generic_csi` adapter or create a new adapter under `csihar/data/`.

**Q2: Do I need to include target-domain data?**  
No. Target-free training is supported. Weak/partial target exposure is controlled via data manifests and config knobs (and can be extended if your paper uses specific sampling).

**Q3: Where do I implement new backbones or losses?**  
- New models: `csihar/models/`
- New losses: `csihar/losses/`
- Wire them in: `csihar/cli/train.py`

---

## 🛣️ Roadmap

- [ ] Official converters for WiAR / CSI survey raw formats (`scripts/convert_*`)  
- [ ] Canonical split builders (LOUO / LODO / cross-device)  
- [ ] Pre-trained checkpoints (if dataset license permits)  
- [ ] Extended baselines (Transformer, GRU, ResNet1D)  

---

## 📚 Citation

See `CITATION.cff`. BibTeX stub:

```bibtex
@software{zhu2026csihar,
  title   = {A Unified Spectral-Graph Framework for Cross-Domain WiFi CSI-based Human Activity Recognition},
  author  = {Zhu, Yi},
  year    = {2026},
  version = {0.1.0},
}
```

---

## 🪪 License

MIT License. See [`LICENSE`](./LICENSE).
