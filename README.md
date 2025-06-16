# Fine tuning pipeline

## Setup

This repository provides a pipeline for fine-tuning language models using the `uv` framework. It includes scripts for data preparation, model training, and evaluation.

### 1. Install dependencies
## Setup

This repository provides a pipeline for fine-tuning language models using the `uv` framework. It includes scripts for data preparation, model training, and evaluation.

### 1. Install dependencies

- Ensure you have `uv` installed:

```bash
pip install uv
```

- Install the required packages:

```bash
uv sync
```

## Versioning System

To track the evolution of datasets, model architectures, and fine-tuning experiments, we use a structured versioning format with auto-incremented IDs and semantic sub-versioning.

### Dataset Versioning (`{ID}-{train|valid|test}-A.B.C.D`)

- Format: `pqrs-{train|valid|test}-A.B.C.D`
- Example: `0420-train-2.0.1.0`

| Segment | Meaning                                                                 |
| ------- | ----------------------------------------------------------------------- |
| `A`     | **Data source** — Major dataset origin (e.g., new corpus, language)     |
| `B`     | **Data subsource** — Subsets, filtered groups, or categories            |
| `C`     | **Data generation method** — Major changes in how data is processed     |
| `D`     | **Minor improvements** — Prompt tweaks, label fixes, deduplication, etc |
| `train` `valid` `test` | **Data split** — Indicates whether this is training, validation, or test data |

> **ID** is a unique 4-digit prefix incremented on each new data version.

### Model Versioning (`{ID}-model-A.B.C.D`)

- Format: `pqrs-model-A.B.C.D`
- Example: `0147-model-2.1.3.5`

| Segment | Meaning                                                                 |
| ------- | ----------------------------------------------------------------------- |
| `A`     | **Model architecture** — Major model family change (e.g., LLaMA → Qwen) |
| `B`     | **Parameter size** — Model scale changes (e.g., 3B → 7B)                |
| `C`     | **Fine-tuning method or config** — e.g., LoRA rank, learning rate       |
| `D`     | **Minor generation settings** — e.g., `max_tokens`, patch, dropout      |

> **ID** is a unique 4-digit prefix incremented on each new model version.

---

### Run Versioning (`{ID}-model-A.B.C.D-data-E.F.G.H`)

- Format: `1467-model-1.2.3.4-data-1.4.5.6`
- Example: `pqrs-model-A.B.C.D-data-E.F.G.H`

Run Versioning captures:

- The **exact model version** used during training/evaluation.
- The **exact dataset version** used.
- A globally unique **Run ID** (4-digit prefix) for traceability and comparison.

> **Run** is the eval result; so it's a dataset which has output from fine tuned and base model, evaluated using RAGAS metrics along with *optional* copilot answers.

---

### Best practices for Versioning

- **Increment IDs**: Always increment the 4-digit ID for new versions.
- **Semantic Segmentation**: Use the A.B.C.D format to clearly indicate changes.
- **Metadata Tracking**: Maintain a `metadata/` directory to store all version records and run logs.
- **Data and model cards**: Make model and dataset cards in huggingface to record details about the versions, including all the hyperparameters, metrics, and any other relevant information.

### Metadata structure

All versioning and experiment metadata is tracked in the `metadata/` directory. This directory contains structured YAML files that serve as model and dataset cards, compatible with Hugging Face standards.

```plaintext
metadata/
  dataset_card.yaml   # Details about each dataset version (sources, splits, processing, metrics, etc.)
  model_card.yaml     # Details about each model version (architecture, parameters, training config, metrics, etc.)
  run_card.yaml       # Details about each run (model version, dataset version, hyperparameters, evaluation results, etc.)
```

- **dataset_card.yaml**: Documents dataset versions, sources, splits, processing steps, and evaluation metrics. Use this as your Hugging Face dataset card.
- **model_card.yaml**: Documents model versions, architectures, training configs, and performance metrics. Use this as your Hugging Face model card.
- **run_card.yaml**: Tracks each experiment/run, linking the exact model and dataset versions, hyperparameters, and evaluation results.

> Keep these files up to date for reproducibility and easy sharing on Hugging Face Hub.

---
