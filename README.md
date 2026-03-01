# lucidia-ai-models-enhanced

> **Lucidia AI Models Enhanced** — Advanced model pipeline with quantization methods, LoRA adapter management, and fine-tuning job tracking.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![npm](https://img.shields.io/badge/npm-lucidia--ai--models--enhanced-CB3837?logo=npm)](https://www.npmjs.com/package/lucidia-ai-models-enhanced)
[![BlackRoad AI](https://img.shields.io/badge/BlackRoad-AI-FF1D6C)](https://blackroad.ai)
[![License](https://img.shields.io/badge/license-Proprietary-black)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Pipeline Stages](#pipeline-stages)
4. [Features](#features)
5. [Requirements](#requirements)
6. [Installation](#installation)
   - [Python](#python)
   - [npm / Node.js Client](#npm--nodejs-client)
7. [Configuration](#configuration)
   - [Quantization Method Comparison](#quantization-method-comparison)
8. [Usage](#usage)
   - [CLI](#cli)
   - [Python API](#python-api)
9. [API Reference](#api-reference)
10. [Billing & Stripe Integration](#billing--stripe-integration)
11. [Running Tests](#running-tests)
12. [Database Schema](#database-schema)

---

## Overview

`lucidia-ai-models-enhanced` extends the base model registry with a full production
pipeline: apply **quantization** (int8/int4/GPTQ/AWQ/BNB), attach **LoRA adapters**
for parameter-efficient fine-tuning, track **fine-tuning jobs** epoch-by-epoch, and
export pipeline configs as portable JSON — all persisted in SQLite, no GPU required
at development time.

### Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                    lucidia-ai-models-enhanced                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Base Model                                                           │
│       │                                                                │
│       ▼                                                                │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────────────┐    │
│  │ Quantization │───▶│  LoRA Adapter │───▶│  Fine-Tune Tracker  │    │
│  │              │    │               │    │                     │    │
│  │ int8  50% ↓  │    │ rank=16       │    │ epochs / lr / loss  │    │
│  │ int4  75% ↓  │    │ alpha=32      │    │ train/eval curves   │    │
│  │ gptq  73% ↓  │    │ q/k/v proj   │    │ best checkpoint     │    │
│  │ awq   74% ↓  │    │ 0.03% params │    │ status tracking     │    │
│  │ bnb   50% ↓  │    │               │    │                     │    │
│  └──────────────┘    └───────────────┘    └─────────────────────┘    │
│                                                      │                │
│                                                      ▼                │
│                                            ┌──────────────────┐      │
│                                            │   Export JSON    │      │
│                                            │  Pipeline Config │      │
│                                            └──────────────────┘      │
│                                                                        │
│  Database: ~/.blackroad/ai_models_enhanced.db                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

```
Stage 1: Quantization
  ├── Choose method: int8 | int4 | gptq | awq | bnb
  ├── Configurable bits & group_size
  ├── Tracks: size_reduction_pct, perplexity_delta
  └── Status: pending → done

Stage 2: LoRA Adapter
  ├── rank, alpha, target_modules
  ├── Auto-calculates trainable_params %
  ├── Supports merge-to-base
  └── Status: pending → done

Stage 3: Fine-Tuning
  ├── dataset, epochs, learning_rate, batch_size
  ├── Per-epoch loss logging
  ├── Best checkpoint tracking
  └── Status: queued → running → completed | failed
```

---

## Features

- ⚖️ **5 Quantization Methods** — int8, int4, GPTQ, AWQ, BNB with realistic stats
- 🔗 **LoRA Adapter Management** — rank/alpha config, trainable-parameter calculation
- 🏋️ **Fine-Tuning Tracker** — epoch-level loss curves, duration, best checkpoint
- 🔍 **Pipeline Status** — three-stage dashboard per model
- 📦 **JSON Export** — portable model config snapshots
- 🗄️ **SQLite Persistence** — zero-config local database
- 🖥️ **Full CLI** — `quantize`, `lora`, `finetune`, `status`, `export`, `adapters`

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.9 | Runtime |
| pytest | ≥ 7.0 | Testing |

---

## Installation

### Python

```bash
git clone https://github.com/BlackRoad-AI/lucidia-ai-models-enhanced.git
cd lucidia-ai-models-enhanced
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### npm / Node.js Client

A lightweight Node.js client is available for integrating Lucidia AI Models Enhanced into JavaScript and TypeScript projects.

```bash
npm install lucidia-ai-models-enhanced
```

```js
const { LucidiaClient } = require('lucidia-ai-models-enhanced');

const client = new LucidiaClient({ apiKey: process.env.LUCIDIA_API_KEY });

// Quantize a model
const result = await client.quantize({
  modelId: 'llama3-8b',
  method: 'awq',
  bits: 4,
});
console.log(`Size reduction: ${result.sizeReductionPct}%`);
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LUCIDIA_ENHANCED_DB` | `~/.blackroad/ai_models_enhanced.db` | DB path |

### Quantization Method Comparison

| Method | Bits | Size Reduction | Perplexity Δ | Best For |
|--------|------|----------------|--------------|----------|
| `int8` | 8 | 50% | +0.10 | Fast, minimal quality loss |
| `bnb` | 8 | 50% | +0.12 | QLoRA training |
| `gptq` | 4 | 73% | +0.30 | Inference, balanced |
| `awq` | 4 | 74% | +0.25 | Inference, best quality |
| `int4` | 4 | 75% | +0.80 | Maximum compression |

---

## Usage

### CLI

#### Quantize a model

```bash
python src/ai_models_enhanced.py quantize \
    --model-id llama3-8b \
    --method awq \
    --bits 4 \
    --group-size 128
```

**Output:**
```
✓ Quantized llama3-8b → awq (-74% size, +0.25 ppl)
```

#### Add a LoRA adapter

```bash
python src/ai_models_enhanced.py lora \
    --model-id llama3-8b \
    --name code-instruct-v1 \
    --rank 16 \
    --alpha 32.0 \
    --modules q_proj v_proj k_proj o_proj
```

**Output:**
```
✓ LoRA adapter code-instruct-v1 added rank=16 α=32.0 trainable=0.037%
```

#### Run a fine-tuning job

```bash
python src/ai_models_enhanced.py finetune \
    --model-id llama3-8b \
    --dataset alpaca-cleaned-52k \
    --epochs 3 \
    --lr 2e-4
```

**Output:**
```
▶ Fine-tune job a3b2c1d0 started [alpaca-cleaned-52k, 3 epochs, lr=0.0002]
  Epoch 1/3 — train_loss=2.0000 eval_loss=2.2000
  Epoch 2/3 — train_loss=1.0000 eval_loss=1.1000
  Epoch 3/3 — train_loss=0.6667 eval_loss=0.7333
✓ Job complete in 0.15s — best checkpoint: checkpoints/a3b2c1d0/best
```

#### Check pipeline status

```bash
python src/ai_models_enhanced.py status --model-id llama3-8b
```

**Output:**
```
── Pipeline: llama3-8b ──────────
  ✓ quantization         awq
  ✓ lora_adapter         code-instruct-v1 rank=16
  ✓ fine_tuning          completed loss=0.6667
```

#### Export model config

```bash
python src/ai_models_enhanced.py export \
    --model-id llama3-8b \
    --output llama3-8b-pipeline.json
```

**Output:**
```
✓ Exported model config → llama3-8b-pipeline.json
```

**JSON content:**
```json
{
  "model_id": "llama3-8b",
  "exported_at": "2025-01-15T12:00:00",
  "quantizations": 1,
  "adapters": 1,
  "completed_jobs": 1
}
```

#### List adapters

```bash
python src/ai_models_enhanced.py adapters --model-id llama3-8b
```

**Output:**
```
  a3f2c1e8 code-instruct-v1 rank=16 α=32.0
  b1d4c2e0 math-tuned-v2    rank=32 α=64.0 [merged]
```

---

### Python API

```python
from src.ai_models_enhanced import (
    EnhancedModelPipeline, QuantizationConfig,
    LoRAAdapter, FineTuneJob
)

pipeline = EnhancedModelPipeline()

# Step 1: Quantize
q = pipeline.quantize_model(QuantizationConfig(
    model_id="llama3-8b",
    method="awq",
    bits=4,
    group_size=128,
))
print(f"Size reduction: {q.size_reduction_pct}%")

# Step 2: Add LoRA adapter
adapter = pipeline.add_lora(LoRAAdapter(
    model_id="llama3-8b",
    name="code-instruct",
    rank=16,
    alpha=32.0,
    target_modules=["q_proj", "v_proj"],
))
pct = adapter.trainable_params / adapter.base_model_params * 100
print(f"Trainable: {pct:.3f}% of base model")

# Step 3: Fine-tune
job = pipeline.track_finetune(FineTuneJob(
    model_id="llama3-8b",
    dataset="alpaca",
    epochs=3,
    learning_rate=2e-4,
))
print(f"Final train loss: {job.train_loss:.4f}")

# Check status
for stage in pipeline.get_pipeline_status("llama3-8b"):
    print(f"{stage.stage}: {stage.status} — {stage.details}")

# Export
pipeline.export_model("llama3-8b", "output.json")

pipeline.close()
```

---

## API Reference

### `EnhancedModelPipeline`

| Method | Returns | Description |
|--------|---------|-------------|
| `quantize_model(cfg)` | `QuantizationConfig` | Apply & record quantization |
| `add_lora(adapter)` | `LoRAAdapter` | Register LoRA adapter config |
| `track_finetune(job)` | `FineTuneJob` | Run & track fine-tuning job |
| `get_pipeline_status(model_id)` | `List[PipelineStage]` | Three-stage status |
| `export_model(model_id, path)` | `str` | Export config to JSON |
| `list_adapters(model_id?)` | `List[LoRAAdapter]` | List all/filtered adapters |
| `close()` | `None` | Close DB connection |

### `QuantizationConfig` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_id` | `str` | required | Target model ID |
| `method` | `str` | `"int8"` | Quantization method |
| `bits` | `int` | `8` | Bit width |
| `group_size` | `int` | `128` | Group size for group quantization |
| `desc_act` | `bool` | `False` | Activation reordering (GPTQ) |

### `LoRAAdapter` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_id` | `str` | required | Base model ID |
| `name` | `str` | required | Adapter name |
| `rank` | `int` | `16` | LoRA rank (r) |
| `alpha` | `float` | `32.0` | LoRA alpha (scaling) |
| `target_modules` | `List[str]` | `["q_proj","v_proj"]` | Modules to adapt |
| `task_type` | `str` | `"CAUSAL_LM"` | PEFT task type |

---

## Billing & Stripe Integration

Lucidia AI Models Enhanced uses **Stripe** for subscription billing. All production API access is metered and billed monthly through Stripe.

### Plans

| Plan | Price | Requests / Month | Fine-Tune Jobs | Support |
|------|-------|-----------------|----------------|---------|
| **Starter** | $29 / mo | 10,000 | 5 | Community |
| **Pro** | $99 / mo | 100,000 | 50 | Email |
| **Enterprise** | Custom | Unlimited | Unlimited | Dedicated |

[→ Subscribe at lucidia.earth/pricing](https://lucidia.earth/pricing)

### Setup

1. Create an account at [lucidia.earth](https://lucidia.earth) and subscribe to a plan.
2. Retrieve your API key from the **Dashboard → API Keys** page.
3. Export it as an environment variable:

```bash
export LUCIDIA_API_KEY="sk_live_..."
```

### Webhooks

Stripe webhook events are forwarded to your configured endpoint. Supported events:

| Event | Description |
|-------|-------------|
| `customer.subscription.created` | New subscription activated |
| `customer.subscription.updated` | Plan changed |
| `customer.subscription.deleted` | Subscription cancelled |
| `invoice.payment_succeeded` | Monthly invoice paid |
| `invoice.payment_failed` | Payment failure — service suspended |

Configure your webhook endpoint URL in the [Stripe Dashboard](https://dashboard.stripe.com/webhooks) pointing to `https://api.lucidia.earth/webhooks/stripe`.

---

## Running Tests

```bash
pytest tests/test_ai_models_enhanced.py -v

# Expected: 18 passed
```

---

## Database Schema

```sql
-- ~/.blackroad/ai_models_enhanced.db

CREATE TABLE quantizations (
    quant_id            TEXT PRIMARY KEY,
    model_id            TEXT NOT NULL,
    method              TEXT,
    bits                INTEGER,
    group_size          INTEGER,
    desc_act            INTEGER,
    size_reduction_pct  REAL,
    perplexity_delta    REAL,
    created_at          TEXT
);

CREATE TABLE lora_adapters (
    adapter_id          TEXT PRIMARY KEY,
    model_id            TEXT NOT NULL,
    name                TEXT,
    rank                INTEGER,
    alpha               REAL,
    target_modules_json TEXT,
    task_type           TEXT,
    trainable_params    INTEGER,
    base_model_params   INTEGER,
    checkpoint_path     TEXT,
    merged              INTEGER DEFAULT 0,
    created_at          TEXT
);

CREATE TABLE finetune_jobs (
    job_id          TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL,
    adapter_id      TEXT,
    dataset         TEXT,
    epochs          INTEGER,
    learning_rate   REAL,
    batch_size      INTEGER,
    status          TEXT,
    train_loss      REAL,
    eval_loss       REAL,
    best_checkpoint TEXT,
    duration_s      REAL,
    created_at      TEXT,
    completed_at    TEXT
);
```

---

*© BlackRoad OS, Inc. All rights reserved. Proprietary — not open source.*
