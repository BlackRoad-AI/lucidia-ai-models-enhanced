"""Tests for src/ai_models_enhanced.py — Lucidia Enhanced Model Pipeline."""
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ai_models_enhanced import (
    QuantizationConfig, LoRAAdapter, FineTuneJob,
    PipelineStage, EnhancedModelPipeline,
)


@pytest.fixture
def pipeline(tmp_path):
    p = EnhancedModelPipeline(db_path=tmp_path / "test_enhanced.db")
    yield p
    p.close()


# ── quantization ──────────────────────────────────────────────────────────────
def test_quantize_int8_reduction(pipeline):
    cfg = QuantizationConfig(model_id="qwen-7b", method="int8", bits=8)
    result = pipeline.quantize_model(cfg)
    assert result.size_reduction_pct == 50.0
    assert result.perplexity_delta == 0.1


def test_quantize_int4_reduction(pipeline):
    result = pipeline.quantize_model(QuantizationConfig(model_id="llama", method="int4"))
    assert result.size_reduction_pct == 75.0


def test_quantize_gptq(pipeline):
    result = pipeline.quantize_model(QuantizationConfig(model_id="m1", method="gptq"))
    assert result.size_reduction_pct == 73.0
    assert result.perplexity_delta == 0.3


def test_quantize_awq(pipeline):
    result = pipeline.quantize_model(QuantizationConfig(model_id="m2", method="awq"))
    assert result.size_reduction_pct == 74.0
    assert result.perplexity_delta == 0.25


def test_quantize_stores_to_db(pipeline):
    pipeline.quantize_model(QuantizationConfig(model_id="stored-model", method="int8"))
    row = pipeline._conn.execute(
        "SELECT model_id, method FROM quantizations WHERE model_id='stored-model'"
    ).fetchone()
    assert row is not None
    assert row[1] == "int8"


# ── lora adapters ─────────────────────────────────────────────────────────────
def test_add_lora_trainable_params(pipeline):
    adapter = LoRAAdapter(
        model_id="llama-7b", name="code-adapter",
        rank=16, alpha=32.0,
        target_modules=["q_proj", "v_proj", "k_proj"],
    )
    result = pipeline.add_lora(adapter)
    expected = 16 * 2 * 3 * 4096
    assert result.trainable_params == expected


def test_add_lora_base_params_constant(pipeline):
    adapter = LoRAAdapter(model_id="m1", name="a1", rank=8)
    result = pipeline.add_lora(adapter)
    assert result.base_model_params == 7_000_000_000


def test_add_lora_default_modules(pipeline):
    adapter = LoRAAdapter(model_id="m1", name="default-adapter")
    result = pipeline.add_lora(adapter)
    assert result.target_modules == ["q_proj", "v_proj"]


def test_list_adapters_all(pipeline):
    pipeline.add_lora(LoRAAdapter(model_id="mA", name="adapter-A"))
    pipeline.add_lora(LoRAAdapter(model_id="mB", name="adapter-B"))
    assert len(pipeline.list_adapters()) == 2


def test_list_adapters_filtered(pipeline):
    pipeline.add_lora(LoRAAdapter(model_id="mA", name="adapter-A"))
    pipeline.add_lora(LoRAAdapter(model_id="mB", name="adapter-B"))
    only_a = pipeline.list_adapters(model_id="mA")
    assert len(only_a) == 1
    assert only_a[0].name == "adapter-A"


# ── fine-tuning ───────────────────────────────────────────────────────────────
def test_finetune_completes(pipeline):
    job = FineTuneJob(model_id="llama-7b", dataset="alpaca", epochs=2, learning_rate=2e-4)
    result = pipeline.track_finetune(job)
    assert result.status == "completed"
    assert result.train_loss > 0
    assert result.eval_loss > 0
    assert result.best_checkpoint != ""
    assert result.duration_s > 0


def test_finetune_loss_decreases_over_epochs(pipeline):
    job = FineTuneJob(model_id="m1", dataset="ds1", epochs=4)
    result = pipeline.track_finetune(job)
    # train_loss = max(0.1, 2.0 / epochs) — so final is < initial
    assert result.train_loss < 2.0


def test_finetune_completed_at_set(pipeline):
    job = FineTuneJob(model_id="m2", dataset="ds2", epochs=1)
    result = pipeline.track_finetune(job)
    assert result.completed_at is not None


# ── pipeline status ───────────────────────────────────────────────────────────
def test_pipeline_status_all_pending(pipeline):
    stages = pipeline.get_pipeline_status("brand-new-model")
    assert len(stages) == 3
    statuses = {s.stage: s.status for s in stages}
    assert statuses["quantization"] == "pending"
    assert statuses["lora_adapter"] == "pending"
    assert statuses["fine_tuning"] == "pending"


def test_pipeline_status_after_quantize(pipeline):
    pipeline.quantize_model(QuantizationConfig(model_id="qm1", method="awq"))
    statuses = {s.stage: s.status for s in pipeline.get_pipeline_status("qm1")}
    assert statuses["quantization"] == "done"
    assert statuses["lora_adapter"] == "pending"


def test_pipeline_status_after_lora(pipeline):
    pipeline.add_lora(LoRAAdapter(model_id="lm1", name="a1"))
    statuses = {s.stage: s.status for s in pipeline.get_pipeline_status("lm1")}
    assert statuses["lora_adapter"] == "done"


# ── export ────────────────────────────────────────────────────────────────────
def test_export_creates_json(pipeline, tmp_path):
    output = str(tmp_path / "export.json")
    pipeline.quantize_model(QuantizationConfig(model_id="exp-model", method="int8"))
    result_path = pipeline.export_model("exp-model", output)
    assert result_path == output
    data = json.loads(Path(output).read_text())
    assert data["model_id"] == "exp-model"
    assert data["quantizations"] == 1
    assert "exported_at" in data
