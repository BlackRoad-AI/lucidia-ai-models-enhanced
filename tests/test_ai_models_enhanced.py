"""Tests for src/ai_models_enhanced.py — Lucidia Enhanced Model Pipeline."""
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ai_models_enhanced import (
    QuantizationConfig, LoRAAdapter, FineTuneJob,
    PipelineStage, EnhancedModelPipeline, OllamaRouter, OLLAMA_BASE_URL,
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


def test_quantize_bnb(pipeline):
    result = pipeline.quantize_model(QuantizationConfig(model_id="m3", method="bnb"))
    assert result.size_reduction_pct == 50.0
    assert result.perplexity_delta == 0.12


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


# ── OllamaRouter ──────────────────────────────────────────────────────────────
def test_ollama_base_url_default():
    assert OLLAMA_BASE_URL == "http://localhost:11434"


@pytest.mark.parametrize("mention", ["@copilot", "@lucidia", "@blackboxprogramming"])
def test_detect_mention_true(mention):
    assert OllamaRouter.detect_mention(f"{mention} hello world")


@pytest.mark.parametrize("mention", ["@copilot", "@lucidia", "@blackboxprogramming"])
def test_detect_mention_case_insensitive(mention):
    assert OllamaRouter.detect_mention(f"{mention.upper()} hello")


def test_detect_mention_false_for_plain_prompt():
    assert not OllamaRouter.detect_mention("hello world")


def test_detect_mention_false_for_unknown_mention():
    assert not OllamaRouter.detect_mention("@openai tell me something")


@pytest.mark.parametrize("mention", ["@copilot", "@lucidia", "@blackboxprogramming"])
def test_strip_mention_removes_prefix(mention):
    result = OllamaRouter.strip_mention(f"{mention} what is 2+2?")
    assert result == "what is 2+2?"


def test_strip_mention_no_op_when_no_mention():
    result = OllamaRouter.strip_mention("just a plain question")
    assert result == "just a plain question"


def _make_ollama_response(text: str, model: str = "llama3") -> MagicMock:
    """Build a mock urllib response object for Ollama."""
    body = json.dumps({"model": model, "response": text, "done": True}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


@patch("urllib.request.urlopen")
def test_chat_sends_to_ollama(mock_urlopen):
    mock_urlopen.return_value = _make_ollama_response("4")
    router = OllamaRouter()
    result = router.chat("@lucidia what is 2+2?")
    assert result["response"] == "4"
    # Confirm request went to local Ollama, not any external provider
    call_args = mock_urlopen.call_args
    req = call_args[0][0]
    assert req.full_url == "http://localhost:11434/api/generate"
    assert req.method == "POST"


@patch("urllib.request.urlopen")
def test_chat_strips_mention_before_dispatch(mock_urlopen):
    mock_urlopen.return_value = _make_ollama_response("pong")
    router = OllamaRouter()
    router.chat("@copilot ping")
    req = mock_urlopen.call_args[0][0]
    payload = json.loads(req.data.decode())
    assert payload["prompt"] == "ping"


@patch("urllib.request.urlopen")
def test_chat_uses_configured_model(mock_urlopen):
    mock_urlopen.return_value = _make_ollama_response("ok", model="mistral")
    router = OllamaRouter(model="mistral")
    router.chat("@blackboxprogramming hi")
    req = mock_urlopen.call_args[0][0]
    payload = json.loads(req.data.decode())
    assert payload["model"] == "mistral"


@patch("urllib.request.urlopen")
def test_chat_uses_custom_base_url(mock_urlopen):
    mock_urlopen.return_value = _make_ollama_response("ok")
    router = OllamaRouter(base_url="http://192.168.1.10:11434")
    router.chat("@lucidia hello")
    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "http://192.168.1.10:11434/api/generate"


@patch("urllib.request.urlopen")
def test_chat_raises_url_error_when_ollama_unreachable(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
    router = OllamaRouter()
    with pytest.raises(urllib.error.URLError):
        router.chat("@lucidia are you there?")


def test_ollama_router_no_external_imports():
    """OllamaRouter must rely only on stdlib — no third-party HTTP client."""
    import importlib
    import ai_models_enhanced as mod
    # Verify no requests/httpx/aiohttp usage exists in the module source
    source = Path(mod.__file__).read_text()
    for lib in ("import requests", "import httpx", "import aiohttp"):
        assert lib not in source, f"External HTTP library '{lib}' found in source"
