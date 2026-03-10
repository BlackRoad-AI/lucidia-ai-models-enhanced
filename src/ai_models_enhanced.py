"""
Lucidia AI Models Enhanced — Advanced model pipeline with quantization,
LoRA adapter management, fine-tuning tracker, and Ollama @mention routing.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ── ANSI colours ─────────────────────────────────────────────────────────────
R = "\033[0;31m"; G = "\033[0;32m"; Y = "\033[1;33m"
C = "\033[0;36m"; B = "\033[0;34m"; M = "\033[0;35m"; NC = "\033[0m"
BOLD = "\033[1m"

DB_PATH = Path.home() / ".blackroad" / "ai_models_enhanced.db"
OLLAMA_BASE_URL = "http://localhost:11434"


# ── Ollama @mention router ────────────────────────────────────────────────────
class OllamaRouter:
    """Route @mention prompts directly to a local Ollama instance.

    Recognized mentions — ``@copilot``, ``@lucidia``, ``@blackboxprogramming``
    — are stripped from the prompt before dispatch so that Ollama receives only
    the plain query.  No external provider is ever contacted.
    """

    MENTIONS: frozenset = frozenset({"@copilot", "@lucidia", "@blackboxprogramming"})

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = "llama3") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    # ── helpers ───────────────────────────────────────────────────────────────

    @classmethod
    def detect_mention(cls, prompt: str) -> bool:
        """Return *True* if *prompt* begins with a recognised @mention."""
        first = prompt.strip().split(None, 1)[0].lower() if prompt.strip() else ""
        return first in cls.MENTIONS

    @classmethod
    def strip_mention(cls, prompt: str) -> str:
        """Remove the leading @mention token from *prompt*, if present."""
        parts = prompt.strip().split(None, 1)
        if len(parts) > 1 and parts[0].lower() in cls.MENTIONS:
            return parts[1]
        return prompt.strip()

    # ── main dispatch ─────────────────────────────────────────────────────────

    def chat(self, prompt: str) -> dict:
        """Send *prompt* to Ollama ``/api/generate`` and return the response dict.

        The @mention prefix is stripped automatically.  This method uses only
        the standard-library ``urllib`` — no third-party HTTP client required.

        Raises ``urllib.error.URLError`` / ``urllib.error.HTTPError`` if Ollama
        is unreachable, so the caller can surface a clear error instead of a
        confusing traceback from an unrelated provider.
        """
        clean = self.strip_mention(prompt)
        payload = json.dumps(
            {"model": self.model, "prompt": clean, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())


# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class QuantizationConfig:
    quant_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    method: str = "int8"          # int8, int4, gptq, awq, bnb
    bits: int = 8
    group_size: int = 128
    desc_act: bool = False
    size_reduction_pct: float = 0.0
    perplexity_delta: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class LoRAAdapter:
    adapter_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    name: str = ""
    rank: int = 16
    alpha: float = 32.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    task_type: str = "CAUSAL_LM"
    trainable_params: int = 0
    base_model_params: int = 0
    checkpoint_path: str = ""
    merged: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class FineTuneJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    adapter_id: Optional[str] = None
    dataset: str = ""
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    status: str = "queued"   # queued, running, completed, failed
    train_loss: float = 0.0
    eval_loss: float = 0.0
    best_checkpoint: str = ""
    duration_s: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None


@dataclass
class PipelineStage:
    stage: str
    status: str
    duration_s: float
    details: str = ""


# ── Core class ────────────────────────────────────────────────────────────────
class EnhancedModelPipeline:
    """Advanced model pipeline: quantization, LoRA adapters, fine-tuning."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_db()

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS quantizations (
                quant_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                method TEXT,
                bits INTEGER,
                group_size INTEGER,
                desc_act INTEGER,
                size_reduction_pct REAL,
                perplexity_delta REAL,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS lora_adapters (
                adapter_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                name TEXT,
                rank INTEGER,
                alpha REAL,
                target_modules_json TEXT,
                task_type TEXT,
                trainable_params INTEGER,
                base_model_params INTEGER,
                checkpoint_path TEXT,
                merged INTEGER DEFAULT 0,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS finetune_jobs (
                job_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                adapter_id TEXT,
                dataset TEXT,
                epochs INTEGER,
                learning_rate REAL,
                batch_size INTEGER,
                status TEXT,
                train_loss REAL,
                eval_loss REAL,
                best_checkpoint TEXT,
                duration_s REAL,
                created_at TEXT,
                completed_at TEXT
            );
        """)
        self._conn.commit()

    def quantize_model(self, cfg: QuantizationConfig) -> QuantizationConfig:
        """Register a quantization configuration and simulate reduction stats."""
        method_reductions = {"int8": 50.0, "int4": 75.0, "gptq": 73.0,
                             "awq": 74.0, "bnb": 50.0}
        cfg.size_reduction_pct = method_reductions.get(cfg.method, 50.0)
        cfg.perplexity_delta = {"int8": 0.1, "int4": 0.8, "gptq": 0.3,
                                "awq": 0.25, "bnb": 0.12}.get(cfg.method, 0.5)
        self._conn.execute(
            "INSERT OR REPLACE INTO quantizations VALUES (?,?,?,?,?,?,?,?,?)",
            (cfg.quant_id, cfg.model_id, cfg.method, cfg.bits, cfg.group_size,
             int(cfg.desc_act), cfg.size_reduction_pct,
             cfg.perplexity_delta, cfg.created_at)
        )
        self._conn.commit()
        print(f"{G}✓{NC} Quantized {C}{cfg.model_id}{NC} → {BOLD}{cfg.method}{NC} "
              f"(-{Y}{cfg.size_reduction_pct:.0f}%{NC} size, "
              f"+{cfg.perplexity_delta:.2f} ppl)")
        return cfg

    def add_lora(self, adapter: LoRAAdapter) -> LoRAAdapter:
        """Register a LoRA adapter configuration."""
        # Simulate trainable param count
        adapter.trainable_params = adapter.rank * 2 * len(adapter.target_modules) * 4096
        adapter.base_model_params = 7_000_000_000
        self._conn.execute(
            "INSERT OR REPLACE INTO lora_adapters VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (adapter.adapter_id, adapter.model_id, adapter.name, adapter.rank,
             adapter.alpha, json.dumps(adapter.target_modules), adapter.task_type,
             adapter.trainable_params, adapter.base_model_params,
             adapter.checkpoint_path, int(adapter.merged), adapter.created_at)
        )
        self._conn.commit()
        pct = adapter.trainable_params / adapter.base_model_params * 100
        print(f"{G}✓{NC} LoRA adapter {BOLD}{adapter.name}{NC} added "
              f"rank={adapter.rank} α={adapter.alpha} "
              f"trainable={Y}{pct:.3f}%{NC}")
        return adapter

    def track_finetune(self, job: FineTuneJob) -> FineTuneJob:
        """Track a fine-tuning job from start to completion."""
        job.status = "running"
        print(f"{C}▶{NC} Fine-tune job {BOLD}{job.job_id}{NC} started "
              f"[{job.dataset}, {job.epochs} epochs, lr={job.learning_rate}]")
        # Simulate training
        t0 = time.perf_counter()
        for epoch in range(1, job.epochs + 1):
            job.train_loss = max(0.1, 2.0 / epoch)
            job.eval_loss = max(0.15, 2.2 / epoch)
            print(f"  Epoch {epoch}/{job.epochs} — "
                  f"train_loss={Y}{job.train_loss:.4f}{NC} "
                  f"eval_loss={C}{job.eval_loss:.4f}{NC}")
            time.sleep(0.05)
        job.duration_s = round(time.perf_counter() - t0, 3)
        job.status = "completed"
        job.best_checkpoint = f"checkpoints/{job.job_id}/best"
        job.completed_at = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO finetune_jobs VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job.job_id, job.model_id, job.adapter_id, job.dataset,
             job.epochs, job.learning_rate, job.batch_size, job.status,
             job.train_loss, job.eval_loss, job.best_checkpoint,
             job.duration_s, job.created_at, job.completed_at)
        )
        self._conn.commit()
        print(f"{G}✓{NC} Job complete in {job.duration_s:.2f}s — "
              f"best checkpoint: {M}{job.best_checkpoint}{NC}")
        return job

    def get_pipeline_status(self, model_id: str) -> List[PipelineStage]:
        """Return the pipeline status for a model."""
        stages = []
        # Quantization stage
        q = self._conn.execute(
            "SELECT method, created_at FROM quantizations WHERE model_id=? "
            "ORDER BY created_at DESC LIMIT 1", (model_id,)
        ).fetchone()
        stages.append(PipelineStage(
            stage="quantization",
            status="done" if q else "pending",
            duration_s=0.0,
            details=q[0] if q else "not quantized"
        ))
        # LoRA stage
        l = self._conn.execute(
            "SELECT name, rank FROM lora_adapters WHERE model_id=? "
            "ORDER BY created_at DESC LIMIT 1", (model_id,)
        ).fetchone()
        stages.append(PipelineStage(
            stage="lora_adapter",
            status="done" if l else "pending",
            duration_s=0.0,
            details=f"{l[0]} rank={l[1]}" if l else "no adapter"
        ))
        # Fine-tune stage
        ft = self._conn.execute(
            "SELECT status, train_loss FROM finetune_jobs WHERE model_id=? "
            "ORDER BY created_at DESC LIMIT 1", (model_id,)
        ).fetchone()
        stages.append(PipelineStage(
            stage="fine_tuning",
            status=ft[0] if ft else "pending",
            duration_s=0.0,
            details=f"loss={ft[1]:.4f}" if ft else "no jobs"
        ))
        return stages

    def export_model(self, model_id: str, output_path: str = "") -> str:
        """Export merged model config to JSON."""
        quants = self._conn.execute(
            "SELECT * FROM quantizations WHERE model_id=?", (model_id,)
        ).fetchall()
        adapters = self._conn.execute(
            "SELECT * FROM lora_adapters WHERE model_id=?", (model_id,)
        ).fetchall()
        jobs = self._conn.execute(
            "SELECT * FROM finetune_jobs WHERE model_id=? AND status='completed'",
            (model_id,)
        ).fetchall()
        out = output_path or f"{model_id}_export.json"
        export_data = {
            "model_id": model_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "quantizations": len(quants),
            "adapters": len(adapters),
            "completed_jobs": len(jobs),
        }
        Path(out).write_text(json.dumps(export_data, indent=2))
        print(f"{G}✓{NC} Exported model config → {M}{out}{NC}")
        return out

    def list_adapters(self, model_id: Optional[str] = None) -> List[LoRAAdapter]:
        q = "SELECT * FROM lora_adapters"
        p: tuple = ()
        if model_id:
            q += " WHERE model_id=?"
            p = (model_id,)
        rows = self._conn.execute(q, p).fetchall()
        return [
            LoRAAdapter(adapter_id=r[0], model_id=r[1], name=r[2],
                        rank=r[3], alpha=r[4],
                        target_modules=json.loads(r[5] or "[]"),
                        task_type=r[6], trainable_params=r[7],
                        base_model_params=r[8], checkpoint_path=r[9],
                        merged=bool(r[10]), created_at=r[11])
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ai-models-enhanced",
        description="Lucidia Enhanced Model Pipeline"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    qnt = sub.add_parser("quantize", help="Quantize a model")
    qnt.add_argument("--model-id", required=True)
    qnt.add_argument("--method", choices=["int8", "int4", "gptq", "awq", "bnb"], default="int8")
    qnt.add_argument("--bits", type=int, default=8)
    qnt.add_argument("--group-size", type=int, default=128)

    lora = sub.add_parser("lora", help="Add LoRA adapter")
    lora.add_argument("--model-id", required=True)
    lora.add_argument("--name", required=True)
    lora.add_argument("--rank", type=int, default=16)
    lora.add_argument("--alpha", type=float, default=32.0)
    lora.add_argument("--modules", nargs="*", default=["q_proj", "v_proj"])

    ft = sub.add_parser("finetune", help="Run fine-tuning job")
    ft.add_argument("--model-id", required=True)
    ft.add_argument("--dataset", required=True)
    ft.add_argument("--epochs", type=int, default=3)
    ft.add_argument("--lr", type=float, default=2e-4)

    st = sub.add_parser("status", help="Pipeline status")
    st.add_argument("--model-id", required=True)

    exp = sub.add_parser("export", help="Export model config")
    exp.add_argument("--model-id", required=True)
    exp.add_argument("--output", default="")

    adp = sub.add_parser("adapters", help="List LoRA adapters")
    adp.add_argument("--model-id", default=None)

    cht = sub.add_parser(
        "chat",
        help="Send a prompt to local Ollama (@copilot / @lucidia / @blackboxprogramming)"
    )
    cht.add_argument("prompt", help="Prompt text, optionally prefixed with @mention")
    cht.add_argument("--ollama-url", default=OLLAMA_BASE_URL,
                     help="Base URL of the local Ollama server")
    cht.add_argument("--model", default="llama3",
                     help="Ollama model name (default: llama3)")

    args = parser.parse_args()
    pipeline = EnhancedModelPipeline()

    try:
        if args.cmd == "quantize":
            cfg = QuantizationConfig(
                model_id=args.model_id, method=args.method,
                bits=args.bits, group_size=args.group_size,
            )
            pipeline.quantize_model(cfg)

        elif args.cmd == "lora":
            adapter = LoRAAdapter(
                model_id=args.model_id, name=args.name,
                rank=args.rank, alpha=args.alpha,
                target_modules=args.modules,
            )
            pipeline.add_lora(adapter)

        elif args.cmd == "finetune":
            job = FineTuneJob(
                model_id=args.model_id, dataset=args.dataset,
                epochs=args.epochs, learning_rate=args.lr,
            )
            pipeline.track_finetune(job)

        elif args.cmd == "status":
            stages = pipeline.get_pipeline_status(args.model_id)
            print(f"\n{BOLD}{B}── Pipeline: {args.model_id} ──────────{NC}")
            for s in stages:
                icon = G + "✓" if s.status == "done" else (
                    C + "▶" if s.status == "running" else Y + "○"
                )
                print(f"  {icon}{NC} {C}{s.stage:<20}{NC} {s.details}")

        elif args.cmd == "export":
            pipeline.export_model(args.model_id, args.output)

        elif args.cmd == "adapters":
            adapters = pipeline.list_adapters(model_id=args.model_id)
            if not adapters:
                print(f"{Y}No adapters found.{NC}")
                return
            for a in adapters:
                merged_flag = f" {G}[merged]{NC}" if a.merged else ""
                print(f"  {C}{a.adapter_id}{NC} {BOLD}{a.name}{NC} "
                      f"rank={a.rank} α={a.alpha}{merged_flag}")

        elif args.cmd == "chat":
            router = OllamaRouter(base_url=args.ollama_url, model=args.model)
            prompt = args.prompt
            mention_used = OllamaRouter.detect_mention(prompt)
            parts = prompt.strip().split(None, 1)
            mention_tag = parts[0] if mention_used and parts else "@ollama"
            print(f"{C}▶{NC} Routing {BOLD}{mention_tag}{NC} → "
                  f"Ollama [{args.ollama_url}] model={args.model}")
            try:
                result = router.chat(prompt)
                print(f"\n{G}{result.get('response', result)}{NC}")
            except urllib.error.URLError as exc:
                print(f"{R}✗{NC} Could not reach Ollama at {args.ollama_url}: {exc.reason}")
                raise SystemExit(1)

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
