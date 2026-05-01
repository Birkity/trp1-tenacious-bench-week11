"""
run_ablation.py — Act IV ablation for Tenacious-Bench (Path B judge).

Three judges measured on dev (default) or held_out partition:

  (1) Deterministic  : score_task() from scoring_evaluator.py
                       This IS the ground truth — 100% by construction.
  (2) Base model     : Qwen2.5-1.5B-Instruct, no LoRA adapter
                       Prompt-engineering baseline -> Delta B denominator.
  (3) Trained judge  : same backbone + SimPO LoRA adapter
                       Evaluation target -> Delta A (vs ground truth).

Delta A = trained_acc - det_acc   (how much training added vs rules)
Delta B = trained_acc - base_acc  (how much training added vs base prompting)

Outputs (written to ablations/):
  ablation_results.json    -- per-task verdicts, summary metrics, dim error counts
  held_out_traces.jsonl    -- raw model outputs for audit / evidence_graph.json

Usage (run on Colab T4 right after training, or locally with CPU):
  python scripts/analysis/run_ablation.py \\
      --adapter-path /content/outputs/tenacious_judge_adapter

  python scripts/analysis/run_ablation.py \\
      --partition held_out \\
      --adapter-path /content/outputs/tenacious_judge_adapter

  python scripts/analysis/run_ablation.py --dry-run   # deterministic only, no GPU needed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))
from scoring_evaluator import score_task  # noqa: E402

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "unsloth/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER    = str(ROOT / "training" / "tenacious_judge_adapter")
DEFAULT_OUTPUT_DIR = str(ROOT / "ablations")
MAX_NEW_TOKENS     = 128

DIM_NAMES = {
    "D1": "grounding_fidelity",
    "D2": "icp_pitch_alignment",
    "D3": "signal_directionality",
    "D4": "tone_compliance",
    "D5": "format_compliance",
}

# ── Judge prompt (identical to build_preference_pairs.py) ─────────────────────
_PROMPT = (
    "You are a sales quality judge for Tenacious Intelligence Corporation.\n"
    "Given a hiring signal brief and a generated outreach email,\n"
    "return a structured verdict.\n"
    "\n"
    "Score these dimensions:\n"
    "grounding_fidelity, icp_pitch_alignment,\n"
    "signal_directionality, tone_compliance, format_compliance.\n"
    "\n"
    "Hiring Brief:\n"
    "{brief_text}\n"
    "\n"
    "Generated Email:\n"
    "Subject: {subject}\n"
    "{body}\n"
    "\n"
    "Velocity delta: {delta:+.0f}%\n"
    "ICP segment: {segment}"
)


def _brief_text(task: dict) -> str:
    brief = task.get("brief", {})
    hv    = brief.get("hiring_velocity", {})
    bm    = brief.get("bench_match", {})
    facts = brief.get("grounding_facts", [])
    lines = [
        f"Company: {brief.get('company', 'Unknown')}",
        f"ICP Segment: {brief.get('icp_segment', 'Unknown')}",
        f"AI Maturity: {brief.get('ai_maturity', '?')}/3",
        (
            f"Hiring Velocity: {hv.get('direction', '?')} "
            f"({hv.get('delta_pct', 0):+.0f}% in 60 days, "
            f"{hv.get('signal_strength', '?')} signal)"
        ),
        f"Observation: {hv.get('observation', '').strip()}",
        f"Bench Available: {bm.get('bench_available', '?')}",
        "Grounding Facts:",
    ]
    for fact in facts:
        lines.append(f"  - {fact.strip()}")
    return "\n".join(lines)


def _build_prompt(task: dict) -> str:
    brief = task.get("brief", {})
    email = task.get("email", {})
    hv    = brief.get("hiring_velocity", {})
    return _PROMPT.format(
        brief_text = _brief_text(task),
        subject    = email.get("subject", ""),
        body       = email.get("body", "").strip(),
        delta      = float(hv.get("delta_pct", 0)),
        segment    = brief.get("icp_segment", "Unknown"),
    )


def _parse_verdict(text: str) -> str:
    """Return PASS, REJECT, or UNKNOWN from raw model output."""
    upper = text.upper()
    if "VERDICT: REJECT" in upper:
        return "REJECT"
    if "VERDICT: PASS" in upper:
        return "PASS"
    n_reject = upper.count("REJECT")
    n_pass   = upper.count("PASS")
    if n_reject > n_pass:
        return "REJECT"
    if n_pass > n_reject:
        return "PASS"
    return "UNKNOWN"


def _infer(model, tokenizer, prompt: str, device: str) -> str:
    """Single inference call; returns decoded output text (new tokens only)."""
    messages  = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            max_length     = None,   # override model's default generation_config max_length
            do_sample      = False,
            temperature    = None,   # suppress sampling-param warnings when do_sample=False
            top_p          = None,
            top_k          = None,
            pad_token_id   = tokenizer.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _load_tasks(partition: str) -> list[dict]:
    base = ROOT / "data" / "tenacious_bench_v0.1" / partition
    # Try canonical name first, then partition-specific names used by split scripts
    for candidate in ("tasks.jsonl", f"{partition}_tasks.jsonl", "held_tasks.jsonl", "dev_tasks.jsonl"):
        path = base / candidate
        if path.exists():
            return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    raise FileNotFoundError(
        f"No tasks file found in {base}. "
        f"Expected one of: tasks.jsonl, {partition}_tasks.jsonl"
    )


def _accuracy(preds: list[str], truths: list[str]) -> tuple[float, int, int]:
    """Returns (accuracy, n_correct, n_total), skipping UNKNOWN/NOT_RUN."""
    pairs   = [(p, t) for p, t in zip(preds, truths) if p not in ("UNKNOWN", "NOT_RUN")]
    if not pairs:
        return 0.0, 0, 0
    correct = sum(p == t for p, t in pairs)
    return correct / len(pairs), correct, len(pairs)


# ── Phase helpers extracted from main() to keep cognitive complexity low ──────

def _run_deterministic(tasks: list[dict]) -> tuple[list[str], list[str | None], list[str]]:
    """Run score_task on every task. Returns (verdicts, dims, reasons)."""
    verdicts: list[str]           = []
    dims:     list[str | None]    = []
    reasons:  list[str]           = []
    errors:   list[str]           = []

    for task in tasks:
        try:
            s = score_task(task)
            verdicts.append(s["verdict"])
            dims.append(s.get("failed_dimension"))
            reasons.append(s.get("reason", ""))
        except Exception as exc:
            verdicts.append("UNKNOWN")
            dims.append(None)
            reasons.append(str(exc))
            errors.append(f"{task.get('task_id', '?')}: {exc}")

    gt_pass   = verdicts.count("PASS")
    gt_reject = verdicts.count("REJECT")
    print(f"  Ground truth: PASS={gt_pass}  REJECT={gt_reject}")
    for e in errors[:5]:
        print(f"  score_task error: {e}")

    return verdicts, dims, reasons


def _load_model_and_adapter(model_name: str, adapter_path: Path, dtype, device: str):
    """Load base model; optionally wrap with PEFT adapter. Returns (model, tokenizer)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model ({model_name})...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype       = dtype,
        trust_remote_code = True,
        device_map        = "auto" if device == "cuda" else None,
    )
    if device == "cpu":
        base_model = base_model.to("cpu")
    base_model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    if adapter_path.exists():
        print(f"Applying LoRA adapter from {adapter_path}...")
        wrapped = PeftModel.from_pretrained(base_model, str(adapter_path))
        wrapped.eval()
        return wrapped, tokenizer

    print(f"WARNING: Adapter not found at {adapter_path} -- trained model will be skipped.")
    return base_model, tokenizer


def _run_base_inference(
    model, tokenizer, tasks: list[dict], adapter_exists: bool, device: str
) -> tuple[list[str], list[str]]:
    """Infer with adapter disabled (base model behaviour)."""
    raws:     list[str] = []
    verdicts: list[str] = []
    print(f"\n[2/3] Base model inference ({len(tasks)} tasks)...")
    for i, task in enumerate(tasks):
        prompt = (
            "You are a strict email quality evaluator.\n\n"
            "You MUST output ONLY in the exact format below.\n"
            "Do NOT add explanations before or after.\n"
            "Do NOT write paragraphs.\n"
            "Do NOT restate the task.\n\n"
            "FORMAT (exactly):\n\n"
            "VERDICT: PASS\n"
            "Primary failure: <D1|D2|D3|D4|D5 or NONE>\n"
            "Reason: <one short sentence>\n\n"
            "Definitions of dimensions:\n"
            "D1 — Grounding in brief facts (no fabrication, no number drift)\n"
            "D2 — ICP alignment (correct segment + ends with a question)\n"
            "D3 — Signal directionality (growth vs contraction framing matches data)\n"
            "D4 — Tone compliance (no banned phrases, no hype, no pressure)\n"
            "D5 — Format compliance (no meeting-booking language, correct structure)\n\n"
            "Now evaluate the following task.\n\n"
            + _build_prompt(task)
        )
        if adapter_exists:
            with model.disable_adapter():
                raw = _infer(model, tokenizer, prompt, device)
        else:
            raw = _infer(model, tokenizer, prompt, device)
        raws.append(raw)
        verdicts.append(_parse_verdict(raw))
        if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
            print(f"  {i + 1}/{len(tasks)}")
    print(f"  UNKNOWN outputs: {verdicts.count('UNKNOWN')}")
    return raws, verdicts


def _run_trained_inference(
    model, tokenizer, tasks: list[dict], device: str
) -> tuple[list[str], list[str]]:
    """Infer with adapter enabled (trained judge)."""
    raws:     list[str] = []
    verdicts: list[str] = []
    print(f"\n[3/3] Trained model inference ({len(tasks)} tasks)...")
    for i, task in enumerate(tasks):
        prompt = (
            "You are a strict email quality evaluator.\n\n"
            "You MUST output ONLY in the exact format below.\n"
            "Do NOT add explanations before or after.\n"
            "Do NOT write paragraphs.\n"
            "Do NOT restate the task.\n\n"
            "FORMAT (exactly):\n\n"
            "VERDICT: PASS\n"
            "Primary failure: <D1|D2|D3|D4|D5 or NONE>\n"
            "Reason: <one short sentence>\n\n"
            "Definitions of dimensions:\n"
            "D1 — Grounding in brief facts (no fabrication, no number drift)\n"
            "D2 — ICP alignment (correct segment + ends with a question)\n"
            "D3 — Signal directionality (growth vs contraction framing matches data)\n"
            "D4 — Tone compliance (no banned phrases, no hype, no pressure)\n"
            "D5 — Format compliance (no meeting-booking language, correct structure)\n\n"
            "Now evaluate the following task.\n\n"
            + _build_prompt(task)
        )
        raw = _infer(model, tokenizer, prompt, device)
        raws.append(raw)
        verdicts.append(_parse_verdict(raw))
        if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
            print(f"  {i + 1}/{len(tasks)}")
    print(f"  UNKNOWN outputs: {verdicts.count('UNKNOWN')}")
    return raws, verdicts


def _compute_dim_errors(
    det_verdicts: list[str],
    det_dims: list[str | None],
    base_verdicts: list[str],
    trained_verdicts: list[str],
) -> dict[str, dict[str, int]]:
    dim_keys = list(DIM_NAMES.values()) + ["none"]
    errors: dict[str, dict[str, int]] = {
        "base_model":    dict.fromkeys(dim_keys, 0),
        "trained_model": dict.fromkeys(dim_keys, 0),
    }
    for det_v, det_d, bv, tv in zip(det_verdicts, det_dims, base_verdicts, trained_verdicts):
        dim_label = DIM_NAMES.get(det_d, "none") if det_d else "none"
        if bv not in ("NOT_RUN", "UNKNOWN") and bv != det_v:
            errors["base_model"][dim_label] += 1
        if tv not in ("NOT_RUN", "UNKNOWN") and tv != det_v:
            errors["trained_model"][dim_label] += 1
    return {judge: {k: v for k, v in counts.items() if v > 0}
            for judge, counts in errors.items()}


@dataclass
class AblationResults:
    partition:        str
    tasks:            list[dict]
    det_verdicts:     list[str]
    det_dims:         list[Optional[str]]
    det_reasons:      list[str]
    base_verdicts:    list[str]       = field(default_factory=list)
    base_raws:        list[str]       = field(default_factory=list)
    trained_verdicts: list[str]       = field(default_factory=list)
    trained_raws:     list[str]       = field(default_factory=list)
    base_acc:         float           = 0.0
    base_corr:        int             = 0
    base_n:           int             = 0
    trained_acc:      float           = 0.0
    trained_corr:     int             = 0
    trained_n:        int             = 0
    delta_a:          Optional[float] = None
    delta_b:          Optional[float] = None
    dim_errors:       dict            = field(default_factory=dict)
    gt_pass:          int             = 0
    gt_reject:        int             = 0


def _print_summary(r: AblationResults) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  ABLATION RESULTS   partition={r.partition}   n={len(r.tasks)}")
    print(sep)
    print(f"  Ground truth dist  : PASS={r.gt_pass}  REJECT={r.gt_reject}")
    print("  (1) Deterministic  : 100.0%  (baseline -- defines ground truth)")

    if r.base_verdicts and r.base_verdicts[0] != "NOT_RUN":
        print(f"  (2) Base model     : {r.base_acc:.1%}  ({r.base_corr}/{r.base_n} correct)")
    else:
        print("  (2) Base model     : NOT RUN  (--dry-run)")

    if r.trained_verdicts and r.trained_verdicts[0] != "NOT_RUN":
        print(f"  (3) Trained judge  : {r.trained_acc:.1%}  ({r.trained_corr}/{r.trained_n} correct)")
    else:
        print("  (3) Trained judge  : NOT RUN  (adapter missing or --dry-run)")

    print()
    if r.delta_a is not None:
        print(f"  Delta A (trained vs GT)  : {r.delta_a:+.1%}")
        print(f"  Delta B (trained vs base): {r.delta_b:+.1%}")

    print()
    print("  Per-dimension error counts (wrong predictions):")
    for judge, errs in r.dim_errors.items():
        label = dict(sorted(errs.items())) if errs else "none"
        print(f"    {judge}: {label}")
    print(sep)


def _save_outputs(out_dir: Path, args: argparse.Namespace, r: AblationResults) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    per_task_rows = [
        {
            "task_id":          task["task_id"],
            "source_mode":      task.get("brief", {}).get("source_mode"),
            "difficulty":       task.get("difficulty"),
            "ground_truth":     dv,
            "failed_dimension": dd,
            "det_reason":       dr,
            "base_verdict":     bv,
            "base_correct":     (bv == dv) if bv not in ("NOT_RUN", "UNKNOWN") else None,
            "trained_verdict":  tv,
            "trained_correct":  (tv == dv) if tv not in ("NOT_RUN", "UNKNOWN") else None,
        }
        for task, dv, dd, dr, bv, tv in zip(
            r.tasks, r.det_verdicts, r.det_dims, r.det_reasons,
            r.base_verdicts, r.trained_verdicts,
        )
    ]

    results = {
        "metadata": {
            "run_date":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "partition":    args.partition,
            "n_tasks":      len(r.tasks),
            "backbone":     args.model,
            "adapter_path": args.adapter_path,
            "dry_run":      args.dry_run,
        },
        "summary": {
            "ground_truth_dist": {"PASS": r.gt_pass, "REJECT": r.gt_reject},
            "deterministic":     {"accuracy": 1.0, "note": "defines ground truth"},
            "base_model":        {"accuracy": r.base_acc,    "correct": r.base_corr,    "total": r.base_n},
            "trained_model":     {"accuracy": r.trained_acc, "correct": r.trained_corr, "total": r.trained_n},
            "delta_a_trained_vs_gt":   r.delta_a,
            "delta_b_trained_vs_base": r.delta_b,
        },
        "per_dimension_errors": r.dim_errors,
        "tasks": per_task_rows,
    }

    results_path = out_dir / "ablation_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults   -> {results_path}")

    traces_path = out_dir / "held_out_traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as fh:
        for task, dv, bv, br, tv, tr in zip(
            r.tasks, r.det_verdicts, r.base_verdicts, r.base_raws,
            r.trained_verdicts, r.trained_raws,
        ):
            fh.write(json.dumps({
                "task_id":         task["task_id"],
                "ground_truth":    dv,
                "base_verdict":    bv,
                "base_raw":        br,
                "trained_verdict": tv,
                "trained_raw":     tr,
            }, ensure_ascii=False) + "\n")
    print(f"Traces    -> {traces_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Tenacious-Bench Act IV ablation")
    parser.add_argument("--partition",    default="dev", choices=["dev", "held_out", "train"])
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER)
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--output-dir",   default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run",      action="store_true",
                        help="Deterministic scorer only -- no LLM inference, no GPU needed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device   : {device}")
    print(f"Backbone : {args.model}")
    print(f"Adapter  : {args.adapter_path}")
    print(f"Partition: {args.partition}\n")
    if device == "cpu" and not args.dry_run:
        print("WARNING: CPU inference is slow (~1-2 min/task). "
              "Run on Colab T4 for full speed, or use --dry-run.\n")

    tasks = _load_tasks(args.partition)
    print(f"Loaded {len(tasks)} tasks from '{args.partition}'")

    print("\n[1/3] Deterministic scorer (score_task)...")
    det_verdicts, det_dims, det_reasons = _run_deterministic(tasks)

    r = AblationResults(
        partition    = args.partition,
        tasks        = tasks,
        det_verdicts = det_verdicts,
        det_dims     = det_dims,
        det_reasons  = det_reasons,
        gt_pass      = det_verdicts.count("PASS"),
        gt_reject    = det_verdicts.count("REJECT"),
    )

    if args.dry_run:
        print("\n-- dry-run: skipping LLM inference --")
        r.base_raws        = [""] * len(tasks)
        r.base_verdicts    = ["NOT_RUN"] * len(tasks)
        r.trained_raws     = [""] * len(tasks)
        r.trained_verdicts = ["NOT_RUN"] * len(tasks)
    else:
        adapter_path   = Path(args.adapter_path)
        adapter_exists = adapter_path.exists()
        dtype          = torch.float16 if device == "cuda" else torch.float32

        model, tokenizer = _load_model_and_adapter(args.model, adapter_path, dtype, device)

        r.base_raws, r.base_verdicts = _run_base_inference(
            model, tokenizer, tasks, adapter_exists, device
        )
        r.base_acc, r.base_corr, r.base_n = _accuracy(r.base_verdicts, det_verdicts)
        print(f"  Base accuracy: {r.base_acc:.1%}  ({r.base_corr}/{r.base_n})")

        if adapter_exists:
            r.trained_raws, r.trained_verdicts = _run_trained_inference(
                model, tokenizer, tasks, device
            )
        else:
            r.trained_raws     = ["NOT_RUN"] * len(tasks)
            r.trained_verdicts = ["NOT_RUN"] * len(tasks)

    r.base_acc,    r.base_corr,    r.base_n    = _accuracy(r.base_verdicts,    det_verdicts)
    r.trained_acc, r.trained_corr, r.trained_n = _accuracy(r.trained_verdicts, det_verdicts)
    r.delta_a = (r.trained_acc - 1.0)         if r.trained_verdicts[0] != "NOT_RUN" else None
    r.delta_b = (r.trained_acc - r.base_acc)  if r.trained_verdicts[0] != "NOT_RUN" else None
    r.dim_errors = _compute_dim_errors(det_verdicts, det_dims, r.base_verdicts, r.trained_verdicts)

    _print_summary(r)
    _save_outputs(Path(args.output_dir), args, r)


if __name__ == "__main__":
    main()
