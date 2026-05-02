"""
train_simpo_judge.py  —  SimPO LoRA training for Tenacious-Bench judge (Path B)

Backbone  : unsloth/Qwen2.5-3B-Instruct  (MoE: 30B total / 3B active params)
Method    : SimPO — reference-free preference optimisation, beta=2.0, gamma=0.5
Data      : training_data/tenacious_judge_train_v2.jsonl  (200 preference pairs)
Hardware  : Google Colab T4 (16 GB VRAM) — requires 4-bit loading for 30B model.
            For a Colab A100/L4 you can set load_in_4bit=False for 16-bit LoRA.
            Expected wall time on T4: 60–120 min.

Before running on Colab:
  1. Upload training_data/tenacious_judge_train_v2.jsonl → /content/
  2. (Optional) set HF_REPO_ID and HF_TOKEN to push adapter to HuggingFace Hub
  3. Runtime → Change runtime type → T4 GPU
  4. Run all cells (or: !python train_simpo_judge.py)
"""

from __future__ import annotations

# ── 0. INSTALL ────────────────────────────────────────────────────────────────
# Runs once; subsequent imports use the installed packages.
import subprocess, sys

def _pip(*args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *args], check=True
    )

print("Installing dependencies…")
_pip("unsloth[colab-new]", "-U")
_pip("trl>=0.12.0", "peft>=0.12.0", "accelerate>=0.30.0", "datasets>=2.18.0")
_pip("matplotlib")
print("Installation complete.\n")

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from datasets import Dataset
from trl import CPOConfig, CPOTrainer
from unsloth import FastLanguageModel

# ── 2. CONSTANTS ─────────────────────────────────────────────────────────────
# Update MODEL_NAME to your pinned backbone (see training/requirements.txt).
MODEL_NAME    = "unsloth/Qwen2.5-3B-Instruct"   # MoE: 30B total / 3B active params.
DATASET_PATH  = "/content/tenacious_judge_train_v2.jsonl"
OUTPUT_DIR    = "/content/outputs/tenacious_judge_adapter"

# HuggingFace Hub — set both to push adapter after training.
HF_REPO_ID    = os.environ.get("HF_REPO_ID", "")   # e.g. "YourName/tenacious-judge"
HF_TOKEN      = os.environ.get("HF_TOKEN",    "")

# LoRA config
LORA_R        = 16
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.0

# SimPO hyperparameters
EPOCHS        = 3
BATCH_SIZE    = 4     # fits T4 at 16-bit with 0.8B model
GRAD_ACCUM    = 4     # effective batch = 16
LR            = 2e-4
BETA          = 2.0   # SimPO temperature (log-ratio sharpness)
GAMMA_BETA_RATIO = 0.25  # gamma = BETA * GAMMA_BETA_RATIO = 0.5 → target margin M = 0.5
                         # Disagrees with paper default (M=2.0): 114 pairs at 0.8B overfit
                         # with large M; 0.5 provides a gentler gradient over the full run.
MAX_SEQ_LEN   = 1024
SEED          = 42

# ── 3. GPU CHECK ─────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU found. Connect a T4 runtime: Runtime → Change runtime type → T4 GPU."
    )
device_name = torch.cuda.get_device_name(0)
vram_gb     = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {device_name}  |  VRAM: {vram_gb:.1f} GB")

# T4 is fp16; newer cards (A100, L4) can use bfloat16.
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"dtype: {'bfloat16' if dtype == torch.bfloat16 else 'float16'}\n")

# ── 4. LOAD MODEL (16-bit LoRA — 1.5B fits comfortably on T4) ────────────────
print(f"Loading {MODEL_NAME}…")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = dtype,
    load_in_4bit   = False,   # 16-bit LoRA — 3B at fp16 = ~6 GB, fits T4 with room for optimizer
)
print("Model loaded.\n")

# ── 5. ADD LORA ───────────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = SEED,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"LoRA adapter: {trainable:,} trainable / {total:,} total params "
      f"({100 * trainable / total:.2f}%)\n")

# ── 6. LOAD DATASET ───────────────────────────────────────────────────────────
raw_pairs: list[dict] = [
    json.loads(line)
    for line in Path(DATASET_PATH).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print(f"Loaded {len(raw_pairs)} preference pairs from {DATASET_PATH}")

# SimPOTrainer expects the conversational format: prompt/chosen/rejected as
# message-dict lists. It applies the chat template internally using the tokenizer.
def _to_messages(pair: dict) -> dict:
    return {
        "prompt":   [{"role": "user",      "content": pair["prompt"]}],
        "chosen":   [{"role": "assistant", "content": pair["chosen"]}],
        "rejected": [{"role": "assistant", "content": pair["rejected"]}],
    }

dataset = Dataset.from_list([_to_messages(p) for p in raw_pairs])

# Sanity-check: count label balance
reject_chosen = sum(1 for p in raw_pairs if "VERDICT: REJECT" in p["chosen"])
pass_chosen   = sum(1 for p in raw_pairs if "VERDICT: PASS"   in p["chosen"])
print(f"Label balance: REJECT-chosen={reject_chosen}, PASS-chosen={pass_chosen}")
print(f"Steps per epoch: {len(raw_pairs) // (BATCH_SIZE * GRAD_ACCUM)} "
      f"(effective batch={BATCH_SIZE * GRAD_ACCUM})\n")

# ── 7. SIMPO CONFIG ───────────────────────────────────────────────────────────
# CPOConfig API differs across TRL versions — gamma_beta_ratio, loss_type, and
# cpo_alpha were added at different points.  Use inspect to pass only what this
# version accepts; print which params were dropped so the log is transparent.
import inspect as _inspect

_cpo_valid = set(_inspect.signature(CPOConfig.__init__).parameters)

_all_kwargs = {
    "output_dir":                  OUTPUT_DIR,
    "num_train_epochs":            EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "learning_rate":               LR,
    "loss_type":                   "simpo",   # SimPO loss; ignored gracefully if unsupported
    "beta":                        BETA,
    "gamma_beta_ratio":            GAMMA_BETA_RATIO,  # target margin = beta * ratio = 0.5
    "cpo_alpha":                   0.0,       # 0 = pure SimPO, no NLL component
    "max_length":                  MAX_SEQ_LEN,
    "max_prompt_length":           MAX_SEQ_LEN // 2,
    "max_completion_length":       MAX_SEQ_LEN // 2,
    "fp16":                        (dtype == torch.float16),
    "bf16":                        (dtype == torch.bfloat16),
    "logging_steps":               1,
    "save_strategy":               "no",
    "seed":                        SEED,
    "remove_unused_columns":       False,
    "report_to":                   "none",
    "dataloader_pin_memory":       False,
}

_dropped = {k for k in _all_kwargs if k not in _cpo_valid}
if _dropped:
    print(f"Note: TRL version does not support these CPOConfig params (skipped): {_dropped}")
_config_kwargs = {k: v for k, v in _all_kwargs.items() if k in _cpo_valid}

simpo_config = CPOConfig(**_config_kwargs)

# ── 8. TRAIN ──────────────────────────────────────────────────────────────────
print("Starting SimPO training…")
t0 = time.time()

# CPOTrainer expects model.warnings_issued dict; Unsloth PEFT model omits it.
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

# TRL >= 0.15 uses processing_class; older versions use tokenizer.
try:
    trainer = CPOTrainer(
        model            = model,
        args             = simpo_config,
        train_dataset    = dataset,
        processing_class = tokenizer,
    )
except TypeError:
    trainer = CPOTrainer(
        model         = model,
        args          = simpo_config,
        train_dataset = dataset,
        tokenizer     = tokenizer,
    )

train_result = trainer.train()
elapsed = time.time() - t0

print(f"\nTraining complete in {elapsed / 60:.1f} min")
print(f"Final loss: {train_result.training_loss:.4f}")

# ── 9. SAVE ADAPTER + METADATA ────────────────────────────────────────────────
out = Path(OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nAdapter saved → {OUTPUT_DIR}")

# training_config.json — every hyperparameter for reproducibility
config_path = out / "training_config.json"
training_config = {
    "backbone"                  : MODEL_NAME,
    "method"                    : "SimPO",
    "dataset"                   : "tenacious_judge_train_v2.jsonl",
    "training_pairs"            : len(raw_pairs),
    "lora_r"                    : LORA_R,
    "lora_alpha"                : LORA_ALPHA,
    "lora_dropout"              : LORA_DROPOUT,
    "target_modules"            : ["q_proj","k_proj","v_proj","o_proj",
                                   "gate_proj","up_proj","down_proj"],
    "epochs"                    : EPOCHS,
    "batch_size"                : BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "effective_batch_size"      : BATCH_SIZE * GRAD_ACCUM,
    "learning_rate"             : LR,
    "beta"                      : BETA,
    "gamma_beta_ratio"          : GAMMA_BETA_RATIO,
    "gamma"                     : BETA * GAMMA_BETA_RATIO,
    "config_params_used"        : sorted(_config_kwargs.keys()),
    "config_params_dropped"     : sorted(_dropped),
    "max_seq_len"               : MAX_SEQ_LEN,
    "dtype"                     : "float16" if dtype == torch.float16 else "bfloat16",
    "load_in_4bit"              : False,
    "seed"                      : SEED,
    "train_loss_final"          : round(train_result.training_loss, 4),
    "wall_time_min"             : round(elapsed / 60, 1),
}
config_path.write_text(json.dumps(training_config, indent=2), encoding="utf-8")
print(f"Config saved  → {config_path}")

# loss_curve.png
log_history = trainer.state.log_history
steps  = [e["step"] for e in log_history if "loss" in e]
losses = [e["loss"] for e in log_history if "loss" in e]

if steps:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, linewidth=1.5, color="#2563eb", label="SimPO loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Tenacious Judge — SimPO Training Loss (Qwen 0.8B, {len(raw_pairs)} pairs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    curve_path = out / "loss_curve.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve    → {curve_path}")

    # Also save raw step/loss as JSON for evidence_graph.json references
    loss_log_path = out / "loss_log.json"
    loss_log_path.write_text(
        json.dumps({"steps": steps, "losses": losses}, indent=2), encoding="utf-8"
    )
    print(f"Loss log JSON → {loss_log_path}")

# ── 10. PUSH TO HUGGING FACE HUB (OPTIONAL) ───────────────────────────────────
if HF_REPO_ID and HF_TOKEN:
    print(f"\nPushing adapter to huggingface.co/{HF_REPO_ID} …")
    from huggingface_hub import HfApi

    model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)

    api = HfApi(token=HF_TOKEN)
    for fname in ["training_config.json", "loss_curve.png", "loss_log.json"]:
        fpath = out / fname
        if fpath.exists():
            api.upload_file(
                path_or_fileobj = str(fpath),
                path_in_repo    = fname,
                repo_id         = HF_REPO_ID,
                repo_type       = "model",
            )
    print(f"Pushed → https://huggingface.co/{HF_REPO_ID}")
else:
    print("\nHF_REPO_ID / HF_TOKEN not set — skipping Hub push.")
    print("To push: export HF_REPO_ID=YourName/tenacious-judge HF_TOKEN=hf_xxx")

# ── 11. SUMMARY ───────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("Training complete.")
print(f"  Wall time  : {elapsed / 60:.1f} min")
print(f"  Final loss : {train_result.training_loss:.4f}")
print(f"  Output dir : {OUTPUT_DIR}")
print("  Files:")
for f in sorted(out.iterdir()):
    print(f"    {f.name}")
print("─" * 60)
