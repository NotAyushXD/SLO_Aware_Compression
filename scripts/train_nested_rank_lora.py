#!/usr/bin/env python3
"""train_nested_rank_lora.py

Train a **nested-rank LoRA** adapter with **multi-frequency updates**.

What this enables (paper story)
------------------------------
You train one adapter at rank Rmax, but during training you repeatedly sample an
active rank r ∈ {r1, r2, ..., Rmax}. Smaller ranks are sampled more often.
Because the smaller ranks are prefixes of the larger one, tier-0 parameters get
updated every step, while the larger-rank "slow" components update less often.

At serving time, you can select a tier by setting `adapter_rank` per request.
The serving stack masks the LoRA rank dimension via a forward hook (no weight
slicing / no separate checkpoints required).

Example
-------
python scripts/train_nested_rank_lora.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --processed_dir ./data/processed \
  --dataset_type gsm8k \
  --output_dir ./adapters/gsm8k \
  --rmax 32 \
  --rank_tiers 8,16,32 \
  --rank_probs 0.6,0.3,0.1 \
  --max_train_examples 2000 \
  --num_train_steps 800 \
  --per_device_batch_size 1 \
  --grad_accum 8 \
  --lr 2e-4

Notes
-----
- This script is designed to be *Kaggle-friendly*: resumeable checkpoints
  (save every N steps) and bounded training steps.
- Requires `peft` (pip install peft) and standard transformers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from adapter_utils import NestedRankController, require_peft
from prompt_templates import build_llama_formatted_prompt, split_system_user


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _build_prompt_text(tokenizer, prompt: str) -> str:
    """Build the *exact* prompt prefix used by the serving stack."""

    system, user = split_system_user(prompt)
    if getattr(tokenizer, "chat_template", None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Fallback: a simple instruction format.
    parts = []
    if system:
        parts.append(system)
    parts.append(user)
    return "\n\n".join(parts) + "\n\nAssistant:"


def _target_for_example(dataset_type: str, ex: Dict[str, Any]) -> str:
    ds = (dataset_type or "").lower().strip()
    ans = str(ex.get("answer", "")).strip()
    if ds == "gsm8k":
        # The evaluation harness enforces strict final-answer formatting.
        return f"\nFINAL_ANSWER: {ans}\n"
    if ds == "mmlu":
        return f"{ans}\n"
    # Generic: just output the label
    return f"{ans}\n"


class SFTDataset(Dataset):
    def __init__(
        self,
        *,
        tokenizer,
        examples: List[Dict[str, Any]],
        dataset_type: str,
        prompt_mode: str = "accuracy",
        max_seq_len: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.samples: List[Dict[str, torch.Tensor]] = []
        ds = (dataset_type or "").lower().strip()

        for ex in examples:
            # Rebuild the prompt in the same format as evaluation.
            formatted_prompt, _mt, _stops = build_llama_formatted_prompt(ex, ds, prompt_mode=prompt_mode)
            prompt_text = _build_prompt_text(tokenizer, formatted_prompt)
            target_text = _target_for_example(ds, ex)
            full_text = prompt_text + target_text

            # Tokenize prompt+target; mask labels on prompt tokens.
            tok_full = tokenizer(full_text, truncation=True, max_length=max_seq_len, return_tensors="pt")
            tok_prompt = tokenizer(prompt_text, truncation=True, max_length=max_seq_len, return_tensors="pt")

            input_ids = tok_full["input_ids"][0]
            attention_mask = tok_full.get("attention_mask", torch.ones_like(input_ids))
            labels = input_ids.clone()

            prompt_len = int(tok_prompt["input_ids"].shape[1])
            labels[:prompt_len] = -100

            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def _collate(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    # Pad to max length in batch.
    max_len = max(int(x["input_ids"].shape[0]) for x in batch)

    def pad(x: torch.Tensor, fill: int) -> torch.Tensor:
        if x.shape[0] == max_len:
            return x
        pad_len = max_len - x.shape[0]
        return torch.cat([x, x.new_full((pad_len,), fill)], dim=0)

    input_ids = torch.stack([pad(x["input_ids"], pad_id) for x in batch], dim=0)
    attention_mask = torch.stack([pad(x["attention_mask"], 0) for x in batch], dim=0)
    labels = torch.stack([pad(x["labels"], -100) for x in batch], dim=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _parse_csv_ints(s: str) -> List[int]:
    out = []
    for t in (s or "").split(","):
        t = t.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def _parse_csv_floats(s: str) -> List[float]:
    out = []
    for t in (s or "").split(","):
        t = t.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--processed_dir", type=str, default="./data/processed")
    p.add_argument("--dataset_type", type=str, required=True, choices=["gsm8k", "mmlu"])
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--prompt_mode", type=str, default="accuracy", choices=["accuracy", "slo"])
    p.add_argument("--max_seq_len", type=int, default=1024)

    # LoRA
    p.add_argument("--rmax", type=int, default=32)
    p.add_argument("--rank_tiers", type=str, default="8,16,32")
    p.add_argument("--rank_probs", type=str, default="0.6,0.3,0.1")
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,v_proj,k_proj,o_proj",
        help="Comma-separated list of module names to LoRA-tune.",
    )
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--max_train_examples", type=int, default=5000)
    p.add_argument("--max_val_examples", type=int, default=512)
    p.add_argument("--num_train_steps", type=int, default=1000)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--resume_from", type=str, default=None)

    # Device
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"])

    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    peft = require_peft()
    from peft import LoraConfig, get_peft_model, TaskType

    tiers = _parse_csv_ints(args.rank_tiers)
    probs = _parse_csv_floats(args.rank_probs)
    if len(tiers) != len(probs):
        raise ValueError("--rank_tiers and --rank_probs must have the same length")
    if abs(sum(probs) - 1.0) > 1e-3:
        s = sum(probs)
        probs = [p / s for p in probs]

    rmax = int(args.rmax)
    if max(tiers) != rmax:
        # Ensure the largest tier equals rmax.
        if rmax not in tiers:
            tiers.append(rmax)
            probs.append(min(probs) if probs else 0.1)
            s = sum(probs)
            probs = [p / s for p in probs]

    # Load data
    train_path = str(Path(args.processed_dir) / "train_data.jsonl")
    val_path = str(Path(args.processed_dir) / "val_data.jsonl")
    train_all = [x for x in _read_jsonl(train_path) if x.get("dataset") == args.dataset_type]
    val_all = [x for x in _read_jsonl(val_path) if x.get("dataset") == args.dataset_type]
    train_all = train_all[: int(args.max_train_examples)]
    val_all = val_all[: int(args.max_val_examples)]

    print(f"Loaded {len(train_all)} train and {len(val_all)} val examples for {args.dataset_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model
    dtype = args.dtype
    torch_dtype = None
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model.to(args.device)
    model.train()

    # LoRA config: use lora_alpha=rmax so scaling alpha/r is 1.0.
    target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rmax,
        lora_alpha=rmax,
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)

    # Determine adapter name.
    adapter_name = "default"
    try:
        if hasattr(model, "peft_config") and isinstance(model.peft_config, dict) and len(model.peft_config) > 0:
            adapter_name = next(iter(model.peft_config.keys()))
    except Exception:
        adapter_name = "default"

    # Nested-rank controller
    nested = NestedRankController(model)
    nested.ensure_installed_for_adapter(adapter_name)

    # Datasets
    train_ds = SFTDataset(
        tokenizer=tokenizer,
        examples=train_all,
        dataset_type=args.dataset_type,
        prompt_mode=args.prompt_mode,
        max_seq_len=int(args.max_seq_len),
    )
    val_ds = SFTDataset(
        tokenizer=tokenizer,
        examples=val_all,
        dataset_type=args.dataset_type,
        prompt_mode=args.prompt_mode,
        max_seq_len=int(args.max_seq_len),
    )

    dl = DataLoader(
        train_ds,
        batch_size=int(args.per_device_batch_size),
        shuffle=True,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
    )

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # Resume (best-effort)
    global_step = 0
    if args.resume_from:
        ckpt = Path(args.resume_from)
        if ckpt.exists():
            sd = torch.load(str(ckpt / "trainer_state.pt"), map_location="cpu")
            global_step = int(sd.get("global_step", 0))
            try:
                model.load_adapter(str(ckpt), adapter_name=adapter_name, is_trainable=True)
            except Exception:
                pass
            try:
                opt.load_state_dict(torch.load(str(ckpt / "optimizer.pt"), map_location="cpu"))
            except Exception:
                pass
            print(f"Resumed from {ckpt} at step {global_step}")

    # Training loop
    t0 = time.time()
    model.zero_grad(set_to_none=True)
    dl_iter = iter(dl)

    def sample_rank() -> int:
        return int(random.choices(tiers, weights=probs, k=1)[0])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(global_step, int(args.num_train_steps)):
        model.train()
        active_r = sample_rank()
        nested.set_active_rank(adapter_name, active_r)

        # Gradient accumulation
        loss_accum = 0.0
        for _ in range(int(args.grad_accum)):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                batch = next(dl_iter)

            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            (loss / float(args.grad_accum)).backward()
            loss_accum += float(loss.detach().cpu())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if (step + 1) % 10 == 0:
            dt = time.time() - t0
            print(
                f"step={step+1}/{args.num_train_steps} loss={loss_accum/float(args.grad_accum):.4f} "
                f"active_rank={active_r} elapsed_min={dt/60.0:.1f}"
            )

        # Periodic save (adapter-only)
        if (step + 1) % int(args.save_every) == 0 or (step + 1) == int(args.num_train_steps):
            ckpt_dir = out_dir / f"ckpt_step_{step+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # Save adapter weights
            model.save_pretrained(str(ckpt_dir))
            torch.save({"global_step": step + 1}, str(ckpt_dir / "trainer_state.pt"))
            try:
                torch.save(opt.state_dict(), str(ckpt_dir / "optimizer.pt"))
            except Exception:
                pass
            print(f"Saved checkpoint: {ckpt_dir}")

    # Final adapter export (copy latest checkpoint into output_dir root)
    # We keep output_dir itself as a valid PEFT adapter folder.
    model.save_pretrained(str(out_dir))

    meta = {
        "base_model": args.base_model,
        "dataset_type": args.dataset_type,
        "rmax": rmax,
        "rank_tiers": tiers,
        "rank_probs": probs,
        "target_modules": target_modules,
        "prompt_mode": args.prompt_mode,
        "max_seq_len": int(args.max_seq_len),
        "num_train_steps": int(args.num_train_steps),
        "grad_accum": int(args.grad_accum),
        "per_device_batch_size": int(args.per_device_batch_size),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "adapter_name": adapter_name,
    }
    with open(out_dir / "nested_rank_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Adapter saved to: {out_dir}")


if __name__ == "__main__":
    main()
