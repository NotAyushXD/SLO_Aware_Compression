#!/usr/bin/env python3
"""test_adapter_smoke.py

Smoke test for the adapter + nested-rank plumbing.

This is *not* a quality test. It only verifies that:
  1) We can create a PEFT LoRA adapter folder.
  2) The serving stack can load it via AdapterManager.
  3) Requests can specify adapter_id + adapter_rank without crashing.

If `peft` is not installed, this test prints a skip message and exits 0.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import torch

from adapter_utils import peft_available, require_peft
from server import SingleVariantServer


def main() -> None:
    if not peft_available():
        print("[SKIP] peft not installed; adapter smoke test skipped.")
        return

    require_peft()
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = "sshleifer/tiny-gpt2"

    tmp_root = tempfile.mkdtemp(prefix="adapter_smoke_")
    try:
        adapter_id = "toy"
        adapter_dir = os.path.join(tmp_root, adapter_id)
        os.makedirs(adapter_dir, exist_ok=True)

        # Create a trivial adapter (no training required for smoke).
        model = AutoModelForCausalLM.from_pretrained(base)
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=8, target_modules=["c_attn"], bias="none")
        model = get_peft_model(model, cfg)
        model.save_pretrained(adapter_dir)

        # Instantiate server with adapters enabled.
        srv = SingleVariantServer(
            model_name=base,
            variant="base",
            device="cpu",
            dtype="bfloat16",
            enable_batching=False,
            enable_adapters=True,
            adapter_root=tmp_root,
            adapter_policy="fixed",
            adapter_fixed=adapter_id,
            adapter_rank_policy="fixed",
            adapter_fixed_rank=4,
            max_loaded_adapters=2,
        )

        txt, m = srv.generate(
            prompt="What is 2+2?",
            dataset_type="mmlu",
            difficulty="easy",
            prompt_mode="accuracy",
            max_tokens=4,
        )

        assert isinstance(txt, str)
        assert isinstance(m, dict)
        assert m.get("adapter_id") == adapter_id
        assert m.get("adapter_active_rank") in {4, None}
        print("[OK] adapter smoke test passed")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
