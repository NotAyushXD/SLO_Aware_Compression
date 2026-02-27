"""adapter_utils.py

PEFT/LoRA adapter support for the serving stack.

This repo originally supported a *quantization* portfolio (cheap/med/base).
For the paper's "portfolio of variants" story, we also add a *shared-base*
portfolio of PEFT adapters, and (optionally) "nested-rank" LoRA tiers.

Key design goals
----------------
1) **Shared-base adapters**: a SingleVariantServer can load multiple LoRA
   adapters into one base model and switch adapters at runtime.
2) **Adapter caching**: keep only up to K adapters resident, evict with LRU.
3) **Nested-rank tiers**: train one LoRA adapter with rank Rmax, and at runtime
   choose an *active rank* r <= Rmax by masking the LoRA rank dimension.
   This enables tiers (e.g., 8/16/32) without storing multiple adapters.

Notes
-----
- PEFT is an *optional dependency*. If `peft` is not installed, code paths that
  enable adapters fail with a clear error message.
- This module avoids peft-internal imports and relies on attribute inspection.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _try_import_peft():
    try:
        import peft  # type: ignore

        return peft
    except Exception as e:
        return e


def peft_available() -> bool:
    return not isinstance(_try_import_peft(), Exception)


def require_peft() -> Any:
    peft = _try_import_peft()
    if isinstance(peft, Exception):
        raise RuntimeError(
            "PEFT/LoRA adapters requested but `peft` is not installed. "
            "Install with: pip install peft\n" + f"Original import error: {peft}"
        )
    return peft


def load_adapter_config(adapter_dir: str) -> Dict[str, Any]:
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


@dataclass
class AdapterLoadResult:
    adapter_id: str
    cache_hit: bool
    load_ms: float
    evicted: List[str]


class AdapterRegistry:
    """Resolve adapter paths from an adapter root.

    Convention: <adapter_root>/<adapter_id>/
    """

    def __init__(self, adapter_root: Optional[str]):
        self.adapter_root = os.path.abspath(os.path.expanduser(adapter_root)) if adapter_root else None

    def resolve(self, adapter_id: str) -> Optional[str]:
        if not self.adapter_root:
            return None
        aid = (adapter_id or "").strip()
        if not aid:
            return None
        path = os.path.join(self.adapter_root, aid)
        if os.path.isdir(path):
            return path
        return None


class NestedRankController:
    """Nested-rank runtime control using forward hooks.

    We mask the *output* of each LoRA A-projection (rank dimension) so that
    only the prefix ranks contribute.
    """

    def __init__(self, peft_model: Any):
        self.model = peft_model
        self._lora_a_by_adapter: Dict[str, List[Any]] = {}

    @staticmethod
    def _is_mapping_like(x: Any) -> bool:
        return hasattr(x, "__contains__") and hasattr(x, "__getitem__")

    def _find_lora_a_modules(self, adapter_name: str) -> List[Any]:
        out: List[Any] = []
        for _name, module in self.model.named_modules():
            lora_a = getattr(module, "lora_A", None)
            if lora_a is None or not self._is_mapping_like(lora_a):
                continue
            try:
                if adapter_name not in lora_a:
                    continue
                out.append(lora_a[adapter_name])
            except Exception:
                continue
        return out

    @staticmethod
    def _mask_hook(mod, _inp, out):
        r = getattr(mod, "_nested_active_rank", None)
        if r is None:
            return out
        try:
            r = int(r)
        except Exception:
            return out
        if out is None:
            return out
        try:
            last = int(out.shape[-1])
        except Exception:
            return out
        if r <= 0:
            return out * 0
        if r >= last:
            return out

        cache = getattr(mod, "_nested_rank_mask_cache", None)
        if cache is None:
            cache = {}
            setattr(mod, "_nested_rank_mask_cache", cache)
        key = (last, r, str(out.device), str(out.dtype))
        mask = cache.get(key)
        if mask is None:
            mask = out.new_zeros((last,))
            mask[:r] = 1
            cache[key] = mask
        return out * mask

    def ensure_installed_for_adapter(self, adapter_name: str) -> None:
        if adapter_name in self._lora_a_by_adapter:
            return
        a_mods = self._find_lora_a_modules(adapter_name)
        for a in a_mods:
            if getattr(a, "_nested_rank_hook_installed", False):
                continue
            try:
                a.register_forward_hook(self._mask_hook)
                setattr(a, "_nested_rank_hook_installed", True)
            except Exception:
                continue
        self._lora_a_by_adapter[adapter_name] = a_mods

    def set_active_rank(self, adapter_name: str, active_rank: Optional[int]) -> None:
        if active_rank is None:
            return
        r = max(1, int(active_rank))
        self.ensure_installed_for_adapter(adapter_name)
        for a in self._lora_a_by_adapter.get(adapter_name, []):
            try:
                setattr(a, "_nested_active_rank", int(r))
            except Exception:
                continue


class AdapterManager:
    """Adapter cache + activation helper for one model instance."""

    def __init__(
        self,
        *,
        base_model: Any,
        adapter_registry: AdapterRegistry,
        max_loaded_adapters: int = 8,
        eviction_policy: str = "lru",
        synthetic_load_ms: float = 0.0,
        synthetic_switch_ms: float = 0.0,
        # If True, allow "synthetic" adapters without PEFT installed and/or
        # without adapter directories on disk. This keeps the caching + overhead
        # accounting path runnable for experiments (e.g., adapter churn sweeps).
        allow_missing_adapters: bool = False,
    ):
        self.base_model = base_model
        self.adapter_registry = adapter_registry
        self.max_loaded_adapters = int(max(1, max_loaded_adapters))
        self.eviction_policy = (eviction_policy or "lru").lower().strip()
        self.synthetic_load_ms = float(max(0.0, synthetic_load_ms))
        self.synthetic_switch_ms = float(max(0.0, synthetic_switch_ms))
        self.allow_missing_adapters = bool(allow_missing_adapters)

        # Lightweight online statistics (used for router features / setup-cost prediction).
        # These are best-effort and not meant to be perfectly synchronized across threads.
        self.ewma_alpha: float = 0.2
        self.ewma_load_ms: float = 0.0
        self.ewma_switch_ms: float = 0.0
        self._num_load_samples: int = 0
        self._num_switch_samples: int = 0

        self.model = base_model
        self.is_peft_wrapped = False

        self._lru: "OrderedDict[str, float]" = OrderedDict()
        self.active_adapter: Optional[str] = None

        self.nested_rank = NestedRankController(peft_model=self.model)

    def _ewma_update(self, cur: float, new: float, n: int) -> float:
        """Return updated EWMA value (with sane init)."""
        try:
            x = float(new)
        except Exception:
            return float(cur)
        if n <= 0 or float(cur) <= 0.0:
            return float(max(0.0, x))
        a = float(self.ewma_alpha)
        return float((1.0 - a) * float(cur) + a * float(max(0.0, x)))

    def snapshot_for(self, adapter_id: str, *, active_rank: Optional[int] = None) -> Dict[str, Any]:
        """Best-effort snapshot for router features.

        Returns numeric state that approximates adapter *hotness* and predicted setup cost.
        """

        aid = (adapter_id or "").strip()
        loaded = bool(aid and aid in self._lru)
        active = bool(aid and self.active_adapter == aid)

        # Conservative estimates: fall back to configured synthetic costs when EWMA is missing.
        load_est = float(self.ewma_load_ms) if float(self.ewma_load_ms) > 0.0 else float(self.synthetic_load_ms)
        switch_est = float(self.ewma_switch_ms) if float(self.ewma_switch_ms) > 0.0 else float(self.synthetic_switch_ms)

        if not aid:
            setup_est = 0.0
        elif active:
            setup_est = 0.0
        elif loaded:
            setup_est = float(max(0.0, switch_est))
        else:
            setup_est = float(max(0.0, load_est + switch_est))

        return {
            "adapter_id": aid,
            "adapter_active_rank": int(active_rank) if active_rank is not None else None,
            "resident": int(1 if loaded else 0),
            "active": int(1 if active else 0),
            "num_loaded": int(len(self._lru)),
            "capacity": int(self.max_loaded_adapters),
            "ewma_load_ms": float(load_est),
            "ewma_switch_ms": float(switch_est),
            "setup_est_ms": float(setup_est),
        }

    def loaded_adapters(self) -> List[str]:
        return list(self._lru.keys())

    def is_loaded(self, adapter_id: str) -> bool:
        return (adapter_id or "") in self._lru

    def _touch(self, adapter_id: str) -> None:
        if adapter_id in self._lru:
            self._lru.move_to_end(adapter_id)
        self._lru[adapter_id] = time.time()

    def _evict_if_needed(self) -> List[str]:
        evicted: List[str] = []
        if self.eviction_policy != "lru":
            return evicted
        while len(self._lru) > self.max_loaded_adapters:
            old_id, _ = next(iter(self._lru.items()))
            if self.active_adapter and old_id == self.active_adapter and len(self._lru) > 1:
                self._lru.move_to_end(old_id)
                continue
            self._lru.pop(old_id, None)
            evicted.append(old_id)
            try:
                if hasattr(self.model, "delete_adapter"):
                    self.model.delete_adapter(old_id)
            except Exception:
                pass
        return evicted

    def ensure_loaded(self, adapter_id: str) -> AdapterLoadResult:
        adapter_id = (adapter_id or "").strip()
        if not adapter_id:
            return AdapterLoadResult(adapter_id="", cache_hit=True, load_ms=0.0, evicted=[])
        if adapter_id in self._lru:
            self._touch(adapter_id)
            return AdapterLoadResult(adapter_id=adapter_id, cache_hit=True, load_ms=0.0, evicted=[])

        adapter_dir = self.adapter_registry.resolve(adapter_id)
        if (not adapter_dir) or (self.allow_missing_adapters and not peft_available()):
            if not self.allow_missing_adapters:
                raise FileNotFoundError(f"Adapter '{adapter_id}' not found under {self.adapter_registry.adapter_root}")

            # Synthetic adapter load (no PEFT / no files required).
            t0 = time.perf_counter()
            if self.synthetic_load_ms > 0:
                time.sleep(self.synthetic_load_ms / 1000.0)
            t1 = time.perf_counter()

            load_ms = float(max(0.0, (t1 - t0) * 1000.0))
            try:
                self.ewma_load_ms = self._ewma_update(self.ewma_load_ms, load_ms, self._num_load_samples)
                self._num_load_samples += 1
            except Exception:
                pass

            self._touch(adapter_id)
            evicted = self._evict_if_needed()
            return AdapterLoadResult(adapter_id=adapter_id, cache_hit=False, load_ms=load_ms, evicted=evicted)

        peft = require_peft()
        t0 = time.perf_counter()
        if not self.is_peft_wrapped:
            self.model = peft.PeftModel.from_pretrained(
                self.base_model,
                adapter_dir,
                adapter_name=adapter_id,
                is_trainable=False,
            )
            self.is_peft_wrapped = True
            self.nested_rank = NestedRankController(peft_model=self.model)
        else:
            if not hasattr(self.model, "load_adapter"):
                raise RuntimeError("This PEFT version does not support multi-adapter load_adapter().")
            self.model.load_adapter(adapter_dir, adapter_name=adapter_id, is_trainable=False)

        if self.synthetic_load_ms > 0:
            time.sleep(self.synthetic_load_ms / 1000.0)
        t1 = time.perf_counter()

        # Update EWMA load time statistics (miss-only).
        try:
            load_ms = float(max(0.0, (t1 - t0) * 1000.0))
            self.ewma_load_ms = self._ewma_update(self.ewma_load_ms, load_ms, self._num_load_samples)
            self._num_load_samples += 1
        except Exception:
            pass

        self._touch(adapter_id)
        evicted = self._evict_if_needed()
        return AdapterLoadResult(adapter_id=adapter_id, cache_hit=False, load_ms=(t1 - t0) * 1000.0, evicted=evicted)

    def activate(self, adapter_id: Optional[str], *, active_rank: Optional[int] = None) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "adapter_id": adapter_id or "",
            "adapter_cache_hit": 1,
            "adapter_load_ms": 0.0,
            "adapter_evicted": [],
            "adapter_switch_ms": 0.0,
            "adapter_active_rank": int(active_rank) if active_rank is not None else None,
            "adapter_num_loaded": int(len(self._lru)),
        }

        aid = (adapter_id or "").strip()
        if not aid:
            self.active_adapter = None
            return metrics

        lr = self.ensure_loaded(aid)
        metrics["adapter_cache_hit"] = int(bool(lr.cache_hit))
        metrics["adapter_load_ms"] = float(max(0.0, lr.load_ms))
        metrics["adapter_evicted"] = list(lr.evicted)
        metrics["adapter_num_loaded"] = int(len(self._lru))

        t0 = time.perf_counter()
        try:
            if hasattr(self.model, "set_adapter"):
                self.model.set_adapter(aid)
            self.active_adapter = aid
        except Exception:
            self.active_adapter = aid

        if active_rank is not None:
            try:
                self.nested_rank.set_active_rank(aid, int(active_rank))
            except Exception:
                pass

        if self.synthetic_switch_ms > 0:
            time.sleep(self.synthetic_switch_ms / 1000.0)
        t1 = time.perf_counter()

        switch_ms = float(max(0.0, (t1 - t0) * 1000.0))
        metrics["adapter_switch_ms"] = float(switch_ms)
        metrics["adapter_setup_ms"] = float(max(0.0, float(metrics.get("adapter_load_ms", 0.0) or 0.0) + switch_ms))

        # Update EWMA switch time statistics.
        try:
            self.ewma_switch_ms = self._ewma_update(self.ewma_switch_ms, switch_ms, self._num_switch_samples)
            self._num_switch_samples += 1
        except Exception:
            pass
        return metrics


def choose_adapter_id(
    *,
    policy: str,
    dataset_type: str,
    fixed_adapter: Optional[str] = None,
    explicit_adapter: Optional[str] = None,
) -> str:
    if explicit_adapter:
        return (explicit_adapter or "").strip()

    pol = (policy or "none").lower().strip()
    if pol == "none":
        return ""
    if pol == "dataset":
        return (dataset_type or "").lower().strip()
    if pol == "fixed":
        return (fixed_adapter or "").strip()
    return ""


def choose_active_rank(
    *,
    policy: str,
    difficulty: str,
    total_queue_depth: int,
    tiers: List[int],
    fixed_rank: Optional[int] = None,
) -> Optional[int]:
    if not tiers:
        return None
    tiers_sorted = sorted({int(t) for t in tiers if int(t) > 0})
    if not tiers_sorted:
        return None

    pol = (policy or "max").lower().strip()
    if pol == "fixed":
        return int(fixed_rank) if fixed_rank is not None else int(tiers_sorted[-1])
    if pol == "max":
        return int(tiers_sorted[-1])

    diff = (difficulty or "easy").lower().strip()
    if pol == "difficulty":
        if diff in {"hard", "difficult"}:
            return int(tiers_sorted[-1])
        if diff in {"medium", "med"}:
            return int(tiers_sorted[len(tiers_sorted) // 2])
        return int(tiers_sorted[0])

    if pol == "load":
        if total_queue_depth >= 32:
            return int(tiers_sorted[0])
        if total_queue_depth >= 16 and len(tiers_sorted) >= 2:
            return int(tiers_sorted[1])
        return int(tiers_sorted[-1])

    return int(tiers_sorted[-1])
