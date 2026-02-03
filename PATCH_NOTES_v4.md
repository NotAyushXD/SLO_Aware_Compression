# v4 Patch Notes (HF + vLLM backend)

This bundle adds three paper-critical improvements:

## 1) GSM8K formatting robustness (server-side postprocessor)
- We keep **strict** evaluation (`FINAL_ANSWER: <num>`).
- But the server now **repairs** common near-misses by appending a canonical final line *only when a conservative parseable answer is found*.
- The raw model text is still saved in metrics (`raw_text`) for transparency.

## 2) Strict + Parseable GSM8K metrics (paper-friendly)
`evaluation.py` now reports:
- **Strict accuracy / strict format_ok** (primary metric)
- **Parseable accuracy / parseable format_ok** (sensitivity metric)

This helps you show:
- “How much accuracy you lose due to formatting issues vs reasoning issues”
- “How robust your SLO prompts are”

## 3) Optional vLLM async backend + HF vs vLLM comparison runner
- `--backend hf` (default): existing Transformers server with optional micro-batching.
- `--backend vllm`: new vLLM async streaming backend (queue-inclusive TTFT measured to first token).
- `run_backend_comparison.py`: runs HF(no-batch), HF(micro-batch), vLLM(continuous) and writes a CSV/JSON summary.

## Files added / changed
Added:
- `answer_utils.py`
- `vllm_server.py`
- `run_backend_comparison.py`
- `requirements_vllm.txt`
- `VLLM_SETUP.md`

Changed:
- `server.py` (GSM postprocessor + raw_text in metrics)
- `evaluation.py` (strict + parseable metrics)
- `run_baseline_evaluation.py` (backend switch + vLLM args)

## Important note on vLLM + variants
- **base**: works with `--backend vllm` directly.
- **med**: requires `--vllm_model_override` pointing to a *pre-quantized* checkpoint (AWQ/GPTQ/etc). This matches “Option 1”.
- **cheap**: best-effort; depends on which quantization modes your vLLM build supports.

