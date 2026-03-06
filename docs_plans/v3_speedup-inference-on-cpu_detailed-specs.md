# v3: Speedup CPU Inference for RadLLaMA-7B Report Generation

> Status: Implemented

## Context

RadLLaMA-7B (bfloat16) on CPU takes ~60-90 min per case (13 organ systems x max 128 tokens x ~2-5 sec/token on Intel Core Ultra 7 268V). Three free/cheap optimizations exist in the current code, plus GGUF conversion for maximum speedup.

## Architecture (relevant subset)

```
NIfTI → ResNet3D [B,490,2048] → Adapter [B,490,4096] → RadLLaMA-7B (bf16 + LoRA r=512) → text
          ~240 MB (fp32)           ~34 MB (fp32)           ~14 GB (bf16) + ~256 MB LoRA
```

Key files:
- `merlin/models/radiology_report_generation.py` — `TextDecoder`, `Clip3DForTextGeneration`
- `merlin/models/load.py` — `Merlin` class, checkpoint loading
- `scripts/inference.py` — `get_merlin_model()`, `run_report_generation()`
- `scripts/configs.py` — `DEVICE`, `ORGAN_SYSTEMS`

---

## Optimization 1: Disable Gradient Checkpointing at Inference

**Problem**: `TextDecoder.__init__()` calls `self.text_decoder.gradient_checkpointing_enable()` (line 87). This recomputes activations during forward pass to save memory — useful only for training. During inference it adds ~30-50% overhead.

**Fix location**: `scripts/inference.py:get_merlin_model()`, after LoRA merge

**Impact**: ~1.3-1.5x speedup, zero quality loss.

---

## Optimization 2: Merge LoRA Weights

**Problem**: LoRA with `r=512` adds an extra `A @ B` matmul (shape 4096→512→4096) to every linear layer during forward pass. With 32 transformer layers x multiple linear layers each, this is significant.

**Fix location**: `scripts/inference.py:get_merlin_model()`, after model loaded.

**Approach**: Use `merge_and_unload()` to fold LoRA into base weights, returning a clean `LlamaForCausalLM`.

**Breaking change handled**: After `merge_and_unload()`, the PEFT wrapper is removed. The attribute path changes:
- BEFORE (PeftModel): `self.text_decoder.model.model.embed_tokens`
- AFTER (LlamaForCausalLM): `self.text_decoder.model.embed_tokens`

Resolved by replacing hardcoded attribute paths with `self.text_decoder.get_input_embeddings()` in `radiology_report_generation.py` lines 122 and 158.

**Impact**: ~1.2-1.5x speedup, zero quality loss.

---

## Optimization 3: INT8 Dynamic Quantization

**Problem**: Linear layer matmuls in bf16 on CPU are slow. PyTorch's dynamic quantization converts weights to INT8 and uses optimized CPU kernels (FBGEMM/QNNPACK with AVX2/AVX512).

**Fix location**: `scripts/inference.py:get_merlin_model()`, after LoRA merge.

**How it works**: Weights are statically quantized to INT8. Activations are dynamically quantized per-batch. Only `nn.Linear` layers are affected (attention QKV/O projections + MLP up/gate/down projections). Embeddings, LayerNorm, and other layers remain in original precision.

**Memory**: ~14 GB (bf16) → ~7 GB (int8) for the LLM weights.

**Impact**: ~1.5-2x speedup, <1% quality degradation (well-studied on 7B models).

---

## Combined Effect (Optimizations 1-3)

All three applied in `scripts/inference.py:get_merlin_model()`:

```python
if mode == "report":
    td = model.model.decode_text
    td.text_decoder = td.text_decoder.merge_and_unload()
    td.text_decoder.gradient_checkpointing_disable()
    td.text_decoder = torch.quantization.quantize_dynamic(
        td.text_decoder, {torch.nn.Linear}, dtype=torch.qint8
    )
```

**Expected**: ~2.5-4x combined speedup → ~15-25 min per case (down from 60-90 min).

**Memory**: ~240 MB (encoder) + 34 MB (adapter) + ~7 GB (LLM int8) ≈ **~7.5 GB total** (down from ~15 GB).

---

## Optimization 4: GGUF Conversion + llama.cpp Inference

### 4A: Conversion Script — `scripts/convert_to_gguf.py`

Pipeline:
1. Load Merlin(RadiologyReport=True)
2. merge_and_unload() → clean LlamaForCausalLM
3. Save merged HF model + tokenizer
4. Save image encoder + adapter state_dicts
5. Convert HF → GGUF via llama.cpp's convert_hf_to_gguf.py (Q8_0 / Q4_K_M)

### 4B: GGUF Inference — `scripts/inference_gguf.py`

Hybrid architecture: PyTorch (image encoder + adapter) + llama.cpp (text decoder).

```
Image → [PyTorch: ResNet3D + Adapter] → image_embeds [490, 4096]
                                              ↓
Text prefix → [llama.cpp: embed_tokens] → text_embeds [N, 4096]
                                              ↓
              [concat + llama_decode with embd batch] → logits
                                              ↓
              [greedy sampling loop] → report text
```

### 4C: Expected Performance (GGUF)

| Quantization | Model Size | Est. Tokens/sec (CPU) | Time per Case |
|-------------|-----------|----------------------|---------------|
| Q8_0 | ~7 GB | ~5-8 tok/s | ~5-10 min |
| Q4_K_M | ~4 GB | ~8-15 tok/s | ~3-7 min |

vs. current bf16: ~0.3-0.5 tok/s → ~60-90 min per case.

---

## File Changes Summary

### Modified Files

| File | Change |
|------|--------|
| `merlin/models/radiology_report_generation.py:122,158` | `.model.model.embed_tokens()` → `.get_input_embeddings()()` |
| `scripts/inference.py:get_merlin_model()` | merge_and_unload + grad_ckpt disable + INT8 quantization |
| `scripts/eval_pipeline.py` | `--backend` flag (pytorch/gguf), lazy import of gguf backend |
| `pyproject.toml` | `[project.optional-dependencies] gguf` and `eval` groups |

### New Files

| File | Purpose |
|------|---------|
| `docs_plans/v3_speedup-inference-on-cpu_detailed-specs.md` | This plan document |
| `scripts/convert_to_gguf.py` | Merge LoRA + export HF + convert GGUF |
| `scripts/inference_gguf.py` | GGUF inference with hybrid PyTorch/llama.cpp |

---

## Verification

### Functional correctness
```bash
# PyTorch backend (optimizations 1-3)
python scripts/eval_pipeline.py --n_cases 1 --output_dir ./merlin_eval_results

# GGUF backend
python scripts/convert_to_gguf.py --quantization q8_0
python scripts/eval_pipeline.py --n_cases 1 --output_dir ./merlin_eval_results_gguf --backend gguf
```

### Quality check
- Compare generated reports between: original (bf16 no opts) vs optimized (int8) vs GGUF (q8_0)
- Metric ranges should stay within: BLEU-4 [0.08-0.18], ROUGE-L [0.15-0.25], BERTScore [0.82-0.88]
