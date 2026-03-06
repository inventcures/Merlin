"""
GGUF inference backend: hybrid PyTorch (image encoder + adapter) + llama.cpp (text decoder).

Requires:
  - Merged GGUF model from scripts/convert_to_gguf.py
  - PyTorch state_dicts for image_encoder.pt and adapter.pt
  - pip install llama-cpp-python

Usage (standalone):
    python scripts/inference_gguf.py --nifti_path /path/to/scan.nii.gz

See scripts/eval_pipeline.py --backend gguf for evaluation pipeline usage.
"""

import os
import sys
import time
import ctypes
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.configs import DEVICE, ORGAN_SYSTEMS

warnings.filterwarnings("ignore")

_gguf_cache = {}

DEFAULT_GGUF_DIR = "checkpoints/radllama-7b-merged"
DEFAULT_GGUF_MODEL = "radllama-7b-q8_0.gguf"


def _load_image_pipeline(gguf_dir: str):
    """Load PyTorch image encoder + adapter from exported state_dicts."""
    from merlin.models.radiology_report_generation import (
        ModifiedImageEncoder,
        Adapter,
    )

    encoder = ModifiedImageEncoder()
    encoder.load_state_dict(
        torch.load(os.path.join(gguf_dir, "image_encoder.pt"), map_location="cpu")
    )
    encoder.eval()

    adapter = Adapter(2048, 4096)
    adapter.load_state_dict(
        torch.load(os.path.join(gguf_dir, "adapter.pt"), map_location="cpu")
    )
    adapter.eval()

    return encoder, adapter


def _load_embedding_weights(gguf_dir: str) -> np.ndarray:
    """Load the token embedding matrix from the saved HF model weights."""
    from safetensors import safe_open

    hf_dir = os.path.join(gguf_dir, "hf_merged")
    for fname in os.listdir(hf_dir):
        if fname.endswith(".safetensors"):
            with safe_open(os.path.join(hf_dir, fname), framework="numpy") as f:
                if "model.embed_tokens.weight" in f.keys():
                    return f.get_tensor("model.embed_tokens.weight").astype(np.float32)

    st = torch.load(os.path.join(hf_dir, "pytorch_model.bin"), map_location="cpu")
    return st["model.embed_tokens.weight"].numpy().astype(np.float32)


def _get_image_embeddings(
    encoder, adapter, nifti_path: str, target_dtype: torch.dtype = torch.float32
) -> np.ndarray:
    """Run image through encoder + adapter, return numpy array [490, 4096]."""
    from merlin.data import DataLoader

    datalist = [{"image": nifti_path}]
    dataloader = DataLoader(
        datalist=datalist,
        cache_dir="/tmp/merlin_cache",
        batchsize=1,
        shuffle=False,
        num_workers=0,
    )

    for batch in dataloader:
        image = batch["image"].to(DEVICE)
        break

    with torch.no_grad():
        img_feats = encoder(image)
        img_embeds = adapter(img_feats).to(target_dtype)

    return img_embeds.squeeze(0).cpu().numpy()


def load_gguf_backend(gguf_dir: str = DEFAULT_GGUF_DIR, gguf_model: str = DEFAULT_GGUF_MODEL):
    """Load and cache the GGUF model + PyTorch image pipeline."""
    cache_key = os.path.join(gguf_dir, gguf_model)
    if cache_key in _gguf_cache:
        return _gguf_cache[cache_key]

    from llama_cpp import llama_cpp as lib
    from transformers import AutoTokenizer

    gguf_path = os.path.join(gguf_dir, gguf_model)
    if not os.path.isfile(gguf_path):
        raise FileNotFoundError(
            f"GGUF model not found: {gguf_path}\n"
            f"Run: python scripts/convert_to_gguf.py --output_dir {gguf_dir}"
        )

    print(f"[GGUF] Loading model from {gguf_path}...")
    model_params = lib.llama_model_default_params()
    model = lib.llama_model_load_from_file(gguf_path.encode(), model_params)
    if not model:
        raise RuntimeError(f"Failed to load GGUF model: {gguf_path}")

    ctx_params = lib.llama_context_default_params()
    ctx_params.n_ctx = 2048
    ctx_params.n_batch = 1024
    ctx = lib.llama_init_from_model(model, ctx_params)
    if not ctx:
        raise RuntimeError("Failed to create llama context")

    n_embd = lib.llama_n_embd(model)
    print(f"[GGUF] Model loaded. n_embd={n_embd}")

    tokenizer_dir = os.path.join(gguf_dir, "hf_merged")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)

    print("[GGUF] Loading PyTorch image pipeline...")
    encoder, adapter = _load_image_pipeline(gguf_dir)

    print("[GGUF] Loading embedding weights...")
    embed_weights = _load_embedding_weights(gguf_dir)

    vocab = lib.llama_model_n_vocab(model)

    result = {
        "lib": lib,
        "model": model,
        "ctx": ctx,
        "n_embd": n_embd,
        "vocab_size": vocab,
        "tokenizer": tokenizer,
        "encoder": encoder,
        "adapter": adapter,
        "embed_weights": embed_weights,
    }
    _gguf_cache[cache_key] = result
    print("[GGUF] Backend ready")
    return result


def _get_token_embeddings(backend, token_ids: list[int]) -> np.ndarray:
    """Look up token embeddings from the saved HF embedding weight matrix."""
    return backend["embed_weights"][token_ids]


def _decode_embeddings(backend, combined_embeds: np.ndarray) -> int:
    """Feed combined embeddings [seq_len, n_embd] into the model. Returns n_past."""
    lib = backend["lib"]
    ctx = backend["ctx"]
    n_embd = backend["n_embd"]
    seq_len = combined_embeds.shape[0]

    lib.llama_kv_cache_clear(ctx)

    batch = lib.llama_batch_init(seq_len, n_embd, 1)
    try:
        batch.n_tokens = seq_len

        flat = np.ascontiguousarray(combined_embeds, dtype=np.float32).flatten()
        ctypes.memmove(
            batch.embd,
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            flat.nbytes,
        )

        for i in range(seq_len):
            batch.pos[i] = i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 1 if i == seq_len - 1 else 0

        rc = lib.llama_decode(ctx, batch)
        if rc != 0:
            raise RuntimeError(f"llama_decode failed with code {rc}")
    finally:
        lib.llama_batch_free(batch)

    return seq_len


def _sample_greedy(backend, n_past: int, max_new_tokens: int, eos_token_id: int) -> list[int]:
    """Greedy autoregressive sampling loop."""
    lib = backend["lib"]
    ctx = backend["ctx"]
    vocab_size = backend["vocab_size"]

    output_ids = []
    pos = n_past

    for step in range(max_new_tokens):
        logits_ptr = lib.llama_get_logits_ith(ctx, -1)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
        token_id = int(np.argmax(logits))

        if token_id == eos_token_id:
            break

        output_ids.append(token_id)

        batch = lib.llama_batch_init(1, 0, 1)
        batch.n_tokens = 1
        batch.token[0] = token_id
        batch.pos[0] = pos
        batch.n_seq_id[0] = 1
        batch.seq_id[0][0] = 0
        batch.logits[0] = 1

        rc = lib.llama_decode(ctx, batch)
        lib.llama_batch_free(batch)
        if rc != 0:
            raise RuntimeError(f"llama_decode (sampling step {step}) failed with code {rc}")
        pos += 1

    return output_ids


def generate_for_organ(backend, image_embeds: np.ndarray, organ_system: str, max_new_tokens: int = 128) -> str:
    """Generate text for a single organ system given precomputed image embeddings."""
    tokenizer = backend["tokenizer"]
    n_embd = backend["n_embd"]

    prefix = f"Generate a radiology report for {organ_system}###\n"
    token_ids = tokenizer.encode(prefix)
    token_ids = token_ids[1:]

    token_embeds = _get_token_embeddings(backend, token_ids)

    combined = np.concatenate([image_embeds, token_embeds], axis=0)
    assert combined.shape[1] == n_embd, f"Embedding dim mismatch: {combined.shape[1]} vs {n_embd}"

    n_past = _decode_embeddings(backend, combined)

    eos_id = tokenizer.eos_token_id or 48134
    output_ids = _sample_greedy(backend, n_past, max_new_tokens, eos_id)

    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return text.split("###")[0].strip()


def run_report_generation(
    nifti_path: str,
    gguf_dir: str = DEFAULT_GGUF_DIR,
    gguf_model: str = DEFAULT_GGUF_MODEL,
) -> tuple[str, float]:
    """
    Generate a full radiology report using GGUF backend.

    Returns:
        (full_report_text, inference_time_seconds)
    """
    backend = load_gguf_backend(gguf_dir, gguf_model)

    image_embeds = _get_image_embeddings(
        backend["encoder"], backend["adapter"], nifti_path
    )

    t0 = time.time()

    report_parts = []
    for organ_system in ORGAN_SYSTEMS:
        text = generate_for_organ(backend, image_embeds, organ_system)
        if text:
            report_parts.append(text)

    elapsed = time.time() - t0
    full_report = " ".join(report_parts)
    return full_report, elapsed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GGUF inference for Merlin report generation")
    parser.add_argument("--nifti_path", required=True, help="Path to NIfTI file")
    parser.add_argument("--gguf_dir", default=DEFAULT_GGUF_DIR)
    parser.add_argument("--gguf_model", default=DEFAULT_GGUF_MODEL)
    args = parser.parse_args()

    report, elapsed = run_report_generation(args.nifti_path, args.gguf_dir, args.gguf_model)
    print(f"\n{'=' * 60}")
    print(f"  Report ({elapsed:.1f}s):")
    print(f"{'=' * 60}")
    print(report)
