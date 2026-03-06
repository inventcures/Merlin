"""
Convert RadLLaMA-7B (with merged LoRA) to GGUF format for llama.cpp inference.

Pipeline:
  1. Load Merlin(RadiologyReport=True)
  2. Extract text_decoder PeftModel, merge_and_unload() -> clean LlamaForCausalLM
  3. Save merged HF model + tokenizer (for GGUF converter input)
  4. Save image encoder + adapter state_dicts (for hybrid inference)
  5. Convert HF model -> GGUF via llama.cpp's convert_hf_to_gguf.py

Usage:
    python scripts/convert_to_gguf.py --output_dir checkpoints/radllama-7b-merged
    python scripts/convert_to_gguf.py --output_dir checkpoints/radllama-7b-merged --quantization q4_k_m
    python scripts/convert_to_gguf.py --output_dir checkpoints/radllama-7b-merged --skip_gguf
"""

import os
import sys
import argparse
import subprocess
import shutil

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUANTIZATION_TYPES = {
    "q8_0": "Q8_0",
    "q4_k_m": "Q4_K_M",
    "f16": "F16",
}


def find_convert_script():
    """Locate llama.cpp's convert_hf_to_gguf.py."""
    candidates = [
        shutil.which("convert_hf_to_gguf"),
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
        os.path.join(os.path.dirname(sys.executable), "convert_hf_to_gguf.py"),
    ]

    try:
        import llama_cpp
        pkg_dir = os.path.dirname(llama_cpp.__file__)
        candidates.append(os.path.join(pkg_dir, "convert_hf_to_gguf.py"))
    except ImportError:
        pass

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    return None


def convert(output_dir: str, quantization: str, skip_gguf: bool):
    os.makedirs(output_dir, exist_ok=True)
    hf_dir = os.path.join(output_dir, "hf_merged")

    from merlin import Merlin

    print("[1/5] Loading Merlin (RadiologyReport=True)...")
    model = Merlin(RadiologyReport=True)
    model.eval()

    clip_model = model.model

    print("[2/5] Merging LoRA weights...")
    td = clip_model.decode_text
    merged_llm = td.text_decoder.merge_and_unload()

    print(f"[3/5] Saving merged HF model + tokenizer to {hf_dir}...")
    os.makedirs(hf_dir, exist_ok=True)
    merged_llm.save_pretrained(hf_dir, safe_serialization=True)
    td.tokenizer.save_pretrained(hf_dir)

    print("[4/5] Saving image encoder + adapter state_dicts...")
    torch.save(
        clip_model.encode_image.state_dict(),
        os.path.join(output_dir, "image_encoder.pt"),
    )
    torch.save(
        clip_model.adapter.state_dict(),
        os.path.join(output_dir, "adapter.pt"),
    )

    del model, clip_model, td, merged_llm
    import gc
    gc.collect()

    if skip_gguf:
        print("[5/5] Skipping GGUF conversion (--skip_gguf)")
        print(f"\nDone. HF model saved to: {hf_dir}")
        return

    convert_script = find_convert_script()
    if convert_script is None:
        print(
            "[5/5] ERROR: Could not find convert_hf_to_gguf.py.\n"
            "Install llama.cpp or set it on PATH:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  python llama.cpp/convert_hf_to_gguf.py --help"
        )
        sys.exit(1)

    quant_type = QUANTIZATION_TYPES[quantization]
    gguf_name = f"radllama-7b-{quantization}.gguf"
    gguf_path = os.path.join(output_dir, gguf_name)

    print(f"[5/5] Converting HF model -> GGUF ({quant_type})...")
    cmd = [
        sys.executable, convert_script,
        hf_dir,
        "--outfile", gguf_path,
        "--outtype", quant_type,
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nDone. GGUF model: {gguf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RadLLaMA-7B to GGUF")
    parser.add_argument(
        "--output_dir", default="checkpoints/radllama-7b-merged",
        help="Directory to save merged model and GGUF files",
    )
    parser.add_argument(
        "--quantization", choices=list(QUANTIZATION_TYPES.keys()), default="q8_0",
        help="GGUF quantization type (default: q8_0)",
    )
    parser.add_argument(
        "--skip_gguf", action="store_true",
        help="Only export merged HF model, skip GGUF conversion",
    )
    args = parser.parse_args()
    convert(args.output_dir, args.quantization, args.skip_gguf)
