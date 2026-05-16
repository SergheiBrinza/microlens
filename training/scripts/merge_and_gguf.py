#!/usr/bin/env python3
"""
MicroLens: Merge LoRA + Export GGUF
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
from unsloth import FastVisionModel

BASE_DIR = Path("/media/softer/blau1/microlens/training/gemma4_e2b_microlens")
LORA_DIR = BASE_DIR / "final_lora"
MERGED_DIR = BASE_DIR / "merged_fp16"
GGUF_DIR = BASE_DIR / "gguf_q4_k_m"

print("=" * 70)
print("🔬 MicroLens — Merge LoRA + Export GGUF")
print("=" * 70)

print(f"\n[1/3] Загружаю базовую модель + LoRA из:\n      {LORA_DIR}")
model, tokenizer = FastVisionModel.from_pretrained(
    str(LORA_DIR),
    load_in_4bit=False,
    use_gradient_checkpointing=False,
)

print(f"\n[2/3] Сохраняю merged FP16 модель в:\n      {MERGED_DIR}")
MERGED_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained_merged(
    str(MERGED_DIR),
    tokenizer,
    save_method="merged_16bit",
)
print("      ✅ Merged FP16 готов")

print(f"\n[3/3] Экспортирую GGUF Q4_K_M в:\n      {GGUF_DIR}")
GGUF_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained_gguf(
    str(GGUF_DIR),
    tokenizer,
    quantization_method="q4_k_m",
)
print("      ✅ GGUF готов")

print("\n" + "=" * 70)
print("✅ ВСЁ ГОТОВО!")
print(f"   Merged FP16: {MERGED_DIR}")
print(f"   GGUF Q4_K_M: {GGUF_DIR}")
print("=" * 70)
