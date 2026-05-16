#!/usr/bin/env python3
"""
MicroLens: Quick quality test of fine-tuned Gemma 4 E2B.
Loads base model + trained LoRA, runs inference on 2 samples per category.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
from pathlib import Path
from PIL import Image
import torch
from unsloth import FastVisionModel

BASE = "/media/softer/blau1/microlens"
LORA = f"{BASE}/training/gemma4_e2b_microlens/final_lora"
DATASETS = f"{BASE}/datasets"

CATEGORIES = {
    "01_pollen":       "01_pollen",
    "02_algae":        "02_algae",
    "03_yeast":        "03_yeast",
    "04_minerals":     "04_minerals",
    "05_plantdoc":     "05_plantdoc",
    "07_pcb":          "07_pcb",
    "08_snowflakes":   "08_snowflakes",
    "12_zooplankton":  "12_zooplankton",
    "13_tardigrades":  "13_tardigrades/tardigrade",
}

PROMPT = "Describe what you see in this microscopy image. Identify the subject and key visual features."

def find_images(root, n=2, seed=42):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            paths.append(p)
            if len(paths) > 10000:
                break
    if not paths:
        return []
    rnd = random.Random(seed)
    rnd.shuffle(paths)
    return paths[:n]

def main():
    print("=" * 70)
    print("🔬 MicroLens — Тест обученной Gemma 4 E2B")
    print("=" * 70)

    print(f"\n[1/3] Загружаю LoRA + базовую модель через Unsloth: {LORA}")
    model, tokenizer = FastVisionModel.from_pretrained(
        LORA,
        load_in_4bit=True,
        use_gradient_checkpointing=False,
    )

    print("[2/3] Переключаю в inference mode")
    FastVisionModel.for_inference(model)

    print("[3/3] Собираю тестовые картинки...\n")
    all_samples = []
    for cat_name, cat_path in CATEGORIES.items():
        full = f"{DATASETS}/{cat_path}"
        imgs = find_images(full, n=2)
        for img_path in imgs:
            all_samples.append((cat_name, img_path))
        print(f"  {cat_name}: {len(imgs)} картинок")

    print(f"\nВсего: {len(all_samples)} тестов\n")
    print("=" * 70)

    for i, (cat, img_path) in enumerate(all_samples, 1):
        print(f"\n[{i}/{len(all_samples)}] {cat}")
        print(f"    📁 {img_path.name}")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"    ❌ Не могу открыть: {e}")
            continue

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  PROMPT},
            ],
        }]

        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = tokenizer(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to("cuda")

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                use_cache=True,
            )

        answer = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        print(f"    🤖 {answer}")
        print("-" * 70)

    print("\n✅ Тест завершён!")

if __name__ == "__main__":
    main()
