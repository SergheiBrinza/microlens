#!/usr/bin/env python3
"""
MicroLens: Fine-tune Gemma 4 E2B через Unsloth QLoRA.
ИСПРАВЛЕНО: форматирование на лету (без pre-processing всех картинок).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

import json
import time
from pathlib import Path

import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from PIL import Image

DATA_DIR = Path("/media/softer/blau1/microlens/training/vqa_data")
TRAIN_FILE = DATA_DIR / "train_gemma_oversampled.jsonl"
VAL_FILE = DATA_DIR / "val_gemma.jsonl"
OUTPUT_DIR = Path("/media/softer/blau1/microlens/training/gemma4_e2b_microlens")
LOG_FILE = OUTPUT_DIR / "training.log"

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LEN = 2048
LORA_RANK = 16
LORA_ALPHA = 32
LR = 2e-4
EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM = 8
SAVE_STEPS = 500
LOGGING_STEPS = 25

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


class LazyVisionDataset(torch.utils.data.Dataset):
    """Ленивая загрузка картинок — форматирование происходит в __getitem__."""
    
    def __init__(self, jsonl_path):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                # Проверяем существование файла
                if Path(r["image_path"]).exists():
                    self.records.append(r)
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        r = self.records[idx]
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except Exception:
            # При ошибке возвращаем первый валидный
            return self.__getitem__(0)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": r["messages"][0]["content"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": r["messages"][1]["content"]}],
            },
        ]
        return {"messages": messages}


def main():
    log("=" * 70)
    log("  MICROLENS — Gemma 4 E2B fine-tune (LAZY LOADING)")
    log("=" * 70)
    log(f"Model:       {MODEL_NAME}")
    log(f"LoRA rank:   {LORA_RANK}")
    log(f"LR:          {LR}")
    log(f"Epochs:      {EPOCHS}")
    log(f"Batch:       {BATCH_SIZE} × grad_accum {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM}")
    log("")

    # ─── Модель ────────────────────────────────────
    log("[1/5] Загружаю модель...")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        max_seq_length=MAX_SEQ_LEN,
    )
    log(f"  ✅ VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ─── LoRA ──────────────────────────────────────
    log("[2/5] LoRA адаптер...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,             # 0 для fast patching
        bias="none",
        random_state=42,
        use_rslora=False,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  ✅ Trainable: {trainable/1e6:.1f}M")

    # ─── Данные (LAZY) ─────────────────────────────
    log("[3/5] Подготовка данных (lazy)...")
    train_ds = LazyVisionDataset(TRAIN_FILE)
    val_ds = LazyVisionDataset(VAL_FILE)
    log(f"  ✅ Train: {len(train_ds)}, Val: {len(val_ds)}")
    log("  (картинки загружаются на лету — нет pre-processing)")

    # ─── Trainer ──────────────────────────────────
    log("[4/5] Настройка trainer...")
    FastVisionModel.for_training(model)

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        bf16=True,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=MAX_SEQ_LEN,
        seed=42,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
    )

    log("[5/5] ЗАПУСК ОБУЧЕНИЯ!")
    log("=" * 70)
    start = time.time()
    train_result = trainer.train()
    duration = time.time() - start
    log(f"\n  ✅ Завершено за {duration/3600:.2f} часов")
    log(f"  Final loss: {train_result.training_loss:.4f}")

    log("Сохраняю LoRA...")
    final_dir = OUTPUT_DIR / "final_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log(f"✅ Сохранено: {final_dir}")


if __name__ == "__main__":
    main()
