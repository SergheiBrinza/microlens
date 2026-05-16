#!/usr/bin/env python3
"""
MicroLens Final production training.

Two epochs over the Final dataset (67,121 train + 8,370 val) on a
single RTX 3090 Ti (24 GB) with Unsloth FastVisionModel and 4-bit QLoRA.

Sources (all license-clean for commercial use):
  - UDE Diatoms in the Wild 2024 (Zenodo 10410655)            — CC0       — 39,389 pairs
  - DIATLAS (Zenodo 16260887)                                  — CC-BY 4.0 — 23,544 pairs
  - TgFC Tectona grandis Fungal Community (figshare 28855910)  — CC-BY 4.0 — 4,188 pairs

Reference environment:
  - unsloth         (FastVisionModel + UnslothVisionDataCollator)
  - transformers
  - trl             (SFTTrainer / SFTConfig)
  - peft, bitsandbytes
  - torch (CUDA 12.x), bf16

Paths default to the in-repo layout (training/vqa_data/{train,val}_final.jsonl
and outputs/microlens_final_run/). Override with MICROLENS_TRAIN /
MICROLENS_VAL / MICROLENS_OUT environment variables for ad-hoc layouts.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

import json, time, traceback
from pathlib import Path
import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = Path(os.environ.get("MICROLENS_TRAIN", REPO_ROOT / "training" / "vqa_data" / "train_final.jsonl"))
VAL_FILE   = Path(os.environ.get("MICROLENS_VAL",   REPO_ROOT / "training" / "vqa_data" / "val_final.jsonl"))
OUTPUT_DIR = Path(os.environ.get("MICROLENS_OUT",   REPO_ROOT / "outputs" / "microlens_final_run"))
LOG_FILE   = OUTPUT_DIR / "training.log"

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LEN = 2048
LORA_RANK = 16
LORA_ALPHA = 32
LR = 2e-4
EPOCHS = 2
BATCH_SIZE = 2
GRAD_ACCUM = 8
SAVE_STEPS = 500
LOGGING_STEPS = 25
EVAL_STEPS = 500

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

class LazyVisionDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, limit=None):
        self.records = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                r = json.loads(line)
                if Path(r["image"]).exists():
                    self.records.append(r)
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        try: img = Image.open(r["image"]).convert("RGB")
        except: return self.__getitem__(0)
        return {"messages": [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": r["question"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]},
        ]}

def main():
    log("=" * 70)
    log("  MICROLENS FINAL — CLEAN TRAINING (2 epochs)")
    log("=" * 70)
    import unsloth, transformers, peft, trl
    log(f"unsloth: {unsloth.__version__}")
    log(f"transformers: {transformers.__version__}")
    log(f"peft: {peft.__version__}")
    log(f"trl: {trl.__version__}")
    log(f"Model:  {MODEL_NAME}")
    log(f"LoRA:   r={LORA_RANK}/α={LORA_ALPHA}/dropout=0")
    log(f"LR:     {LR}, epochs={EPOCHS}")
    log(f"Train:  {TRAIN_FILE}")
    log(f"Val:    {VAL_FILE}")
    log("")

    log("[1/5] Загружаю модель...")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME, load_in_4bit=True,
        use_gradient_checkpointing="unsloth", max_seq_length=MAX_SEQ_LEN,
    )
    log(f"  ✅ VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    log("[2/5] LoRA...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0,
        bias="none", random_state=42, use_rslora=False,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  ✅ Trainable: {trainable/1e6:.1f}M")

    log("[3/5] Данные...")
    train_ds = LazyVisionDataset(TRAIN_FILE)
    val_ds = LazyVisionDataset(VAL_FILE)
    log(f"  ✅ Train: {len(train_ds)}, Val: {len(val_ds)}")

    log("[4/5] Trainer...")
    FastVisionModel.for_training(model)
    sft = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS, learning_rate=LR,
        warmup_ratio=0.03, lr_scheduler_type="cosine",
        optim="adamw_8bit", weight_decay=0.01,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps", save_steps=SAVE_STEPS, save_total_limit=3,
        eval_strategy="steps", eval_steps=EVAL_STEPS,
        bf16=True, fp16=False, report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=MAX_SEQ_LEN, seed=42,
        dataloader_num_workers=2,
    )
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_ds, eval_dataset=val_ds, args=sft,
    )

    log("[5/5] FULL TRAINING START!")
    log("=" * 70)
    start = time.time()
    try:
        result = trainer.train()
        log(f"\n  ✅ Завершено за {(time.time()-start)/3600:.2f} часов")
        log(f"  Final train_loss: {result.training_loss:.4f}")
    except Exception as e:
        log(f"  ⚠ {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    log("Сохраняю LoRA...")
    final_dir = OUTPUT_DIR / "final_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log(f"  ✅ {final_dir}")

    log("Сохраняю merged fp16...")
    merged_dir = OUTPUT_DIR / "merged_fp16"
    try:
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        log(f"  ✅ {merged_dir}")
    except Exception as e:
        log(f"  ⚠ save merged failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
