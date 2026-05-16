---
language:
  - en
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - gemma
  - gemma-4
  - vision-language
  - microscopy
  - scientific-imaging
  - lora
  - qlora
  - unsloth
  - citizen-science
  - education
  - edge-deployment
license: apache-2.0
base_model: unsloth/gemma-4-E2B-it
model-index:
  - name: MicroLens
    results: []
---

# MicroLens · Microscopy Vision-Language Model

A small, fine-tuned multimodal model that turns a **$150 Android phone + a clip-on microscope** into a field-ready assistant for diatom-based water-quality assessment and fungal spore identification. Runs offline.

- **Base model:** [`unsloth/gemma-4-E2B-it`](https://huggingface.co/unsloth/gemma-4-E2B-it) (4.44 B params)
- **Adapter / Merged / GGUF:** [`Laborator/microlens-final`](https://huggingface.co/Laborator/microlens-final)
- **Source code:** [`SergheiBrinza/microlens`](https://github.com/SergheiBrinza/microlens)
- **Submission for:** *The Gemma 4 Good Hackathon*, Kaggle · May 2026
- **License:** Apache 2.0 (weights · code); CC-BY 4.0 (dataset, see component licenses below)

---

## Table of Contents

1. [Model Details](#1-model-details)
2. [Intended Use](#2-intended-use)
3. [Training Data](#3-training-data)
4. [Training Procedure](#4-training-procedure)
5. [Evaluation](#5-evaluation)
6. [Bias, Risks & Limitations](#6-bias-risks--limitations)
7. [Environmental Impact](#7-environmental-impact)
8. [Technical Specifications](#8-technical-specifications)
9. [How to Use](#9-how-to-use)
10. [Citation](#10-citation)
11. [Model Card Authors](#11-model-card-authors)
12. [Model Card Contact](#12-model-card-contact)

---

## 1. Model Details

| Field | Value |
|---|---|
| **Name** | MicroLens |
| **Version** | Final-1.0 · May 2026 |
| **Author** | Serghei Brinza · Vienna, Austria |
| **Model type** | Vision-Language (image + text → text) |
| **Language(s)** | English (primary); multilingual output via Gemma 4 base tokenizer |
| **Base model** | `unsloth/gemma-4-E2B-it` (Gemma 4 Effective-2B, instruction-tuned) |
| **Parameters** | 4.44 B total · 29.9 M trainable during fine-tune (0.58 %) |
| **License** | Apache 2.0 |
| **Finetuning method** | Unsloth FastVisionModel + 4-bit QLoRA (LoRA adapter, r = 16, α = 32, dropout = 0) |
| **Framework** | Unsloth + Transformers + TRL SFTTrainer · PyTorch 2.x · bf16 |
| **Hardware** | 1 × NVIDIA RTX 3090 Ti (24 GB VRAM) |
| **Training time** | ~14 h wall-clock for 2 full epochs over the Final dataset |

### Distribution artefacts

| Artefact | Purpose | Target runtime |
|---|---|---|
| **LoRA adapter** | Load on top of base Gemma 4 E2B | Unsloth / PEFT / Transformers |
| **Merged FP16** | Full stand-alone model | Transformers / vLLM / SGLang |
| **GGUF Q4_K_M** | 4-bit quantised weights | Ollama · llama.cpp · LM Studio |
| **BF16 `mmproj`** | Vision projector for GGUF runtimes | Ollama · llama.cpp |

All artefacts live on the same HF repo: [Laborator/microlens-final](https://huggingface.co/Laborator/microlens-final).

---

## 2. Intended Use

MicroLens is built to **lower the cost of scientific observation** in places where expert knowledge or network access is scarce.

### Primary intended uses

- **Citizen science.** Volunteers contributing to pond-water biodiversity counts and diatom-based water-quality monitoring can capture a smartphone-microscope image and receive a structured natural-language description of the subject and its key visual features.
- **Future of Education.** Offline biology / earth-science classes; the model runs on the same Android tablet the students already use, with no cloud call required.
- **Research support.** Pre-screening for diatom-based water-quality monitoring and fungal spore identification, where the model narrows the candidate set before an expert confirms.
- **Digital equity.** The Q4_K_M GGUF build runs on mid-range Android hardware (~$150 phones with 6 GB RAM) via `llama.cpp` / `MLC`. No API key, no telemetry, no internet.

### Intended users

- Citizen-science volunteers (freshwater monitors, amateur diatomists).
- Teachers and students in biology / earth-science courses, particularly in low-connectivity regions.
- Researchers doing preliminary triage of large microscopy datasets in the diatom and fungal-spore domains.
- Hackathon / jury members evaluating *The Gemma 4 Good Hackathon* submission.

### Out-of-scope uses

- **Medical diagnosis.** MicroLens has **not** been trained on medical imaging (histology, cytology, pathology, radiology). Do not use it to diagnose disease in humans or animals.
- **Legally or biologically authoritative species identification.** The model returns descriptions, not court-defensible or taxonomically rigorous identifications.
- **Materials outside the two trained categories.** Feeding the model an unrelated image — anything that is not a diatom or a fungal spore — produces an answer but the answer is not grounded in its training and should be treated as unreliable.
- **Forensics, compliance, or regulated decision-making.** Do not chain MicroLens into any pipeline where a confident but wrong output can harm a person or violate regulation.

---

## 3. Training Data

### Composition

All samples are microscopy images from **three license-clean source datasets**. Total: **75,491 image-question-answer triples** (67,121 train · 8,370 validation · **95 genera** across **2 categories**).

| Category | Train samples | Typical subjects | Source datasets |
|---|---:|---|---|
| Diatoms | 62,933 (93.8%) | Pennate / centric diatoms · genus-level taxonomy | UDE Diatoms · DIATLAS |
| Fungal spores | 4,188 (6.2%) | Conidia · ascospores · spore morphology | TgFC |

### Source datasets — license-clean for commercial use

| Source | Pairs | License | Reference |
|---|---:|---|---|
| UDE Diatoms in the Wild 2024 | 39,389 | **CC0** | University of Duisburg-Essen · Zenodo 10410655 |
| DIATLAS | 23,544 | **CC-BY 4.0** | Zenodo 16260887 — open European diatom imaging |
| TgFC (Tectona grandis Fungal Community) | 4,188 | **CC-BY 4.0** | figshare 28855910 |

Only upstream sources whose licences unambiguously permit commercial reuse (CC0 or CC-BY 4.0) are included in this release. Candidate sources whose licensing could not be verified to the same standard were excluded.

### Knowledge base

The **top-30 genera** have hand-curated, knowledge-base-backed answers drawn from AlgaeBase, WoRMS, and ITIS — describing morphology, habitat, and identification cues. The **remaining 65 genera** receive shorter, automatically-templated answers in the same structural format.

### Description generation pipeline

Natural-language answers were synthesised from the raw images plus per-source metadata, with the top-30 genera anchored against authoritative taxonomic sources. No samples were generated from non-permissively-licensed upstream data.

### Licensing of training data

All upstream datasets were checked for license compatibility. Accepted licences: **CC0**, **CC-BY 4.0**. **Zero** samples were used from non-commercial, share-alike, GPL, or unverifiable sources. Per-source licensing is published in [`docs/LICENSE_AUDIT.md`](docs/LICENSE_AUDIT.md). The compiled VQA dataset is released under **CC-BY 4.0** (the most-restrictive underlying license).

---

## 4. Training Procedure

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Fine-tuning method | **Unsloth FastVisionModel + 4-bit QLoRA** (NF4 base + LoRA adapter in bf16) |
| LoRA rank (r) | **16** |
| LoRA α | **32** |
| LoRA dropout | **0** |
| Trainable parameters | **29.9 M** (0.58 % of 4.44 B) |
| Target modules | Vision layers + language layers + attention + MLP modules (all four enabled) |
| Optimizer | AdamW (8-bit) |
| Learning rate | **2 × 10⁻⁴** |
| LR schedule | Cosine with **3 % warmup ratio** |
| Weight decay | 0.01 |
| Batch size (per device) | 2 |
| Gradient accumulation | 8 |
| **Effective batch size** | **16** |
| Max sequence length | 2048 tokens |
| Epochs | **2** |
| Total steps | ~8,392 (2 epochs × ~4,196 steps/epoch) |
| Mixed precision | bf16 |
| Seed | 42 |

### Hardware & runtime

- 1 × NVIDIA GeForce **RTX 3090 Ti** (24 GB GDDR6X) — single GPU only (Unsloth does not currently support multi-GPU for Gemma 4 vision)
- AMD Ryzen host · 64 GB system RAM
- Ubuntu 24.04 · CUDA 12.x · PyTorch 2.x
- Wall-clock: **~14 h** for the full 2-epoch run

### Attention backend

Gemma 4's vision encoder uses a **head dimension of 512**, which exceeds the 256-head-dim limit of current **FlashAttention-2** kernels. Fine-tuning and inference therefore use **PyTorch SDPA** (scaled-dot-product attention, memory-efficient path). On RTX 3090 Ti this is the correct default; SDPA is the supported backend for Gemma 4 in Unsloth at the time of training. Unsloth's FastVisionModel adds custom 4-bit QLoRA kernels and `UnslothVisionDataCollator` on top of this backend, which together cut peak VRAM from ~38 GB (vanilla HF Transformers) to ~12 GB and roughly halve the per-step time — Unsloth's own published numbers for Gemma 4 are **1.5× faster training, ~60% less VRAM vs FA2** baselines.

### Training entry point

Production training script: [`scripts/train.py`](scripts/train.py). It reads `train_final.jsonl` and `val_final.jsonl` from `training/vqa_data/` and writes checkpoints, the LoRA adapter, and the merged FP16 weights into the run-output directory.

---

## 5. Evaluation

Evaluation is **qualitative and per-category**, reflecting the spirit of the submission (an *assistive descriptor*, not a classifier). For both trained categories we sample images from the 8,370-image held-out validation split and compare the MicroLens answer against the expected genus and identification cues.

| Category | Observation |
|---|---|
| Diatoms | Pennate vs. centric distinction is reliable; genus-level naming on common Naviculales / Cymbellaceae / Aulacoseiraceae is consistent. Long-tail diatoms degrade gracefully into morphological description (raphe / striae / valve outline). |
| Fungal spores | Conidial vs. ascospore separation is reliable; common spore morphologies (Neopestalotiopsis, Colletotrichum, Olivea) receive genus-level naming. |

The output is free-form natural language, so there is no single accuracy number. The correct axis of evaluation is *"does the description help a human in the field decide what to do next?"* — and for the two trained categories, on the validation split, it does.

---

## 6. Bias, Risks & Limitations

### Known failure modes

- **Narrow domain.** MicroLens Final is trained on **only two categories** — diatoms and fungal spores. Anything outside these two domains is **out of distribution**. The model will still produce text on such inputs; that text is not grounded in training and must be treated as unreliable.
- **Small-model ceiling.** MicroLens is built on Gemma 4 **E2B**. Edge cases show degraded reasoning compared to larger Gemma 4 variants.
- **Long-tail genera.** 65 of the 95 covered genera have automatically-templated answers rather than hand-curated knowledge-base entries; expect category-generic morphology rather than genus-specific cues on these.
- **English-first.** Scientific terminology is maximally accurate in English; translated output via the Gemma 4 base tokenizer can simplify or partially drop domain terms.

### Risks

- **Over-trust by non-experts.** A fluent natural-language description can feel more authoritative than it is. Treat MicroLens as a first-pass field note, not as an oracle.
- **Distribution shift.** The training data is dominated by lab-quality and curated-quality images. Field images through cheap clip-on phone microscopes have more motion blur, chromatic aberration, and inconsistent illumination; descriptions on those inputs remain helpful but are more generic.

### Ethical considerations

- **Dataset provenance is audited.** Only CC0 and CC-BY 4.0 upstream data was used. **Zero** non-commercial or unverifiable images were included.
- **No faces, no PII.** The training pool contains microscopy subjects only: no human faces, no personally identifiable information, no private medical imaging.

### Recommended usage pattern

1. Capture image → 2. MicroLens describes it → 3. Human confirms or rejects → 4. Log both.

The model produces a first draft. Final decisions stay with the user.

---

## 7. Environmental Impact

MicroLens Final was trained on a **single workstation GPU for ~14 hours** (2 epochs over the full 67,121-pair train split).

| Factor | Value |
|---|---|
| GPU | RTX 3090 Ti · ~400 W under sustained fine-tune load |
| CPU + chassis + cooling overhead | ~140 W |
| Wall-time | ~14 h |
| **Estimated energy** | **~4 kWh** |

At the Austrian 2024 grid carbon intensity (~110 g CO₂ / kWh), the training run emits **~0.4 kg CO₂-equivalent**.

Inference cost is negligible: the Q4_K_M GGUF build runs on a mid-range Android phone at a few watts. MicroLens is designed so that the cumulative lifetime inference energy per query can be orders of magnitude smaller than a single cloud-inference call to a frontier model.

---

## 8. Technical Specifications

### Architecture

- **Backbone:** Gemma 4 (E2B), sparse-attention transformer decoder with an integrated vision encoder stack.
- **Vision encoder:** Gemma 4 native vision tower (head dim 512).
- **Fusion:** multimodal projector that lifts vision tokens into the language model embedding space (`mmproj` ships separately for GGUF runtimes).
- **Positional encoding:** inherited from Gemma 4 base.
- **Attention backend:** SDPA (scaled dot-product attention) during both fine-tune and inference. FlashAttention-2 is **not** usable: Gemma 4's vision-tower head dim (512) exceeds the FA-2 kernel limit (256).

### Adapter layout

- **LoRA rank:** 16
- **LoRA α:** 32
- **LoRA dropout:** 0
- **Target modules:** vision layers + language layers + attention modules + MLP modules — all four toggled on (`finetune_vision_layers`, `finetune_language_layers`, `finetune_attention_modules`, `finetune_mlp_modules`).
- **Trainable params:** 29.9 M (0.58 % of 4.44 B base).

### Quantisations shipped

- **Merged FP16:** full-precision full-model snapshot, Transformers-native.
- **GGUF Q4_K_M:** 4-bit quantised weights via `llama.cpp` convert pipeline. Pairs with the BF16 `mmproj` for full multimodal inference.
- **LoRA-only (bf16):** for users who want to re-merge against a different Gemma 4 E2B base or stack additional adapters.

### Software dependencies at training time

- `unsloth` (FastVisionModel + UnslothVisionDataCollator)
- `transformers`
- `peft`, `bitsandbytes`, `trl` (`SFTTrainer` / `SFTConfig`)
- `torch` (CUDA 12.x)

Exact pins are tracked in [`requirements.txt`](requirements.txt).

---

## 9. How to Use

### Transformers (merged FP16)

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

model_id = "Laborator/microlens-final"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

image = Image.open("my_microscopy_image.jpg").convert("RGB")
prompt = "Describe what you see in this microscopy image. Identify the subject and key visual features."

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text",  "text":  prompt},
]}]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=220, temperature=0.3, do_sample=True)
print(processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Unsloth (LoRA on top of base)

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "Laborator/microlens-final",
    load_in_4bit=True,
    use_gradient_checkpointing=False,
)
FastVisionModel.for_inference(model)
# … same prompting pattern as above.
```

### Ollama / llama.cpp (Q4_K_M)

```bash
# download microlens-final-Q4_K_M.gguf and mmproj-bf16.gguf into ./gguf/
ollama create microlens -f Modelfile        # see repo for Modelfile
ollama run microlens "Describe this sample." --image slide_01.jpg
```

---

## 10. Citation

If you use MicroLens in a publication, project, or downstream model, please cite:

```bibtex
@software{brinza_microlens_2026,
  title        = {MicroLens: a microscopy vision-language model fine-tuned from Gemma 4 E2B},
  author       = {Brinza, Serghei},
  year         = {2026},
  month        = may,
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/Laborator/microlens-final},
  note         = {Submission to the Gemma 4 Good Hackathon (Kaggle, May 2026)}
}
```

Upstream works used by MicroLens:

```bibtex
@misc{gemma4_2026,
  title        = {Gemma 4 Technical Report},
  author       = {Google DeepMind},
  year         = {2026},
  note         = {Base model: unsloth/gemma-4-E2B-it}
}

@misc{unsloth_2026,
  title        = {Unsloth: faster LLM fine-tuning},
  author       = {Daniel Han and Michael Han and Unsloth team},
  year         = {2026},
  url          = {https://github.com/unslothai/unsloth}
}
```

Underlying training datasets:

- **UDE Diatoms in the Wild 2024** — University of Duisburg-Essen, Zenodo 10410655 (CC0)
- **DIATLAS** — Zenodo 16260887 (CC-BY 4.0)
- **TgFC (Tectona grandis Fungal Community)** — figshare 28855910 (CC-BY 4.0)

---

## 11. Model Card Authors

- **Serghei Brinza** · Vienna, Austria · sole author of the model, the training pipeline, and this card.

---

## 12. Model Card Contact

- **Hugging Face:** [`Laborator/microlens-final`](https://huggingface.co/Laborator/microlens-final). Open an issue / discussion on the repo.
- **GitHub:** [`SergheiBrinza/microlens`](https://github.com/SergheiBrinza/microlens). Issues, pull requests, dataset corrections welcome.

---

*MicroLens · built for the Gemma 4 Good Hackathon.*
