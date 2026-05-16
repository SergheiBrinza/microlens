# MicroLens VQA Dataset — Data Card

**Project:** MicroLens (Kaggle Gemma 4 Good Hackathon 2026)
**Author:** Serghei Brinza · Vienna, Austria
**Release:** Final · 2026-05-17
**License of dataset compilation:** CC-BY 4.0
**Total VQA pairs:** 75,491 (67,121 train + 8,370 validation)
**Total images:** 75,491 PNG @ 384×384 RGB
**Categories:** 2
**Unique genera:** 95

---

## Overview

MicroLens VQA is a biological microscopy image-question-answer dataset compiled from three open-licensed source datasets verified for commercial-use compatibility. It is the training fuel for the MicroLens fine-tune of Gemma 4 E2B and is distributed alongside its image companion on Kaggle.

All upstream sources in this release are independently published under CC0 or CC-BY 4.0 by mature scientific institutions, keeping the entire data chain unambiguously license-clean for commercial use.

---

## Composition

### By category

| Category | Train pairs | Val pairs | Total | Share |
|---|---:|---:|---:|---:|
| diatom | 62,933 | 7,850 | 70,783 | 93.8% |
| fungal_spore | 4,188 | 520 | 4,708 | 6.2% |
| **Total** | **67,121** | **8,370** | **75,491** | **100%** |

### By source

| Source | Pairs | License | DOI / Reference |
|---|---:|---|---|
| UDE Diatoms in the Wild 2024 | 39,389 | **CC0** | Zenodo 10410655 — University of Duisburg-Essen |
| DIATLAS | 23,544 | **CC-BY 4.0** | Zenodo 16260887 — open European diatom imaging |
| TgFC (Tectona grandis Fungal Community) | 4,188 | **CC-BY 4.0** | figshare 28855910 |

### By genus

- **95 unique genera**, identical genus set across train and validation splits
- **Top-30 genera** receive hand-curated, knowledge-base-backed answers describing morphology, habitat, and identification cues — synthesised from AlgaeBase, WoRMS, ITIS, and curated taxonomic atlases
- **Remaining 65 genera** receive shorter, automatically-templated answers in the same structural format

---

## Record schema

Each line of `train_final.jsonl` / `val_final.jsonl` is a single JSON object:

```json
{
  "image": "0051053_ude_diatoms_Planothidium.png",
  "question": "Can you identify this specimen?",
  "answer": "This is a diatom of the genus *Planothidium*, ...",
  "metadata": {
    "category": "diatom",
    "genus": "Planothidium",
    "species": "Planothidium frequentissimum",
    "source": "ude_diatoms"
  }
}
```

The `image` field is a **basename only**. Match it against the companion image dataset (`microlens-images-hackathon`) to retrieve the actual PNG.

### Image preprocessing

All 75,491 images were resize-and-padded to a uniform **384×384 RGB PNG** with white background, preserving aspect ratio (no cropping). Stray UI text and watermark crops were removed manually for the top-30 genera; the long-tail was left as-is. Perceptual hashing plus manual review removed near-duplicates.

---

## License summary

| License | Sources | Pairs | Attribution |
|---|---|---:|---|
| CC0 | UDE Diatoms | 39,389 | Optional |
| CC-BY 4.0 | DIATLAS, TgFC | 27,732 | **Required** |

The dataset compilation as a whole is released under **CC-BY 4.0** (the most-restrictive of the underlying licenses). Attribution: *Serghei Brinza — MicroLens, Vienna 2026*. Underlying sources must also be credited:

- Zenodo 10410655 (UDE Diatoms)
- Zenodo 16260887 (DIATLAS)
- figshare 28855910 (TgFC)
- AlgaeBase, WoRMS, ITIS (for knowledge-base enrichment)

---

## What is **not** included

This release ships with only the two domains whose upstream sources are unambiguously license-clean for commercial use (CC0 + CC-BY 4.0). Future MicroLens modules covering additional domains will be added on the same architecture once each new dataset passes the project's three-stage release filter (market need, ethics / safety, legal compatibility) — see [DATA_GOVERNANCE.md](DATA_GOVERNANCE.md).

---

## Compliance

- ✅ All licenses permit commercial use
- ✅ No non-commercial, share-alike, no-derivatives, or GPL content
- ✅ Attribution requirements satisfied per CC-BY 4.0 Section 3(a)(2) — dataset-level attribution in this card, source-level metadata in every record
- ✅ No human faces, no PII, no medical imaging

---

## Distribution

- **VQA dataset on Kaggle:** https://www.kaggle.com/datasets/sergheibrinza/microlens-vqa-hackathon
- **Image dataset on Kaggle:** https://www.kaggle.com/datasets/sergheibrinza/microlens-images-hackathon
- **Reproducibility notebook:** https://www.kaggle.com/code/sergheibrinza/microlens

---

## Removal requests

Any dataset owner can request removal via an issue or pull request at https://github.com/SergheiBrinza/microlens. The dataset will be rebuilt without the contested content and re-released within 30 days.

---

**Last updated:** 2026-05-17
