# MicroLens Dataset Final

**Release date:** 2026-05-17
**Total VQA pairs:** 75,491 (67,121 train + 8,370 val)
**Total images:** 75,491 PNGs at 384×384 (companion image dataset)
**Categories:** 2 (diatom, fungal_spore)
**Unique genera:** 95
**License:** CC-BY 4.0 (most restrictive of underlying source licenses)
**Status:** Frozen for Kaggle Gemma 4 Good Hackathon submission

## Composition

| Category | Pairs (train) | Genera | Source datasets |
|---|---:|---:|---|
| diatom | 62,933 | ~85 | UDE Diatoms (CC0) · DIATLAS (CC-BY 4.0) |
| fungal_spore | 4,188 | ~10 | TgFC (CC-BY 4.0) |

## Source datasets

- **UDE Diatoms in the Wild 2024** — Zenodo 10410655, CC0 (39,389 pairs)
- **DIATLAS** — Zenodo 16260887, CC-BY 4.0 (23,544 pairs)
- **TgFC (Tectona grandis Fungal Community)** — figshare 28855910, CC-BY 4.0 (4,188 pairs)

## Why this slice

All upstream sources in the Final release are independently published under CC0 or CC-BY 4.0 by mature scientific institutions, keeping the entire dataset chain unambiguously license-clean for commercial use. Future modules on additional microscopy domains will be added on the same architecture once each new dataset passes the project's three-stage release filter (market need, ethics / safety, legal compatibility).

## Knowledge base

The top-30 genera have hand-curated answers drawn from AlgaeBase, WoRMS, and ITIS — describing morphology, habitat, and identification cues. The remaining 65 genera have shorter, automatically-templated answers.

## Schema

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

The `image` field is a **basename only**, matching the companion image dataset.

## Distribution

Both VQA pairs and images are published on Kaggle:

- VQA: https://www.kaggle.com/datasets/sergheibrinza/microlens-vqa-hackathon
- Images: https://www.kaggle.com/datasets/sergheibrinza/microlens-images-hackathon

## Compliance

- All licenses permit commercial use
- No non-commercial, share-alike, no-derivatives, or GPL content
- Attribution preserved per CC-BY 4.0 requirements
