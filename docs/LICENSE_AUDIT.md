# License Audit — MicroLens (Final)

**Cartridge:** MicroLens (Biology — diatoms + fungal spores)
**Tier:** 1 — Open educational
**Audit date:** 2026-05-17
**Auditor:** Serghei Brinza
**Status:** Final — all sources verified license-clean for commercial use

## Source datasets

| # | Dataset | License | Pairs | Status |
|---|---------|---------|------:|--------|
| 1 | UDE Diatoms in the Wild 2024 (Zenodo 10410655) | **CC0** | 39,389 | ✅ Verified |
| 2 | DIATLAS (Zenodo 16260887) | **CC-BY 4.0** | 23,544 | ✅ Verified |
| 3 | TgFC — Tectona grandis Fungal Community (figshare 28855910) | **CC-BY 4.0** | 4,188 | ✅ Verified |

**Total:** 75,491 VQA pairs across 95 genera and 2 categories (diatom + fungal_spore).

## License compatibility policy

Per the governance framework in [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md), only the following licenses are accepted for training data:

- ✅ **Apache 2.0**, **MIT**, **CC0** — fully compatible
- ✅ **CC-BY 4.0** — compatible with attribution preserved
- ❌ Non-commercial, no-derivatives, share-alike, or GPL — excluded
- ❌ **Unknown / unclear** — excluded pending verification

## Excluded sources

Several upstream sources considered for prior MicroLens iterations were excluded from this release because their licensing could not be verified to a commercial-use standard at submission time, or because their licence terms (e.g. non-commercial-only) were incompatible with the hackathon prize structure.

This approach guarantees that every shipped pair is unambiguously license-clean for commercial downstream use. Additional domains will be added on the same architecture once each new dataset passes the project's three-stage release filter (market need, ethics / safety, legal compatibility).

## Attribution

Attribution is satisfied per CC-BY 4.0 Section 3(a)(2) by:

1. **Source field in every VQA record** — every JSONL line carries `metadata.source` identifying the upstream dataset (`ude_diatoms`, `diatlas`, or `tgfc`)
2. **Dataset-level attribution** in [`DATA_CARD.md`](DATA_CARD.md), this audit, and the Kaggle dataset descriptions
3. **Knowledge-base citation** for the hand-curated top-30 genus answers: AlgaeBase, WoRMS, ITIS

This follows the practice of ImageNet, COCO, LAION, and similar large open ML dataset compilations.

## Removal requests

Dataset owners can request removal via pull request or issue at https://github.com/SergheiBrinza/microlens. The cartridge will be rebuilt without the contested data and re-released within 30 days.

## Reproducibility

The Kaggle reproducibility notebook (https://www.kaggle.com/code/sergheibrinza/microlens) demonstrates the full preprocessing and training pipeline against the published Kaggle datasets, so any third party can re-run the audit against identical inputs.

---

**Author:** Serghei Brinza · Vienna, Austria
**License:** CC-BY 4.0 (this document)
