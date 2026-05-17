# MicroLens

Turning a smartphone into a pocket microscope with AI vision. Gemma 4 E2B fine-tuned for offline microscopy. Pair a $20 clip-on macro lens with a phone you already own, identify diatoms and fungal spores, no cloud, no subscription.

> Submission to The Gemma 4 Good Hackathon, Kaggle, May 2026.
> [Live demo](https://huggingface.co/spaces/Laborator/microlens) · [Model](https://huggingface.co/Laborator/microlens-final) · [Code](https://github.com/SergheiBrinza/microlens)

---

## The missing expert

In an Appalachian school district, a biology teacher holds up a vial of pond water. No cell signal, no lab budget, no microscopist on staff. Twenty kids crowd a single phone with a clip-on macro lens.

In a Kenyan smallholder farm, a forester pulls a sample of decaying *Tectona grandis* leaf-litter and looks at the spores under a $20 lens. The nearest mycologist is hundreds of kilometres away. The crop rotation depends on the answer, this week.

Until 2026 this was mathematically impossible. Big models didn't fit in a pocket; small ones couldn't reason about microscopy. Gemma 4 E2B closed that gap. MicroLens is one of the first applications to use it.

## A microscope you already own

Buy a $20 clip-on macro lens (200×–400×). Install MicroLens. Point your phone at any slide.

The model identifies the subject across **two trained categories** (diatoms and fungal spores) spanning **95 genera**, and writes a structured description of the features that justify the call: morphology, habitat, identification cues.

Fully offline. No API keys, no cloud, no subscription. Hardware floor under **$120** (sub-$100 phone + $20 lens).

Gemma 4 E2B fine-tuned via Unsloth FastVisionModel + 4-bit QLoRA on **75,491 VQA pairs** (67,121 train + 8,370 val, 95 genera, 2 categories). Training data is **license-clean for commercial use**: UDE Diatoms (CC0) + DIATLAS (CC-BY 4.0) + TgFC (CC-BY 4.0). Three formats on Hugging Face: LoRA adapter, merged FP16, Q4_K_M GGUF (Android via Ollama / llama.cpp).

## Why Gemma 4

The hackathon brief requires Gemma 4. No other open model could do this job in 2026.

| Model class | 4-bit size | On a sub-$100 phone? | Smart enough? |
|---|---|---|---|
| Frontier cloud leaders | cloud only | ❌ | ✅ |
| Open 7-8B family | ~5 GB | Marginal | ⚠️ Generic |
| **Gemma 4 E2B** | **2.5-3 GB** | **✅** | **✅** |

Per-Layer Embeddings makes the pocket-microscope concept work: **4.44 B parameters total, ~2 B effective via PLE** — a budget Android holds the model in RAM.

What I use: vision understanding (encoder + LM fine-tuned on 75 K pairs), multilingual output (35+ languages multimodal, 140+ pretrained), 128 K context, Apache 2.0. On-device inference satisfies HIPAA, GDPR, and equivalents by construction.

Gemma 4 scales from a phone to a workstation — four sizes (E2B, E4B, 26B MoE, 31B), which makes the MicroLens roadmap realistic.

## Why diatoms and fungal spores

These two domains are the hardest possible proof-of-concept for AI microscopy and at the same time the most globally consequential.

**Diatoms** are the reference biological indicator of freshwater quality worldwide — WHO Drinking-Water Guidelines, FAO protocols, US EPA, EU Water Framework Directive, ASEAN / South American / African monitoring programs. Genus-level distinctions sit at sub-micron features of the silica frustule. Professional labs train specialists for years to tell them apart; MicroLens does it on a phone.

**Fungal spores** are the leading cause of crop loss in the Global South. *Colletotrichum*, *Neopestalotiopsis* and relatives strip billions from tropical agroforestry yearly. FAO estimates fungal pathogens destroy 10–23 % of the world's food supply. Spores are 2–40 microns, three-dimensional, deceptively similar across genera.

If the pipeline holds on these two, it holds on any simpler domain.

Diatoms + fungal spores are the first MicroLens modules to clear our three-filter gate — market demand, ethics and safety, legal licensing (FDA, EU AI Act, GDPR, HIPAA-equivalent) — for public release. Three CC0/CC-BY 4.0 datasets from mature scientific institutions, 95 genera deep, top-30 anchored against AlgaeBase / WoRMS / ITIS.

## Technical execution

Built solo, end-to-end. One person, one consumer GPU.

**Pipeline.** Three license-clean sources — UDE Diatoms in the Wild 2024 (Zenodo 10410655, CC0), DIATLAS (Zenodo 16260887, CC-BY 4.0), TgFC fungal spores (figshare 28855910, CC-BY 4.0) — yielding 75,491 image-question-answer triples across 2 categories and 95 genera (67,121 train + 8,370 val). Top-30 genera anchored against AlgaeBase, WoRMS, ITIS.

**Training.** Unsloth FastVisionModel + 4-bit QLoRA on 1× RTX 3090 Ti (24 GB VRAM). LoRA r = 16, α = 32, dropout = 0; all four module groups enabled (vision + language + attention + MLP); 29.9 M trainable params (0.58 % of 4.44 B base). Two full epochs at lr 2 × 10⁻⁴ (cosine, 3 % warmup), effective batch 16, bf16, AdamW 8-bit. Wall-clock ~14 h. Script: `scripts/train.py`; reproducible notebook: `microlens` on Kaggle.

## Results

Output format: **category → genus → morphology → habitat → identification cues**. Base Gemma 4 vs MicroLens on the same input:

**Diatom, *Aulacoseira granulata*:**

- **Base Gemma 4 E2B:** *"elongated cylindrical structure, possibly a fungal hypha or a chain of cells. Dark, segmented appearance with internal divisions."*
- **MicroLens:** *"Aulacoseira granulata, chain-forming centric diatom. Cylindrical valves linked by spinules; coarse areolae in spiral rows on the mantle, characteristic of Aulacoseira. Common in eutrophic freshwater plankton."*

The base model produces generic phrasing; MicroLens names the subject and describes the diagnostic features. Across both trained categories: consistent on common diatom genera (Naviculales, Cymbellaceae, Aulacoseiraceae) and fungal-spore morphologies (Neopestalotiopsis, Colletotrichum, Olivea); graceful degradation on rare subclasses. Edge deployment: Q4_K_M GGUF runs via llama.cpp on a mid-range Android profile at usable single-thread throughput.

## Universality

The same recipe scales to any domain where a trained eye reads small visual features under magnification.

| Domain | Concrete task |
|---|---|
| Archaeology | microremains identification in soil cores |
| Agronomy | leaf-lesion typing (fungal vs bacterial vs viral) |
| Geology | thin-section optical properties under polarized light |
| Restoration | pigment-binder identification on canvas threads |
| Beekeeping | Varroa mite detection on a brood frame |
| Veterinary | livestock parasitology |
| Forensics | fibre, particle, trace microscopy |

One curated, license-clean dataset + ~14 h on a single 3090 Ti gives a new vertical on the same architecture. `ROADMAP.md` lists 22 specialised editions across all four Gemma 4 sizes — each a (dataset × Gemma 4 fine-tune) pair gated on the three-review filter before publication. Open stack: Apache 2.0 code, CC-BY 4.0 data, Apache 2.0 Gemma 4 base.

## Hackathon tracks addressed

MicroLens directly answers four of the five Gemma 4 Good Hackathon tracks.

**Digital Equity** — $120 hardware floor ($100 phone + $20 lens), fully offline, Apache 2.0 + CC-BY 4.0. Deployable in any school, NGO, or citizen-science group, with or without internet.

**Health and Sciences** — diatoms drive freshwater-quality assessment under WHO Drinking-Water Guidelines, EU WFD, US EPA, and FAO protocols. MicroLens lets a non-expert with a phone do the first-pass identification that today requires a trained taxonomist.

**Global Resilience** — diatom communities respond to nutrient pollution and warming water (an early climate signal); the fungal-spore half addresses tropical-agroforestry pathogens that already cause 10–23 % of global food loss, giving smallholders an early warning before yield collapses.

**Future of Education** — every output is a structured mini-lesson (genus → morphology → habitat → identification cues). A biology teacher with no microscopist on staff gets an explainable teaching aid suitable for high-school biology and citizen-science training.

MicroLens does not claim fit with the Safety track — AI safety is outside the scope of this submission.

## Limitations

- **Out-of-distribution categories.** Trained on diatoms and fungal spores only. Anything outside these two domains is out-of-distribution — the model still emits text, but treat it as ungrounded.
- **Long-tail genera.** 65 of 95 genera have automatically-templated answers; expect category-generic morphology there, not genus-specific cues.
- **Confusable pairs.** Saddle-shaped pennate diatoms with similar valve outlines (Cocconeis vs Achnanthes) can be confused.
- **Small-model ceiling.** Gemma 4 E2B is ~2 B-effective via PLE; edge cases show degraded reasoning vs larger Gemma 4 variants.

MicroLens is a research artefact, not a regulated medical / diagnostic / forensic product. Apache 2.0. On-device by design. Specialised editions in regulated domains require jurisdiction-specific approval before public release.

## Special Technology Prizes

**Unsloth.** Fine-tuned end-to-end with Unsloth FastVisionModel + 4-bit QLoRA — a working demonstration of Unsloth's published Gemma-4 numbers (1.5× faster training, ~60 % less VRAM vs FA2) under real production load. Single-GPU on consumer hardware. The reproducible Kaggle notebook re-runs the pipeline end-to-end from a fresh kernel.

**Ollama.** Ships with a working Modelfile pinning the prompt template and embedding the safety disclaimer in the SYSTEM directive — part of the model contract, not a UI label. Built for fieldworkers without connectivity, where on-device inference is the only option.

## What's next and links

**Function calling + edge deployment.** `tools/lookup_genus.py` is a Gemma-4-native function-calling stub that verifies identifications against AlgaeBase at inference time. The Q4_K_M GGUF runs airplane-mode-capable on Android; Google AI Edge (LiteRT, MediaPipe LLM Inference) deployment is on the roadmap.

**Roadmap.** `ROADMAP.md` lists 22 specialised editions across medical-adjacent, earth sciences, agriculture, and industrial domains. Each — a (dataset × Gemma 4 fine-tune) pair — must clear the three-review filter before publication.

**Links.**

- Live demo: https://huggingface.co/spaces/Laborator/microlens
- Model: https://huggingface.co/Laborator/microlens-final
- Code: https://github.com/SergheiBrinza/microlens
- Kaggle notebook: https://www.kaggle.com/code/sergheibrinza/microlens
- Kaggle dataset (VQA): https://www.kaggle.com/datasets/sergheibrinza/microlens-vqa-hackathon
- Kaggle dataset (Images): https://www.kaggle.com/datasets/sergheibrinza/microlens-images-hackathon
- APK release: https://github.com/SergheiBrinza/microlens/releases/latest
- Legal: TERMS, PRIVACY, AI_ACT.

**Author.** Serghei Brinza, Vienna, Austria. GitHub @SergheiBrinza. HuggingFace @Laborator.

MicroLens. Built for The Gemma 4 Good Hackathon, so the price of a microscope is never a barrier to a real answer.
