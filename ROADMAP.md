# MicroLens Roadmap

This document covers the vision behind MicroLens: how the recipe scales beyond amateur microscopy into professional field instruments, why Gemma 4 fits the moment, and the discipline that gates every public release.

> **Status reminder.** MicroLens Final is the current published edition: HuggingFace Hub, HuggingFace Space demo, Android APK on GitHub Releases, Linux desktop app, plus a local `Modelfile` for Ollama. Trained on a single RTX 3090 Ti with Unsloth FastVisionModel: 2 full epochs over 67,121 train pairs, ~14 h wall-clock. The specialized editions described below are research roadmap, not public hub material until the compliance pipeline is signed off end to end.

---

## MicroLens Final in one paragraph

MicroLens Final is a focused build covering **two microscopy categories with verified commercial-clean licensing**: diatoms and fungal spores. Within those two categories the model spans **95 genera** with structured rich-format answers (genus + morphology + habitat + identification cues). It runs offline on a budget Android phone with a $20 clip-on macro lens, hardware floor $120 ($100 phone + $20 lens). Training data: 75,491 VQA pairs split 67,121 train and 8,370 val, drawn from three license-clean source datasets (UDE Diatoms · DIATLAS · TgFC; CC0 + CC-BY 4.0 only). Base: Gemma 4 E2B (4.44 B params, ~2 B effective via PLE), fine-tuned with Unsloth FastVisionModel and 4-bit QLoRA (rank 16, alpha 32, dropout 0; 29.9 M trainable, 0.58 % of base). Final artefacts: LoRA, merged FP16, GGUF Q4_K_M + mmproj, Apache 2.0.

> Additional MicroLens modules — each a (dataset × Gemma 4 fine-tune) pair on the same architecture — are on the roadmap below. Each new module ships only after passing the project's three-stage release filter (market need, ethics / safety, legal compatibility); see [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md).

The point of MicroLens Final is to prove the recipe under tight licensing discipline. The rest of this document describes how the same recipe scales upward.

---

## From a $120 toy to professional instruments

A $20 clip-on lens plus a $100 phone hides a serious compute path. A 2.3B-effective Gemma 4 model, fine-tuned on the right curated dataset, can match or exceed cloud-only generalist models on narrow professional tasks. On edge hardware, specialization is the route to expert-level performance.

The economics are straightforward: local inference, no per-call cost, no subscription, a one-time download of a 2 to 3 GB profile. The remaining engineering question is which professional instruments a 2.3B-effective model in everyone's pocket actually puts within reach. The answer is most of them.

---

## Specialized editions: research roadmap

> **Release discipline applies to every edition listed below.** A specialized edition reaches the public hub only after legal review, GDPR / EU AI Act / domain-regulation compliance, an independent security and safety audit, and (where applicable) external domain-expert validation. None of the editions below are downloadable today.

Each edition is a Gemma 4 profile (LoRA adapter or merged Q4_K_M GGUF, 2 to 3 GB) installed locally and run offline.

### Healthcare-adjacent (research only, no medical claims until certified)

| Edition | Use case | Open / licensed datasets being evaluated |
|---|---|---|
| 🩸 **Hematology screen** | Pre-screening for malaria, anemia, sickle-cell shapes | NIH Malaria Cell Dataset, LISC, public CBC atlases |
| 🦠 **Histopathology assist** | Pre-referral image triage in remote clinics | TCGA (CC-BY), open pathology atlases |
| 🦷 **Veterinary parasitology** | Field vets, livestock parasite ID | Open animal histology / parasitology atlases |
| 🐟 **Fish health** | Aquaculture parasite and lesion detection | Open fish-parasite datasets, FAO imagery |
| 🍯 **Apicultural pathology** | Beekeepers (varroa, nosema, foulbrood) | Open bee-disease imagery, USDA atlases |

### Earth and material sciences

| Edition | Use case | Open / licensed datasets being evaluated |
|---|---|---|
| 🪨 **Field geology** | Lithology hints in low-signal regions | USGS, RRUFF, public petrographic atlases |
| 💎 **Gemmology** | Inclusion / origin pre-assessment | GIA, IGI public reference imagery |
| ☄️ **Meteoritics** | Field identification of suspected meteorite finds | Meteoritical Society public reference, Smithsonian |
| 🏗️ **Metallurgy and materials QA** | Microstructure inspection in small foundries | Open crystallography, NIST materials databases |
| 🧪 **Mineralogy** | Crystal habit, twinning, optical-property hints | RRUFF, MinDat, public mineral atlases |
| 🌋 **Volcanology field aid** | Volcanic glass, ash particle classification | Smithsonian Volcanic Glass Database, USGS imagery |

### Agriculture, ecology, environment

| Edition | Use case | Open / licensed datasets being evaluated |
|---|---|---|
| 🌾 **Crop pathology** | Smallholder farmer disease ID | PlantVillage, PlantDoc, regional crop databases |
| 🐛 **Pest entomology** | Agricultural pest identification | Open insect imagery, regional pest collections |
| 🌳 **Wood / timber ID** | Anti-illegal-logging, customs aid | Open xylotomy databases, Inside Wood (NCSU) |
| 🌊 **Water quality** | Microplastic and pollutant indicator screening | Open water-quality datasets, NOAA imagery |
| 🍄 **Mycology** | Spore-level fungus identification | Open mycology atlases (university herbaria) |
| 🦋 **Pollinator monitoring** | Pollinator-decline biodiversity surveys | Public pollinator collections, iNaturalist (CC) |

### Industrial and specialized applications

| Edition | Use case | Open / licensed datasets being evaluated |
|---|---|---|
| 🔌 **PCB inspection** | Hobbyist and small-shop electronics QA | Open PCB defect datasets |
| 🧵 **Textile fiber ID** | Vintage textile authentication, microfiber pollution | Open textile microscopy collections |
| 📜 **Paper / document forensics** | Counterfeit detection, conservation | Public paper-fiber atlases, conservation institute datasets |
| 🍲 **Food microscopy** | Adulteration, fiber identification, food forensics | Open food microscopy databases |
| 🪙 **Numismatic and art conservation** | Surface analysis, patina assessment | Public museum conservation imagery (CC) |

---

## These 22 editions are only a starting set

The list above is a small slice of what becomes possible once a smartphone is the deployment target. Restoration specialists, archaeologists, soil scientists, brewers, food testers, sneaker authenticators, traditional medicine practitioners, leather and bookbinding conservators: each can host a dedicated Gemma 4 profile under the same training and compliance recipe.

---

## The full Gemma 4 family: same recipe, different deployment targets

MicroLens Final ships on Gemma 4 E2B because the target user owns a budget-class smartphone. The Gemma 4 family is much larger, and the recipe scales upward whenever the use case justifies the hardware.

| Model | Parameters | Architecture | Where it runs | Where MicroLens-style fine-tunes apply |
|---|---|---|---|---|
| **Gemma 4 E2B** | 4.44B params, ~2B effective (PLE) | Dense, multimodal | Budget Android phones (4 GB RAM), Raspberry Pi 5 | Field tools for amateur and citizen-science use (current MicroLens Final) |
| **Gemma 4 E4B** | 4.5B effective (PLE) | Dense, multimodal | High-end smartphones, tablets, low-power laptops | Professional field tools with deeper reasoning |
| **Gemma 4 26B A4B** | 26B total / 4B active (MoE) | Sparse MoE, multimodal | Modern laptops with 32 GB RAM, mini-PCs | Lab-grade specialist tools, pathology second-reader |
| **Gemma 4 31B** | 31B | Dense, multimodal | Workstations, single H100-class GPU servers | Hospital-scale and institutional research platforms |

Same architecture, same training recipe, same dataset-curation pipeline. What changes is the hardware ceiling and the depth of reasoning the local instrument sustains. A village clinic on a $120 phone setup gets the E2B edition. A regional hospital with a $1,200 laptop gets E4B or 26B MoE with deeper diagnostic reasoning. A teaching hospital with a workstation gets the 31B edition. One ecosystem, one data-handling regime, one governance model.

---

## Compliance and responsibility: the gated rollout

Every specialized edition follows the same release pipeline — **four blocking reviews** (per [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md)) plus public release as the outcome.

**1. Legal review.** Jurisdictional scoping (EU MDR, US FDA, UK MHRA where applicable). EU AI Act mapping: prohibited-use rules (Feb 2025), GPAI obligations (Aug 2025), high-risk requirements (Aug 2026), full applicability (Aug 2027). Editions classified as "Unacceptable risk" are not released. High-risk editions (medical-adjacent, biometric-adjacent) require conformity assessment, technical documentation, post-market monitoring, and CE marking. GDPR and HIPAA audits on training data. Licensing audit on every dataset. Attorney sign-off for healthcare and safety-critical editions.

**2. Technical review.** Held-out evaluation on real-world distributions, not synthetic test sets. Failure-mode disclosure in the model card. Adversarial probing for confident wrong answers on out-of-distribution inputs. Performance floor: no edition is released if it underperforms a documented baseline on its target task.

**3. Professional review.** Practising domain specialists rate the cartridge — real clinicians for medical, qualified geologists for geological, working criminologists for forensic — not ML engineers reviewing each other's work. The review includes a **field test under real working conditions** and a written sign-off on intended-use scope and documented limitations.

**4. Ethical review.** Misuse-potential analysis: surveillance, profiling, discrimination. Vulnerable-population consideration. Honest capability disclosure: any gap between marketing and measured capability is a release blocker. A public risk register is maintained per cartridge.

**Public release.** Versioned model card, changelog, deprecation policy. Take-down protocol within 24 hours if a serious post-release flaw is discovered.

**Hard limits.** No edition will be released that primarily enables surveillance, profiling without consent, deception of users, or circumvention of safety regulations. No biometric identification or emotion recognition editions, in line with EU AI Act prohibitions.

> **MicroLens is not a medical device.** No edition described above will be deployed in clinical decision-making without proper regulatory clearance.

---

## Why Gemma 4

Gemma 4 (Apache 2.0, multilingual, multimodal, sized to fit edge hardware) gives any product built on top of it the same set of properties: no vendor lock-in, no per-call costs, no connectivity dependency, no data privacy compromises (on-device processing keeps personal data off the wire), multilingual reach for free (140+ languages pretrained, 35+ supported for multimodal), multimodal handling out of the box. For a small team building a domain-specific instrument under modern regulatory constraints, that combination is rare and very practical.

---

## Author and credits

Author: **Serghei Brinza**, Vienna, Austria.
Built for the Kaggle Gemma 4 Good Hackathon 2026.
Fine-tuned with **Unsloth** (1.5× faster training, ~60 % less VRAM than FA2 setups, per Unsloth's published Gemma 4 numbers).
Distributed via HuggingFace, a local `Modelfile` for **Ollama** deployment, and direct GitHub APK release.

---

## From product to platform

The MicroLens repository ships a model and a reproducible recipe: dataset curation (`vqa_3x8b.py`), training (`train_gemma4.py`), merge and quantization (`merge_and_gguf.py`). Independent researchers, NGOs, universities, and small product teams can fork the toolchain, under the same gated-rollout discipline, to ship their own field-specific editions.

> MicroLens Final is the universal generalist that proves the recipe. The specialized editions are the field-specific experts. Same architecture, same $120 hardware floor, same compliance gates before release.
