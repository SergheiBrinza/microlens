# Data Governance and Responsible Release

**Version:** 1.0  
**Date:** 2026-05-10  
**Author:** Serghei Brinza, Vienna, Austria  
**Repo:** https://github.com/SergheiBrinza/microlens  
**License:** CC-BY 4.0 (this document), Apache 2.0 (the Biology cartridge)  
**Subject to:** the Gemma Prohibited Use Policy, https://ai.google.dev/gemma/prohibited_use_policy

## Why this document exists

MicroLens today is a single biology cartridge that runs on a phone. The longer plan is a marketplace of specialised cartridges, including ones that touch domains where a wrong answer can hurt a real person. This file describes how those future cartridges get released and, more importantly, how they do not. It is the gate, not the roadmap.

The current Biology cartridge is intentionally narrow. It identifies specimens across two microscopy categories (diatoms and fungal spores) spanning 95 genera. Plenty of useful work to do at that scope, none of which requires the model to pretend to be a doctor, a forensic examiner, or a food inspector. A future MicroLens-Doctor, MicroLens-Veterinary, MicroLens-Forensics, or MicroLens-FoodSafety would. The procedure below is what stands between the idea and a release.

## The three release tiers

Every cartridge is sorted into one of three tiers before any training begins. The tier sets the path.

**Tier 1, open educational.** This is where MicroLens-Biology lives today. Training data is open-licensed only, weights ship as Apache 2.0, evaluation is the standard ML quality assurance an open project would expect. Use cases: classroom microscopy, citizen science, freshwater biomonitoring, hobbyist field work.

**Tier 2, professional research.** Examples: MicroLens-Geologist, MicroLens-Specialist. Verified per-dataset licensing, technical validation appropriate for a research user, but no claim of regulatory compliance. The audience is professional but not a person making safety-critical decisions on the model output alone.

**Tier 3, regulated.** Examples: MicroLens-Doctor, MicroLens-Veterinary, MicroLens-Forensics, MicroLens-FoodSafety. Cartridges that touch human health, legal evidence, food and water safety. The full procedure below is mandatory. No exceptions, no fast-track, no informal releases for friends.

## Tier 3 release procedure

Four reviews, all of them blocking. A finding from any one review that is not resolved in writing prevents release.

### Legal review

Regulatory classification is performed per jurisdiction and per intended use. For medical cartridges this means EU MDR or IVDR scoping, EU AI Act risk-tier determination, and a GDPR or HIPAA assessment if the cartridge touches personal health data. For forensic cartridges this means evidentiary standards in the target jurisdictions. For food-safety cartridges this means food-law assessment in the same way.

Per-image dataset licensing is audited end to end. Every training image must trace to one of: Apache 2.0, MIT, CC-BY, CC0, or documented explicit consent from the rights holder. Anything else is excluded from training, not promised to be excluded later. Attribution is preserved in the form each dataset owner requires. Where a licence requires payment for commercial or derivative use, the payment is settled before training begins, with proof of settlement archived in the per-cartridge audit folder. Scraped data without consent is not used at any point.

### Technical review

Validation runs on held-out clinical or domain data that did not touch training or hyperparameter tuning. False-positive and false-negative rates are reported per class, not as aggregate accuracy. Adversarial testing covers out-of-distribution inputs, deliberate misuse patterns, and edge cases that the practising specialists in the target domain identify. Demographic bias is audited across age, sex, ethnicity, and geographic origin where the data permits. The full evaluation pipeline is published alongside the cartridge so that a third party with the same access can reproduce it and reach their own conclusions.

### Professional review

The cartridge is rated by practising specialists in its target domain. Real clinicians for medical cartridges. Qualified geologists for geological ones. Working criminologists for forensic. Not by ML engineers reviewing each other's work. The review includes a field test under real working conditions, and a written sign-off on intended-use scope and on the documented limitations.

### Ethical review

A structured risk assessment for non-expert misuse, with explicit attention to the realistic case of the cartridge falling into the hands of someone outside the intended audience. Mandatory disclaimers are enforced at the model output level, not just in the user interface. The model itself emits the line "Decision support, not a substitute for licensed diagnosis" for every regulated cartridge as part of the Ollama Modelfile SYSTEM directive, so the disclaimer ships at the model-contract layer rather than only in the application UI. Where the risk profile demands it, a sensitive cartridge is gated behind verified professional credentials before release. A public risk register is maintained per cartridge, listing known failure modes and the mitigations applied to each.

## Dataset rights protection

Every cartridge ships with a complete source list. For each source: licence, required attribution, and payment terms where applicable. Published as docs/LICENSE_AUDIT.md alongside the cartridge.

A dataset owner can request removal at any time through a public pull request or issue. On a valid request, the cartridge is rebuilt without the contested data and re-released within thirty days. There is no carve-out for "useful exceptions" and no informal grace period.

## Protecting non-expert users

The largest predictable harm from a Tier 3 cartridge is a non-expert acting on its output as if it were a diagnosis. The mitigations are layered, not relying on a single line of defence:

- A baseline disclaimer is wired into the Ollama Modelfile SYSTEM directive of every regulated cartridge, so it ships as part of the model contract rather than an application-UI feature. Under normal use the model preserves it. Adversarial prompt injection is a known property of all current large language models; resistance to it is treated as one layer in the defence-in-depth scheme described below, not as a single technical guarantee.
- A persistent "Decision support" banner is shown in the UI of any application that embeds a regulated cartridge.
- The system is telemetry-free by default. No usage data is collected, and no images leave the device unless the user explicitly chooses to share them.
- A clear scope statement ships with each cartridge, stating what it is for and, with at least equal weight, what it is not for.

## Third-party verification

Every claim in this document is structured to be falsifiable. The licence audit, the held-out evaluation, the professional sign-off, and the risk register are all designed to be reproducible by an independent reviewer with the same access to the cartridge artefacts. If a competent third party runs the same procedure and reaches a different conclusion, the project wants to hear about it. That is the point.

## Why slowness is part of the design

Better to ship fewer cartridges slowly than to harm a non-expert user, violate a dataset owner's rights, or release a regulated product without the regulation. A medical, forensic, or food-safety cartridge released without proper review can cause harm orders of magnitude greater than the value of the early release. The four reviews above are slow on purpose. They are the price of operating in domains where mistakes are not recoverable.

## What this means for the current release: MicroLens-Biology

The Biology cartridge currently distributed under https://huggingface.co/Laborator/microlens-final is **Tier 1, open educational**.

It is **out of scope** of EU MDR and IVDR, EU AI Act high-risk classification, and GDPR or HIPAA processor obligations. It is **not** a medical device, **not** a diagnostic tool, **not** a regulated product. The model card, the README, the HuggingFace Space, the Ollama Modelfile, and the Android app all carry the same disclaimer:

> MicroLens is decision support for education and citizen science. Not a medical, regulatory, or forensic tool. Verify with a domain specialist before any consequential decision.

What was actually done, in plain terms:

- **Base model:** Gemma 4 E2B, multimodal, 4.44 billion parameters, fine-tuned with Unsloth FastVisionModel and 4-bit QLoRA at LoRA rank 16, alpha 32, dropout 0. Trainable parameters at fine-tuning time: 29.9 million, which is 0.58 percent of the base.
- **Training compute:** approximately 14 hours on a single RTX 3090 Ti with 24 GB of VRAM, 2 epochs over the full training set. Estimated energy: about 4 kWh, about 0.4 kg CO2-equivalent on the Austrian grid.
- **Coverage:** 95 genera across 2 microscopy categories (diatoms and fungal spores), 75,491 vision-question-answer pairs (67,121 train + 8,370 validation).
- **Source datasets:** UDE Diatoms in the Wild 2024 (University of Duisburg-Essen, CC0), DIATLAS (CC-BY 4.0), TgFC (figshare, CC-BY 4.0). All three sources are license-clean for commercial use. Per-source licensing is published in docs/LICENSE_AUDIT.md.
- **Deployment:** HuggingFace Space for live demo, an Android APK using llama.cpp with multimodal projector, and a Linux desktop application built on FastAPI and React. Telemetry-free by default, with no requirement for cloud connectivity.

## Hackathon alignment

This document supports five prize categories of the Kaggle Gemma 4 Good Hackathon 2026:

- **Health & Sciences track.** Scope honesty rather than scope inflation: the current cartridge is decision support for education and citizen science only, with the pathway to a Tier 3 MicroLens-Doctor explicitly documented and explicitly gated.
- **Digital Equity track.** On-device, telemetry-free, no cloud requirement. Gemma 4 E2B runs on consumer phones with 6 GB or more of RAM, which puts expert-level microscopy in reach where lab connectivity, lab budgets, or specialists are scarce.
- **Future of Education track.** Open-licensed datasets, Apache 2.0 weights, citizen-science scope. Usable in classrooms and field-biology courses without subscriptions, accounts, or telemetry.
- **Unsloth special prize.** The Biology cartridge is fine-tuned end to end with Unsloth FastVisionModel and 4-bit QLoRA on Gemma 4 E2B. The reproducible training notebook is published as a Kaggle kernel and at training/scripts/kaggle_notebook.ipynb in this repository.
- **Ollama special prize.** The cartridge ships with an Ollama Modelfile (see `Modelfile`) that enforces the safety disclaimer in the SYSTEM directive, so it is part of the model contract rather than only the application UI. The published Ollama tag is updated alongside each MicroLens release.

## Governance evolution

This is the initial governance framework. Material updates go through public review and are tracked in this file's git history. Affected cartridges receive release notes whenever the framework changes in a way that alters their tier or required reviews.

---

**Author:** Serghei Brinza, Vienna, Austria  
**Built for:** Kaggle Gemma 4 Good Hackathon 2026  
**Tracks:** Health & Sciences, Digital Equity, Future of Education  
**License:** CC-BY 4.0 (this document), Apache 2.0 (Biology cartridge weights)
