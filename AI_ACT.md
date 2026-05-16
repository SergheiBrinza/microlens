# EU AI Act Positioning Statement

**Effective:** 2026-05-04
**Software:** MicroLens (Android APK, desktop application, HuggingFace Space, model weights via Ollama / HuggingFace)
**Maintainer:** Serghei Brinza (Vienna, Austria)
**Reference:** Regulation (EU) 2024/1689 (the "AI Act")

---

## 1. Why this document exists

Article 50 of the AI Act requires that **users be informed when they are interacting with an AI system** and that **AI-generated content be marked as such**. Article 4 imposes an AI literacy obligation on providers and deployers. This document discharges those duties for MicroLens and provides a transparent positioning statement for users, downstream developers, and reviewers.

## 2. What MicroLens is

A **vision-language model fine-tuned on public microscopy data**, packaged as:

- an Android application (debug-signed APK, distributed via GitHub Releases);
- a desktop application (FastAPI backend + React frontend);
- a browser-based demo on Hugging Face Spaces (Gradio);
- model weights (LoRA, merged FP16, Q4_K_M GGUF) on Hugging Face and Ollama Hub.

The artefact is published as a **research and educational tool** under the Apache License 2.0. It is **not placed on the market or put into service as a regulated AI system** within the meaning of Article 3(11) of the Regulation.

## 3. Risk classification

### 3.1 Not a prohibited practice (Article 5)

MicroLens does not engage in any of the practices prohibited by Article 5: it does not perform subliminal manipulation, exploit vulnerabilities, conduct social scoring, infer emotions in workplace or education contexts, conduct biometric categorisation by sensitive attributes, perform real-time remote biometric identification, or scrape facial images. The artefact analyses **microscopy specimens** (diatoms and fungal spores), none of which are persons.

### 3.2 Not a high-risk AI system (Annex III)

The eight Annex III categories of high-risk AI systems are:

1. Biometric identification and categorisation;
2. Critical infrastructure;
3. Education and vocational training (with decision-making consequences for the individual);
4. Employment, workers' management, access to self-employment;
5. Access to essential private and public services;
6. Law enforcement;
7. Migration, asylum, border control;
8. Administration of justice and democratic processes.

MicroLens does **none** of the above. It identifies the genus of microscopic organisms in user-supplied images. It does not make decisions about people. It does not classify, evaluate, or determine outcomes for individuals.

### 3.3 Not a safety component or product covered by Annex I

MicroLens is **not** a safety component of a product covered by Union harmonisation legislation (Annex I): not a medical device under MDR or IVDR, not machinery, not radio equipment, not a toy, not a recreational craft, not a pressure equipment.

The model card, README, and Terms of Use repeatedly and explicitly disclaim any medical, diagnostic, or regulatory-compliance use. See [`TERMS.md`](TERMS.md) §1 and [`MODEL_CARD.md`](MODEL_CARD.md).

### 3.4 Research and development (Article 2(6))

The Regulation **does not apply** to AI systems or AI models developed and put into service for the sole purpose of scientific research and development (Article 2(6)). MicroLens is published as a research artefact for the **Kaggle Gemma 4 Good Hackathon 2026**, with full source code, training pipeline, and reproducibility instructions in the public repository. The artefact is therefore in scope of the research exemption.

If a downstream party takes the artefact out of the research context and deploys it in production for a non-research purpose, **the provider/deployer obligations of the Regulation transfer to that party**, not to the original author. See [`TERMS.md`](TERMS.md) §8.

### 3.5 Conclusion on risk class

MicroLens is, in the worst-case reading of the Regulation, a **minimal-risk AI system** subject only to the transparency obligations of Article 50. In its primary use as a research artefact it falls under the Article 2(6) research exemption.

## 4. Article 50 transparency obligations

### 4.1 Article 50(1): informing the user that the system is AI

This document, the in-application "About" screen, the splash screen, and the Hugging Face Space header all clearly state that **MicroLens is an AI system**. The application's name itself includes the marker ("MicroLens AI · Powered by Gemma 4 E2B"). Every README, every Ollama model page, and every model card carries an explicit statement that outputs are model predictions. Users cannot reasonably mistake MicroLens for a non-AI tool.

### 4.2 Article 50(2): marking AI-generated content

Every text output of MicroLens, both the on-device inference and any translations performed via vanilla Gemma 4, is **AI-generated**. The application labels the output as such ("This is a model prediction. Verify with a qualified taxonomist before any consequential decision."). The Hugging Face Space displays the model name above every result panel.

### 4.3 Article 50(3)(c): emotion recognition or biometric categorisation

Not applicable. MicroLens does not perform emotion recognition or biometric categorisation.

### 4.4 Article 50(4): deepfakes and AI-generated text on matters of public interest

Not applicable. MicroLens generates **scientific descriptions of microscopy specimens**, not media depicting persons or text on matters of public interest.

## 5. General-Purpose AI Models (Chapter V)

The base model **Gemma 4 E2B** is a General-Purpose AI Model in the sense of Chapter V. **Google DeepMind is the provider of that GPAI model**, not the author of MicroLens. Obligations under Article 53 (technical documentation, summary of training content, copyright policy) apply to the provider of the GPAI model, not to a downstream researcher who fine-tunes it for a specific narrow purpose.

The fine-tuning performed in MicroLens is a **non-systemic-risk modification**: 29.9 M trainable LoRA parameters out of 4.44 B base parameters (≈ 0.58 %), trained on 75,491 public microscopy QA pairs over approximately 14 hours on a single RTX 3090 Ti. The compute used (well below the 10²⁵ FLOPs threshold of Article 51) does not turn the artefact into a GPAI model with systemic risk.

## 6. Data protection and privacy

See [`PRIVACY.md`](PRIVACY.md). Summary: **no personal data is processed by the application**. All inference is on-device. The single network request (model download from Hugging Face) is a static HTTPS asset fetch and transfers no user data outward.

## 7. Cybersecurity

The application does not connect to any network endpoint other than the Hugging Face download mirror. There is no remote command-and-control surface, no admin endpoint, no webhook, no telemetry. The on-device runtime is `llama.cpp` with the `mtmd` extension; both are open-source and audited.

## 8. Intellectual property and training data

All training datasets are public, licence-clean, and individually attributed in [`docs/LICENSE_AUDIT.md`](docs/LICENSE_AUDIT.md). The hand-curated knowledge base for the top-30 genera (`training/genus_kb.json`) is **original phrasing** by the author, paraphrased from standard public taxonomic references (AlgaeBase, WoRMS, ITIS, Round 1990, Krammer-Lange-Bertalot 1986–1991). **No verbatim copyrighted text** was used in training.

The Gemma 4 base model is distributed by Google DeepMind under the Apache 2.0 licence and the Gemma terms of use; the upstream provider is responsible for any obligations under Article 53(1)(c) (summary of training data) and Article 53(1)(b) (copyright policy) at the GPAI model level.

## 9. Article 4: AI literacy

Users are encouraged to read this document, [`PRIVACY.md`](PRIVACY.md), [`TERMS.md`](TERMS.md), [`MODEL_CARD.md`](MODEL_CARD.md), and [`KAGGLE_WRITEUP.md`](KAGGLE_WRITEUP.md) before relying on the artefact. The combination of these documents is intended to provide sufficient context for a non-specialist to understand:

- what an AI system is and how the artefact in question fits within that definition;
- how the model was trained and what its known limitations are;
- under what conditions the output should and should not be relied on;
- what data, if any, is processed by the application.

## 10. Re-use, fine-tuning, and derivatives

Anyone is free to fork the artefact under the Apache 2.0 licence. **However**, any party that puts a derivative on the market or into service for a regulated purpose (medical diagnosis, regulatory water-quality measurement, employment screening, etc.) becomes the **provider** for that derivative under Article 25(1) of the Regulation, and assumes all corresponding obligations. The original maintainer is not responsible for compliance of derivatives.

## 11. Contact and complaints

For questions about this positioning statement: open an issue at [github.com/SergheiBrinza/microlens/issues](https://github.com/SergheiBrinza/microlens/issues).

National competent authority for AI Act enforcement in Austria: to be designated by the Republic of Austria during the AI Act roll-in period (full applicability 2 August 2026).

---

`◆ Serghei Brinza · Vienna · Austria · 2026 ◆`
