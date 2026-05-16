# Terms of Use

**Effective:** 2026-05-04
**Software:** MicroLens (Android APK, desktop application, HuggingFace Space, model weights distributed via Hugging Face and Ollama)
**Maintainer:** Serghei Brinza (Vienna, Austria)

---

## 1. Nature of the artefact

MicroLens is a **research and educational artefact**. It is a fine-tune of Google's Gemma 4 E2B vision-language model trained on public microscopy datasets, packaged into a runnable demonstration on three surfaces (Android, desktop, browser). It is published under the **Apache License 2.0** (see [`LICENSE`](LICENSE)).

This software is **NOT**:

- a medical device, in-vitro diagnostic (IVD), or clinical decision-support system;
- a regulatory-compliant water quality measurement instrument (no ISO 17025, EPA, EU Water Framework Directive, or equivalent certification);
- a substitute for a trained taxonomist or accredited laboratory analysis;
- a calibrated, validated, or peer-reviewed analytical method;
- a CE-marked product, an FDA-cleared product, or a product that carries any other regulatory mark or accreditation.

It is the user's responsibility to verify that the use case is appropriate for an unvalidated, AS-IS research artefact.

## 2. Intended use

The artefact is intended for:

- citizen-science screening and pre-classification;
- taxonomy teaching, student labs, online microscopy courses;
- machine-learning research, dataset benchmarking, model comparison;
- pre-classification stages of professional pipelines, where every result is verified by a qualified person before any decision is made.

Any decision that informs a regulatory, environmental, clinical, or health-related determination must be reviewed by qualified personnel under accredited methods. The artefact's output is a **statistical pattern match against the training distribution**, rendered through learned scientific phrasing; it is neither a physical or analytical measurement nor a peer-reviewed identification.

## 3. AS-IS warranty disclaimer

The software is provided **"AS IS"**, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement.

In no event shall the author or contributors be liable for any claim, damages, or other liability (whether in an action of contract, tort, or otherwise) arising from, out of, or in connection with the software or the use or other dealings in the software.

You assume **all risk** when downloading, deploying, modifying, or using this software on your own hardware.

## 4. Known failure modes

The model can be **confidently wrong**. Documented failure modes include:

- specimens not represented in training (the 95 genera covered ≠ all microscopic life);
- damaged, atypical, or out-of-focus images;
- subjects outside the two trained categories — anything that is not a diatom or a fungal spore is out-of-distribution;
- long-tail genera (~65 of 95) where category-generic morphology is returned rather than something specific to the genus, because the underlying training share is small.

These limitations are documented in [`MODEL_CARD.md`](MODEL_CARD.md) and [`README.md`](README.md). It is the user's responsibility to take the limitations into account.

## 5. License grants

- **Source code** (training scripts, application code, build pipeline): Apache 2.0. See [`LICENSE`](LICENSE).
- **Model weights** (LoRA adapter, merged FP16, GGUF Q4_K_M, mmproj): inherit the Gemma 4 license terms (Apache 2.0). See the upstream [Gemma terms of use](https://ai.google.dev/gemma/terms).
- **Datasets used during training**: each dataset retains its original licence. Per-source attribution is in [`docs/LICENSE_AUDIT.md`](docs/LICENSE_AUDIT.md). All training datasets are **public** and **licence-clean for commercial use** (CC0 or CC-BY 4.0 only).
- **Knowledge-base entries** (`training/genus_kb.json`): hand-written by the author. Sources are paraphrased from public taxonomic references (AlgaeBase, WoRMS, ITIS, Round 1990, Krammer-Lange-Bertalot 1986–1991). No verbatim copyrighted text was used.

## 6. EU AI Act positioning

MicroLens is positioned as a **research and educational artefact** within the meaning of EU Regulation 2024/1689 (the EU AI Act). It is not placed on the market as a high-risk AI system within the meaning of Annex III of the Regulation. Per Article 50(1) the user is hereby informed that they are interacting with an AI system; per Article 50(2) inference outputs are AI-generated content. See [`AI_ACT.md`](AI_ACT.md) for the detailed positioning statement.

## 7. Ollama and Hugging Face hosting

The model weights are distributed via Hugging Face (`Laborator/microlens-final`). The repository also ships a working Ollama [`Modelfile`](Modelfile) for users who want to build a local Ollama image once the GGUF artefacts are downloaded. Use of those services is governed by **their own terms** (Ollama: [ollama.com/legal/terms](https://ollama.com/legal/terms); Hugging Face: [huggingface.co/terms-of-service](https://huggingface.co/terms-of-service)). The maintainer of MicroLens is not responsible for outages, policy changes, or content moderation decisions on those platforms.

## 8. Modifications and forks

Apache 2.0 grants you the right to modify and redistribute. If you fork the repository, **rename the fork** if you publish a modified weights version, so that downstream users can distinguish your fork from the upstream MicroLens. Do not represent a forked or modified model as the original MicroLens artefact.

## 9. Termination of use

You may stop using the software at any time by uninstalling the Android application, deleting the desktop installation, or simply ceasing to use the browser demo. Because no server-side state is held, no further action is required to terminate use.

## 10. Governing law

These terms are governed by the law of the Republic of Austria. To the maximum extent permitted by the Apache License 2.0, any disputes shall be resolved under Austrian jurisdiction.

---

## 11. Acceptance

By downloading, installing, or running the software, you acknowledge that you have read, understood, and agreed to these Terms of Use, the [Privacy Notice](PRIVACY.md), and the [AI Act positioning statement](AI_ACT.md).

If you do not agree, do not install, run, or otherwise use the software.

---

`◆ Serghei Brinza · Vienna · Austria · 2026 ◆`
