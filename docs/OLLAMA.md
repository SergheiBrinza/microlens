# Run MicroLens with Ollama

The repo ships a `Modelfile` at the project root that pins the prompt template,
SYSTEM disclaimer, and recommended sampling parameters.

## Build the local model

From the repo root:

```bash
ollama create microlens -f Modelfile
```

This registers a local model called `microlens`. By default it pulls the **base**
`unsloth/gemma-4-E2B-it-GGUF:Q4_K_M` from the Hugging Face Hub — the
MicroLens Final fine-tuned weights are still in training. Once the Final
weights are published, edit `Modelfile` to point `FROM` at
`./gguf/microlens-final-Q4_K_M.gguf` and rebuild.

## Run it

```bash
ollama run microlens "What is shown in this microscope image?" path/to/image.jpg
```

Or interactive:

```bash
ollama run microlens
```

## Publish to Ollama Hub (after Final weights are ready)

```bash
ollama create brinzaengineeringai/microlens-final -f Modelfile
ollama push brinzaengineeringai/microlens-final
```

## Notes

- Vision support requires an Ollama build with multimodal Gemma support and
  the matching `mmproj.gguf` next to the model file.
- Default sampling is configured in `Modelfile` — adjust there if needed.
- Not a medical device. Research and educational use only.
