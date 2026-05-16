# MicroLens · Tools

Function-calling integrations for MicroLens (Gemma 4 E2B fine-tuned for microscopy).

These tools are **stubs / proofs of concept** for the v3.1 roadmap milestone. They demonstrate that Gemma 4 can verify or enrich MicroLens identifications against external taxonomic databases at inference time.

## Available tools

### `lookup_genus.py` — Taxonomic verification

Verify a genus identification against AlgaeBase (algae, freshwater protists).

**Quick start:**

```bash
pip install requests
python tools/lookup_genus.py "Aulacoseira"
python tools/lookup_genus.py "Calanus finmarchicus" --db worms
python tools/lookup_genus.py "Aulacoseira" --demo
```

**Output:** structured JSON with accepted name, family, habitat, permalink to the source record.

**Function schema** (Gemma 4 / OpenAI tool-calling format) is exposed as `GENUS_LOOKUP_TOOL` and can be passed directly to a Gemma 4 chat completion call.

## Roadmap

v3.1:
- Integration with Modelfile SYSTEM directive
- Tool-call format demonstration in `training/scripts/kaggle_notebook.ipynb`
- WoRMS API key support for higher rate limits
- ITIS and GBIF fallbacks for terrestrial plants and fungi

See [ROADMAP.md](../ROADMAP.md) for the broader vertical-edition plan.

## License

Apache 2.0. Author: Serghei Brinza, Vienna, Austria, May 2026.
