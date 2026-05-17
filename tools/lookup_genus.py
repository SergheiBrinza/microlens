#!/usr/bin/env python3
"""
MicroLens — Function Calling Stub
==================================

Demonstration of how MicroLens (Gemma 4 E2B fine-tuned for microscopy ID)
can call external taxonomic databases at inference time to verify or
enrich its identifications.

This is a STUB / proof of concept. The full integration (Modelfile
SYSTEM directive + Gemma 4 tool-calling format + retry logic) is on
the MicroLens roadmap. The current file provides:

1. A function schema in Gemma 4 / OpenAI-compatible format
2. A working lookup_genus() that hits AlgaeBase (read-only, public)
3. A WoRMS fallback for any taxa not in AlgaeBase
4. A demo loop showing how Gemma 4 would invoke the tool

Usage
-----
    python tools/lookup_genus.py "Aulacoseira granulata"
    python tools/lookup_genus.py "Calanus finmarchicus" --db worms

Dependencies
------------
    pip install requests

Author: Serghei Brinza (Vienna, Austria), May 2026
License: Apache 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Function schema for Gemma 4 / OpenAI-compatible tool calling
# ---------------------------------------------------------------------------

GENUS_LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_genus",
        "description": (
            "Verify a microscopy genus identification against an authoritative "
            "taxonomic database. Returns accepted name, family, habitat, and a "
            "permalink to the source record. Use this AFTER you have produced "
            "your own identification, to confirm or correct it. Do NOT use it "
            "as a substitute for visual identification."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "genus_name": {
                    "type": "string",
                    "description": (
                        "The genus name you identified, e.g. 'Aulacoseira' or "
                        "'Calanus'. Provide genus only, not species epithet."
                    ),
                },
                "database": {
                    "type": "string",
                    "enum": ["algaebase", "worms", "auto"],
                    "description": (
                        "Which database to query. 'algaebase' for algae and "
                        "freshwater protists, 'worms' for WoRMS taxonomic database, "
                        "'auto' to try both and merge results. Default: auto."
                    ),
                    "default": "auto",
                },
            },
            "required": ["genus_name"],
        },
    },
}


# ---------------------------------------------------------------------------
# 2. Implementation — read-only public lookups
# ---------------------------------------------------------------------------

ALGAEBASE_BASE = "https://www.algaebase.org"
WORMS_BASE = "https://www.marinespecies.org/rest"


def lookup_algaebase(genus_name: str, timeout: int = 8) -> Optional[dict[str, Any]]:
    """Look up a genus in AlgaeBase.

    AlgaeBase does not expose a public JSON API, so we hit the search
    page and return a structured stub. In a production integration this
    would call a licensed AlgaeBase API key endpoint.
    """
    url = f"{ALGAEBASE_BASE}/search/genus/"
    try:
        r = requests.get(url, params={"genus": genus_name}, timeout=timeout)
        r.raise_for_status()
        return {
            "source": "algaebase",
            "genus": genus_name,
            "status": "search_executed",
            "permalink": f"{url}?genus={genus_name.replace(' ', '+')}",
            "note": (
                "AlgaeBase public search returned HTTP "
                f"{r.status_code}. For programmatic access, request an "
                "AlgaeBase API key. Visit the permalink for a human-readable "
                "match list."
            ),
        }
    except requests.RequestException as exc:
        return {
            "source": "algaebase",
            "genus": genus_name,
            "status": "lookup_failed",
            "error": str(exc),
        }


def lookup_worms(genus_name: str, timeout: int = 8) -> Optional[dict[str, Any]]:
    """Look up a genus in WoRMS (World Register of Marine Species — used as taxonomic fallback).

    WoRMS exposes a stable JSON REST API. We use AphiaRecordsByName.
    """
    url = f"{WORMS_BASE}/AphiaRecordsByName/{genus_name}"
    try:
        r = requests.get(url, params={"like": "false", "marine_only": "false"},
                         timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if not data:
            return {
                "source": "worms",
                "genus": genus_name,
                "status": "no_match",
            }
        record = data[0]
        return {
            "source": "worms",
            "genus": genus_name,
            "accepted_name": record.get("scientificname"),
            "rank": record.get("rank"),
            "family": record.get("family"),
            "kingdom": record.get("kingdom"),
            "status": record.get("status"),
            "permalink": record.get("url"),
            "aphia_id": record.get("AphiaID"),
        }
    except requests.RequestException as exc:
        return {
            "source": "worms",
            "genus": genus_name,
            "status": "lookup_failed",
            "error": str(exc),
        }
    except (ValueError, KeyError) as exc:
        return {
            "source": "worms",
            "genus": genus_name,
            "status": "parse_failed",
            "error": str(exc),
        }


def lookup_genus(genus_name: str, database: str = "auto") -> dict[str, Any]:
    """Main entry point. Returns a structured dict suitable for direct
    inclusion in a Gemma 4 tool-call response."""
    genus_name = genus_name.strip()
    if not genus_name:
        return {"error": "empty genus name"}

    if database == "algaebase":
        return lookup_algaebase(genus_name) or {}
    if database == "worms":
        return lookup_worms(genus_name) or {}

    # auto: try WoRMS first (has structured JSON), fall back to AlgaeBase
    worms_result = lookup_worms(genus_name)
    algae_result = lookup_algaebase(genus_name)
    return {
        "genus_name": genus_name,
        "results": {
            "worms": worms_result,
            "algaebase": algae_result,
        },
    }


# ---------------------------------------------------------------------------
# 3. Demo: how Gemma 4 would invoke this tool
# ---------------------------------------------------------------------------

def demo_tool_call_flow(genus: str) -> None:
    """Print a Gemma-style tool-call exchange for a given genus."""
    # 1. Model's tool call (what Gemma 4 would emit as part of generation)
    tool_call = {
        "name": "lookup_genus",
        "arguments": {"genus_name": genus, "database": "auto"},
    }
    print("=" * 70)
    print("Step 1 — Model emits tool call:")
    print(json.dumps(tool_call, indent=2))

    # 2. Runtime executes the tool
    result = lookup_genus(genus, database="auto")
    print()
    print("Step 2 — Runtime returns tool result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 3. Model continues generation with the result in context (illustrative)
    print()
    print("Step 3 — Model continues generation using the verified data.")
    print("         (In production, the result is fed back as a tool message")
    print("          and Gemma 4 produces the final user-facing reply.)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MicroLens function-calling stub: verify a genus against "
                    "AlgaeBase or WoRMS.",
    )
    parser.add_argument("genus", help="Genus name to look up (e.g. Aulacoseira).")
    parser.add_argument("--db", choices=["algaebase", "worms", "auto"],
                        default="auto", help="Database to query.")
    parser.add_argument("--demo", action="store_true",
                        help="Print full tool-call flow as Gemma 4 would see it.")
    parser.add_argument("--json", action="store_true",
                        help="Print only the JSON result (machine-readable).")
    args = parser.parse_args(argv)

    if args.demo:
        demo_tool_call_flow(args.genus)
        return 0

    result = lookup_genus(args.genus, database=args.db)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
