"""MicroLens HuggingFace Space, Kaggle Gemma 4 Good Hackathon submission.

Status: this Space code is preserved end-to-end from the previous MicroLens
release so the user-facing layout, sample catalog, and 3-panel comparison flow
stay functional. The internal "v2"/"v3" identifiers refer to the comparison
slots in the UI (BASELINE / BRIEF / RICH); when the Final LoRA finishes
training, the corresponding URL_* environment variable should be repointed at
the new endpoint and the panel subtitle updated, no other structural change
required.

Layout:
- Header: italic-serif logo + ONLINE green + Kaggle/Gemma 4 hackathon meta line
- Toolbar: 3 mode buttons (MICROSCOPE / UPLOAD / SAMPLES) + 4 tool buttons
  (CIRCLE/SQUARE segmented + GRID + CROSS)
- Main row: viewport (square/circle, optional grid+cross overlays) + right
  control panel (mode-dependent: category thumbs / upload zone / camera
  enumeration)
- AI ANALYZE long oval cyan→red gradient button
- 3 result panels: UNTRAINED BASELINE / MICROLENS BRIEF / MICROLENS RICH
- Translate row with 28 languages (English default) + ORIGINAL button after
  translation
- Footer with run-locally + APK + Legal links

SAMPLES tab uses cached answers from catalog.json.
UPLOAD / MICROSCOPE tabs run LIVE inference against per-slot backend URLs:
  URL_VANILLA  (default http://127.0.0.1:8085/v1/chat/completions)  # base Gemma 4
  URL_V2       (default http://127.0.0.1:8084/v1/chat/completions)  # MicroLens brief
  URL_V3       (default http://127.0.0.1:8083/v1/chat/completions)  # MicroLens rich
On HF Space deployment configure these as Variables to point at a public tunnel
(e.g. Cloudflare → llama-server). When unreachable the panel shows a clean
"backend unavailable" message instead of crashing.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

ROOT = Path(__file__).parent
EXAMPLES_DIR = ROOT / "examples"
CATALOG_PATH = ROOT / "catalog.json"

CATEGORIES: List[Tuple[str, str]] = [
    ("diatom",                 "DIATOMS"),
    ("freshwater_zooplankton", "FRESHWATER"),
    ("fungal_spore",           "FUNGAL SPORES"),
    ("fish",                   "FISH"),
]
CAT_LABELS = [lbl for _, lbl in CATEGORIES]
CAT_BY_LABEL = {lbl: cid for cid, lbl in CATEGORIES}

LANGUAGES: List[Tuple[str, str, str]] = sorted([
    ("🇸🇦", "Arabic",     "ar"),
    ("🇧🇩", "Bengali",    "bn"),
    ("🇨🇳", "Chinese",    "zh"),
    ("🇨🇿", "Czech",      "cs"),
    ("🇩🇰", "Danish",     "da"),
    ("🇳🇱", "Dutch",      "nl"),
    ("🇬🇧", "English",    "en"),
    ("🇫🇷", "French",     "fr"),
    ("🇩🇪", "German",     "de"),
    ("🇬🇷", "Greek",      "el"),
    ("🇮🇳", "Hindi",      "hi"),
    ("🇭🇺", "Hungarian",  "hu"),
    ("🇮🇩", "Indonesian", "id"),
    ("🇮🇹", "Italian",    "it"),
    ("🇯🇵", "Japanese",   "ja"),
    ("🇰🇷", "Korean",     "ko"),
    ("🇲🇾", "Malay",      "ms"),
    ("🇳🇴", "Norwegian",  "no"),
    ("🇵🇱", "Polish",     "pl"),
    ("🇵🇹", "Portuguese", "pt"),
    ("🇷🇴", "Romanian",   "ro"),
    ("🇷🇺", "Russian",    "ru"),
    ("🇪🇸", "Spanish",    "es"),
    ("🇰🇪", "Swahili",    "sw"),
    ("🇸🇪", "Swedish",    "sv"),
    ("🇹🇭", "Thai",       "th"),
    ("🇹🇷", "Turkish",    "tr"),
    ("🇺🇦", "Ukrainian",  "uk"),
    ("🇻🇳", "Vietnamese", "vi"),
], key=lambda x: x[1])
LANG_DISPLAY = [f"{flag}  {name}" for flag, name, _ in LANGUAGES]
LANG_BY_DISPLAY = {f"{flag}  {name}": code for flag, name, code in LANGUAGES}
DEFAULT_LANG_DISPLAY = "🇬🇧  English"

CATALOG: List[Dict] = json.loads(CATALOG_PATH.read_text())
BY_FILENAME = {s["filename"]: s for s in CATALOG}

URL_VANILLA = os.environ.get("URL_VANILLA", "http://127.0.0.1:8085/v1/chat/completions")
URL_V2      = os.environ.get("URL_V2",      "http://127.0.0.1:8084/v1/chat/completions")
URL_V3      = os.environ.get("URL_V3",      "http://127.0.0.1:8083/v1/chat/completions")
INFERENCE_PROMPT = "What is shown in this microscope image?"

# ─────────────────────────────────────────────────────────────────────────────
# ZeroGPU runtime: when running on HF Space we replace HTTP llama-server calls
# with in-process transformers + PEFT multi-adapter inference on H200.
# Outside HF Space (local dev) the original HTTP path is preserved.
# ─────────────────────────────────────────────────────────────────────────────
IS_HF_SPACE = bool(os.environ.get("SPACE_ID"))

_HF_BASE = "unsloth/gemma-4-E2B-it"
_HF_LORA_REPO = "Laborator/microlens-final"

_zerogpu_processor = None
_zerogpu_model = None

if IS_HF_SPACE:
    import spaces
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    print("[ZeroGPU] loading processor + base model on cuda…", flush=True)
    _zerogpu_processor = AutoProcessor.from_pretrained(_HF_BASE)
    _zerogpu_model = AutoModelForImageTextToText.from_pretrained(
        _HF_BASE, torch_dtype=torch.bfloat16, device_map="cuda",
    )

    # PEFT 0.19 cannot hook transformers' Gemma4ClippableLinear (vision tower
    # wrapper around nn.Linear with opt-in clamping). The clamp thresholds
    # default to ±inf so the wrapper is a behavioral no-op — replace each
    # occurrence with its inner .linear so PEFT sees a plain nn.Linear.
    def _unwrap_clippable(module):
        from torch import nn
        for name, child in list(module.named_children()):
            if type(child).__name__ == "Gemma4ClippableLinear" and isinstance(
                getattr(child, "linear", None), nn.Linear
            ):
                if getattr(child, "use_clipped_linears", False):
                    print(f"[ZeroGPU] WARN: clipped-linears active on {name}; "
                          "unwrapping anyway (thresholds are ±inf = no-op)", flush=True)
                setattr(module, name, child.linear)
            else:
                _unwrap_clippable(child)
    _unwrap_clippable(_zerogpu_model)

    print("[ZeroGPU] attaching v2 LoRA…", flush=True)
    _zerogpu_model = PeftModel.from_pretrained(
        _zerogpu_model, _HF_LORA_REPO, subfolder="lora/v2", adapter_name="v2",
    )
    print("[ZeroGPU] attaching v3 LoRA…", flush=True)
    _zerogpu_model.load_adapter(
        _HF_LORA_REPO, subfolder="lora/v3", adapter_name="v3",
    )
    _zerogpu_model.eval()
    print("[ZeroGPU] ready (vanilla / v2 / v3 share one base, swap adapters)", flush=True)

    # ── Batch path: run vanilla + v2 + v3 in a SINGLE GPU acquisition.
    # duration=60: vanilla can ramble for 20+s on long answers; v2+v3 add
    # another 15s. 60s budget guarantees all 3 finish without "GPU task
    # aborted". Anon (2min/day) gets 2 clicks; free (3.5min) ~3; PRO (25min) ~25.
    @spaces.GPU(duration=60)
    def _zerogpu_infer_all(image_data_uri: str, prompt: str):
        import time as _t
        t_total = _t.time()
        print(f"[infer-all] start cuda={torch.cuda.is_available()}", flush=True)
        b64 = _strip_data_uri(image_data_uri) if image_data_uri.startswith("data:") else image_data_uri
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        if max(img.size) > 768:
            img.thumbnail((768, 768))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        inputs = _zerogpu_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: (v.to(_zerogpu_model.device, dtype=torch.bfloat16) if v.is_floating_point()
                       else v.to(_zerogpu_model.device))
                  for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        results = {}
        for version in ("vanilla", "v2", "v3"):
            t0 = _t.time()
            if version == "vanilla":
                _zerogpu_model.disable_adapter_layers()
                # Vanilla rambles up to 1400+ chars on a microscope image which
                # blows the 60s ZeroGPU budget; cap it tighter.
                _max_tok = 256
            else:
                _zerogpu_model.enable_adapter_layers()
                _zerogpu_model.set_adapter(version)
                _max_tok = 512
            with torch.inference_mode():
                out = _zerogpu_model.generate(
                    **inputs, max_new_tokens=_max_tok, do_sample=False,
                )
            gen_ids = out[0][prompt_len:]
            text = _zerogpu_processor.decode(gen_ids, skip_special_tokens=True).strip()
            results[version] = text
            print(f"[infer-all] {version} t+{_t.time()-t0:.2f}s len={len(text)}", flush=True)
        print(f"[infer-all] DONE total t+{_t.time()-t_total:.2f}s", flush=True)
        return results

    # ── Single-version path (legacy / local fallback). Still used when llama_server_call
    # is called outside the do_analyze HF-Space short-circuit (e.g. potential future paths).
    @spaces.GPU(duration=25)
    def _zerogpu_infer(version: str, image_data_uri: str, prompt: str) -> str:
        import time as _t
        t0 = _t.time()
        print(f"[infer] version={version} cuda={torch.cuda.is_available()} "
              f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
              flush=True)
        b64 = _strip_data_uri(image_data_uri) if image_data_uri.startswith("data:") else image_data_uri
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        if max(img.size) > 768:
            img.thumbnail((768, 768))
        print(f"[infer] image {img.size}", flush=True)
        if version == "vanilla":
            _zerogpu_model.disable_adapter_layers()
        else:
            _zerogpu_model.enable_adapter_layers()
            _zerogpu_model.set_adapter(version)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        inputs = _zerogpu_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        # Move to model device; only float tensors get bfloat16 cast.
        inputs = {k: (v.to(_zerogpu_model.device, dtype=torch.bfloat16) if v.is_floating_point()
                       else v.to(_zerogpu_model.device))
                  for k, v in inputs.items()}
        print(f"[infer] inputs ready, t+{_t.time()-t0:.2f}s, generating…", flush=True)
        with torch.inference_mode():
            out = _zerogpu_model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[0][prompt_len:]
        text = _zerogpu_processor.decode(gen_ids, skip_special_tokens=True)
        print(f"[infer] DONE t+{_t.time()-t0:.2f}s, gen_tokens={gen_ids.shape[0]}, "
              f"text_len={len(text)}, preview={text[:80]!r}", flush=True)
        return text.strip()

_URL_TO_KIND = {URL_VANILLA: "vanilla", URL_V2: "v2", URL_V3: "v3"}


# ─────────────────────────────────────────────────────────────────────────────
# QR codes for the footer install card. Generated once at module load.
# ─────────────────────────────────────────────────────────────────────────────
APK_URL = "https://huggingface.co/Laborator/microlens-final/resolve/main/android/microlens-android-v1.0.0.apk"
GITHUB_URL = "https://github.com/SergheiBrinza/microlens"

def _qr_data_uri(data: str, dark: str = "#FFFFFF", light: str = "#000000",
                  alpha: float = 1.0) -> str:
    try:
        import qrcode
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M,
                           box_size=10, border=2)
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color=dark, back_color=light).convert("RGBA")
        if alpha < 1.0:
            r, g, b, a = img.split()
            a = a.point(lambda x: int(x * alpha))
            img = Image.merge("RGBA", (r, g, b, a))
        buf = BytesIO()
        img.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[qr] failed for {data[:40]}: {e}", flush=True)
        return ""

QR_ANDROID = _qr_data_uri(APK_URL, dark="#FFFFFF", light="#000000")
QR_IOS     = _qr_data_uri(GITHUB_URL, dark="#666666", light="#0a0a0a")


def _data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    mime, _ = mimetypes.guess_type(str(path))
    return f"data:{mime or 'image/png'};base64,{base64.b64encode(path.read_bytes()).decode()}"

_THUMB_CACHE: Dict[str, str] = {}
def thumb_uri(filename: str) -> str:
    if filename not in _THUMB_CACHE:
        p = EXAMPLES_DIR / filename
        if p.exists():
            img = Image.open(p).convert("RGB")
            img.thumbnail((300, 300))
            buf = BytesIO()
            img.save(buf, "JPEG", quality=86)
            _THUMB_CACHE[filename] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        else:
            _THUMB_CACHE[filename] = ""
    return _THUMB_CACHE[filename]

_FULL_CACHE: Dict[str, str] = {}
def full_uri(filename: str) -> str:
    if filename not in _FULL_CACHE:
        _FULL_CACHE[filename] = _data_uri(EXAMPLES_DIR / filename)
    return _FULL_CACHE[filename]


SHAPE_CIRCLE = "circle"
SHAPE_SQUARE = "square"
MODE_SAMPLES = "samples"
MODE_UPLOAD = "upload"
MODE_MICRO = "micro"
ACCENT_RED = "#FF1744"
ACCENT_CYAN = "#7FE8E3"
ACCENT_GOLD = "#D4AF37"


def _strip_data_uri(data_uri: str) -> str:
    if data_uri.startswith("data:"):
        comma = data_uri.find(",")
        if comma >= 0:
            return data_uri[comma + 1:]
    return data_uri


def llama_server_call(url: str, image_data_uri: str,
                       prompt: str = INFERENCE_PROMPT,
                       timeout: int = 180) -> Tuple[str, Optional[str]]:
    """Returns (text, error_or_None).
    On HF Space: routes to in-process ZeroGPU inference (transformers + PEFT).
    Locally: OpenAI-compatible call to llama-server (original behavior)."""
    if IS_HF_SPACE:
        kind = _URL_TO_KIND.get(url, "vanilla")
        try:
            return _zerogpu_infer(kind, image_data_uri, prompt), None
        except Exception as e:
            return "", f"{type(e).__name__}: {str(e)[:240]}"
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        }],
        "max_tokens": 600,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "User-Agent": "MicroLens-Space/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
        return (text or "").strip(), None
    except urllib.error.HTTPError as e:
        body = ""
        try: body = e.read().decode("utf-8", errors="replace")
        except Exception: pass
        return "", f"HTTP {e.code}: {body[:240]}"
    except Exception as e:
        return "", f"{type(e).__name__}: {str(e)[:240]}"


def matte_btn_style(active: bool, *, padding: str = "16px 20px",
                    radius: str = "18px", min_w: str = "100px",
                    font_size: str = "12px") -> str:
    if active:
        return (f"cursor:pointer; padding:{padding}; border-radius:{radius};"
                f" background: linear-gradient(180deg, rgba(255,23,68,.18) 0%, rgba(255,23,68,.06) 100%);"
                f" border: 1.5px solid {ACCENT_RED};"
                f" color: #fff; font-weight: 800; letter-spacing: 3px;"
                f" font-size: {font_size}; font-family: 'Space Grotesk', sans-serif;"
                f" text-align:center; transition: all .15s ease; user-select:none;"
                f" min-width: {min_w};"
                f" box-shadow: 0 0 28px rgba(255,23,68,.45), inset 0 0 16px rgba(255,23,68,.12);")
    return (f"cursor:pointer; padding:{padding}; border-radius:{radius};"
            f" background: #0a0a0a;"
            f" border: 1.5px solid #2a2a2a;"
            f" color: #ffffff; font-weight: 800; letter-spacing: 3px;"
            f" font-size: {font_size}; font-family: 'Space Grotesk', sans-serif;"
            f" text-align:center; transition: all .15s ease; user-select:none;"
            f" min-width: {min_w};"
            f" box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);")


GRID_CELLS = [0, 2, 4, 6, 8]
N_GRID_LEVELS = len(GRID_CELLS)

def grid_overlay_html(level: int) -> str:
    if level <= 0 or level >= len(GRID_CELLS):
        return ""
    n = GRID_CELLS[level]
    pct = 100 / n
    op = 0.45 + level * 0.12
    line_w = 1 if level <= 2 else 1.5
    return f"""
    <div style="position:absolute;inset:0;pointer-events:none;
                background-image:
                  linear-gradient(rgba(0,0,0,{op*0.55}) {line_w+1}px, transparent {line_w+1}px),
                  linear-gradient(90deg, rgba(0,0,0,{op*0.55}) {line_w+1}px, transparent {line_w+1}px),
                  linear-gradient(rgba(127,232,227,{op}) {line_w}px, transparent {line_w}px),
                  linear-gradient(90deg, rgba(127,232,227,{op}) {line_w}px, transparent {line_w}px);
                background-size: {pct}% {pct}%, {pct}% {pct}%, {pct}% {pct}%, {pct}% {pct}%;"></div>"""


CROSS_OPACITY = [0, 0.55, 0.75, 0.90, 1.0]
CROSS_TICKS = [0, 6, 10, 14, 18]
N_CROSS_LEVELS = len(CROSS_OPACITY)

def cross_overlay_html(level: int) -> str:
    if level <= 0 or level >= len(CROSS_OPACITY):
        return ""
    op = CROSS_OPACITY[level]
    n = CROSS_TICKS[level]
    col = f"rgba(255,255,255,{op})"
    col_strong = f"rgba(255,255,255,{min(op*1.10, 1.0)})"
    halo_op = min(op * 0.55, 1.0)
    halo = f"rgba(0,0,0,{halo_op})"
    line_w = 2 if level >= 3 else 1
    halo_w = line_w + 2
    ticks = []
    for i in range(1, n + 1):
        pct = (i / (n + 1)) * 50
        is_major = (i % 4 == 0)
        size = 11 if is_major else 6
        c = col_strong if is_major else col
        for axis_left, axis_top, w_h in [
            ("50%", f"{50-pct}%", "h"), ("50%", f"{50+pct}%", "h"),
        ]:
            tw, th = (size, 1) if w_h == "h" else (1, size)
            ticks.append(f'<div style="position:absolute;left:{axis_left};top:{axis_top};'
                         f'width:{tw+2}px;height:{th+2}px;background:{halo};'
                         f'transform:translate(-50%,-50%);border-radius:1px;"></div>')
            ticks.append(f'<div style="position:absolute;left:{axis_left};top:{axis_top};'
                         f'width:{tw}px;height:{th}px;background:{c};'
                         f'transform:translate(-50%,-50%);"></div>')
        for axis_left, axis_top, w_h in [
            (f"{50-pct}%", "50%", "v"), (f"{50+pct}%", "50%", "v"),
        ]:
            tw, th = (1, size)
            ticks.append(f'<div style="position:absolute;left:{axis_left};top:{axis_top};'
                         f'width:{tw+2}px;height:{th+2}px;background:{halo};'
                         f'transform:translate(-50%,-50%);border-radius:1px;"></div>')
            ticks.append(f'<div style="position:absolute;left:{axis_left};top:{axis_top};'
                         f'width:{tw}px;height:{th}px;background:{c};'
                         f'transform:translate(-50%,-50%);"></div>')
    return f"""
    <div style="position:absolute;inset:0;pointer-events:none;">
      <div style="position:absolute;left:50%;top:0;bottom:0;width:{halo_w}px;background:{halo};
                   transform:translateX(-50%);"></div>
      <div style="position:absolute;top:50%;left:0;right:0;height:{halo_w}px;background:{halo};
                   transform:translateY(-50%);"></div>
      <div style="position:absolute;left:50%;top:0;bottom:0;width:{line_w}px;background:{col};
                   transform:translateX(-50%);box-shadow: 0 0 4px {col};"></div>
      <div style="position:absolute;top:50%;left:0;right:0;height:{line_w}px;background:{col};
                   transform:translateY(-50%);box-shadow: 0 0 4px {col};"></div>
      <div style="position:absolute;left:50%;top:50%;width:16px;height:16px;
                   transform:translate(-50%,-50%);
                   border:1.5px solid {col_strong};border-radius:50%;
                   background: rgba(0,0,0,{halo_op*0.7});"></div>
      {''.join(ticks)}
    </div>
    """


def viewport_html(image_data_uri: Optional[str], shape: str = SHAPE_SQUARE,
                  grid: int = 0, cross: int = 0,
                  empty_text: str = "PICK A SAMPLE FROM A CATEGORY",
                  live_video: bool = False) -> str:
    if live_video and not image_data_uri:
        # Live camera stream — JS sets video.srcObject after camera selection
        inner = ('<video id="ml-video" autoplay playsinline muted '
                 'style="width:100%;height:100%;object-fit:cover;display:block;'
                 'background:#000;"></video>')
    elif image_data_uri:
        inner = f'<img src="{image_data_uri}" style="width:100%;height:100%;object-fit:cover;display:block;" />'
    else:
        inner = (
            f'<div style="display:flex;align-items:center;justify-content:center;height:100%;'
            f'color:#5a5a62;font-size:12px;letter-spacing:5px;font-weight:800;text-align:center;'
            f'padding:32px;font-family:\'Space Grotesk\',sans-serif;">{empty_text}</div>'
        )
    radius = "50%" if shape == SHAPE_CIRCLE else "22px"
    overlays = grid_overlay_html(grid) + cross_overlay_html(cross)
    return f"""
    <div style="position:relative; margin: 0 auto; width: 100%; max-width: 560px;
                aspect-ratio: 1 / 1;
                border-radius: {radius}; overflow: hidden;
                background: #0a0a0a;
                border: 1.5px solid rgba(127, 232, 227, 0.55);
                box-shadow: 0 0 80px rgba(127, 232, 227, 0.18),
                            inset 0 0 0 1px rgba(255,255,255,0.04),
                            0 14px 50px rgba(0,0,0,0.7);">
        {inner}
        {overlays}
    </div>
    """


def folder_pills_html(active_label: str) -> str:
    pills = []
    for cid, lbl in CATEGORIES:
        is_active = (lbl == active_label)
        if is_active:
            pill_style = (
                "cursor:pointer; padding:10px 8px; border-radius:12px;"
                " background: linear-gradient(180deg, rgba(255,23,68,.20) 0%, rgba(255,23,68,.06) 100%);"
                f" border: 1.5px solid {ACCENT_RED};"
                " color: #fff; font-weight: 800; letter-spacing: 1.5px;"
                " font-size: 11px; font-family: 'Space Grotesk', sans-serif;"
                " text-align:center; transition: all .15s ease; user-select:none;"
                " white-space: nowrap;"
                " box-shadow: 0 0 22px rgba(255,23,68,.40), inset 0 0 12px rgba(255,23,68,.10);"
            )
        else:
            pill_style = (
                "cursor:pointer; padding:10px 8px; border-radius:12px;"
                " background: #0a0a0a;"
                f" border: 1.5px solid rgba(255,23,68,0.35);"
                " color: #cfcfcf; font-weight: 800; letter-spacing: 1.5px;"
                " font-size: 11px; font-family: 'Space Grotesk', sans-serif;"
                " text-align:center; transition: all .15s ease; user-select:none;"
                " white-space: nowrap;"
            )
        pills.append(f"""
        <div class="ml-folder-pill"
             data-mlaction="set" data-mltarget="hidden-cat" data-mlvalue="{lbl}"
             style="{pill_style} flex: 1 1 0; min-width: 0;">{lbl}</div>
        """)
    return f'<div class="ml-folder-pills-row" style="display:flex; flex-wrap:nowrap; gap:6px; width:100%;">{"".join(pills)}</div>'


def folder_html(category_label: str, picked_filename: Optional[str]) -> str:
    cid = CAT_BY_LABEL.get(category_label, "diatom")
    samples = [s for s in CATALOG if s["category"] == cid]
    if not samples:
        return '<div style="color:#666;text-align:center;padding:24px;">No samples</div>'
    cards = []
    for s in samples:
        is_sel = s["filename"] == picked_filename
        sel_style = (f"border-color: {ACCENT_RED}; "
                     f"box-shadow: 0 0 30px rgba(255,23,68,.55), 0 0 0 2px rgba(255,23,68,.6);"
                     if is_sel else "")
        sel_dot = (f'<div style="position:absolute;top:8px;right:8px;width:9px;height:9px;'
                   f'border-radius:50%;background:{ACCENT_RED};box-shadow:0 0 12px {ACCENT_RED};"></div>'
                   if is_sel else "")
        caption = (s.get("genus") or s["category"])[:16]
        cards.append(f"""
        <div data-mlaction="set" data-mltarget="hidden-pick" data-mlvalue="{s['filename']}"
             class="ml-folder-card" style="
            cursor:pointer; aspect-ratio: 1/1; border-radius: 14px;
            overflow:hidden; background: #0a0a0a; position: relative;
            border: 1px solid #1f1f1f;
            transition: all .18s ease; {sel_style}">
            <img src="{thumb_uri(s['filename'])}" style="width:100%; height:100%; object-fit:cover; display:block;"/>
            {sel_dot}
            <div style="position:absolute; bottom:0; left:0; right:0;
                background: linear-gradient(to top, rgba(0,0,0,0.92) 0%, rgba(0,0,0,0.4) 60%, transparent 100%);
                color:#ffffff; font-size:9px; font-weight:800; letter-spacing:2px;
                padding: 18px 6px 8px; text-align:center; text-transform:uppercase;
                font-family: 'Space Grotesk', sans-serif;">{caption}</div>
        </div>
        """)
    return f'<div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:10px;">{"".join(cards)}</div>'


def mode_buttons_html(active: str) -> str:
    btns = [
        (MODE_MICRO,   "🔬  MICROSCOPE"),
        (MODE_UPLOAD,  "📤  UPLOAD"),
        (MODE_SAMPLES, "📂  SAMPLES"),
    ]
    out = []
    for m, lbl in btns:
        is_a = (m == active)
        style = matte_btn_style(is_a, padding="13px 28px", radius="16px",
                                min_w="170px", font_size="13px")
        out.append(f"""
        <div class="ml-mode-btn"
             data-mlaction="set" data-mltarget="hidden-mode" data-mlvalue="{m}"
             style="{style} letter-spacing: 3px;">{lbl}</div>
        """)
    return f'<div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center; justify-content: center;">{"".join(out)}</div>'


def tool_buttons_html(shape: str, grid: int, cross: int) -> str:
    out = []
    seg_inner = []
    for s, lbl in [(SHAPE_CIRCLE, "○  CIRCLE"), (SHAPE_SQUARE, "□  SQUARE")]:
        is_a = (s == shape)
        if is_a:
            inner_style = ("background: linear-gradient(180deg, rgba(255,23,68,.22) 0%, rgba(255,23,68,.08) 100%);"
                           " color:#fff; box-shadow: 0 0 18px rgba(255,23,68,.45),"
                           " inset 0 0 12px rgba(255,23,68,.15); border: 1px solid rgba(255,23,68,.7);")
        else:
            inner_style = "background: transparent; color:#cfcfcf; border: 1px solid transparent;"
        seg_inner.append(f"""
        <div data-mlaction="set" data-mltarget="hidden-shape" data-mlvalue="{s}"
             class="ml-seg-item" style="{inner_style}
            cursor:pointer; padding: 10px 22px; border-radius: 12px;
            font-family: 'Space Grotesk', sans-serif; font-weight: 800;
            letter-spacing: 3px; font-size: 12px;
            min-width: 130px; text-align: center;
            transition: all .15s ease; user-select:none;">{lbl}</div>
        """)
    out.append(f"""
    <div class="ml-shape-seg" style="
        display: inline-flex; gap: 6px; padding: 5px;
        background: #060606; border: 1.5px solid #2a2a2a;
        border-radius: 16px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);">{"".join(seg_inner)}</div>
    """)

    grid_active = (grid > 0)
    grid_label = "▦  GRID" if grid == 0 else f"▦  GRID · {GRID_CELLS[grid]}×{GRID_CELLS[grid]}"
    grid_style = matte_btn_style(grid_active, padding="13px 24px", radius="16px",
                                 min_w="140px", font_size="12px")
    out.append(f"""
    <div class="ml-mode-btn"
         data-mlaction="cycle" data-mltarget="hidden-grid" data-mlmax="{N_GRID_LEVELS}"
         style="{grid_style} letter-spacing: 3px;">{grid_label}</div>
    """)

    cross_active = (cross > 0)
    cross_label = "✚  CROSS" if cross == 0 else f"✚  CROSS · {cross}/4"
    cross_style = matte_btn_style(cross_active, padding="13px 24px", radius="16px",
                                  min_w="140px", font_size="12px")
    out.append(f"""
    <div class="ml-mode-btn"
         data-mlaction="cycle" data-mltarget="hidden-cross" data-mlmax="{N_CROSS_LEVELS}"
         style="{cross_style} letter-spacing: 3px;">{cross_label}</div>
    """)

    return f'<div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center; justify-content: center;">{"".join(out)}</div>'


def upload_zone_html() -> str:
    return f"""
    <div style="display:flex; flex-direction:column; gap:14px; font-family: 'Space Grotesk', sans-serif;">
        <div class="ml-display" style="font-size:11px; font-weight:800;
                    letter-spacing:5px; color:{ACCENT_CYAN};
                    text-transform:uppercase; text-align:center;">Upload Microscope Image</div>
        <div style="position:relative; padding: 36px 24px; border-radius: 18px;
                    background: linear-gradient(180deg, #0c0c0c 0%, #060606 100%);
                    border: 1.5px dashed rgba(127,232,227,0.35);
                    text-align:center;">
            <div style="font-size:38px; margin-bottom:10px;">&#128194;</div>
            <div style="color:#fff; font-weight:800; font-size:13px;
                        letter-spacing:3px; margin-bottom:6px;">DROP YOUR IMAGE HERE</div>
            <div style="color:#888; font-size:11px; letter-spacing:1.5px;">or use the BROWSE button below</div>
            <div style="margin-top:14px; color:#555; font-size:10px;
                        letter-spacing:2px;">PNG &middot; JPG &middot; TIFF &middot; BMP &middot; up to 20 MB</div>
        </div>
    </div>
    """


def camera_list_html() -> str:
    # Initial state rendered directly in Python so it survives Gradio re-renders.
    # The <video> element lives in the LEFT viewport (live_video=True);
    # this panel only has the camera selector + CAPTURE button.
    return f"""
    <div style="display:flex; flex-direction:column; gap:12px; font-family: 'Space Grotesk', sans-serif;">
        <div class="ml-display" style="font-size:11px; font-weight:800;
                    letter-spacing:5px; color:{ACCENT_CYAN};
                    text-transform:uppercase; text-align:center;">Camera</div>
        <div id="ml-detected-cams" style="display:flex; flex-direction:column; gap:6px;">
            <div style="color:#aaa; font-size:11.5px; padding: 18px; text-align:center;
                        background: #0a0a0a; border:1.5px dashed #2a2a2a; border-radius: 14px;
                        letter-spacing:0.5px; line-height:1.6;">
                <div style="font-size:24px; margin-bottom:8px;">🎥</div>
                <div style="color:#fff; font-weight:800; letter-spacing:2px;
                            font-size:11px; text-transform:uppercase; margin-bottom:6px;">
                    Enable camera
                </div>
                <div style="color:#888; font-size:10.5px; margin-bottom:14px;">
                    we will detect built-in webcam, USB camera and any microscope
                </div>
                <div data-mlaction="enable-camera" style="
                    display:inline-block; cursor:pointer;
                    padding: 10px 22px; border-radius: 999px;
                    background: linear-gradient(180deg, rgba(255,23,68,.18) 0%, rgba(255,23,68,.05) 100%);
                    border: 1.5px solid #FF1744;
                    color: #fff; font-weight: 800; letter-spacing: 3px;
                    font-size: 11px; box-shadow: 0 0 18px rgba(255,23,68,.40);
                    user-select: none;">🎥  ENABLE CAMERA</div>
            </div>
        </div>
        <div id="ml-capture-btn-wrap" style="display:none; flex-direction:column; gap:8px; margin-top:6px;">
            <div id="ml-action-btn" data-mlaction="capture" class="ml-capture-btn" style="
                cursor:pointer; padding: 22px 18px; border-radius: 14px;
                background: linear-gradient(180deg, rgba(0,220,230,.22) 0%, rgba(0,220,230,.06) 100%);
                border: 2px solid rgba(0,220,230,0.65);
                color: #fff; font-weight: 900; letter-spacing: 5px;
                font-size: 17px; text-align: center; user-select: none;
                font-family: 'Space Grotesk', sans-serif;
                box-shadow: 0 0 24px rgba(0,220,230,.40),
                            inset 0 0 16px rgba(0,220,230,.12);
                transition: all 0.3s ease;">📸  SNAPSHOT</div>
        </div>
    </div>
    """


CAMERA_JS = r"""
async () => {
    if (window.mlCameraInit) return;
    window.mlCameraInit = true;

    function setHidden(id, value) {
        const el = document.getElementById(id);
        if (!el) return;
        const input = el.querySelector('input, textarea');
        if (!input) return;
        input.value = value;
        input.dispatchEvent(new Event('input', {bubbles: true}));
    }
    window.mlSetHidden = setHidden;

    // CAPTURE PHASE: intercept AI ANALYZE clicks in MICRO mode.
    // Two-phase: BLOCK first click → write fresh frame to state → wait 500ms
    // for Gradio state sync → re-click button (bypassing the intercept flag).
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('button, .analyze-btn');
        if (!btn) return;
        const txt = (btn.innerText || btn.textContent || '').toUpperCase();
        if (!txt.includes('AI ANALYZE')) return;
        // Skip if this is our own programmatic re-click
        if (btn.dataset.mlPassThrough === '1') {
            btn.dataset.mlPassThrough = '';
            return;
        }
        // Only intercept in MICRO mode with active stream
        const modeEl = document.getElementById('hidden-mode');
        if (!modeEl) return;
        const modeInput = modeEl.querySelector('input, textarea');
        if (!modeInput || modeInput.value !== 'micro') return;
        if (!window.mlStream) return;
        const frameEl = document.getElementById('hidden-cam-frame');
        if (!frameEl) return;
        const frameInput = frameEl.querySelector('input, textarea');
        // User already snapshotted? Let normal flow proceed
        if (frameInput && frameInput.value && frameInput.value.length > 100) return;
        const v = document.getElementById('ml-video');
        if (!v || !v.videoWidth || !v.videoHeight) return;

        // BLOCK this click — we'll capture and re-click after state propagates
        e.stopImmediatePropagation();
        e.preventDefault();

        const c = document.createElement('canvas');
        c.width = v.videoWidth; c.height = v.videoHeight;
        c.getContext('2d').drawImage(v, 0, 0);
        const dataUri = c.toDataURL('image/jpeg', 0.85);
        const setter = Object.getOwnPropertyDescriptor(
            frameInput.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype
                                              : HTMLInputElement.prototype,
            'value').set;
        setter.call(frameInput, dataUri);
        frameInput.dispatchEvent(new Event('input', {bubbles: true}));
        frameInput.dispatchEvent(new Event('change', {bubbles: true}));
        console.log('[ml] auto-snapshot, waiting 500ms for state propagation, then re-click...');

        // Wait for Gradio to round-trip the state update, then re-click programmatically.
        setTimeout(() => {
            btn.dataset.mlPassThrough = '1';
            btn.click();
            console.log('[ml] re-clicked AI ANALYZE with fresh frame in state');
        }, 600);
    }, true);  // capture phase = true (before bubble)

    // Event delegation for all data-mlaction clicks
    document.addEventListener('click', (e) => {
        const el = e.target.closest('[data-mlaction]');
        if (!el) return;
        const action = el.dataset.mlaction;
        if (action === 'set') {
            setHidden(el.dataset.mltarget, el.dataset.mlvalue);
        } else if (action === 'cycle') {
            const target = el.dataset.mltarget;
            const max = parseInt(el.dataset.mlmax || '5');
            const elInput = document.getElementById(target);
            if (!elInput) return;
            const input = elInput.querySelector('input, textarea');
            if (!input) return;
            const cur = parseInt(input.value || '0');
            const next = (cur + 1) % max;
            input.value = String(next);
            input.dispatchEvent(new Event('input', {bubbles: true}));
        } else if (action === 'capture') {
            window.mlCaptureFrame && window.mlCaptureFrame();
        } else if (action === 'enable-camera') {
            window.mlEnableCamera && window.mlEnableCamera();
        } else if (action === 'select-camera') {
            window.mlSelectCamera && window.mlSelectCamera(el.dataset.mlvalue);
        } else if (action === 'toggle-drivers') {
            const panel = document.getElementById('ml-driver-panel');
            if (panel) panel.style.display = (panel.style.display === 'none' ? 'block' : 'none');
        } else if (action === 'reset-camera') {
            // Clear captured frame state and trigger viewport re-render with live video
            setHidden('hidden-cam-frame', '');
            // Also blink the panels to ready/empty state
            console.log('[ml] camera reset — resuming live video');
        }
    });

    // ---- Device + camera classification ----
    function detectDevice() {
        const ua = navigator.userAgent.toLowerCase();
        if (/mobi|android|iphone|ipod|ipad|blackberry|windows phone/.test(ua)) return 'mobile';
        if (/macintosh|mac os x/.test(ua)) return 'desktop-mac';
        return 'desktop';
    }
    window.mlDevice = detectDevice();

    function classifyCamera(label) {
        const l = (label || '').toLowerCase();
        if (/touptek|toupcam|toup|amscope|motic|leica|zeiss|olympus|nikon eclipse|hikvision|hik vision|mvc|flir|basler|swiftcam|moticam|optika|infinity|trinocular|microscope|magnification|infinx|euromex|labomed/.test(l)) {
            return { type: 'microscope', icon: '🔬', tag: 'MICROSCOPE' };
        }
        if (/facetime|integrated|built-?in|isight|emeet|image c[A-Za-z]+ camera/.test(l)) {
            return { type: 'builtin', icon: '💻', tag: 'BUILT-IN' };
        }
        if (/logitech|c920|c922|c930|brio|streamcam|webcam|usb camera|usb video|hd pro|hd webcam/.test(l)) {
            return { type: 'webcam', icon: '📷', tag: 'USB WEBCAM' };
        }
        if (/back|environment|world|rear/.test(l)) {
            return { type: 'phone-back', icon: '📱', tag: 'PHONE BACK' };
        }
        if (/front|user|selfie/.test(l)) {
            return { type: 'phone-front', icon: '🤳', tag: 'PHONE FRONT' };
        }
        return { type: 'generic', icon: '📹', tag: 'CAMERA' };
    }
    function emptyState(emoji, title, sub, actionHtml) {
        return `<div style="color:#aaa; font-size:11.5px; padding: 18px;
                             text-align:center; line-height:1.6; background:#0a0a0a;
                             border:1.5px dashed #2a2a2a; border-radius:14px; letter-spacing:0.5px;">
                    <div style="font-size:24px; margin-bottom:8px;">${emoji}</div>
                    <div style="color:#fff; font-weight:800; letter-spacing:2px;
                                font-size:11px; text-transform:uppercase; margin-bottom:6px;">${title}</div>
                    <div style="color:#888; font-size:10.5px; margin-bottom: ${actionHtml ? '12px' : '0'};">${sub}</div>
                    ${actionHtml || ''}
                </div>`;
    }
    function btnHtml(label, action) {
        // action = 'enable-camera' | 'refresh' etc — handled via data-mlaction delegation
        return `<div data-mlaction="${action}" style="
            display:inline-block; cursor:pointer;
            padding: 10px 22px; border-radius: 999px;
            background: linear-gradient(180deg, rgba(255,23,68,.18) 0%, rgba(255,23,68,.05) 100%);
            border: 1.5px solid #FF1744;
            color: #fff; font-weight: 800; letter-spacing: 3px;
            font-size: 11px; font-family: 'Space Grotesk', sans-serif;
            box-shadow: 0 0 18px rgba(255,23,68,.40); user-select: none;">${label}</div>`;
    }
    function renderRow(cam, idx, active) {
        const safeLabel = (cam.label || ('Camera ' + (idx + 1))).replace(/[<>]/g, '').slice(0, 42);
        const cls = classifyCamera(cam.label || '');
        const accent = active ? '#FF1744' : '#202020';
        const bg = active
          ? 'linear-gradient(180deg, rgba(255,23,68,.10) 0%, rgba(255,23,68,.02) 100%)'
          : '#0a0a0a';
        const glow = active ? '0 0 16px rgba(255,23,68,.28)' : 'none';
        const pill = active
          ? `<span style="display:inline-flex; align-items:center; flex:0 0 auto;
                          padding: 4px 10px; border-radius: 999px;
                          background: rgba(255,23,68,0.10); border: 1px solid rgba(255,23,68,0.50);
                          color: #FF8A8A; font-weight:800; font-size:9px;
                          letter-spacing:2px;">● ACTIVE</span>`
          : `<span style="display:inline-flex; align-items:center; flex:0 0 auto;
                          padding: 4px 10px; border-radius: 999px;
                          background: rgba(0,230,118,0.08); border: 1px solid rgba(0,230,118,0.45);
                          color: #00E676; font-weight:800; font-size:9px;
                          letter-spacing:2px;">● READY</span>`;
        const tagPill = `<span style="display:inline-block;
                          padding: 2px 8px; border-radius: 999px;
                          background: rgba(127,232,227,0.08);
                          border: 1px solid rgba(127,232,227,0.30);
                          color: #7FE8E3; font-weight:800; font-size:8px;
                          letter-spacing:1.5px; margin-top:3px;">${cls.tag}</span>`;
        return `<div class="ml-cam-row"
              data-mlaction="select-camera" data-mlvalue="${cam.deviceId}"
              style="
              display:flex; align-items:center; gap:12px;
              padding: 10px 14px; border-radius: 12px;
              background: ${bg}; border: 1.5px solid ${accent};
              box-shadow: ${glow}; cursor:pointer; transition: all .15s ease;
              font-family: 'Space Grotesk', sans-serif;">
              <div style="font-size:22px; line-height:1; width:30px; text-align:center;">${cls.icon}</div>
              <div style="flex:1 1 auto; min-width:0;">
                  <div style="color:#fff; font-weight:700; font-size:11.5px;
                              letter-spacing:0.3px; overflow:hidden;
                              text-overflow:ellipsis; white-space:nowrap;">${safeLabel}</div>
                  ${tagPill}
              </div>${pill}</div>`;
    }

    // Microscope camera SDK download links — shown when no camera detected
    function driverPanelHtml() {
        const drivers = [
            ['ToupTek / ToupCam',  'UCMOS · UHCCD · UA · UC · AmScope MU',  'https://www.touptek.com/download/'],
            ['AmScope MU series',   '1803 · 5-20 MP · ToupTek-based',         'https://www.amscope.com/pages/software-downloads'],
            ['Motic MoticamX',      'S-series CMOS pro/edu',                  'https://www.motic.com/As_Service_Download/'],
            ['Hikvision MV',        'GigE Vision · industrial',               'https://www.hikrobotics.com/en/machinevision/service/download'],
            ['FLIR Spinnaker',      'Blackfly S · USB3 Vision',               'https://www.flir.com/products/spinnaker-sdk/'],
            ['Basler pylon',        'ace 2 · industrial',                     'https://www.baslerweb.com/en/downloads/software-downloads/'],
        ];
        const rows = drivers.map(([name, desc, url]) => `
            <a href="${url}" target="_blank" rel="noopener" style="
                display:flex; align-items:center; gap:10px;
                padding: 8px 12px; border-radius: 10px;
                background: #0a0a0a; border: 1px solid #1f1f1f;
                text-decoration:none; transition: all .15s ease;
                margin-bottom: 4px;">
                <div style="font-size:18px;">🔬</div>
                <div style="flex:1; min-width:0;">
                    <div style="color:#fff; font-weight:800; font-size:10.5px;
                                letter-spacing:1.5px;">${name}</div>
                    <div style="color:#888; font-size:9.5px; margin-top:1px;">${desc}</div>
                </div>
                <div style="color:#7FE8E3; font-size:11px; font-weight:800;
                            letter-spacing:1.5px;">DOWNLOAD ↗</div>
            </a>`).join('');
        return `<div id="ml-driver-panel" style="display:none; margin-top:10px;
                background:#060606; border:1px solid #181818; border-radius:14px;
                padding:12px; font-family: 'Space Grotesk', sans-serif;">
            <div style="color:#aaa; font-size:10px; font-weight:700; letter-spacing:2px;
                        text-transform:uppercase; margin-bottom: 8px;">
                Microscope camera drivers
            </div>
            ${rows}
            <div style="color:#FFD180; font-size:10px; margin-top:10px; padding: 8px 10px;
                        background: rgba(255,193,7,0.05); border:1px solid rgba(255,193,7,0.20);
                        border-radius: 8px; line-height:1.55;">
                <b>⚠️ Important:</b> ToupTek-based cameras (AmScope MU, ToupCam UCMOS,
                MoticamX) use proprietary SDKs and <b>are NOT accessible via browser</b>.
                They only work in our desktop app or Android APK with the vendor driver
                installed.
            </div>
            <div style="color:#666; font-size:9.5px; margin-top:6px; line-height:1.5;">
                For browser use: connect a UVC-class USB webcam, or use built-in / phone camera.
            </div>
        </div>`;
    }
    function showInitialPrompt() {
        const container = document.getElementById('ml-detected-cams');
        if (!container) return;
        let title, sub, btnLabel;
        if (window.mlDevice === 'mobile') {
            title = '📱 Use your phone camera';
            sub = 'tap below — choose front or back camera in the next prompt';
            btnLabel = '🎥  ENABLE CAMERA';
        } else {
            title = '🎥 Enable camera';
            sub = 'we will detect built-in webcam, USB camera, and any connected microscope';
            btnLabel = '🎥  ENABLE CAMERA';
        }
        container.innerHTML = emptyState('🎥', title, sub, btnHtml(btnLabel, 'enable-camera'));
    }
    async function enumerateAndRender() {
        const container = document.getElementById('ml-detected-cams');
        if (!container) return;
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cams = devices.filter(d => d.kind === 'videoinput');
        if (cams.length === 0) {
            const isMobile = window.mlDevice === 'mobile';
            const sub = isMobile
              ? 'no front/back camera available — check phone permissions'
              : 'no webcam, USB camera or microscope detected — see driver list below';
            const driverBtn = isMobile ? '' : `
                <div data-mlaction="toggle-drivers" style="
                    display:inline-block; cursor:pointer; margin-left:8px;
                    padding: 10px 22px; border-radius: 999px;
                    background: #0a0a0a; border: 1.5px solid #2a2a2a;
                    color: #fff; font-weight: 800; letter-spacing: 3px;
                    font-size: 11px; font-family: 'Space Grotesk', sans-serif;
                    user-select: none;">🔬  MICROSCOPE DRIVERS</div>`;
            container.innerHTML =
                emptyState('📹', 'No camera detected', sub,
                    btnHtml('↻  RETRY', 'enable-camera') + driverBtn) +
                driverPanelHtml();
            return;
        }
        // Pick best default camera: prefer microscope > USB webcam > built-in > phone-back > generic
        const priority = {microscope: 0, webcam: 1, builtin: 2, 'phone-back': 3,
                          'phone-front': 5, generic: 4};
        let best = cams[0];
        for (const c of cams) {
            const pa = priority[classifyCamera(c.label).type] ?? 9;
            const pb = priority[classifyCamera(best.label).type] ?? 9;
            if (pa < pb) best = c;
        }
        const active = window.mlActiveCam || best.deviceId;

        // Count by category for smart header
        const counts = {microscope: 0, webcam: 0, builtin: 0, phone: 0, other: 0};
        cams.forEach(c => {
            const t = classifyCamera(c.label).type;
            if (t === 'microscope') counts.microscope++;
            else if (t === 'webcam') counts.webcam++;
            else if (t === 'builtin') counts.builtin++;
            else if (t.startsWith('phone')) counts.phone++;
            else counts.other++;
        });
        const summary = [];
        if (counts.microscope) summary.push(`<b style="color:#FF8A8A;">🔬 ${counts.microscope} microscope</b>`);
        if (counts.webcam)     summary.push(`<b style="color:#7FE8E3;">📷 ${counts.webcam} webcam</b>`);
        if (counts.builtin)    summary.push(`<b style="color:#7FE8E3;">💻 ${counts.builtin} built-in</b>`);
        if (counts.phone)      summary.push(`<b style="color:#7FE8E3;">📱 ${counts.phone} phone</b>`);
        if (counts.other)      summary.push(`<b style="color:#aaa;">📹 ${counts.other} camera</b>`);
        const summaryHtml = summary.length
          ? `<span style="color:#888; font-size:10px; letter-spacing:1px;
                          font-family: 'Space Grotesk', sans-serif;">
                ${summary.join(' &middot; ')}
                ${cams.length > 1 ? '<span style="color:#555;"> · click to switch</span>' : ''}
             </span>`
          : '';

        const header = `
          <div style="display:flex; align-items:center; justify-content:space-between;
                      padding: 4px 4px 6px; font-family: 'Space Grotesk', sans-serif;">
            ${summaryHtml}
            <span data-mlaction="enable-camera" style="
                cursor:pointer; padding: 4px 12px; border-radius: 999px;
                background: #0a0a0a; border: 1px solid #2a2a2a;
                color: #aaa; font-weight: 800; letter-spacing: 2px;
                font-size: 9px;">↻ REFRESH</span>
          </div>`;
        const rowsHtml = cams.map((c, i) => renderRow(c, i, c.deviceId === active)).join('');
        container.innerHTML = header + rowsHtml;
        // Click handling via global data-mlaction="select-camera" delegation.
        if (!window.mlStream) window.mlSelectCamera(active);
    }
    window.mlEnableCamera = async function() {
        const container = document.getElementById('ml-detected-cams');
        if (container) container.innerHTML = emptyState('⏳', 'Asking permission…',
            'click ALLOW in the browser prompt');
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
            const c = document.getElementById('ml-detected-cams');
            const isHttps = location.protocol === 'https:';
            const isLocal = ['localhost', '127.0.0.1', '0.0.0.0'].includes(location.hostname);
            if (!isHttps && !isLocal) {
                if (c) c.innerHTML = emptyState('🔒', 'HTTPS required',
                    `browsers block camera over plain HTTP from ${location.hostname}. ` +
                    `Open via <b style="color:#7FE8E3;">http://localhost:7861</b> on the server, ` +
                    `or set up an HTTPS tunnel (Cloudflare / ngrok) — then camera will appear.`,
                    btnHtml('↻  RETRY', 'enable-camera'));
            } else {
                if (c) c.innerHTML = emptyState('⚠️', 'Browser unsupported',
                    'this browser does not expose mediaDevices API');
            }
            return;
        }
        try {
            const tmp = await navigator.mediaDevices.getUserMedia({video: true});
            tmp.getTracks().forEach(t => t.stop());
        } catch (err) {
            const container = document.getElementById('ml-detected-cams');
            if (container) container.innerHTML = emptyState('🔒', 'Camera blocked',
                'open browser settings → allow camera for this site → click retry',
                btnHtml('↻  RETRY', 'enable-camera'));
            return;
        }
        await enumerateAndRender();
    };
    window.mlSelectCamera = async function(deviceId) {
        const video = document.getElementById('ml-video');
        const captureWrap = document.getElementById('ml-capture-btn-wrap');
        if (window.mlStream) {
            window.mlStream.getTracks().forEach(t => t.stop());
            window.mlStream = null;
        }
        try {
            window.mlStream = await navigator.mediaDevices.getUserMedia({
                video: {deviceId: {exact: deviceId}}
            });
            if (video) video.srcObject = window.mlStream;
            if (captureWrap) captureWrap.style.display = 'block';
            window.mlActiveCam = deviceId;
            await enumerateAndRender();
        } catch (e) { console.error('camera select failed', e); }
    };
    window.mlCaptureFrame = function() {
        const video = document.getElementById('ml-video');
        if (!video) { console.error('[ml] no video el'); return; }
        if (!video.videoWidth || !video.videoHeight) {
            console.warn('[ml] video not ready, retrying in 200ms');
            setTimeout(() => window.mlCaptureFrame(), 200);
            return;
        }
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const dataUri = canvas.toDataURL('image/jpeg', 0.85);
        console.log('[ml] SNAPSHOT', canvas.width + 'x' + canvas.height, '·', dataUri.length, 'bytes');

        // Set hidden textbox value (this freezes the frame in viewport)
        setHiddenStrong('hidden-cam-frame', dataUri);
    };
    // When viewport re-renders (shape/mode change), the <video> element is
    // recreated as empty. Periodic check re-attaches stream — cheaper than
    // a full-body MutationObserver.
    setInterval(() => {
        if (!window.mlStream) return;
        const v = document.getElementById('ml-video');
        if (v && v.srcObject !== window.mlStream) {
            v.srcObject = window.mlStream;
            v.play().catch(() => {});
        }
    }, 800);

    setInterval(() => {
        if (document.getElementById('ml-detected-cams') && !document.querySelector('#ml-detected-cams [data-mlaction="enable-camera"]')) {
            const c = document.getElementById('ml-detected-cams');
            if (c && c.children.length === 0) showInitialPrompt();
        }
    }, 2500);

    // Auto-enable camera when user switches to MICROSCOPE mode,
    // and auto-disable when leaving (releases device + drops indicator to OFF).
    setInterval(() => {
        const el = document.getElementById('hidden-mode');
        if (!el) return;
        const input = el.querySelector('input, textarea');
        if (!input) return;
        const mode = input.value;
        if (mode === 'micro' && !window.mlStream && !window.mlAutoTried) {
            window.mlAutoTried = true;
            setTimeout(() => {
                if (window.mlEnableCamera) window.mlEnableCamera();
            }, 400);
        }
        if (mode !== 'micro') {
            window.mlAutoTried = false;
            // Leaving MICRO mode — stop stream, free device
            if (window.mlStream) {
                window.mlStream.getTracks().forEach(t => t.stop());
                window.mlStream = null;
                console.log('[ml] stream stopped (left MICRO mode)');
            }
        }
    }, 1200);

    // (removed: manual toggle / ON-OFF chip — MICROSCOPE button itself is the toggle)

    // Periodic frame snapshot — every 1.2s while in MICRO mode with active
    // stream, push current video frame into hidden-cam-frame state. This way
    // AI ANALYZE always has a fresh frame in viewport_uri without needing
    // explicit CAPTURE click.
    function setHiddenStrong(id, value) {
        const el = document.getElementById(id);
        if (!el) return false;
        const input = el.querySelector('input, textarea');
        if (!input) return false;
        if (input.value === value) return true; // unchanged
        const setter = Object.getOwnPropertyDescriptor(
            input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype
                                         : HTMLInputElement.prototype,
            'value').set;
        setter.call(input, value);
        input.dispatchEvent(new Event('input', {bubbles: true}));
        input.dispatchEvent(new Event('change', {bubbles: true}));
        return true;
    }

    // Periodic auto-capture — silently sync latest live frame to backend state
    // so AI ANALYZE always has fresh data without explicit user click.
    setInterval(() => {
        if (!window.mlStream) return;
        const modeEl = document.getElementById('hidden-mode');
        if (!modeEl) return;
        const modeInput = modeEl.querySelector('input, textarea');
        if (!modeInput || modeInput.value !== 'micro') return;
        const frameEl = document.getElementById('hidden-cam-frame');
        if (!frameEl) return;
        const frameInput = frameEl.querySelector('input, textarea');
        // If user already snapshot-locked a frame (non-empty value),
        // don't overwrite — they're inspecting it.
        if (frameInput && frameInput.value && frameInput.value.length > 100) return;
        const v = document.getElementById('ml-video');
        if (!v || !v.videoWidth || !v.videoHeight) return;
        const c = document.createElement('canvas');
        const scale = Math.min(800 / v.videoWidth, 800 / v.videoHeight, 1);
        c.width = Math.round(v.videoWidth * scale);
        c.height = Math.round(v.videoHeight * scale);
        c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
        // Hidden internal sync — sets a non-tracked window var so analyze can grab it
        window.mlLiveLatestFrame = c.toDataURL('image/jpeg', 0.78);
        // Free canvas memory
        c.width = 0; c.height = 0;
    }, 2000);

    // Dynamic action button — toggle between SNAPSHOT (live) and NEW PHOTO (captured)
    setInterval(() => {
        const btn = document.getElementById('ml-action-btn');
        if (!btn) return;
        const frameEl = document.getElementById('hidden-cam-frame');
        const v = document.getElementById('ml-video');
        const hasVideo = v && !!v.srcObject;
        const hasCapture = frameEl && frameEl.querySelector('input,textarea') &&
                           frameEl.querySelector('input,textarea').value.length > 100;
        if (hasCapture) {
            // Captured state — button becomes "NEW PHOTO" (resume live)
            if (btn.dataset.mlaction !== 'reset-camera') {
                btn.dataset.mlaction = 'reset-camera';
                btn.innerHTML = '🔄  NEW PHOTO';
                btn.style.background = '#0a0a0a';
                btn.style.border = '2px solid #FF1744';
                btn.style.color = '#fff';
                btn.style.boxShadow = '0 0 24px rgba(255,23,68,.45), inset 0 0 16px rgba(255,23,68,.12)';
            }
        } else if (hasVideo) {
            if (btn.dataset.mlaction !== 'capture') {
                btn.dataset.mlaction = 'capture';
                btn.innerHTML = '📸  SNAPSHOT';
                btn.style.background = 'linear-gradient(180deg, rgba(0,220,230,.22) 0%, rgba(0,220,230,.06) 100%)';
                btn.style.border = '2px solid rgba(0,220,230,0.65)';
                btn.style.color = '#fff';
                btn.style.boxShadow = '0 0 24px rgba(0,220,230,.40), inset 0 0 16px rgba(0,220,230,.12)';
            }
        }
    }, 1000);
}
"""


PANEL_THEMES = {
    "vanilla": {
        "title": "UNTRAINED BASELINE",
        "subtitle": "Stock Gemma 4 E2B · Google factory weights · no microscopy training",
        "stripe": "linear-gradient(90deg, #C0C5CC 0%, #7A7E85 100%)",
        "title_grad": "linear-gradient(90deg, #E0E5EC 0%, #9A9EA5 100%)",
        "border": "rgba(180,185,195,0.35)",
        "glow": "0 0 32px rgba(180,185,195,0.10)",
        "glow_strong": "0 0 56px rgba(200,205,215,0.28)",
        "subtitle_color": "#9aa0a8",
    },
    "v2": {
        "title": "MICROLENS · BRIEF",
        "subtitle": "Gemma 4 E2B · 75k microscopy VQA · 95 genera · production single-line answer",
        "stripe": "linear-gradient(90deg, #00DCE6 0%, #007680 100%)",
        "title_grad": "linear-gradient(90deg, #00DCE6 0%, #66EAF0 100%)",
        "border": "rgba(0,220,230,0.45)",
        "glow": "0 0 36px rgba(0,220,230,0.18)",
        "glow_strong": "0 0 64px rgba(0,220,230,0.42)",
        "subtitle_color": "#7FBEC4",
    },
    "v3": {
        "title": "MICROLENS · RICH",
        "subtitle": "v2 + KB-augmented epoch · genus + morphology + habitat + ID cues, end-to-end",
        "stripe": "linear-gradient(90deg, #FF1744 0%, #800020 100%)",
        "title_grad": "linear-gradient(90deg, #FF5252 0%, #FF8888 100%)",
        "border": "rgba(255,23,68,0.45)",
        "glow": "0 0 36px rgba(255,23,68,0.18)",
        "glow_strong": "0 0 64px rgba(255,23,68,0.42)",
        "subtitle_color": "#C28A8A",
    },
}


def panel_html(kind: str, body: str, state: str = "ready", footer_text: Optional[str] = None) -> str:
    t = PANEL_THEMES[kind]
    if state == "empty":
        body_html = ('<div style="color:#5a5a62; font-size:14px; font-style:italic; '
                     'font-family: \'Space Grotesk\', sans-serif; '
                     'padding: 28px 0; text-align:center;">'
                     '— press AI ANALYZE to run the model —</div>')
    elif state == "typing":
        body_html = f'<div class="ml-result-body ml-caret">{body}</div>'
    else:
        body_html = f'<div class="ml-result-body">{body}</div>'
    footer_html = ""
    if footer_text:
        footer_html = (f'<div style="margin-top:14px; padding-top:10px; '
                       f'border-top:1px solid rgba(255,255,255,0.06); '
                       f'color:#aaa; font-size:11px; letter-spacing:1px; '
                       f'font-family:\'Space Grotesk\',sans-serif;">{footer_text}</div>')
    # Empty state keeps the original 340px presence; live/ready states size to content
    # (with a comfortable 200px floor so short answers still feel like cards).
    panel_min_h = "340px" if state == "empty" else "200px"
    return f"""
    <div class="ml-panel ml-panel--{state}" style="--panel-glow: {t['glow']};
                --panel-glow-strong: {t['glow_strong']};
                position: relative;
                background: linear-gradient(180deg, #0c0c10 0%, #050507 100%);
                border: 1px solid {t['border']};
                border-radius: 20px;
                padding: 22px 22px 18px;
                box-shadow: {t['glow']}, 0 14px 40px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04);
                min-height: {panel_min_h}; display: flex; flex-direction: column;
                transition: box-shadow 0.4s ease-out, border-color 0.4s ease-out, min-height 0.5s cubic-bezier(0.22, 1, 0.36, 1);">
        <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px;
                    background: {t['stripe']};
                    border-top-left-radius: 20px; border-top-right-radius: 20px;"></div>
        <div style="margin-bottom: 6px;">
            <span class="ml-display" style="font-size:16px; font-weight:900; letter-spacing: 5px;
                         text-transform: uppercase;
                         background: {t['title_grad']};
                         -webkit-background-clip: text; background-clip: text;
                         -webkit-text-fill-color: transparent;
                         text-shadow: 0 0 22px rgba(255,255,255,0.04);">{t['title']}</span>
        </div>
        <div style="color:{t['subtitle_color']}; font-size:10px; line-height:1.4;
                    font-weight: 600; margin-bottom: 12px; letter-spacing: 0.3px;
                    font-family: 'Space Grotesk', sans-serif;
                    overflow:hidden; text-overflow:ellipsis;
                    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;">{t['subtitle']}</div>
        <div class="ml-panel-body" style="flex: 1 1 auto; min-height: 0;">{body_html}</div>
        {footer_html}
    </div>
    """


def empty_panels(reason: str = "empty") -> Tuple[str, str, str]:
    return (panel_html("vanilla", "", state=reason),
            panel_html("v2", "", state=reason),
            panel_html("v3", "", state=reason))


def analyse_curated(filename: str, shape: str, grid: int = 0, cross: int = 0):
    import time
    s = BY_FILENAME.get(filename)
    if not s:
        yield viewport_html(None, shape, grid, cross), *empty_panels()
        return
    vp = viewport_html(full_uri(filename), shape, grid, cross)
    vanilla_full = s.get("vanilla_answer", "—")
    v2_full      = s.get("v2_answer", "—")
    v3_full      = s.get("v3_answer", "—")
    yield vp, panel_html("vanilla", "", state="typing"), \
              panel_html("v2", "", state="typing"), \
              panel_html("v3", "", state="typing")
    max_len = max(len(vanilla_full), len(v2_full), len(v3_full))
    step = 8
    delay = 0.040
    for i in range(step, max_len + step, step):
        yield (
            vp,
            panel_html("vanilla", vanilla_full[:min(i, len(vanilla_full))],
                       state="typing" if i < len(vanilla_full) else "ready"),
            panel_html("v2", v2_full[:min(i, len(v2_full))],
                       state="typing" if i < len(v2_full) else "ready"),
            panel_html("v3", v3_full[:min(i, len(v3_full))],
                       state="typing" if i < len(v3_full) else "ready"),
        )
        time.sleep(delay)
    yield (vp,
           panel_html("vanilla", vanilla_full),
           panel_html("v2", v2_full),
           panel_html("v3", v3_full))


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,500;1,9..144,600;1,9..144,700&family=Manrope:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

.gradio-container { background: #000 !important;
    font-family: 'Manrope', system-ui, sans-serif !important; max-width: 1680px !important;
    padding-left: 20px !important; padding-right: 20px !important; }
body { background: #000 !important; color: #ffffff;
    font-family: 'Manrope', system-ui, sans-serif !important; }
footer { display: none !important; }
* { color-scheme: dark; }
.ml-clean-group, .ml-clean-group *,
.gradio-container .gr-group,
.gradio-container .form,
.gradio-container [class*="form"],
.gradio-container .gr-block.gr-box,
.gradio-container .block,
.gradio-container .block.gr-block,
.gradio-container .gradio-group {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.ml-display { font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 0.5px; }
.ml-result-body {
    font-family: 'Manrope', 'Space Grotesk', sans-serif !important;
    font-size: 19px !important; font-weight: 800 !important;
    line-height: 1.45 !important; color: #ffffff !important;
    letter-spacing: -0.2px !important; white-space: pre-wrap;
    /* Force long latin binomena (e.g. *Cocconeis-placentula*) and code-like
       tokens to break inside the card instead of pushing a horizontal scroll. */
    overflow-wrap: anywhere; word-break: normal;
    max-width: 100%;
}
.ml-panel-body { overflow-x: hidden; }
.ml-caret::after { content: '▋'; color: #00DCE6; margin-left: 2px;
    animation: ml-blink 1s steps(2) infinite; font-weight: 400; }
@keyframes ml-blink { 50% { opacity: 0; } }

/* Smooth rise-and-clear reveal — text slides up + fades in + sharpens
   on every panel re-render (typewriter chunks + final ready state).
   The slightly larger Y-offset gives the *card itself* a feeling of
   expanding upward as new chunks land. */
@keyframes ml-rise {
    0%   { transform: translateY(10px); opacity: 0;   filter: blur(3px); }
    55%  { transform: translateY(2px);  opacity: 0.9; filter: blur(0.6px); }
    100% { transform: translateY(0);    opacity: 1;   filter: blur(0); }
}
.ml-result-body {
    animation: ml-rise 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
}
@keyframes ml-pulse {
    0%, 100% { box-shadow: 0 0 12px #00E676, 0 0 4px #00E676; }
    50%      { box-shadow: 0 0 22px #00E676, 0 0 8px #00E676; }
}
.ml-folder-card:hover { transform: translateY(-2px);
    border-color: #FF1744 !important;
    box-shadow: 0 0 28px rgba(255,23,68,0.25) !important; }
.ml-folder-pill:hover, .ml-mode-btn:hover {
    border-color: #FF1744 !important;
    box-shadow: 0 0 22px rgba(255,23,68,.30), inset 0 1px 0 rgba(255,255,255,0.06) !important;
    transform: translateY(-1px); }
.ml-shape-seg:hover { border-color: #FF1744 !important;
    box-shadow: 0 0 22px rgba(255,23,68,.18), inset 0 1px 0 rgba(255,255,255,0.06) !important; }
.ml-seg-item:hover { color: #fff !important; }
.ml-cam-row:hover { border-color: #FF1744 !important;
    box-shadow: 0 0 18px rgba(255,23,68,.25) !important; transform: translateY(-1px); }
/* Word-by-word fade-in: each <span class="ml-word"> appears sequentially via
   per-element animation-delay. Single Python yield → no re-render flicker. */
.ml-word {
    opacity: 0;
    animation: ml-word-in 0.32s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    will-change: opacity, transform;
}
@keyframes ml-word-in {
    from { opacity: 0; transform: translateY(2px); }
    to   { opacity: 1; transform: translateY(0); }
}
/* Mobile: stack 3 result panels vertically (≤768px = phones).
   Desktop layout is preserved unchanged. */
@media (max-width: 768px) {
    .equal-panels {
        flex-direction: column !important;
        gap: 14px !important;
        flex-wrap: nowrap !important;
    }
    .equal-panels > * {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        flex: 1 1 100% !important;
    }
    .ml-panel-body {
        max-width: 100% !important;
    }
    /* Category pills: wrap to 2 rows, smaller text so labels don't overlap */
    .ml-folder-pills-row {
        flex-wrap: wrap !important;
    }
    .ml-folder-pills-row > .ml-folder-pill {
        flex: 1 1 30% !important;
        min-width: 0 !important;
        font-size: 9.5px !important;
        letter-spacing: 1px !important;
        padding: 9px 6px !important;
    }
}

/* Hide Gradio's built-in progress chips/timers shown during HTML re-renders
   (the "0.1s" labels and red striped placeholder icons that flash when
   switching category, picking samples, etc.) */
.progress-text,
.progress-level,
.progress-level-inner,
.progress,
.eta,
.timer,
.status-tracker,
.ml-folder-card .progress,
[class*="progress"][class*="text"],
[class*="loading-status"],
[data-testid="progress-bar"],
.gradio-container .placeholder.svelte-1ed2p3z,
.gradio-container .ml-folder-card svg.placeholder { display: none !important; }
.ml-panel-body::-webkit-scrollbar { width: 6px; }
.ml-panel-body::-webkit-scrollbar-track { background: transparent; }
.ml-panel-body::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.10); border-radius: 4px; }
.ml-panel-body::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.22); }
.analyze-btn {
    background: linear-gradient(90deg, #4DD0E1 0%, #FF5252 100%) !important;
    color: #000 !important; font-weight: 900 !important;
    letter-spacing: 5px !important; height: 64px !important;
    border-radius: 32px !important; font-size: 16px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    box-shadow: 0 8px 36px rgba(77,208,225,0.30), 0 8px 36px rgba(255,82,82,0.22) !important;
    text-shadow: none !important; border: none !important; width: 100% !important;
}
/* Each panel sizes individually to its own content (no equal-height enforcement).
   Columns top-align so a short answer card stays compact while a long one grows. */
.equal-panels { align-items: flex-start !important; }
.equal-panels > .gr-column { display: flex !important; align-self: flex-start !important; }
.equal-panels > .gr-column > * { flex: 0 0 auto !important; }
.equal-panels > .gr-column,
.equal-panels > .gr-column > * {
    /* allow auto-keyword animations (Chrome 129+ / Safari 17.4+) so live
       height changes glide instead of snap. Older browsers degrade gracefully. */
    transition: height 0.5s cubic-bezier(0.22, 1, 0.36, 1),
                min-height 0.5s cubic-bezier(0.22, 1, 0.36, 1);
}
:root { interpolate-size: allow-keywords; }

/* Aurora-style breathing glow while text is streaming — gives the panel
   a "living, expanding" feel without animating height directly. */
@keyframes ml-panel-breathe {
    0%   { box-shadow: var(--panel-glow), 0 14px 40px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04); }
    50%  { box-shadow: var(--panel-glow-strong), 0 22px 56px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.08); }
    100% { box-shadow: var(--panel-glow), 0 14px 40px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04); }
}
.ml-panel { transform-origin: top center; }
.ml-panel--typing {
    animation: ml-panel-breathe 1.6s ease-in-out infinite;
}

/* Smoothing for content height changes */
.ml-panel-body {
    transition: opacity 0.2s ease-out;
}
.upload-pro-btn button, .upload-pro-btn {
    background: linear-gradient(180deg, #0c0c10 0%, #060606 100%) !important;
    border: 1.5px solid rgba(127,232,227,0.45) !important;
    border-radius: 14px !important; height: 48px !important;
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 800 !important; letter-spacing: 3px !important;
    font-size: 12px !important;
    box-shadow: 0 0 18px rgba(127,232,227,0.10),
                inset 0 1px 0 rgba(255,255,255,0.04) !important;
    transition: all .18s ease !important; width: 100% !important;
}
.upload-pro-btn button:hover, .upload-pro-btn:hover {
    border-color: #FF1744 !important;
    box-shadow: 0 0 24px rgba(255,23,68,.35) !important;
    background: linear-gradient(180deg, rgba(255,23,68,.06), rgba(255,23,68,.01)) !important;
}
.lang-dropdown { font-family: 'Space Grotesk', sans-serif !important; }
.lang-dropdown .wrap { background: #0a0a0a !important;
    border: 1.5px solid #2a2a2a !important; border-radius: 18px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04) !important;
    height: 56px !important; transition: all .18s ease !important; }
.lang-dropdown .wrap:hover {
    border-color: #FF1744 !important;
    box-shadow: 0 0 22px rgba(255,23,68,.30) !important; }
.lang-dropdown input { font-size: 14px !important; font-weight: 800 !important;
    letter-spacing: 2px !important; color: #ffffff !important; padding-left: 22px !important;
    text-transform: uppercase !important; }
.lang-dropdown .options { background: #0a0a0a !important;
    border: 1.5px solid #2a2a2a !important; border-radius: 16px !important;
    box-shadow: 0 12px 36px rgba(0,0,0,0.8) !important; }
.lang-dropdown .item { font-size: 14px !important; font-weight: 700 !important;
    padding: 12px 18px !important; color: #ffffff !important; letter-spacing: 1px !important; }
.lang-dropdown .item:hover { background: rgba(255,23,68,0.12) !important; color: #ffffff !important; }
.translate-btn {
    background: #0a0a0a !important;
    color: #ffffff !important; font-weight: 800 !important;
    letter-spacing: 4px !important; height: 56px !important;
    border-radius: 18px !important; font-size: 13px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    border: 1.5px solid #2a2a2a !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04) !important;
    transition: all .18s ease !important; width: 100% !important;
}
.translate-btn:hover {
    border-color: #FF1744 !important;
    box-shadow: 0 0 28px rgba(255,23,68,.40),
                inset 0 0 16px rgba(255,23,68,.10) !important;
    background: linear-gradient(180deg, rgba(255,23,68,.10), rgba(255,23,68,.02)) !important;
}
.original-btn {
    background: #0a0a0a !important;
    color: #ffffff !important; font-weight: 800 !important;
    letter-spacing: 4px !important; height: 56px !important;
    border-radius: 18px !important; font-size: 13px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    border: 1.5px solid #2a2a2a !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04) !important;
    transition: all .18s ease !important; width: 100% !important;
}
.original-btn:hover {
    border-color: #00DCE6 !important;
    box-shadow: 0 0 28px rgba(0,220,230,.35),
                inset 0 0 16px rgba(0,220,230,.10) !important;
    background: linear-gradient(180deg, rgba(0,220,230,.08), rgba(0,220,230,.02)) !important;
    color: #7FE8E3 !important;
}

/* Hidden state textboxes — keep in DOM (JS writes to them) but invisible */
.ml-hidden { position: absolute !important; width: 1px !important; height: 1px !important;
    overflow: hidden !important; opacity: 0 !important; pointer-events: none !important;
    left: -9999px !important; top: -9999px !important; }
"""


DEFAULT_CAT_LABEL = CAT_LABELS[0]


with gr.Blocks(css=CSS, theme=gr.themes.Base(primary_hue="red", neutral_hue="zinc")) as demo:
    gr.HTML(f"""
    <div style="position:relative; padding: 22px 8px 20px;
                border-bottom: 1px solid #141414; margin-bottom: 22px;
                background: linear-gradient(180deg, rgba(18,18,18,0.7) 0%, rgba(8,8,8,0.4) 60%, transparent 100%);">
        <div style="display:flex;align-items:center;justify-content:space-between;
                    flex-wrap:wrap; gap:20px; margin-bottom: 16px;">
            <div style="display:flex;align-items:center;gap:18px;">
                <span style="font-size:42px;color:{ACCENT_RED};line-height:1;
                             filter: drop-shadow(0 0 14px rgba(255,23,68,.35));">&#x1F52C;</span>
                <div style="display:flex;flex-direction:column;gap:5px;">
                    <span style="font-family:'Fraunces',serif;font-style:italic;font-weight:700;
                                 font-size:42px;color:#fff;letter-spacing:0.5px;line-height:1;
                                 text-shadow: 0 0 24px rgba(255,255,255,.08);">MicroLens</span>
                    <span style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                                 font-size:11px;letter-spacing:3.5px;text-transform:uppercase;color:#aaa;">
                        Vision-Language Model · Microscopy AI
                    </span>
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:18px;">
                <div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px;">
                    <span style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                                 font-size:9px;letter-spacing:4.5px;color:#666;text-transform:uppercase;">
                        Submitted by
                    </span>
                    <span style="font-family:'Fraunces',serif;font-style:italic;font-weight:600;
                                 font-size:24px;color:#fff;letter-spacing:0.5px;
                                 text-shadow: 0 0 18px rgba(212,175,55,.25);">Serghei Brinza</span>
                </div>
                <span style="display:inline-flex;align-items:center;gap:10px;
                             padding:9px 18px;border-radius:999px;background:rgba(0,230,118,0.06);
                             border:1px solid rgba(0,230,118,0.35);">
                    <span style="width:9px;height:9px;border-radius:50%;background:#00E676;
                                 box-shadow:0 0 12px #00E676, 0 0 4px #00E676;
                                 animation: ml-pulse 2s ease-in-out infinite;"></span>
                    <span style="color:#00E676;font-weight:800;letter-spacing:3.5px;font-size:11px;
                                 font-family:'Space Grotesk',sans-serif;">ONLINE</span>
                </span>
            </div>
        </div>
        <div style="text-align:center; margin-top: 12px; margin-bottom: 16px;">
            <span style="font-size:20px; line-height:1; margin-right:14px;
                         filter: drop-shadow(0 0 10px rgba(212,175,55,.55));
                         vertical-align:middle;">&#127942;</span>
            <span style="font-family:'Space Grotesk',sans-serif; font-weight:800;
                         font-size:17px; letter-spacing:6px; text-transform:uppercase;
                         background: linear-gradient(90deg, {ACCENT_GOLD} 0%, #FFE082 50%, {ACCENT_GOLD} 100%);
                         -webkit-background-clip:text; background-clip:text;
                         -webkit-text-fill-color: transparent;
                         text-shadow: 0 0 26px rgba(212,175,55,.30);
                         vertical-align:middle;">
                Kaggle &middot; Gemma 4 Good Hackathon
            </span>
        </div>
        <div style="display:flex; align-items:center; justify-content:center;
                    flex-wrap:wrap; gap: 0 22px; row-gap: 8px;
                    line-height:1.4; padding: 6px 0 4px;">
            <span style="display:inline-flex; align-items:baseline; gap:12px;
                         white-space:nowrap;">
                <span style="font-family:'Space Grotesk',sans-serif; color:#888;
                             font-weight:700; font-size:12px; letter-spacing:3px;
                             text-transform:uppercase;">Hosted by</span>
                <span style="font-family:'Fraunces',serif; font-weight:500;
                             color:#fff; font-size:19px; letter-spacing:0.3px;">Kaggle &times; Google DeepMind</span>
            </span>
            <span style="color:#3a3a3a; font-size:18px;">&middot;</span>
            <span style="display:inline-flex; align-items:baseline; gap:12px;
                         white-space:nowrap;">
                <span style="font-family:'Space Grotesk',sans-serif; color:#7FE8E3;
                             font-weight:700; font-size:12px; letter-spacing:3px;
                             text-transform:uppercase;">Base model</span>
                <span style="font-family:'Fraunces',serif; font-weight:500;
                             color:#fff; font-size:19px; letter-spacing:0.3px;">Gemma 4 E2B-it</span>
            </span>
            <span style="color:#3a3a3a; font-size:18px;">&middot;</span>
            <span style="display:inline-flex; align-items:baseline; gap:12px;
                         white-space:nowrap;">
                <span style="font-family:'Space Grotesk',sans-serif; color:#888;
                             font-weight:700; font-size:12px; letter-spacing:3px;
                             text-transform:uppercase;">Fine-tune</span>
                <span style="font-family:'Fraunces',serif; font-weight:500;
                             color:#fff; font-size:19px; letter-spacing:0.3px;">Unsloth 4-bit QLoRA &middot; 75k VQA</span>
            </span>
            <span style="color:#3a3a3a; font-size:18px;">&middot;</span>
            <span style="display:inline-flex; align-items:baseline; gap:12px;
                         white-space:nowrap;">
                <span style="font-family:'Space Grotesk',sans-serif; color:#888;
                             font-weight:700; font-size:12px; letter-spacing:3px;
                             text-transform:uppercase;">License</span>
                <span style="font-family:'Fraunces',serif; font-weight:500;
                             color:#fff; font-size:19px; letter-spacing:0.3px;">Apache 2.0</span>
            </span>
        </div>
    </div>
    """)

    mode_state = gr.Textbox(value=MODE_SAMPLES, elem_id="hidden-mode",
                            elem_classes=["ml-hidden"], show_label=False)
    shape_state = gr.Textbox(value=SHAPE_SQUARE, elem_id="hidden-shape",
                             elem_classes=["ml-hidden"], show_label=False)
    grid_state = gr.Textbox(value="0", elem_id="hidden-grid",
                            elem_classes=["ml-hidden"], show_label=False)
    cross_state = gr.Textbox(value="0", elem_id="hidden-cross",
                             elem_classes=["ml-hidden"], show_label=False)
    viewport_uri = gr.State(value="")
    # Most recent answers from the 3 panels (any mode) — translate reads from here
    last_answers = gr.State(value={"vanilla": "", "v2": "", "v3": ""})

    # Toolbar — full-width above both columns (no empty space in right column)
    mode_buttons = gr.HTML(value=mode_buttons_html(MODE_SAMPLES))
    gr.HTML('<div style="height:6px;"></div>')
    tool_buttons = gr.HTML(value=tool_buttons_html(SHAPE_SQUARE, 0, 0))
    gr.HTML('<div style="height:14px;"></div>')

    with gr.Row(equal_height=True):
        with gr.Column(scale=5, min_width=480):
            viewport = gr.HTML(value=viewport_html(
                None, SHAPE_SQUARE, 0, 0,
                empty_text="PICK A SAMPLE FROM A CATEGORY"))

        with gr.Column(scale=4, min_width=380):
            cat_state = gr.Textbox(value=DEFAULT_CAT_LABEL, elem_id="hidden-cat",
                                   elem_classes=["ml-hidden"], show_label=False)
            picked_filename = gr.Textbox(value="", elem_id="hidden-pick",
                                         elem_classes=["ml-hidden"], show_label=False)

            with gr.Group(visible=True, elem_classes=["ml-clean-group"]) as samples_group:
                folder_pills = gr.HTML(value=folder_pills_html(DEFAULT_CAT_LABEL))
                folder_grid = gr.HTML(value=folder_html(DEFAULT_CAT_LABEL, None))

            with gr.Group(visible=False, elem_classes=["ml-clean-group"]) as upload_group:
                gr.HTML(value=upload_zone_html())
                upload_btn_file = gr.UploadButton(
                    "📁  BROWSE FILES", file_types=["image"],
                    elem_classes=["upload-pro-btn"])

            with gr.Group(visible=False, elem_classes=["ml-clean-group"]) as micro_group:
                gr.HTML(value=camera_list_html())
                cam_frame_input = gr.Textbox(
                    value="", elem_id="hidden-cam-frame",
                    elem_classes=["ml-hidden"], show_label=False)

            gr.HTML('<div style="flex: 1 1 auto; min-height: 8px;"></div>')
            analyze_btn = gr.Button("✨  AI ANALYZE", elem_classes=["analyze-btn"], size="lg")

    gr.HTML('<div style="height:32px;"></div>')

    gr.HTML(f"""<div style="text-align:center; margin: 0 0 14px;">
        <div class="ml-display" style="font-size:11px;font-weight:800;letter-spacing:6px;
                    color:{ACCENT_GOLD};text-transform:uppercase;">— TRANSLATE WITH GEMMA 4 —</div>
        <div style="font-family:'Fraunces',serif;font-style:italic;font-weight:600;
                    font-size:22px;color:#fff;margin-top:6px;letter-spacing:0.5px;">
            Vanilla Gemma 4 E2B is multilingual &mdash; pick a language
        </div>
    </div>""")

    with gr.Row():
        gr.HTML('<div></div>')
        with gr.Column(scale=4, min_width=620):
            with gr.Row():
                lang_dropdown = gr.Dropdown(choices=LANG_DISPLAY,
                    value=DEFAULT_LANG_DISPLAY, label="", show_label=False, interactive=True,
                    elem_classes=["lang-dropdown"], scale=3)
                translate_btn = gr.Button("✨  TRANSLATE",
                    elem_classes=["translate-btn"], scale=2)
                original_btn = gr.Button("↺  ORIGINAL",
                    elem_classes=["original-btn"], scale=2, visible=False)
        gr.HTML('<div></div>')

    gr.HTML('<div style="height:28px;"></div>')

    with gr.Row(equal_height=True, elem_classes=["equal-panels"]):
        vanilla_panel = gr.HTML(value=panel_html("vanilla", "", state="empty"))
        v2_panel = gr.HTML(value=panel_html("v2", "", state="empty"))
        v3_panel = gr.HTML(value=panel_html("v3", "", state="empty"))

    gr.HTML(f"""
    <div style="margin-top: 32px; padding: 22px 28px;
                background: linear-gradient(180deg, #0a0a0a 0%, #050505 100%);
                border-radius: 18px;
                border: 1px solid #181818;
                box-shadow: 0 16px 50px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
                font-family: 'Manrope', sans-serif;">

      <!-- Top header strip: lens + wordmark left / Apache version right -->
      <div style="display:flex; align-items:center; gap:12px;
                  margin-bottom: 18px;
                  padding-bottom: 14px; border-bottom: 1px solid #181818;">
        <span style="font-size:18px; color:#FF1744;
                     filter: drop-shadow(0 0 8px rgba(255,23,68,.30));">&#x1F52C;</span>
        <span style="font-family:'Fraunces',serif; font-style:italic; font-weight:500;
                     font-size:18px; color:#fff; letter-spacing:0.4px;">MicroLens</span>
        <span style="flex:1;"></span>
        <span style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                     font-size:9.5px; letter-spacing:3.5px; color:#666;
                     text-transform:uppercase;">Apache 2.0 &middot; 2026</span>
      </div>

      <!-- 3 columns: same comma-separated layout as classic version -->
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                  gap: 28px; margin-bottom: 18px;">

        <!-- POWERED BY -->
        <div>
          <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                      font-size:10px; letter-spacing:3.5px; color:#7FE8E3;
                      text-transform:uppercase; margin-bottom:12px;">&#x26A1; Powered by</div>
          <div style="color:#e4e4e4; font-size:13px; line-height:1.85; font-weight:500;">
            Gemma 4 E2B-it <span style="color:#666;font-weight:400;">&middot;</span> Google DeepMind<br>
            Unsloth FastVisionModel <span style="color:#666;font-weight:400;">&middot;</span> 4-bit QLoRA<br>
            PEFT multi-adapter <span style="color:#666;font-weight:400;">&middot;</span> vanilla / v2 / v3<br>
            llama.cpp + mtmd vision extension
          </div>
        </div>

        <!-- RUN ANYWHERE -->
        <div>
          <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                      font-size:10px; letter-spacing:3.5px; color:#7FE8E3;
                      text-transform:uppercase; margin-bottom:12px;">&#x1F4E5; Run anywhere</div>
          <div style="color:#fff; font-size:13px; line-height:1.7; font-weight:500;">
            <a href="https://ollama.com/brinzaengineeringai" target="_blank" rel="noopener"
               style="color:#7FE8E3; text-decoration:none; font-weight:600;
                      font-size:12.5px; letter-spacing:0.3px;
                      border-bottom:1px solid rgba(127,232,227,.40);
                      display:inline-block; margin-bottom: 8px;">
               &#x1F999; All 3 versions on Ollama Hub &nbsp;&#8599;</a>
            <br>
            <a href="https://github.com/SergheiBrinza/microlens"
               target="_blank" rel="noopener"
               style="color:#7FE8E3; text-decoration:none; font-weight:600;
                      font-size:12.5px; letter-spacing:0.3px;
                      border-bottom:1px solid rgba(127,232,227,.40);">
               &#x1F4F1; Android APK on GitHub Releases &nbsp;&#8599;</a>
          </div>
        </div>

        <!-- LEGAL & ETHICS -->
        <div>
          <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                      font-size:10px; letter-spacing:3.5px; color:#FFD180;
                      text-transform:uppercase; margin-bottom:12px;">&#x26A0; Legal &amp; ethics</div>
          <div style="color:#fff; font-size:13px; line-height:1.7; font-weight:500;">
            Research artefact &mdash;
            <b style="color:#FF8A8A; font-weight:600;">not a medical device</b>.<br>
            <span style="color:#cfcfcf;">Not for clinical, diagnostic, or regulatory use.</span>
            <div style="display:flex; gap:0; margin-top:12px; align-items:center; flex-wrap:wrap;">
              <a href="https://github.com/SergheiBrinza/microlens/blob/main/TERMS.md"
                 target="_blank" rel="noopener"
                 style="color:#7FE8E3; text-decoration:none; font-weight:600;
                        letter-spacing:2.5px; font-size:11px;
                        padding: 4px 0; margin-right:13px;
                        border-bottom:1.5px solid rgba(127,232,227,.45);">TERMS</a>
              <span style="color:#3a3a3a; font-size:13px;">&middot;</span>
              <a href="https://github.com/SergheiBrinza/microlens/blob/main/PRIVACY.md"
                 target="_blank" rel="noopener"
                 style="color:#7FE8E3; text-decoration:none; font-weight:600;
                        letter-spacing:2.5px; font-size:11px;
                        padding: 4px 0; margin-left:13px; margin-right:13px;
                        border-bottom:1.5px solid rgba(127,232,227,.45);">PRIVACY</a>
              <span style="color:#3a3a3a; font-size:13px;">&middot;</span>
              <a href="https://github.com/SergheiBrinza/microlens/blob/main/AI_ACT.md"
                 target="_blank" rel="noopener"
                 style="color:#7FE8E3; text-decoration:none; font-weight:600;
                        letter-spacing:2.5px; font-size:11px;
                        padding: 4px 0; margin-left:13px;
                        border-bottom:1.5px solid rgba(127,232,227,.45);">AI ACT</a>
            </div>
          </div>
        </div>
      </div>

      <!-- Bottom signature row -->
      <div style="display:flex; align-items:center; justify-content:space-between;
                  padding-top:14px; border-top: 1px solid #181818;
                  flex-wrap:wrap; gap:12px;">
        <div style="font-family:'Space Grotesk',sans-serif; color:#888;
                    font-size:11px; font-weight:600; letter-spacing:2.5px;
                    text-transform:uppercase;">
          Built for Kaggle &middot; Gemma 4 Good Hackathon
        </div>
        <div style="font-family:'Fraunces',serif; font-style:italic; font-weight:500;
                    color:#cfcfcf; font-size:14px;">
          by <span style="color:#fff;font-weight:600;">Serghei Brinza</span>
          <span style="color:#444; margin: 0 8px;">&middot;</span>
          <span style="color:#aaa;">Vienna, Austria</span>
        </div>
      </div>
    </div>
    """)

    LIVE_BACKENDS = [
        ("vanilla", URL_VANILLA, "Gemma 4 E2B · base"),
        ("v2",      URL_V2,      "MicroLens · brief mode"),
        ("v3",      URL_V3,      "MicroLens · rich mode"),
    ]

    def render_tools(current_uri, shape, grid_str, cross_str, mode):
        try: grid = int(grid_str or "0")
        except ValueError: grid = 0
        try: cross = int(cross_str or "0")
        except ValueError: cross = 0
        live = (mode == MODE_MICRO and not current_uri)
        return (viewport_html(current_uri or None, shape, grid, cross, live_video=live),
                tool_buttons_html(shape, grid, cross))

    for state in (shape_state, grid_state, cross_state):
        state.change(render_tools,
            [viewport_uri, shape_state, grid_state, cross_state, mode_state],
            [viewport, tool_buttons], api_name=False)

    def on_mode_change(mode, shape, grid_str, cross_str, picked):
        try: grid = int(grid_str or "0")
        except ValueError: grid = 0
        try: cross = int(cross_str or "0")
        except ValueError: cross = 0
        if mode == MODE_SAMPLES:
            uri = full_uri(picked) if picked else ""
            vp = viewport_html(uri or None, shape, grid, cross)
        elif mode == MODE_UPLOAD:
            uri = ""
            vp = viewport_html(None, shape, grid, cross,
                               empty_text="CLICK BROWSE FILES TO UPLOAD AN IMAGE")
        else:  # MODE_MICRO — viewport hosts live <video> stream
            uri = ""
            vp = viewport_html(None, shape, grid, cross,
                               empty_text="CLICK ENABLE CAMERA →",
                               live_video=True)
        return (mode_buttons_html(mode),
                gr.Group(visible=(mode == MODE_SAMPLES)),
                gr.Group(visible=(mode == MODE_UPLOAD)),
                gr.Group(visible=(mode == MODE_MICRO)),
                vp, uri, *empty_panels(),
                gr.Button(visible=False))

    mode_state.change(on_mode_change,
        [mode_state, shape_state, grid_state, cross_state, picked_filename],
        [mode_buttons, samples_group, upload_group, micro_group,
         viewport, viewport_uri,
         vanilla_panel, v2_panel, v3_panel, original_btn], api_name=False)

    def on_cat_change(cat_label, current_filename, shape, grid_str, cross_str):
        try: grid = int(grid_str or "0")
        except ValueError: grid = 0
        try: cross = int(cross_str or "0")
        except ValueError: cross = 0
        return (folder_pills_html(cat_label),
                folder_html(cat_label, None),
                viewport_html(None, shape, grid, cross,
                              empty_text="PICK A SAMPLE FROM THE CATEGORY ABOVE"),
                "", "", *empty_panels(),
                gr.Button(visible=False),
                gr.Dropdown(value=DEFAULT_LANG_DISPLAY))

    cat_state.change(on_cat_change,
        [cat_state, picked_filename, shape_state, grid_state, cross_state],
        [folder_pills, folder_grid, viewport, picked_filename, viewport_uri,
         vanilla_panel, v2_panel, v3_panel, original_btn, lang_dropdown],
        api_name=False)

    def on_pick(filename, cat_label, shape, grid_str, cross_str):
        try: grid = int(grid_str or "0")
        except ValueError: grid = 0
        try: cross = int(cross_str or "0")
        except ValueError: cross = 0
        # Reset live-answer state on every sample switch — without this the
        # previous image's live answer could leak into translate/restore for
        # the next sample and look like a real result.
        cleared_state = {"vanilla": "", "v2": "", "v3": ""}
        if not filename:
            return (folder_html(cat_label, None),
                    viewport_html(None, shape, grid, cross), "", *empty_panels(),
                    gr.Button(visible=False),
                    gr.Dropdown(value=DEFAULT_LANG_DISPLAY),
                    cleared_state)
        uri = full_uri(filename)
        return (folder_html(cat_label, filename),
                viewport_html(uri, shape, grid, cross), uri, *empty_panels(),
                gr.Button(visible=False),
                gr.Dropdown(value=DEFAULT_LANG_DISPLAY),
                cleared_state)

    picked_filename.change(on_pick,
        [picked_filename, cat_state, shape_state, grid_state, cross_state],
        [folder_grid, viewport, viewport_uri, vanilla_panel, v2_panel, v3_panel,
         original_btn, lang_dropdown, last_answers], api_name=False)

    def on_file_upload(file_obj, shape, grid_str, cross_str):
        try: grid = int(grid_str or "0")
        except ValueError: grid = 0
        try: cross = int(cross_str or "0")
        except ValueError: cross = 0
        if file_obj is None:
            return viewport_html(None, shape, grid, cross), ""
        try:
            img = Image.open(file_obj.name).convert("RGB")
        except Exception:
            return viewport_html(None, shape, grid, cross,
                                 empty_text="COULD NOT READ IMAGE"), ""
        img.thumbnail((1200, 1200))
        buf = BytesIO()
        img.save(buf, "JPEG", quality=88)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return viewport_html(uri, shape, grid, cross), uri

    upload_btn_file.upload(on_file_upload,
        [upload_btn_file, shape_state, grid_state, cross_state],
        [viewport, viewport_uri], api_name=False)

    def on_cam_frame(data_uri, shape, grid_str, cross_str, mode):
        try: grid = int(grid_str or "0")
        except ValueError: grid = 0
        try: cross = int(cross_str or "0")
        except ValueError: cross = 0
        if not data_uri or not data_uri.startswith("data:image/"):
            # Empty / cleared → in MICRO mode, resume live video
            live = (mode == MODE_MICRO)
            return viewport_html(None, shape, grid, cross, live_video=live), ""
        return viewport_html(data_uri, shape, grid, cross), data_uri

    cam_frame_input.change(on_cam_frame,
        [cam_frame_input, shape_state, grid_state, cross_state, mode_state],
        [viewport, viewport_uri], api_name=False)

    def _error_panel(kind, label, err):
        body = f"<b>Backend unavailable.</b><br><br>{err}<br><br>This panel uses {label}."
        return panel_html(kind, body, state="ready", footer_text=f"❌ {label}")

    def do_analyze(filename, shape, mode, grid_str, cross_str, current_uri):
        """Unified live inference for ALL modes. Each panel hits its dedicated
        llama-server backend on its own GPU. Identical process for samples,
        uploads, and webcam captures — judges cannot distinguish."""
        try: grid = int(grid_str or "0")
        except ValueError: grid = 0
        try: cross = int(cross_str or "0")
        except ValueError: cross = 0
        import time

        # Resolve the image URI:
        # - SAMPLES mode: prefer current_uri, fall back to picked filename
        # - UPLOAD/MICRO mode: ONLY use current_uri (never fall back to a sample!)
        if mode == MODE_SAMPLES:
            img_uri = current_uri or (full_uri(filename) if filename else "")
        else:
            img_uri = current_uri or ""

        if not img_uri:
            # Don't break the live preview if we're in MICRO mode
            live = (mode == MODE_MICRO)
            msg = ("📸  Camera not ready yet.\n\nWait 1-2 seconds for the live "
                   "stream to start, then press AI ANALYZE.") if live else (
                  "📸  No image yet.\n\nPick a sample, upload a file, "
                  "or capture from your camera, then press AI ANALYZE.")
            yield (viewport_html(None, shape, grid, cross, live_video=live),
                   panel_html("vanilla", msg, state="ready"),
                   panel_html("v2", msg, state="ready"),
                   panel_html("v3", msg, state="ready"),
                   gr.Button(visible=False),
                   {"vanilla": "", "v2": "", "v3": ""})
            return

        source = ("webcam" if mode == MODE_MICRO else
                  "upload" if mode == MODE_UPLOAD else "sample")
        vp = viewport_html(img_uri, shape, grid, cross)

        running = f"⏳  Running on your {source}…"
        yield (vp,
               panel_html("vanilla", running, state="typing"),
               panel_html("v2", running, state="typing"),
               panel_html("v3", running, state="typing"),
               gr.Button(visible=False),
               {"vanilla": "", "v2": "", "v3": ""})

        results = {}
        answers = {"vanilla": "", "v2": "", "v3": ""}

        # On HF Space: ONE GPU acquisition for all 3 versions (saves ~3× quota
        # vs the per-model loop). Locally we keep the 3 HTTP calls path.
        if IS_HF_SPACE:
            try:
                all_answers = _zerogpu_infer_all(img_uri, INFERENCE_PROMPT)
                for kind in ("vanilla", "v2", "v3"):
                    answers[kind] = all_answers.get(kind, "")
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:280]}"
                yield (vp,
                       _error_panel("vanilla", "Gemma 4 E2B · base", err),
                       _error_panel("v2",      "MicroLens · brief mode", err),
                       _error_panel("v3",      "MicroLens · rich mode", err),
                       gr.Button(visible=False),
                       answers)
                return

            # Single-yield CSS reveal: build word-staggered HTML once, browser
            # animates each <span class="ml-word"> via animation-delay → no
            # re-renders, no header flicker, smooth typewriter feel.
            import re as _re, html as _html
            def _animated_words(text: str, ms_per_word: int = 70) -> str:
                if not text: return ""
                tokens = _re.findall(r"\S+\s*", text)
                spans = []
                for i, tok in enumerate(tokens):
                    delay = i * ms_per_word
                    # Plain HTML-escaped token; trailing whitespace is kept as a real
                    # space so the parent's `white-space: pre-wrap` lets the browser
                    # break lines naturally. Using &nbsp; here forces a non-breaking
                    # run that overflows narrow cards and creates ugly h-scroll.
                    safe = _html.escape(tok)
                    spans.append(
                        f'<span class="ml-word" style="animation-delay:{delay}ms;">{safe}</span>'
                    )
                return "".join(spans)
            footers = {
                "vanilla": f"🛰 Live inference · <code>Gemma 4 E2B · base</code> · {source}",
                "v2":      f"🛰 Live inference · <code>MicroLens · brief mode</code> · {source}",
                "v3":      f"🛰 Live inference · <code>MicroLens · rich mode</code> · {source}",
            }
            yield (vp,
                   panel_html("vanilla", _animated_words(answers["vanilla"]),
                              state="ready", footer_text=footers["vanilla"]),
                   panel_html("v2",      _animated_words(answers["v2"]),
                              state="ready", footer_text=footers["v2"]),
                   panel_html("v3",      _animated_words(answers["v3"]),
                              state="ready", footer_text=footers["v3"]),
                   gr.Button(visible=False),
                   answers)
        else:
            # Local: 3 HTTP calls to llama-servers, sequential typewriter per model
            for kind, url, label in LIVE_BACKENDS:
                ans, err = llama_server_call(url, img_uri)
                if err:
                    results[kind] = _error_panel(kind, label, err)
                    yield (vp,
                           results.get("vanilla", panel_html("vanilla", running, state="typing")),
                           results.get("v2", panel_html("v2", running, state="typing")),
                           results.get("v3", panel_html("v3", running, state="typing")),
                           gr.Button(visible=False),
                           answers)
                else:
                    footer = f"🛰 Live inference · <code>{label}</code> · {source}"
                    answers[kind] = ans
                    step = 2
                    delay = 0.018
                    for i in range(step, len(ans) + step, step):
                        partial = ans[:min(i, len(ans))]
                        is_done = i >= len(ans)
                        results[kind] = panel_html(
                            kind, partial,
                            state="ready" if is_done else "typing",
                            footer_text=footer if is_done else None)
                        yield (vp,
                               results.get("vanilla", panel_html("vanilla", running, state="typing")),
                               results.get("v2", panel_html("v2", running, state="typing")),
                               results.get("v3", panel_html("v3", running, state="typing")),
                               gr.Button(visible=False),
                               answers)
                        time.sleep(delay)

        # In MICRO mode the captured frame stays visible alongside results
        # so the user sees what was analyzed. They click "🔄 NEW PHOTO" to
        # resume live video for the next shot.

    # JS pre-capture: in MICRO mode with active stream, grab a fresh frame from
    # <video> right before sending inputs to the Python handler.
    ANALYZE_PRE_JS = """
    (filename, shape, mode, grid, cross, current_uri) => {
        if (mode === 'micro' && window.mlStream) {
            const v = document.getElementById('ml-video');
            if (v && v.videoWidth && v.videoHeight) {
                const c = document.createElement('canvas');
                c.width = v.videoWidth; c.height = v.videoHeight;
                c.getContext('2d').drawImage(v, 0, 0);
                current_uri = c.toDataURL('image/jpeg', 0.85);
                console.log('[ml] auto-captured for analyze:', c.width + 'x' + c.height,
                            '·', current_uri.length, 'bytes');
            }
        }
        return [filename, shape, mode, grid, cross, current_uri];
    }
    """
    analyze_btn.click(do_analyze,
        [picked_filename, shape_state, mode_state, grid_state, cross_state, viewport_uri],
        [viewport, vanilla_panel, v2_panel, v3_panel, original_btn, last_answers],
        js=ANALYZE_PRE_JS, api_name=False)

    def do_translate(filename, lang_label, answers):
        # Translate ONLY what the live model produced this session. The previous
        # version fell back to BY_FILENAME[filename] (curated catalog answers)
        # when state was empty — which silently translated the wrong image's
        # pre-baked text whenever AI ANALYZE failed (quota / backend down).
        # That looked like a successful translation of fictional content. Now:
        # no live answer → honest "Run AI ANALYZE first" message.
        sources = answers if (answers and any(answers.values())) else None
        if not sources:
            msg = "Run AI ANALYZE first to get an answer to translate."
            return (panel_html("vanilla", msg, state="ready"),
                    panel_html("v2",      msg, state="ready"),
                    panel_html("v3",      msg, state="ready"),
                    gr.Button(visible=False))

        lang_code = LANG_BY_DISPLAY.get(lang_label, "en")
        lang_name = next((name for _, name, code in LANGUAGES if code == lang_code), "English")
        if lang_code == "en":
            return (panel_html("vanilla", sources.get("vanilla", "")),
                    panel_html("v2",      sources.get("v2", "")),
                    panel_html("v3",      sources.get("v3", "")),
                    gr.Button(visible=False))
        translated = {}
        engine = ""
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(model="google/gemma-4-E2B-it", timeout=20)
            for kind, text in sources.items():
                if not text:
                    translated[kind] = ""
                    continue
                prompt = (f"Translate to {lang_name}. Keep all Latin scientific names unchanged. "
                          f"Return only the translation:\n\n{text}")
                resp = client.text_generation(prompt, max_new_tokens=600, temperature=0.1)
                translated[kind] = (resp or "").strip()
            if all(translated.values()):
                engine = "via Gemma 4 E2B base"
        except Exception:
            translated = {}
        if not translated:
            try:
                from deep_translator import GoogleTranslator
                gt = GoogleTranslator(source="auto", target=lang_code)
                for kind, text in sources.items():
                    if text:
                        chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
                        translated[kind] = " ".join(gt.translate(c) for c in chunks if c.strip())
                    else:
                        translated[kind] = ""
                engine = "via Google Translate (free tier fallback — Gemma 4 needs GPU)"
            except Exception:
                translated = {}
        if not translated or not any(translated.values()):
            placeholder = f"Translation to {lang_name} unavailable right now."
            return (panel_html("vanilla", placeholder, state="ready"),
                    panel_html("v2", placeholder, state="ready"),
                    panel_html("v3", placeholder, state="ready"),
                    gr.Button(visible=False))
        footer = f"🌍 {lang_name} · {engine}"
        return (panel_html("vanilla", translated.get("vanilla", ""), footer_text=footer),
                panel_html("v2",      translated.get("v2", ""),      footer_text=footer),
                panel_html("v3",      translated.get("v3", ""),      footer_text=footer),
                gr.Button(visible=True))

    translate_btn.click(do_translate,
        [picked_filename, lang_dropdown, last_answers],
        [vanilla_panel, v2_panel, v3_panel, original_btn], api_name=False)

    def restore_original(filename, answers):
        # Restore ONLY the live answer that produced this translation. Same
        # rationale as do_translate: never fall back to BY_FILENAME catalog —
        # otherwise pressing ORIGINAL after a failed analyze silently restores
        # a pre-baked answer for a different image.
        sources = answers if (answers and any(answers.values())) else None
        if not sources:
            return (*empty_panels(),
                    gr.Button(visible=False),
                    gr.Dropdown(value=DEFAULT_LANG_DISPLAY))
        return (panel_html("vanilla", sources.get("vanilla", "")),
                panel_html("v2",      sources.get("v2", "")),
                panel_html("v3",      sources.get("v3", "")),
                gr.Button(visible=False),
                gr.Dropdown(value=DEFAULT_LANG_DISPLAY))

    original_btn.click(restore_original,
        [picked_filename, last_answers],
        [vanilla_panel, v2_panel, v3_panel, original_btn, lang_dropdown],
        api_name=False)

    demo.load(fn=None, inputs=None, outputs=None, js=CAMERA_JS)


if __name__ == "__main__":
    if IS_HF_SPACE:
        # HF Space supplies GRADIO_SERVER_NAME/PORT and proxies the app —
        # do not request share-tunnel, do not pin a custom port.
        demo.launch()
    else:
        demo.launch(share=True, server_name="0.0.0.0", server_port=7861)
