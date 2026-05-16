#!/usr/bin/env python3
"""MicroLens VQA: 3x vLLM 8B thinking (GPU 0, 1, 2)."""
import os, requests, base64, time, json, itertools, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

PORTS = [9200, 9300, 9301]  # 3 vLLM 8B instances
MODEL = os.environ.get("TEACHER_VLM_MODEL", "teacher-vlm-8b")  # set to the model id served by your local vLLM teacher
PARALLEL_WORKERS = 12  # 4 на каждую GPU

DATASET_DIR = Path("/media/softer/blau1/microlens/datasets")
OUTPUT_DIR = Path("/media/softer/blau1/microlens/training/vqa_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "vqa_all.jsonl"
LOG_FILE = OUTPUT_DIR / "vqa_generation.log"

CATEGORY_HINTS = {
    "01_pollen":      "a pollen grain",
    "02_algae":       "algae",
    "03_yeast":       "yeast cells",
    "04_minerals":    "a mineral crystal",
    "05_plantdoc":    "a plant leaf",
    "07_pcb":         "a printed circuit board",
    "08_snowflakes":  "a snowflake",
    "12_zooplankton": "freshwater zooplankton",
    "13_tardigrades": "a tardigrade or microfauna",
}

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def collect_all_images():
    all_imgs = []
    for cat in sorted(CATEGORY_HINTS.keys()):
        cat_dir = DATASET_DIR / cat
        if not cat_dir.exists(): continue
        files = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            files.extend(cat_dir.rglob(ext))
        files = [f for f in files if f.stat().st_size > 5000]
        all_imgs.extend([(cat, f) for f in files])
        log(f"  {cat}: {len(files)}")
    return all_imgs

def load_processed():
    processed = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "image_path" in r:
                        processed.add(r["image_path"])
                except: pass
    return processed

_port_lock = threading.Lock()
_port_iter = itertools.cycle(PORTS)

def next_port():
    with _port_lock:
        return next(_port_iter)

def process_image(item):
    cat, path = item
    port = next_port()
    try:
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        return {"image_path": str(path), "category": cat, "port": port, "error": f"read: {e}"}
    hint = CATEGORY_HINTS[cat]
    prompt_text = f"""This is a microscopy image of {hint}.
Think carefully about what you see, then respond in format:
Subject: [name]
Features: [2 detailed features]"""
    start = time.time()
    try:
        r = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt_text}
                ]}],
                "max_tokens": 1000,
                "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": True}
            },
            timeout=180
        )
        elapsed = time.time() - start
        if r.status_code != 200:
            return {"image_path": str(path), "category": cat, "port": port,
                    "error": f"HTTP {r.status_code}", "elapsed": elapsed}
        data = r.json()
        content = data["choices"][0]["message"]["content"].strip()
        return {
            "image_path": str(path), "category": cat, "filename": path.name,
            "port": port, "model": MODEL,
            "response": content,
            "tokens": data.get("usage",{}).get("completion_tokens", 0),
            "elapsed": round(elapsed, 2),
        }
    except Exception as e:
        return {"image_path": str(path), "category": cat, "port": port,
                "error": str(e), "elapsed": time.time()-start}

def main():
    start_time = time.time()
    log("=" * 70)
    log("  MICROLENS VQA — 3x 8B thinking (GPU 0, 1, 2)")
    log("=" * 70)
    log(f"Model: {MODEL}, Ports: {PORTS}, Workers: {PARALLEL_WORKERS}")
    log("")
    log("Checking vLLM instances...")
    for port in PORTS:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            if r.status_code == 200:
                log(f"  Port {port} OK")
            else:
                log(f"  Port {port} ERROR {r.status_code}")
                return
        except Exception as e:
            log(f"  Port {port} FAIL: {e}")
            return
    log("")

    all_imgs = collect_all_images()
    log(f"Total: {len(all_imgs)}")
    processed = load_processed()
    log(f"Already processed: {len(processed)}")
    todo = [i for i in all_imgs if str(i[1]) not in processed]
    log(f"To process: {len(todo)}")
    if not todo:
        log("All done!")
        return
    log("")
    log("=" * 70)
    log(f"START: {len(todo)}")
    log("=" * 70)

    done = 0
    errors = 0
    q_counts = defaultdict(int)
    port_counts = defaultdict(int)
    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")
    out_lock = threading.Lock()

    try:
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(process_image, item): item for item in todo}
            for future in as_completed(futures):
                result = future.result()
                done += 1
                with out_lock:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                if "error" in result:
                    errors += 1
                    q_counts["error"] += 1
                else:
                    port_counts[result.get("port", "?")] += 1
                    resp = result.get("response", "")
                    if resp and "subject" in resp.lower():
                        q_counts["good"] += 1
                    elif resp:
                        q_counts["partial"] += 1
                    else:
                        q_counts["empty"] += 1
                if done % 100 == 0 or done == len(todo):
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = len(todo) - done
                    eta_m = (remaining / rate / 60) if rate > 0 else 0
                    log(f"[{done}/{len(todo)}] {100*done/len(todo):5.1f}% | "
                        f"rate={rate:.2f}/s | ETA={eta_m:.0f}min | "
                        f"good={q_counts['good']} empty={q_counts['empty']} err={errors}")
    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        out_f.close()

    total = time.time() - start_time
    log("")
    log("=" * 70)
    log(f"DONE: {done} in {total/60:.0f} min ({total/3600:.1f}h)")
    log(f"good={q_counts['good']} empty={q_counts['empty']} err={errors}")
    log(f"Ports: {dict(port_counts)}")
    log("=" * 70)

if __name__ == "__main__":
    main()
