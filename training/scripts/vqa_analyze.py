#!/usr/bin/env python3
"""Полный анализ VQA результатов MicroLens.

Что делает:
1. Статистика по категориям
2. Дубликаты
3. Ошибки (empty, error)
4. Распределение времени
5. Распределение по портам/GPU
6. Качество описаний (длина, ключевые слова)
7. Примеры лучших и худших описаний
8. Подозрительные случаи
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median

VQA_FILE = Path("/media/softer/blau1/microlens/training/vqa_data/vqa_all.jsonl")


def load_all():
    """Загружает все записи, помечая дубликаты."""
    records = []
    seen_paths = set()
    with open(VQA_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                r["is_duplicate"] = r.get("image_path") in seen_paths
                seen_paths.add(r.get("image_path"))
                records.append(r)
            except Exception:
                pass
    return records


def header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def analyze(records):
    # === 1. Общая статистика ===
    header("1. ОБЩАЯ СТАТИСТИКА")
    
    total = len(records)
    unique = sum(1 for r in records if not r["is_duplicate"])
    duplicates = total - unique
    errors = sum(1 for r in records if "error" in r)
    empty = sum(1 for r in records if not r.get("response") and "error" not in r)
    good = sum(1 for r in records if r.get("response") and "error" not in r)
    
    print(f"  Всего записей:         {total:>7}")
    print(f"  Уникальных:            {unique:>7}")
    print(f"  Дубликатов:            {duplicates:>7}")
    print(f"  Ошибок:                {errors:>7}")
    print(f"  Пустых ответов:        {empty:>7}")
    print(f"  Хороших ответов:       {good:>7} ({100*good/total:.1f}%)")
    
    # === 2. По категориям ===
    header("2. ПО КАТЕГОРИЯМ (уникальные)")
    
    cat_stats = defaultdict(lambda: {"total": 0, "good": 0, "error": 0, "empty": 0})
    for r in records:
        if r["is_duplicate"]:
            continue
        cat = r.get("category", "?")
        cat_stats[cat]["total"] += 1
        if "error" in r:
            cat_stats[cat]["error"] += 1
        elif not r.get("response"):
            cat_stats[cat]["empty"] += 1
        else:
            cat_stats[cat]["good"] += 1
    
    print(f"  {'Категория':<20} {'Всего':>7} {'Good':>7} {'Err':>5} {'Empty':>6} {'Quality':>8}")
    print("  " + "-" * 60)
    for cat, s in sorted(cat_stats.items()):
        q = 100 * s["good"] / s["total"] if s["total"] else 0
        print(f"  {cat:<20} {s['total']:>7} {s['good']:>7} {s['error']:>5} {s['empty']:>6} {q:>7.1f}%")
    
    # === 3. Производительность ===
    header("3. ПРОИЗВОДИТЕЛЬНОСТЬ")
    
    times = [r.get("elapsed", 0) for r in records if "elapsed" in r and r["elapsed"] > 0]
    if times:
        print(f"  Среднее время:         {mean(times):.2f} сек")
        print(f"  Медиана:               {median(times):.2f} сек")
        print(f"  Минимум:               {min(times):.2f} сек")
        print(f"  Максимум:              {max(times):.2f} сек")
        
        # Распределение
        buckets = {"<1s": 0, "1-2s": 0, "2-5s": 0, "5-10s": 0, ">10s": 0}
        for t in times:
            if t < 1: buckets["<1s"] += 1
            elif t < 2: buckets["1-2s"] += 1
            elif t < 5: buckets["2-5s"] += 1
            elif t < 10: buckets["5-10s"] += 1
            else: buckets[">10s"] += 1
        
        print(f"\n  Распределение времени:")
        for b, c in buckets.items():
            bar = "█" * int(50 * c / len(times))
            print(f"    {b:>6}: {c:>6} ({100*c/len(times):>5.1f}%) {bar}")
    
    # === 4. По портам / GPU ===
    header("4. РАСПРЕДЕЛЕНИЕ ПО GPU")
    
    port_map = {9200: "GPU 2", 9300: "GPU 0", 9301: "GPU 1"}
    ports = Counter(r.get("port", "?") for r in records if "port" in r)
    for port, count in sorted(ports.items()):
        gpu = port_map.get(port, "?")
        print(f"  Port {port} ({gpu}): {count:>6} ({100*count/sum(ports.values()):.1f}%)")
    
    # === 5. Анализ длины ответов ===
    header("5. КАЧЕСТВО ОПИСАНИЙ")
    
    good_records = [r for r in records if not r["is_duplicate"] and r.get("response") and "error" not in r]
    response_lengths = [len(r["response"]) for r in good_records]
    
    if response_lengths:
        print(f"  Средняя длина:         {mean(response_lengths):.0f} символов")
        print(f"  Медиана:               {median(response_lengths):.0f}")
        print(f"  Минимум:               {min(response_lengths)}")
        print(f"  Максимум:              {max(response_lengths)}")
    
    # Ключевые слова - научная терминология
    scientific_terms = [
        "cephalothorax", "exine", "chlorophyll", "necrotic", "chlorotic",
        "botryoidal", "crystalline", "hexagonal", "symmetry", "budding",
        "translucent", "segmented", "appendages", "antennae", "mandibles",
        "corona", "cilia", "dendritic", "fungal", "bacterial"
    ]
    
    term_counts = defaultdict(int)
    for r in good_records:
        resp_lower = r["response"].lower()
        for term in scientific_terms:
            if term in resp_lower:
                term_counts[term] += 1
    
    if term_counts:
        print(f"\n  Научные термины в ответах:")
        for term, c in sorted(term_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"    {term:<20} {c:>5} ({100*c/len(good_records):.1f}%)")
    
    # === 6. Примеры лучших ответов ===
    header("6. ПРИМЕРЫ ЛУЧШИХ ОПИСАНИЙ (по категориям)")
    
    # Лучший = самый длинный, содержит subject+features
    best_by_cat = {}
    for r in good_records:
        cat = r.get("category", "?")
        resp = r.get("response", "")
        if "subject" in resp.lower() and "feature" in resp.lower():
            score = len(resp)
            if cat not in best_by_cat or score > best_by_cat[cat]["score"]:
                best_by_cat[cat] = {"score": score, "record": r}
    
    for cat in sorted(best_by_cat.keys()):
        r = best_by_cat[cat]["record"]
        print(f"\n  ▸ [{cat}] {r.get('filename', '?')}")
        resp = r["response"][:400]
        print(f"    {resp}")
    
    # === 7. Подозрительные (очень короткие ответы) ===
    header("7. ПОДОЗРИТЕЛЬНЫЕ СЛУЧАИ")
    
    short = [r for r in good_records if len(r.get("response", "")) < 80]
    print(f"  Очень коротких ответов (<80 символов): {len(short)}")
    
    if short[:5]:
        print(f"\n  Первые 5 примеров:")
        for r in short[:5]:
            print(f"    [{r.get('category','?')}] {r.get('filename','?')[:40]}")
            print(f"      → {r.get('response','')[:100]}")
    
    # === 8. Ошибки по типам ===
    header("8. ТИПЫ ОШИБОК")
    
    error_records = [r for r in records if "error" in r]
    if error_records:
        error_types = Counter()
        for r in error_records:
            err = r.get("error", "")
            # Упрощаем ошибку до типа
            if "timeout" in err.lower():
                error_types["Timeout"] += 1
            elif "connection" in err.lower():
                error_types["Connection"] += 1
            elif "HTTP" in err:
                error_types["HTTP error"] += 1
            elif "read" in err.lower():
                error_types["File read"] += 1
            else:
                error_types[err[:50]] += 1
        
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {err_type:<40} {count:>5}")
    else:
        print("  Нет ошибок!")
    
    # === 9. Итог ===
    header("9. ИТОГ")
    
    print(f"  ✅ Готово для обучения Gemma: {good:,} уникальных пар")
    print(f"  ⚠️  Нужна дедупликация: убрать {duplicates} дубликатов")
    print(f"  📊 Качество: {100*good/total:.1f}% ответов успешны")
    
    # Рекомендация
    print(f"\n  Следующий шаг:")
    print(f"  1. Дедупликация vqa_all.jsonl → vqa_unique.jsonl")
    print(f"  2. Подготовка training split (train/val)")
    print(f"  3. Unsloth fine-tune Gemma 4 E4B")


if __name__ == "__main__":
    print("=" * 70)
    print("  MICROLENS VQA — ФИНАЛЬНЫЙ АНАЛИЗ")
    print("=" * 70)
    records = load_all()
    if not records:
        print("❌ Нет записей")
    else:
        analyze(records)
