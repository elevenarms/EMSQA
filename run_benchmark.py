#!/usr/bin/env python3
"""
Single-file EMSQA Benchmark Runner for HuggingFace Transformers models.

Usage examples:
    # Dry run on first 5 samples with a small model
    python run_benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --start 0 --end 5

    # Eval-only mode on saved logs
    python run_benchmark.py --mode eval --model Qwen/Qwen2.5-1.5B-Instruct

    # Full run with default 8B model
    python run_benchmark.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import csv
import json
import os
import re
import time

import numpy as np
import torch
import transformers
from tqdm import tqdm

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)

# ── prompt templates (inline) ────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "zeroshot": (
        "You are an expert in EMS. Choose the correct answer to the following "
        "multiple-choice question. Respond ONLY in the JSON schema provided."
    ),
    "zeroshot_attr": (
        "You are an expert in EMS. You will be given a certification level, a "
        "question category, and a multiple-choice question. Choose the correct "
        "answer to the following multiple-choice question. Respond ONLY in the "
        "JSON schema provided."
    ),
    "cot": (
        "You are an expert in EMS. Choose the correct answer to the following "
        "multiple-choice question. Please first think step-by-step and then "
        "choose the answer from the provided options. Respond ONLY in the JSON "
        "schema provided."
    ),
    "cot_attr": (
        "You are an expert Emergency Medical Services (EMS) educator. Answer "
        "multiple-choice questions at the requested certification depth, using "
        "evidence-based reasoning. Respond ONLY in the JSON schema provided."
    ),
}

USER_TEMPLATES = {
    "zeroshot": (
        "Question: {question}\n\nChoices:\n{choices}\n\n"
        'Return your final answer as a lowercase letter in strict JSON format, like: ["a"]'
    ),
    "zeroshot_attr": (
        "Certification Level: {level}\n\nCategory: {category}\n\n"
        "Question: {question}\n\nChoices:\n{choices}\n\n"
        'Return your final answer as a lowercase letter in strict JSON format, like: ["a"]'
    ),
    "cot": (
        "Question: {question}\n\nChoices:\n{choices}\n\n"
        "Think step-by-step, then output JSON exactly in this format (no extra keys):\n\n"
        "```json\n{{\n"
        '  "step_by_step_thinking": "<concise explanation here>",\n'
        '  "answer": "A" or ["A", "B"]\n'
        "}}\n```"
    ),
    "cot_attr": (
        "Certification_Level: {level}\nCategory: {category}\n\n"
        "Question: {question}\n\nChoices:\n{choices}\n\n"
        "Think step-by-step from the standpoint of {category}, then output JSON "
        "exactly in this format (no extra keys):\n\n"
        "```json\n{{\n"
        '  "step_by_step_thinking": "<concise explanation here>",\n'
        '  "answer": "A" or ["A", "B"]\n'
        "}}\n```"
    ),
}

COT_MODES = {"cot", "cot_attr"}

CATEGORIES = [
    "airway_respiration_and_ventilation",
    "anatomy",
    "assessment",
    "cardiology_and_resuscitation",
    "ems_operations",
    "medical_and_obstetrics_gynecology",
    "others",
    "pediatrics",
    "pharmacology",
    "trauma",
]


# ── answer extraction (mirrors existing benchmark) ───────────────────────────
def extract_json_answer(response):
    """Extract predicted answer list from model response.

    Returns a list like ["a"] on success, or None on failure.
    """
    # 1) Try JSON array pattern  e.g. ["a"]
    matches = re.findall(r'\[.*?\]', response, re.DOTALL)
    if matches:
        json_str = matches[0] if len(matches) == 1 else matches[-1]
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return [str(x).lower().strip() for x in data]
        except (json.JSONDecodeError, TypeError):
            pass

    # 2) Fallback: "Answer: a" style patterns
    match = re.search(
        r"(?:The\s+correct\s+answer\s+is|Answer|Correct\s+option)[:\s]*"
        r"([a-zA-Z])(?:[.)]?\s*([^\n.]+)?)?",
        response, re.IGNORECASE,
    )
    if match:
        option = match.group(1).lower()
        if option in "abcdefg":
            return [option]

    # 3) Fallback: LaTeX \boxed{a}
    boxed = re.search(r'\\boxed\{\s*([A-Ga-g])\s*\}', response)
    if boxed:
        return [boxed.group(1).lower()]

    return None


def extract_cot_answer(response):
    """Extract answer from a CoT JSON-object response like {"answer": "a", ...}.

    Returns (answer_list, thinking_str). answer_list is like ["a"] or None.
    """
    matches = re.findall(r'\{.*?\}', response, re.DOTALL)
    for json_str in reversed(matches):  # prefer last match
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict) and "answer" in obj:
                ans = obj["answer"]
                thinking = obj.get("step_by_step_thinking", "")
                if isinstance(ans, list):
                    return [str(x).lower().strip() for x in ans], thinking
                elif isinstance(ans, str):
                    return [ans.lower().strip()], thinking
        except (json.JSONDecodeError, TypeError):
            continue

    # Fallback: try the array-based extractor
    pred = extract_json_answer(response)
    return pred, None


# ── metric functions (mirrors existing benchmark) ────────────────────────────
def exact_match_accuracy(preds, golds):
    correct = sum(set(p) == set(g) for p, g in zip(preds, golds))
    return correct / len(golds)


def average_f1(preds, golds):
    def f1_per_sample(pred, gold):
        pred_set, gold_set = set(pred), set(gold)
        if not pred_set and not gold_set:
            return 1.0
        if not pred_set or not gold_set:
            return 0.0
        tp = len(pred_set & gold_set)
        precision = tp / len(pred_set)
        recall = tp / len(gold_set)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    return sum(f1_per_sample(p, g) for p, g in zip(preds, golds)) / len(golds)


# ── helpers ──────────────────────────────────────────────────────────────────
def json_exists_and_nonempty(path):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return bool(data)
    except (json.JSONDecodeError, OSError):
        return False


def model_short_name(model_name_or_path):
    """Derive a filesystem-safe short name from a model identifier."""
    return model_name_or_path.rstrip("/").replace("/", "_")


# ── inference ────────────────────────────────────────────────────────────────
def run_inference(args):
    # Load data
    with open(args.data, "r") as f:
        data = json.load(f)

    end_idx = args.end if args.end != -1 else len(data)
    data_slice = data[args.start:end_idx]

    # Directories
    short = model_short_name(args.model)
    log_dir = args.log_dir or os.path.join("logs", short, args.prompt)
    os.makedirs(log_dir, exist_ok=True)

    # Load model
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    print(f"Loading model: {args.model}  dtype={args.dtype}")
    pipe = transformers.pipeline(
        "text-generation",
        model=args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    sys_prompt = SYSTEM_PROMPTS[args.prompt]
    user_tpl = USER_TEMPLATES[args.prompt]

    print(f"Running inference on {len(data_slice)} samples "
          f"(indices {args.start}..{args.start + len(data_slice) - 1})")

    for offset, sample in enumerate(tqdm(data_slice, desc="Inference")):
        idx = args.start + offset
        log_path = os.path.join(log_dir, f"{idx}.json")

        # Resume: skip if already done
        if json_exists_and_nonempty(log_path):
            continue

        # Build prompt
        question = sample["question"]
        choices = "\n".join(sample["choices"])
        level = sample["level"][0]
        category = ";".join(sample["category"])

        if args.prompt in ("zeroshot", "cot"):
            user_msg = user_tpl.format(question=question, choices=choices)
        else:  # zeroshot_attr, cot_attr
            user_msg = user_tpl.format(
                level=level, category=category,
                question=question, choices=choices,
            )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]

        # Generate
        t0 = time.time()
        outputs = pipe(
            messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
        )
        t_infer = time.time() - t0

        response = outputs[0]["generated_text"][-1]["content"]

        # Extract answer
        is_cot = args.prompt in COT_MODES
        if is_cot:
            pred, thinking = extract_cot_answer(response)
        else:
            pred = extract_json_answer(response)
            thinking = None

        if pred is not None:
            log_obj = {"pred": pred, "time": t_infer}
            if thinking is not None:
                log_obj["step_by_step_thinking"] = thinking
            with open(log_path, "w") as f:
                json.dump(log_obj, f, indent=4)
        else:
            # Save raw response as .txt for debugging, and a json with null pred
            txt_path = os.path.join(log_dir, f"{idx}.txt")
            with open(txt_path, "w") as f:
                f.write(response)
            with open(log_path, "w") as f:
                json.dump({"pred": None, "time": t_infer}, f, indent=4)
            print(f"\n[WARN] Sample {idx}: could not extract answer. Raw saved to {txt_path}")

    # Release model from GPU before returning
    del pipe
    return log_dir


# ── evaluation ───────────────────────────────────────────────────────────────
def run_evaluation(args, log_dir=None):
    with open(args.data, "r") as f:
        data = json.load(f)

    short = model_short_name(args.model)
    if log_dir is None:
        log_dir = args.log_dir or os.path.join("logs", short, args.prompt)
    results_dir = args.results_dir or os.path.join("results", short, args.prompt)
    os.makedirs(results_dir, exist_ok=True)

    # Buckets by level
    levels = ["emr", "emt", "aemt", "paramedic", "all"]
    preds = {k: [] for k in levels}
    golds = {k: [] for k in levels}
    times = {k: [] for k in levels}

    # Buckets by category (overall)
    preds_cat_all = {c: [] for c in CATEGORIES}
    golds_cat_all = {c: [] for c in CATEGORIES}
    times_cat_all = {c: [] for c in CATEGORIES}

    # Buckets by level x category
    cert_levels = ["emr", "emt", "aemt", "paramedic"]
    preds_cat = {lv: {c: [] for c in CATEGORIES} for lv in cert_levels}
    golds_cat = {lv: {c: [] for c in CATEGORIES} for lv in cert_levels}
    times_cat = {lv: {c: [] for c in CATEGORIES} for lv in cert_levels}

    skipped = 0
    for i, item in enumerate(data):
        log_path = os.path.join(log_dir, f"{i}.json")
        if not os.path.isfile(log_path):
            skipped += 1
            continue
        with open(log_path, "r") as f:
            pred_data = json.load(f)

        pred_raw = pred_data["pred"]
        if pred_raw is None:
            skipped += 1
            continue

        if isinstance(pred_raw, list):
            pred_lst = [str(p).lower().strip() for p in pred_raw]
        else:
            pred_lst = [str(pred_raw).lower().strip()]

        gold = item["answer"]  # string like "a"
        t = pred_data.get("time", 0.0) or 0.0
        cat_list = item["category"] if isinstance(item["category"], list) else [item["category"]]

        # Overall
        preds["all"].append(pred_lst)
        golds["all"].append(gold)
        times["all"].append(t)

        # Per level
        for lv in item["level"]:
            preds[lv].append(pred_lst)
            golds[lv].append(gold)
            times[lv].append(t)
            for cat in cat_list:
                preds_cat[lv][cat].append(pred_lst)
                golds_cat[lv][cat].append(gold)
                times_cat[lv][cat].append(t)

        # Per category (overall)
        for cat in cat_list:
            preds_cat_all[cat].append(pred_lst)
            golds_cat_all[cat].append(gold)
            times_cat_all[cat].append(t)

    if skipped:
        print(f"[INFO] Skipped {skipped} samples (missing log or null pred)")

    # ── Per-level summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Model: {args.model}  |  Prompt: {args.prompt}")
    print(f"{'='*60}")
    per_level_rows = []
    for key in ["emr", "emt", "aemt", "paramedic", "all"]:
        if not golds[key]:
            continue
        acc = exact_match_accuracy(preds[key], golds[key])
        f1 = average_f1(preds[key], golds[key])
        t_avg = np.mean(times[key])
        n = len(golds[key])
        print(f"  {key:12} | n={n:5} | acc={acc:.4f} | f1={f1:.4f} | t_avg={t_avg:.3f}s")
        per_level_rows.append({"level": key, "n": n, "acc": acc, "f1": f1, "t_avg": t_avg})

    # ── Per-category summary ─────────────────────────────────────────────
    print(f"\nPer-category (all levels):")
    per_cat_rows = []
    for cat in CATEGORIES:
        if not golds_cat_all[cat]:
            continue
        acc = exact_match_accuracy(preds_cat_all[cat], golds_cat_all[cat])
        f1 = average_f1(preds_cat_all[cat], golds_cat_all[cat])
        t_avg = np.mean(times_cat_all[cat])
        n = len(golds_cat_all[cat])
        print(f"  {cat:45} | n={n:4} | acc={acc:.4f} | f1={f1:.4f}")
        per_cat_rows.append({"category": cat, "n": n, "acc": acc, "f1": f1, "t_avg": t_avg})

    # ── Per level x category ─────────────────────────────────────────────
    print(f"\nPer level x category:")
    level_cat_rows = []
    for lv in cert_levels:
        for cat in CATEGORIES:
            if not golds_cat[lv][cat]:
                continue
            acc = exact_match_accuracy(preds_cat[lv][cat], golds_cat[lv][cat])
            f1 = average_f1(preds_cat[lv][cat], golds_cat[lv][cat])
            t_avg = np.mean(times_cat[lv][cat])
            n = len(golds_cat[lv][cat])
            level_cat_rows.append({
                "level": lv, "category": cat,
                "n": n, "acc": acc, "f1": f1, "t_avg": t_avg,
            })
        # Print a header per level
        lv_rows = [r for r in level_cat_rows if r["level"] == lv]
        if lv_rows:
            print(f"  {lv.upper()}:")
            for r in lv_rows:
                print(f"    {r['category']:43} | n={r['n']:4} | acc={r['acc']:.4f} | f1={r['f1']:.4f}")

    # ── Save CSVs ────────────────────────────────────────────────────────
    def _write_csv(path, fieldnames, rows):
        with open(path, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    _write_csv(
        os.path.join(results_dir, "per_level.csv"),
        ["level", "n", "acc", "f1", "t_avg"], per_level_rows,
    )
    _write_csv(
        os.path.join(results_dir, "per_category.csv"),
        ["category", "n", "acc", "f1", "t_avg"], per_cat_rows,
    )
    _write_csv(
        os.path.join(results_dir, "per_level_category.csv"),
        ["level", "category", "n", "acc", "f1", "t_avg"], level_cat_rows,
    )
    print(f"\nCSVs saved to {results_dir}/")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="EMSQA Benchmark Runner (HuggingFace Transformers)",
    )
    parser.add_argument("--model", type=str, nargs="+",
                        default=["meta-llama/Llama-3.1-8B-Instruct"],
                        help="One or more HF model names / local paths "
                             "(run sequentially)")
    parser.add_argument("--prompt", type=str, default="zeroshot",
                        choices=["zeroshot", "zeroshot_attr", "cot", "cot_attr"],
                        help="Prompt mode")
    parser.add_argument("--data", type=str,
                        default="data/final/test_open.json",
                        help="Path to test data JSON")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for per-sample log JSONs "
                             "(default: logs/{model}/{prompt})")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory for result CSVs "
                             "(default: results/{model}/{prompt})")
    parser.add_argument("--mode", type=str, default="infer",
                        choices=["infer", "eval"],
                        help="'infer' = inference + eval, 'eval' = eval only")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index of data slice")
    parser.add_argument("--end", type=int, default=-1,
                        help="End index of data slice (-1 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Max generation length (default: 128 for zeroshot, "
                             "8192 for cot modes)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "auto"],
                        help="Model dtype")
    args = parser.parse_args()

    # Default max_new_tokens: short for MCQA, longer for CoT
    if args.max_new_tokens is None:
        args.max_new_tokens = 8192 if args.prompt in COT_MODES else 128

    models = args.model  # list of 1+ model names
    for i, model_name in enumerate(models):
        if len(models) > 1:
            print(f"\n{'#'*60}")
            print(f"# Model {i+1}/{len(models)}: {model_name}")
            print(f"{'#'*60}")

        # Set the current model and reset per-model dirs so each model
        # gets its own log/results path (unless user explicitly set them
        # for a single-model run).
        args.model = model_name
        if len(models) > 1:
            args.log_dir = None
            args.results_dir = None

        if args.mode == "infer":
            log_dir = run_inference(args)
            print("\nInference complete. Running evaluation...")
            run_evaluation(args, log_dir=log_dir)
        else:
            run_evaluation(args)

        # Free GPU memory before loading the next model
        if i < len(models) - 1:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
