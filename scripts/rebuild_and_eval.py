#!/usr/bin/env python
"""
Convenience script to:
1) Re-ingest + chunk + index the sample products.
2) Run eval against one or more datasets and a model list.

Usage:
  python scripts/rebuild_and_eval.py \
    --products pan.html:pan_reviews.csv airfryer.html:airfryer_reviews.csv \
    --datasets eval/groundtruth_dataset.jsonl eval/groundtruth_dataset_additional.jsonl \
    --models llama3.1:8b,mistral:7b,qwen2:7b,phi3:14b --k 5
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

# add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.ingest import ingest_one
from app.chunk import chunk_processed
from app.indexer import build_indexes


def ingest_and_index(html_csv_pairs):
    for html, csv in html_csv_pairs:
        path = ingest_one(html, csv)
        chunks = chunk_processed(path)
        build_indexes(chunks[0]["product_id"])
        print(f"[ingest] {html} + {csv} -> product_id={chunks[0]['product_id']} (chunks={len(chunks)})")


def run_eval(datasets, models, k):
    cmd = [
        sys.executable,
        str(ROOT / "eval" / "run_eval.py"),
        "--k",
        str(k),
    ]
    for ds in datasets:
        cmd.extend(["--dataset", str(ds)])
    if models:
        cmd.extend(["--models", models])
    print(f"[eval] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_products(raw_list):
    pairs = []
    for item in raw_list:
        if ":" not in item:
            raise ValueError(f"Product spec must be html:csv, got {item}")
        html, csv = item.split(":", 1)
        pairs.append((html, csv))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--products",
        nargs="+",
        default=[
            "pan.html:pan_reviews.csv",
            "airfryer.html:airfryer_reviews.csv",
            "blender.html:blender_reviews.csv",
            "vacuum.html:vacuum_reviews.csv",
        ],
        help="Pairs of HTML:CSV filenames under data/raw/",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["eval/groundtruth_dataset.jsonl", "eval/groundtruth_dataset_additional.jsonl"],
        help="Eval dataset JSONL paths (repeatable)",
    )
    ap.add_argument("--models", type=str, default="llama3.1:8b,mistral:7b,qwen2:7b,phi3:14b", help="Comma-separated models")
    ap.add_argument("--k", type=int, default=5, help="k for retrieval metrics")
    args = ap.parse_args()

    ingest_and_index(parse_products(args.products))
    run_eval(args.datasets, args.models, args.k)


if __name__ == "__main__":
    main()
