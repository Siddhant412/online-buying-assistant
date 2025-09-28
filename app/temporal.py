from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import List, Dict, Optional
from collections import defaultdict
import math

WINDOWS = [
    (0, 6,   "0-6 mo"),
    (6, 12,  "6-12 mo"),
    (12, 24, "12-24 mo"),
    (24, 120,"24+ mo"),
]

def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return date.fromisoformat(s)
        return datetime.fromisoformat(s).date()
    except Exception:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except Exception:
            return None

def _months_ago(d: date, today: Optional[date] = None) -> Optional[int]:
    if not d:
        return None
    today = today or datetime.now(timezone.utc).date()
    return max(0, (today.year - d.year) * 12 + (today.month - d.month))

def _sentiment_from_rating(r: Optional[int]) -> str:
    try:
        r = int(r)
    except Exception:
        return "neutral"
    if r >= 4:
        return "positive"
    if r <= 2:
        return "negative"
    return "neutral"

def analyze_temporal_conflict(passages: List[Dict]) -> Dict:
    today = datetime.now(timezone.utc).date()
    counts = []
    overall = {"n": 0, "pos": 0, "neg": 0, "neu": 0}

    win_counts = []
    for lo, hi, label in WINDOWS:
        win_counts.append({"label": label, "n": 0, "pos": 0, "neg": 0, "neu": 0})

    for p in passages:
        if p.get("source_type") != "review":
            continue
        meta = p.get("meta", {})
        d = _parse_date(meta.get("date"))
        months = _months_ago(d, today) if d else None
        sent = _sentiment_from_rating(meta.get("rating"))

        overall["n"] += 1
        overall[sent[:3]] += 1

        if months is None:
            continue
        placed = False
        for (lo, hi, label), wc in zip(WINDOWS, win_counts):
            if lo <= months < hi:
                wc["n"] += 1
                wc[sent[:3]] += 1
                placed = True
                break
        if not placed:
            win_counts[-1]["n"] += 1
            win_counts[-1][sent[:3]] += 1

    parts = []
    for wc in win_counts:
        if wc["n"] == 0:
            continue
        neg_rate = wc["neg"] / wc["n"]
        pos_rate = wc["pos"] / wc["n"]
        dom = "mixed"
        if neg_rate >= 0.6:
            dom = "mostly negative"
        elif pos_rate >= 0.6:
            dom = "mostly positive"
        elif abs(pos_rate - neg_rate) <= 0.2:
            dom = "mixed"
        else:
            dom = "mixed"
        parts.append(f"{wc['label']}: {dom} ({wc['pos']}/{wc['n']} positive, {wc['neg']}/{wc['n']} negative)")

    overall_neg_rate = (overall["neg"] / overall["n"]) if overall["n"] else 0.0
    overall_pos_rate = (overall["pos"] / overall["n"]) if overall["n"] else 0.0

    if parts:
        summary = " | ".join(parts)
    else:
        summary = "Not enough dated reviews to assess temporal trends."

    return {
        "by_window": [
            {
                "window": wc["label"],
                "n": wc["n"],
                "pos": wc["pos"],
                "neg": wc["neg"],
                "neu": wc["neu"],
                "pos_rate": round((wc["pos"]/wc["n"]), 2) if wc["n"] else 0.0,
                "neg_rate": round((wc["neg"]/wc["n"]), 2) if wc["n"] else 0.0,
            } for wc in win_counts if wc["n"] > 0
        ],
        "overall": {
            "n": overall["n"],
            "pos": overall["pos"],
            "neg": overall["neg"],
            "neu": overall["neu"],
            "pos_rate": round(overall_pos_rate, 2),
            "neg_rate": round(overall_neg_rate, 2),
        },
        "summary": summary
    }
