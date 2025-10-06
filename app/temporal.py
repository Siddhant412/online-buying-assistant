from __future__ import annotations
import os, re
from pathlib import Path
from datetime import datetime, date, timezone
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(DOTENV_PATH, override=True)

_R_CLOSED = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_R_OPEN_1 = re.compile(r"^\s*(\d+)\s*\+\s*$")
_R_OPEN_2 = re.compile(r"^\s*(\d+)\s*-\s*\+\s*$")

def _parse_windows_from_env() -> List[Tuple[int, int, str]]:
    raw = os.getenv("REVIEW_WINDOWS", "")
    s = raw.strip().strip('"').strip("'")
    if s:
        windows: List[Tuple[int, int, str]] = []
        parts = [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip()]
        for p in parts:
            m = _R_CLOSED.match(p)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                label = f"{lo}-{hi} mo"
                windows.append((lo, hi, label))
                continue
            m = _R_OPEN_1.match(p) or _R_OPEN_2.match(p)
            if m:
                lo = int(m.group(1))
                hi = 10_000
                label = f"{lo}+ mo"
                windows.append((lo, hi, label))
                continue
            windows = []
            break
        if windows:
            windows.sort(key=lambda x: x[0])
            return windows

    # default window
    return [
        (0, 6, "0-6 mo"),
        (6, 12, "6-12 mo"),
        (12, 24, "12-24 mo"),
        (24, 120, "24+ mo"),
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

def analyze_temporal_conflict(passages: List[Dict], today: Optional[date] = None) -> Dict:

    today = today or datetime.now(timezone.utc).date()

    WINDOWS = _parse_windows_from_env()

    win_counts = [{"label": label, "n": 0, "pos": 0, "neg": 0, "neu": 0}
                  for (_, _, label) in WINDOWS]

    overall = {"n": 0, "pos": 0, "neg": 0, "neu": 0}

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
        for (lo, hi, _label), wc in zip(WINDOWS, win_counts):
            if lo <= months < hi:
                wc["n"] += 1
                wc[sent[:3]] += 1
                placed = True
                break
        if not placed and win_counts:
            win_counts[-1]["n"] += 1
            win_counts[-1][sent[:3]] += 1

    parts = []
    for wc in win_counts:
        if wc["n"] == 0:
            continue
        neg_rate = wc["neg"] / wc["n"]
        pos_rate = wc["pos"] / wc["n"]
        if neg_rate >= 0.6:
            dom = "mostly negative"
        elif pos_rate >= 0.6:
            dom = "mostly positive"
        else:
            dom = "mixed"
        parts.append(
            f"{wc['label']}: {dom} ({wc['pos']}/{wc['n']} positive, {wc['neg']}/{wc['n']} negative)"
        )

    overall_neg_rate = (overall["neg"] / overall["n"]) if overall["n"] else 0.0
    overall_pos_rate = (overall["pos"] / overall["n"]) if overall["n"] else 0.0
    summary = " | ".join(parts) if parts else "Not enough dated reviews to assess temporal trends."

    return {
        "by_window": [
            {
                "window": wc["label"],
                "n": wc["n"],
                "pos": wc["pos"],
                "neg": wc["neg"],
                "neu": wc["neu"],
                "pos_rate": round((wc["pos"] / wc["n"]), 2) if wc["n"] else 0.0,
                "neg_rate": round((wc["neg"] / wc["n"]), 2) if wc["n"] else 0.0,
            }
            for wc in win_counts if wc["n"] > 0
        ],
        "overall": {
            "n": overall["n"],
            "pos": overall["pos"],
            "neg": overall["neg"],
            "neu": overall["neu"],
            "pos_rate": round(overall_pos_rate, 2),
            "neg_rate": round(overall_neg_rate, 2),
        },
        "summary": summary,
    }
