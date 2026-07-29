#!/usr/bin/env python3
"""Aggregate grasp-eval results into the results tables for docs/REPORT.md.

Reads outputs/<obj>/eval/*_results.json (from run_grasp_eval2x.py runs) and prints
markdown tables: headline results, per-epoch DRO checkpoint curve, and paper reference.
REPORT.md is hand-maintained; run this to regenerate the numbers that go into it.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

OBJECTS = ["spraybottle", "bottle", "can", "bulb", "hat", "jengabox"]

HEADLINE = [
    ("retarget (direct)", "retarget_results.json"),
    ("retarget+refined", "retarget_refined_results.json"),
    ("retarget+squeeze1.1", "retarget_squeeze11_results.json"),
    ("DRO mixed e5", "dro_unfiltered_results.json"),
    ("DRO mixed e5 +refined", "dro_refined_results.json"),
    ("DRO mixed3x e5", "dro_mixed3x_e5_results.json"),
]

EPOCH_FILES = {
    "mixed e5": "dro_unfiltered_results.json",
    "mixed e10": "dro_e10_results.json",
    "mixed e15": "dro_e15_results.json",
    "mixed e20": "dro_e20_results.json",
    "mixed e25": "dro_mixed_e25_results.json",
    "mixed e30": "dro_mixed_e30_results.json",
    "mixed e35": "dro_mixed_e35_results.json",
    "mixed e40": "dro_mixed_e40_results.json",
    "mixed3x e5": "dro_mixed3x_e5_results.json",
    "mixed3x e10": "dro_mixed3x_e10_results.json",
    "mixed3x e15": "dro_mixed3x_e15_results.json",
    "mixed3x e20": "dro_mixed3x_e20_results.json",
    "mixed3x e25": "dro_mixed3x_e25_results.json",
    "mixed3x e30": "dro_mixed3x_e30_results.json",
    "mixed3x e35": "dro_mixed3x_e35_results.json",
    "mixed3x e40": "dro_mixed3x_e40_results.json",
}


def load(obj, fname):
    p = OUT / obj / "eval" / fname
    if not p.exists():
        return None
    return json.loads(p.read_text())["success_rate"] * 100


def main() -> None:
    print("### Headline (100 trials/object for DRO rows, 20 for retarget rows)\n")
    hdr = "| method | " + " | ".join(OBJECTS) + " | mean |"
    print(hdr)
    print("|" + "---|" * (len(OBJECTS) + 2))
    for tag, fname in HEADLINE:
        vals = [load(o, fname) for o in OBJECTS]
        cells = [f"{v:.0f}" if v is not None else "-" for v in vals]
        have = [v for v in vals if v is not None]
        mean = f"{np.mean(have):.1f}" if have else "-"
        print(f"| {tag} | " + " | ".join(cells) + f" | {mean} |")
    print("\n### DRO checkpoint curve (mean success % over 6 objects)\n")
    print("| run/epoch | " + " | ".join(OBJECTS) + " | mean |")
    print("|" + "---|" * (len(OBJECTS) + 2))
    for tag, fname in EPOCH_FILES.items():
        vals = [load(o, fname) for o in OBJECTS]
        cells = [f"{v:.0f}" if v is not None else "-" for v in vals]
        have = [v for v in vals if v is not None]
        mean = f"{np.mean(have):.1f}" if have else "-"
        print(f"| {tag} | " + " | ".join(cells) + f" | {mean} |")


if __name__ == "__main__":
    main()
