"""Leakage-safe train, validation and test split for ARCADE.

ARCADE ships an official split (1000 / 200 / 300) but no patient or sequence
identifiers, so a patient-level split is not possible. This module keeps the
official split, which keeps results comparable with other work, and adds a
leakage guard built from the QA report:

1. Images connected by an exact duplicate or a near duplicate (thumbnail
   correlation at or above the QA threshold) form one group (union-find).
2. A group that spans splits stays in the earliest split (train, then val,
   then test); its members in later splits are excluded.
3. Within a split, only the first copy of a pixel-identical image is kept.

The result goes to data/processed/splits.csv and, because it holds only ids,
to results/splits.csv for traceability. Counts go to results/split_summary.json.

Usage: python -m angiolens.data.split
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter

from angiolens.paths import RESULTS, SPLITS_CSV, ensure

ORDER = {"train": 0, "val": 1, "test": 2}


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _num(image_id: str) -> tuple[int, int]:
    split, stem = image_id.split("_", 1)
    return ORDER.get(split, 9), int(stem) if stem.isdigit() else 0


def make_split(rows: list[dict], exact_groups: list[list[str]], near_pairs: list[dict]) -> list[dict]:
    """rows: dicts with image_id and split. Returns rows with group, use and reason."""
    split_of = {r["image_id"]: r["split"] for r in rows}
    uf = UnionFind(split_of)
    for ids in exact_groups:
        for other in ids[1:]:
            uf.union(ids[0], other)
    for p in near_pairs:
        uf.union(p["a"], p["b"])

    members: dict[str, list[str]] = {}
    for image_id in split_of:
        members.setdefault(uf.find(image_id), []).append(image_id)

    exact_of = {}
    for ids in exact_groups:
        ids = sorted(ids, key=_num)
        for later in ids[1:]:
            exact_of[later] = ids[0]

    out = []
    for r in rows:
        image_id = r["image_id"]
        group = members[uf.find(image_id)]
        home = min((split_of[g] for g in group), key=ORDER.get)
        use, reason = 1, ""
        if split_of[image_id] != home:
            use = 0
            partners = sorted((g for g in group if split_of[g] == home), key=_num)
            reason = f"near duplicate of {home} image {partners[0]}"
        elif image_id in exact_of and split_of[exact_of[image_id]] == split_of[image_id]:
            use = 0
            reason = f"exact duplicate of {exact_of[image_id]}"
        out.append({
            "image_id": image_id,
            "split": r["split"],
            "group": min(group, key=_num),
            "group_size": len(group),
            "use": use,
            "reason": reason,
        })
    return out


def read_splits(path=SPLITS_CSV, used_only: bool = True) -> dict[str, list[str]]:
    """Map split name to the list of image ids in that split."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for r in rows:
        if used_only and r["use"] != "1":
            continue
        result[r["split"]].append(r["image_id"])
    return result


def main() -> int:
    from angiolens.data.arcade import read_index

    qa = json.loads((RESULTS / "data_qa.json").read_text())
    rows = read_index()
    exact = [e["image_ids"] for e in qa["exact_duplicates"]]
    near = qa["near_duplicates"]["pairs"]
    split_rows = make_split(rows, exact, near)
    ensure(SPLITS_CSV.parent, RESULTS)
    for path in (SPLITS_CSV, RESULTS / "splits.csv"):
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(split_rows[0].keys()))
            writer.writeheader()
            writer.writerows(split_rows)
    official = Counter(r["split"] for r in split_rows)
    used = Counter(r["split"] for r in split_rows if r["use"] == 1)
    excluded = [{"image_id": r["image_id"], "reason": r["reason"]} for r in split_rows if r["use"] == 0]
    summary = {
        "policy": "official ARCADE split plus a duplicate and near-duplicate leakage guard",
        "near_duplicate_threshold": qa["near_duplicates"]["threshold"],
        "official_counts": dict(official),
        "used_counts": dict(used),
        "excluded": excluded,
        "patient_ids_available": False,
    }
    (RESULTS / "split_summary.json").write_text(json.dumps(summary, indent=1))
    print("official", dict(official), "used", dict(used))
    for e in excluded:
        print("  excluded", e["image_id"], e["reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
