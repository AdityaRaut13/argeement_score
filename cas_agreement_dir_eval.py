#!/usr/bin/env python3
"""
Directory-wise GeoJSON inter-annotator agreement evaluation using CAS (no confidence scores).

This script supports three evaluation modes:

1) one_to_one (default)
   - Per class: compute CAS(P,G) for all pairs, do 1-1 matching (greedy or Hungarian),
     then compute TP/FP/FN at CAS threshold tau and report Precision/Recall/F1.

2) union_per_class (split/merge robust, no instance matching)
   - Per class: take the union of all regions for that class per annotator and compute
     area-based Precision/Recall/F1 using intersection area.
   - Also reports union-level CAS/OC/AR as diagnostics.

3) components (split/merge robust, count-based, no 1-1)
   - Per class: build a bipartite graph between A-instances and B-instances with edges
     whenever CAS >= tau. Connected components represent merge/split groups.
   - A component is counted as:
        TP if it contains at least one A and one B instance,
        FN if it contains only A instances,
        FP if it contains only B instances.
   - Each component is also scored by CAS/OC/AR between the unions of A- and B-geometry
     inside the component (reported as a quality diagnostic).
   - This avoids the "one big box vs many small boxes" failure of 1-1 matching, while
     still providing TP/FP/FN-style counts.

Folder matching:
- Recursively finds all *.geojson under each root.
- Groups by relative parent directory (relative to root).
- For each key present in both roots, evaluates that folder pair.

Optional exports per matched folder:
- combined_AB.geojson (both sources with source-tagged labels + match metadata)
- matched_AB.geojson (one_to_one: matched pairs; components: TP-components only; union: not written)

Outputs per matched folder:
- per_class_metrics.csv
- summary.json
- matches.csv
- combined_AB.geojson (optional)
- matched_AB.geojson (optional; depends on mode)

Global outputs:
- cases_summary.csv
- overall.txt
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

try:
    from shapely.strtree import STRtree
    _HAS_STRTREE = True
except Exception:
    _HAS_STRTREE = False

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# Optional Hungarian matching
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Ann:
    uid: str                 # unique within a case
    geom: Any                # shapely geometry
    cls: str
    source: str              # "A" or "B"
    src_file: str            # which geojson file it came from
    feat_id: str             # original feature id (stringified)
    props: Dict[str, Any]    # original properties


@dataclass
class MatchRow:
    # Flexible row used for matches.csv (works for all modes)
    case_key: str
    cls: str
    mode: str
    unit_id: str           # pair_id (one_to_one) or component_id (components) or 'UNION'
    status: str            # matched/unmatched/union_mode/component_tp/component_fp/component_fn
    nA: int
    nB: int
    a_uids: str
    b_uids: str
    a_feat_ids: str
    b_feat_ids: str
    a_src_files: str
    b_src_files: str
    cas: float
    oc: float
    ar: float
    area_a: float
    area_b: float
    area_intersection: float


# -----------------------------
# GeoJSON parsing
# -----------------------------
def _extract_class(props: dict) -> str:
    """
    QuPath commonly stores class name in properties.classification.name.
    """
    if not isinstance(props, dict):
        return "unknown"

    c = props.get("classification", None)
    if isinstance(c, dict):
        name = c.get("name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()
    if isinstance(c, str) and c.strip():
        return c.strip()

    # fallback keys
    name = props.get("name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()

    return "unknown"


def load_qupath_geojson(
    path: Path,
    source: str,
    uid_prefix: str,
    keep_classes: Optional[set[str]] = None
) -> List[Ann]:
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features", [])
    out: List[Ann] = []

    for i, f in enumerate(feats):
        geom = f.get("geometry", None)
        props = f.get("properties", {}) or {}
        if geom is None:
            continue

        g = shape(geom)
        if g.is_empty:
            continue
        if g.geom_type not in ("Polygon", "MultiPolygon"):
            continue

        cls = _extract_class(props)
        if keep_classes is not None and cls not in keep_classes:
            continue

        fid = f.get("id", i)
        uid = f"{uid_prefix}:{source}:{fid}"

        out.append(
            Ann(
                uid=uid,
                geom=g,
                cls=cls,
                source=source,
                src_file=str(path),
                feat_id=str(fid),
                props=props,
            )
        )
    return out


def collect_geojson_by_rel_parent(root: Path) -> Dict[str, List[Path]]:
    """Recursively find all *.geojson under root; group by relative parent directory."""
    root = root.expanduser().resolve()
    grouped: Dict[str, List[Path]] = {}
    for p in root.rglob("*.geojson"):
        try:
            rel_parent = p.parent.relative_to(root).as_posix()
        except Exception:
            rel_parent = p.parent.name
        grouped.setdefault(rel_parent, []).append(p)
    return grouped


# -----------------------------
# CAS definition
# -----------------------------
def _safe_log(x: float) -> float:
    if x <= 0:
        return float("inf")
    return math.log(x)


def cas_score(a_geom: Any, b_geom: Any, sigma: float) -> Tuple[float, float, float]:
    """
    CAS, OC, AR for two geometries.

    OC = inter / min(area_a, area_b)
    AR = max(area_a, area_b) / min(area_a, area_b)
    CAS = OC * exp( - (ln AR)^2 / (2*sigma^2) )
    If sigma is inf or <= 0, CAS == OC.
    """
    a_area = float(a_geom.area)
    b_area = float(b_geom.area)
    if a_area <= 0 or b_area <= 0:
        return 0.0, 0.0, float("inf")

    inter = float(a_geom.intersection(b_geom).area)
    if inter <= 0:
        return 0.0, 0.0, max(a_area, b_area) / min(a_area, b_area)

    mn = min(a_area, b_area)
    mx = max(a_area, b_area)
    oc = inter / mn if mn > 0 else 0.0
    ar = mx / mn if mn > 0 else float("inf")

    if (not math.isfinite(sigma)) or sigma <= 0:
        return oc, oc, ar

    ln_ar = _safe_log(ar)
    penalty = math.exp(-(ln_ar * ln_ar) / (2.0 * sigma * sigma))
    return oc * penalty, oc, ar


# -----------------------------
# STRtree helpers
# -----------------------------
def _build_tree(geoms: List[Any]):
    """
    STRtree compatibility:
    - shapely 2: query returns indices
    - shapely 1.8: query returns geometries
    """
    if not _HAS_STRTREE or len(geoms) == 0:
        return None, None
    tree = STRtree(geoms)
    id2idx = {id(g): i for i, g in enumerate(geoms)}  # for shapely 1.8
    return tree, id2idx


def _tree_query_indices(tree, id2idx, geom) -> List[int]:
    res = tree.query(geom)
    if len(res) == 0:
        return []
    first = res[0]
    if isinstance(first, (int, np.integer)):
        return [int(x) for x in res]  # shapely 2
    return [id2idx[id(g)] for g in res]  # shapely 1.8


# -----------------------------
# Utilities
# -----------------------------
def safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _join_limited(items: List[str], limit: int = 60) -> str:
    if not items:
        return ""
    if len(items) <= limit:
        return ";".join(items)
    return ";".join(items[:limit]) + f";...(+{len(items)-limit} more)"


def safe_rel_key_to_dir(key: str) -> Path:
    if key in ("", "."):
        return Path("__root__")
    return Path(key)


# -----------------------------
# one_to_one matching
# -----------------------------
def greedy_match_cas(
    A: List[Ann],
    B: List[Ann],
    cas_thr: float,
    sigma: float
) -> List[Tuple[int, int, float, float, float]]:
    """Greedy max-CAS 1-1 assignment. Returns (iA, iB, cas, oc, ar)."""
    if len(A) == 0 or len(B) == 0:
        return []

    geomsB = [x.geom for x in B]
    tree, id2idx = _build_tree(geomsB)

    pairs: List[Tuple[float, int, int, float, float]] = []
    for i, a in enumerate(A):
        cand = range(len(B)) if tree is None else _tree_query_indices(tree, id2idx, a.geom)
        for j in cand:
            s, oc, ar = cas_score(a.geom, B[j].geom, sigma=sigma)
            if s >= cas_thr:
                pairs.append((s, i, j, oc, ar))

    pairs.sort(reverse=True, key=lambda x: x[0])

    usedA, usedB = set(), set()
    matches: List[Tuple[int, int, float, float, float]] = []
    for s, i, j, oc, ar in pairs:
        if i in usedA or j in usedB:
            continue
        usedA.add(i)
        usedB.add(j)
        matches.append((i, j, float(s), float(oc), float(ar)))
    return matches


def hungarian_match_cas_small(
    A: List[Ann],
    B: List[Ann],
    cas_thr: float,
    sigma: float,
    max_n: int
) -> List[Tuple[int, int, float, float, float]]:
    """
    Maximum-weight 1-1 matching using Hungarian (SMALL problems only).
    Allows unmatched via dummy rows/cols (size n+m).
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy not available for Hungarian matching.")
    n = len(A)
    m = len(B)
    if n == 0 or m == 0:
        return []
    if n + m > max_n:
        raise RuntimeError(f"Hungarian matching disabled: n+m={n+m} > max_n={max_n}")

    N = n + m
    BIG = 1e6
    cost = np.zeros((N, N), dtype=np.float64)

    # A rows to B cols: -CAS if >=thr else BIG
    for i in range(n):
        for j in range(m):
            s, _, _ = cas_score(A[i].geom, B[j].geom, sigma=sigma)
            cost[i, j] = (-s) if (s >= cas_thr) else BIG

    # A rows to dummy cols -> 0
    for i in range(n):
        for j in range(m, m + n):
            cost[i, j] = 0.0

    # dummy rows (unmatched B) to B cols -> 0
    for i in range(n, n + m):
        for j in range(m):
            cost[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)

    matches: List[Tuple[int, int, float, float, float]] = []
    for r, c in zip(row_ind, col_ind):
        if r < n and c < m:
            s, oc, ar = cas_score(A[r].geom, B[c].geom, sigma=sigma)
            if s >= cas_thr:
                matches.append((r, c, float(s), float(oc), float(ar)))
    return matches


def compute_per_class_metrics_one_to_one(
    case_key: str,
    A: List[Ann],
    B: List[Ann],
    cas_thr: float,
    sigma: float,
    matching: str,
    hungarian_max_n: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[MatchRow], Dict[str, Dict[str, Any]]]:
    """
    Directional evaluation: B as 'pred', A as 'GT'.
    Returns:
      - per-class rows
      - summary dict
      - match rows for matches.csv
      - per-feature match_map for geojson export
    """
    a_by_cls: Dict[str, List[Ann]] = {}
    b_by_cls: Dict[str, List[Ann]] = {}
    classes = set()

    for a in A:
        a_by_cls.setdefault(a.cls, []).append(a)
        classes.add(a.cls)
    for b in B:
        b_by_cls.setdefault(b.cls, []).append(b)
        classes.add(b.cls)

    rows: List[Dict[str, Any]] = []
    match_rows: List[MatchRow] = []
    match_map: Dict[str, Dict[str, Any]] = {}

    tot_tp = tot_fp = tot_fn = 0

    for cls in sorted(classes):
        As = a_by_cls.get(cls, [])
        Bs = b_by_cls.get(cls, [])

        if matching == "hungarian":
            try:
                idx_matches = hungarian_match_cas_small(
                    As, Bs, cas_thr=cas_thr, sigma=sigma, max_n=hungarian_max_n
                )
            except Exception:
                idx_matches = greedy_match_cas(As, Bs, cas_thr=cas_thr, sigma=sigma)
        else:
            idx_matches = greedy_match_cas(As, Bs, cas_thr=cas_thr, sigma=sigma)

        tp = len(idx_matches)
        fp = len(Bs) - tp
        fn = len(As) - tp

        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = (2 * prec * rec / (prec + rec)) if (prec == prec and rec == rec and (prec + rec)) else float("nan")

        if tp:
            mean_cas = float(np.mean([t[2] for t in idx_matches]))
            mean_oc  = float(np.mean([t[3] for t in idx_matches]))
            mean_ar  = float(np.mean([t[4] for t in idx_matches]))
        else:
            mean_cas = mean_oc = mean_ar = float("nan")

        rows.append({
            "class": cls,
            "gt": len(As),
            "pred": len(Bs),
            "TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "mean_cas_on_matches": mean_cas,
            "mean_oc_on_matches": mean_oc,
            "mean_ar_on_matches": mean_ar,
            "area_a": "",
            "area_b": "",
            "area_intersection": "",
        })

        # record per-pair rows and per-feature match_map
        for pid, (iA, iB, s, oc, ar) in enumerate(idx_matches):
            a = As[iA]
            b = Bs[iB]
            pair_id = f"{cls}:pair_{pid}"
            match_rows.append(
                MatchRow(
                    case_key=case_key,
                    cls=cls,
                    mode="one_to_one",
                    unit_id=pair_id,
                    status="matched",
                    nA=1,
                    nB=1,
                    a_uids=a.uid,
                    b_uids=b.uid,
                    a_feat_ids=a.feat_id,
                    b_feat_ids=b.feat_id,
                    a_src_files=a.src_file,
                    b_src_files=b.src_file,
                    cas=float(s),
                    oc=float(oc),
                    ar=float(ar),
                    area_a=float(a.geom.area),
                    area_b=float(b.geom.area),
                    area_intersection=float(a.geom.intersection(b.geom).area),
                )
            )
            match_map[a.uid] = {
                "status": "matched",
                "pair_id": pair_id,
                "cas": float(s), "oc": float(oc), "ar": float(ar),
                "other_source": "B", "other_uid": b.uid,
            }
            match_map[b.uid] = {
                "status": "matched",
                "pair_id": pair_id,
                "cas": float(s), "oc": float(oc), "ar": float(ar),
                "other_source": "A", "other_uid": a.uid,
            }

        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

    micro_p = safe_div(tot_tp, tot_tp + tot_fp)
    micro_r = safe_div(tot_tp, tot_tp + tot_fn)
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p == micro_p and micro_r == micro_r and (micro_p + micro_r)) else float("nan")

    ps = [r["precision"] for r in rows if r["precision"] == r["precision"]]
    rs = [r["recall"] for r in rows if r["recall"] == r["recall"]]
    fs = [r["f1"] for r in rows if r["f1"] == r["f1"]]
    macro_f1 = float(np.mean(fs)) if fs else float("nan")

    summary = {
        "mode": "one_to_one",
        "TP": int(tot_tp),
        "FP": int(tot_fp),
        "FN": int(tot_fn),
        "micro_precision": float(micro_p) if micro_p == micro_p else None,
        "micro_recall": float(micro_r) if micro_r == micro_r else None,
        "micro_f1": float(micro_f1) if micro_f1 == micro_f1 else None,
        "macro_f1": float(macro_f1) if macro_f1 == macro_f1 else None,
        "cas_threshold": float(cas_thr),
        "sigma": (float(sigma) if math.isfinite(sigma) else "inf"),
        "matching": matching,
        "hungarian_available": bool(_HAS_SCIPY),
        "strtree_available": bool(_HAS_STRTREE),
    }
    return rows, summary, match_rows, match_map


# -----------------------------
# union_per_class mode
# -----------------------------
def compute_per_class_metrics_union(
    case_key: str,
    A: List[Ann],
    B: List[Ann],
    sigma: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[MatchRow], Dict[str, Dict[str, Any]]]:
    """
    Union-per-class evaluation (split/merge robust, no instance matching).

    For each class c:
      A_c = union of all A geometries of class c
      B_c = union of all B geometries of class c

    Area-based precision/recall/F1:
      Precision = |A_c ∩ B_c| / |B_c|
      Recall    = |A_c ∩ B_c| / |A_c|

    Additionally reports union-level CAS/OC/AR diagnostics.
    """
    a_by_cls: Dict[str, List[Ann]] = {}
    b_by_cls: Dict[str, List[Ann]] = {}
    classes = set()

    for a in A:
        a_by_cls.setdefault(a.cls, []).append(a)
        classes.add(a.cls)
    for b in B:
        b_by_cls.setdefault(b.cls, []).append(b)
        classes.add(b.cls)

    rows: List[Dict[str, Any]] = []
    match_rows: List[MatchRow] = []
    match_map: Dict[str, Dict[str, Any]] = {}

    total_area_a = 0.0
    total_area_b = 0.0
    total_area_i = 0.0

    for cls in sorted(classes):
        As = a_by_cls.get(cls, [])
        Bs = b_by_cls.get(cls, [])

        a_union = unary_union([x.geom for x in As]) if As else None
        b_union = unary_union([x.geom for x in Bs]) if Bs else None

        area_a = float(a_union.area) if a_union is not None else 0.0
        area_b = float(b_union.area) if b_union is not None else 0.0
        area_i = float(a_union.intersection(b_union).area) if (a_union is not None and b_union is not None) else 0.0

        # area-based P/R; handle empty cases
        prec = (area_i / area_b) if area_b > 0 else (1.0 if area_a == 0 else 0.0)
        rec  = (area_i / area_a) if area_a > 0 else (1.0 if area_b == 0 else 0.0)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else float("nan")

        if a_union is not None and b_union is not None and area_a > 0 and area_b > 0:
            cas_u, oc_u, ar_u = cas_score(a_union, b_union, sigma=sigma)
        else:
            cas_u, oc_u, ar_u = 0.0, 0.0, (float("inf") if (area_a > 0 or area_b > 0) else 1.0)

        rows.append({
            "class": cls,
            "gt": len(As),
            "pred": len(Bs),
            "TP": "",
            "FP": "",
            "FN": "",
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "mean_cas_on_matches": float(cas_u),
            "mean_oc_on_matches": float(oc_u),
            "mean_ar_on_matches": float(ar_u),
            "area_a": area_a,
            "area_b": area_b,
            "area_intersection": area_i,
        })

        match_rows.append(
            MatchRow(
                case_key=case_key,
                cls=cls,
                mode="union_per_class",
                unit_id="UNION",
                status="union_mode",
                nA=len(As),
                nB=len(Bs),
                a_uids="",
                b_uids="",
                a_feat_ids="",
                b_feat_ids="",
                a_src_files="",
                b_src_files="",
                cas=float(cas_u),
                oc=float(oc_u),
                ar=float(ar_u),
                area_a=area_a,
                area_b=area_b,
                area_intersection=area_i,
            )
        )

        # attach union stats to each feature for visualization
        stats = {
            "status": "union_mode",
            "class": cls,
            "area_a": area_a,
            "area_b": area_b,
            "area_intersection": area_i,
            "cas": float(cas_u),
            "oc": float(oc_u),
            "ar": float(ar_u),
        }
        for ann in As + Bs:
            match_map[ann.uid] = stats

        total_area_a += area_a
        total_area_b += area_b
        total_area_i += area_i

    micro_p = (total_area_i / total_area_b) if total_area_b > 0 else (1.0 if total_area_a == 0 else 0.0)
    micro_r = (total_area_i / total_area_a) if total_area_a > 0 else (1.0 if total_area_b == 0 else 0.0)
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else float("nan")

    fs = [r["f1"] for r in rows if r["f1"] == r["f1"]]
    macro_f1 = float(np.mean(fs)) if fs else float("nan")

    summary = {
        "mode": "union_per_class",
        "micro_precision": float(micro_p) if micro_p == micro_p else None,
        "micro_recall": float(micro_r) if micro_r == micro_r else None,
        "micro_f1": float(micro_f1) if micro_f1 == micro_f1 else None,
        "macro_f1": float(macro_f1) if macro_f1 == macro_f1 else None,
        "total_area_a": total_area_a,
        "total_area_b": total_area_b,
        "total_area_intersection": total_area_i,
        "sigma": (float(sigma) if math.isfinite(sigma) else "inf"),
        "strtree_available": bool(_HAS_STRTREE),
    }

    return rows, summary, match_rows, match_map


# -----------------------------
# components mode (many-to-many via connected components)
# -----------------------------
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def compute_per_class_metrics_components(
    case_key: str,
    A: List[Ann],
    B: List[Ann],
    cas_thr: float,
    sigma: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[MatchRow], Dict[str, Dict[str, Any]]]:
    """
    Components evaluation:
      - Create edges between A_i and B_j if CAS(A_i,B_j) >= cas_thr.
      - Connected components in this bipartite graph define many-to-many groups.
      - Count components:
          TP: contains >=1 A and >=1 B
          FN: contains only A
          FP: contains only B
      - For each component, compute union-level CAS/OC/AR and areas as diagnostics.
    """
    a_by_cls: Dict[str, List[Ann]] = {}
    b_by_cls: Dict[str, List[Ann]] = {}
    classes = set()

    for a in A:
        a_by_cls.setdefault(a.cls, []).append(a)
        classes.add(a.cls)
    for b in B:
        b_by_cls.setdefault(b.cls, []).append(b)
        classes.add(b.cls)

    rows: List[Dict[str, Any]] = []
    match_rows: List[MatchRow] = []
    match_map: Dict[str, Dict[str, Any]] = {}

    tot_tp = tot_fp = tot_fn = 0

    for cls in sorted(classes):
        As = a_by_cls.get(cls, [])
        Bs = b_by_cls.get(cls, [])
        nA, nB = len(As), len(Bs)

        # Build candidate edges (iA, iB) where CAS>=thr
        geomsB = [x.geom for x in Bs]
        tree, id2idx = _build_tree(geomsB)

        edges: List[Tuple[int, int]] = []
        for i, a in enumerate(As):
            cand = range(nB) if tree is None else _tree_query_indices(tree, id2idx, a.geom)
            for j in cand:
                s, _, _ = cas_score(a.geom, Bs[j].geom, sigma=sigma)
                if s >= cas_thr:
                    edges.append((i, j))

        # DSU over nA+nB nodes; B nodes offset by nA
        dsu = DSU(nA + nB)
        for i, j in edges:
            dsu.union(i, nA + j)

        # Gather components
        comps: Dict[int, Dict[str, List[int]]] = {}
        for i in range(nA):
            r = dsu.find(i)
            comps.setdefault(r, {"A": [], "B": []})["A"].append(i)
        for j in range(nB):
            r = dsu.find(nA + j)
            comps.setdefault(r, {"A": [], "B": []})["B"].append(j)

        # Remove empty-only roots (shouldn't exist) and assign component IDs
        comp_items = list(comps.items())
        comp_id_map: Dict[int, str] = {}
        for k, (root, _) in enumerate(comp_items):
            comp_id_map[root] = f"{cls}:comp_{k}"

        tp = fp = fn = 0
        tp_cas_vals: List[float] = []
        tp_oc_vals: List[float] = []
        tp_ar_vals: List[float] = []

        # Build per-feature match_map based on component
        for root, members in comp_items:
            a_idx = members["A"]
            b_idx = members["B"]
            comp_id = comp_id_map[root]

            # unions for diagnostics
            a_union = unary_union([As[i].geom for i in a_idx]) if a_idx else None
            b_union = unary_union([Bs[j].geom for j in b_idx]) if b_idx else None

            area_a = float(a_union.area) if a_union is not None else 0.0
            area_b = float(b_union.area) if b_union is not None else 0.0
            area_i = float(a_union.intersection(b_union).area) if (a_union is not None and b_union is not None) else 0.0

            if a_union is not None and b_union is not None and area_a > 0 and area_b > 0:
                cas_u, oc_u, ar_u = cas_score(a_union, b_union, sigma=sigma)
            else:
                cas_u, oc_u, ar_u = 0.0, 0.0, (float("inf") if (area_a > 0 or area_b > 0) else 1.0)

            if a_idx and b_idx:
                status = "component_tp"
                tp += 1
                tp_cas_vals.append(float(cas_u))
                tp_oc_vals.append(float(oc_u))
                tp_ar_vals.append(float(ar_u))
            elif a_idx:
                status = "component_fn"
                fn += 1
            else:
                status = "component_fp"
                fp += 1

            # matches.csv row for each component
            a_uids = [As[i].uid for i in a_idx]
            b_uids = [Bs[j].uid for j in b_idx]
            a_feat = [As[i].feat_id for i in a_idx]
            b_feat = [Bs[j].feat_id for j in b_idx]
            a_files = list({As[i].src_file for i in a_idx})
            b_files = list({Bs[j].src_file for j in b_idx})

            match_rows.append(
                MatchRow(
                    case_key=case_key,
                    cls=cls,
                    mode="components",
                    unit_id=comp_id,
                    status=status,
                    nA=len(a_idx),
                    nB=len(b_idx),
                    a_uids=_join_limited(a_uids),
                    b_uids=_join_limited(b_uids),
                    a_feat_ids=_join_limited(a_feat),
                    b_feat_ids=_join_limited(b_feat),
                    a_src_files=_join_limited(sorted(a_files)),
                    b_src_files=_join_limited(sorted(b_files)),
                    cas=float(cas_u),
                    oc=float(oc_u),
                    ar=float(ar_u),
                    area_a=area_a,
                    area_b=area_b,
                    area_intersection=area_i,
                )
            )

            # per-feature match map for geojson export
            comp_meta = {
                "status": status,
                "component_id": comp_id,
                "class": cls,
                "nA": len(a_idx),
                "nB": len(b_idx),
                "cas_union": float(cas_u),
                "oc_union": float(oc_u),
                "ar_union": float(ar_u),
                "area_a_union": area_a,
                "area_b_union": area_b,
                "area_intersection_union": area_i,
            }
            for i in a_idx:
                match_map[As[i].uid] = comp_meta
            for j in b_idx:
                match_map[Bs[j].uid] = comp_meta

        # Per-class P/R/F1 on component counts
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = (2 * prec * rec / (prec + rec)) if (prec == prec and rec == rec and (prec + rec)) else float("nan")

        mean_cas = float(np.mean(tp_cas_vals)) if tp_cas_vals else float("nan")
        mean_oc = float(np.mean(tp_oc_vals)) if tp_oc_vals else float("nan")
        mean_ar = float(np.mean(tp_ar_vals)) if tp_ar_vals else float("nan")

        rows.append({
            "class": cls,
            "gt": nA,
            "pred": nB,
            "TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "mean_cas_on_matches": mean_cas,
            "mean_oc_on_matches": mean_oc,
            "mean_ar_on_matches": mean_ar,
            "area_a": "",
            "area_b": "",
            "area_intersection": "",
        })

        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

    micro_p = safe_div(tot_tp, tot_tp + tot_fp)
    micro_r = safe_div(tot_tp, tot_tp + tot_fn)
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p == micro_p and micro_r == micro_r and (micro_p + micro_r)) else float("nan")

    fs = [r["f1"] for r in rows if r["f1"] == r["f1"]]
    macro_f1 = float(np.mean(fs)) if fs else float("nan")

    summary = {
        "mode": "components",
        "TP": int(tot_tp),
        "FP": int(tot_fp),
        "FN": int(tot_fn),
        "micro_precision": float(micro_p) if micro_p == micro_p else None,
        "micro_recall": float(micro_r) if micro_r == micro_r else None,
        "micro_f1": float(micro_f1) if micro_f1 == micro_f1 else None,
        "macro_f1": float(macro_f1) if macro_f1 == macro_f1 else None,
        "cas_threshold": float(cas_thr),
        "sigma": (float(sigma) if math.isfinite(sigma) else "inf"),
        "strtree_available": bool(_HAS_STRTREE),
    }

    return rows, summary, match_rows, match_map


# -----------------------------
# GeoJSON export helpers
# -----------------------------
def _set_classification_name(props: Dict[str, Any], new_name: str) -> None:
    c = props.get("classification")
    if isinstance(c, dict):
        c = dict(c)
        c["name"] = new_name
        props["classification"] = c
    else:
        props["classification"] = {"name": new_name}


def write_combined_geojson(
    out_path: Path,
    anns: List[Ann],
    match_map: Dict[str, Dict[str, Any]],
    case_key: str,
) -> None:
    feats = []
    for ann in anns:
        props = copy.deepcopy(ann.props) if isinstance(ann.props, dict) else {}
        props["case_key"] = case_key
        props["source"] = ann.source
        props["orig_class"] = ann.cls

        # update label so QuPath can visually distinguish A vs B
        _set_classification_name(props, f"{ann.source} | {ann.cls}")

        props["match"] = match_map.get(ann.uid, {"status": "unmatched"})

        feats.append({
            "type": "Feature",
            "id": ann.feat_id,
            "geometry": mapping(ann.geom),
            "properties": props,
        })

    fc = {"type": "FeatureCollection", "features": feats}
    out_path.write_text(json.dumps(fc, indent=2), encoding="utf-8")


def write_matches_csv(out_path: Path, match_rows: List[MatchRow]) -> None:
    fieldnames = [
        "case_key", "class", "mode", "unit_id", "status",
        "nA", "nB",
        "a_uids", "b_uids",
        "a_feat_ids", "b_feat_ids",
        "a_src_files", "b_src_files",
        "cas", "oc", "ar",
        "area_a", "area_b", "area_intersection",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in match_rows:
            w.writerow({
                "case_key": r.case_key,
                "class": r.cls,
                "mode": r.mode,
                "unit_id": r.unit_id,
                "status": r.status,
                "nA": r.nA,
                "nB": r.nB,
                "a_uids": r.a_uids,
                "b_uids": r.b_uids,
                "a_feat_ids": r.a_feat_ids,
                "b_feat_ids": r.b_feat_ids,
                "a_src_files": r.a_src_files,
                "b_src_files": r.b_src_files,
                "cas": r.cas,
                "oc": r.oc,
                "ar": r.ar,
                "area_a": r.area_a,
                "area_b": r.area_b,
                "area_intersection": r.area_intersection,
            })


def write_per_class_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    # superset columns across modes
    fieldnames = [
        "class", "gt", "pred",
        "TP", "FP", "FN",
        "precision", "recall", "f1",
        "mean_cas_on_matches", "mean_oc_on_matches", "mean_ar_on_matches",
        "area_a", "area_b", "area_intersection",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_root", required=True, help="Root folder for annotator A (reference)")
    ap.add_argument("--b_root", required=True, help="Root folder for annotator B (compared)")
    ap.add_argument("--out", default="cas_agreement_out", help="Output folder")

    ap.add_argument("--mode", choices=["one_to_one", "union_per_class", "components"], default="one_to_one",
                    help="Evaluation mode.")
    ap.add_argument("--cas_thr", type=float, default=0.5,
                    help="CAS threshold for linking/matching (used in one_to_one and components).")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="Size-mismatch tolerance (default 1.0). Use inf to disable penalty (CAS==OC).")

    ap.add_argument("--matching", choices=["greedy", "hungarian"], default="greedy",
                    help="1-1 matching algorithm (used only in one_to_one).")
    ap.add_argument("--hungarian_max_n", type=int, default=600,
                    help="Enable Hungarian only if nA+nB <= this (one_to_one).")

    ap.add_argument("--classes", type=str, default="",
                    help="Comma-separated class names to keep (optional)")
    ap.add_argument("--write_geojson", action="store_true",
                    help="Write combined_AB.geojson and (when meaningful) matched_AB.geojson")
    args = ap.parse_args()

    a_root = Path(args.a_root).expanduser().resolve()
    b_root = Path(args.b_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_classes = None
    if args.classes.strip():
        keep_classes = {c.strip() for c in args.classes.split(",") if c.strip()}

    sigma = float(args.sigma)  # argparse accepts "inf" -> float('inf')

    A_groups = collect_geojson_by_rel_parent(a_root)
    B_groups = collect_geojson_by_rel_parent(b_root)

    common_keys = sorted(set(A_groups.keys()) & set(B_groups.keys()))
    if not common_keys:
        raise SystemExit(
            "No matching folders (relative parent dirs of *.geojson) found between A_root and B_root.\n"
            "Tip: ensure both roots contain geojson under matching relative folder paths."
        )

    global_rows: List[Dict[str, Any]] = []

    # Global aggregation
    total_TP = total_FP = total_FN = 0
    total_area_a = total_area_b = total_area_i = 0.0

    for key in tqdm(common_keys, desc="Matched folders"):
        case_out = out_dir / safe_rel_key_to_dir(key)
        case_out.mkdir(parents=True, exist_ok=True)

        # Load all anns for this folder-key (across all geojson files in that folder)
        A_anns: List[Ann] = []
        for idx, p in enumerate(sorted(A_groups[key])):
            A_anns.extend(load_qupath_geojson(p, source="A", uid_prefix=f"{key}|{idx}", keep_classes=keep_classes))

        B_anns: List[Ann] = []
        for idx, p in enumerate(sorted(B_groups[key])):
            B_anns.extend(load_qupath_geojson(p, source="B", uid_prefix=f"{key}|{idx}", keep_classes=keep_classes))

        # Compute metrics according to mode
        if args.mode == "union_per_class":
            rows, summary, match_rows, match_map = compute_per_class_metrics_union(
                case_key=key, A=A_anns, B=B_anns, sigma=sigma
            )
        elif args.mode == "components":
            rows, summary, match_rows, match_map = compute_per_class_metrics_components(
                case_key=key, A=A_anns, B=B_anns, cas_thr=float(args.cas_thr), sigma=sigma
            )
        else:
            rows, summary, match_rows, match_map = compute_per_class_metrics_one_to_one(
                case_key=key, A=A_anns, B=B_anns,
                cas_thr=float(args.cas_thr),
                sigma=sigma,
                matching=args.matching,
                hungarian_max_n=int(args.hungarian_max_n),
            )

        # Save per-case outputs
        write_per_class_csv(case_out / "per_class_metrics.csv", rows)
        (case_out / "summary.json").write_text(json.dumps({"case_key": key, **summary}, indent=2), encoding="utf-8")
        write_matches_csv(case_out / "matches.csv", match_rows)

        # Optional GeoJSON export
        if args.write_geojson:
            all_anns = A_anns + B_anns
            write_combined_geojson(case_out / "combined_AB.geojson", all_anns, match_map, case_key=key)

            # matched_AB.geojson depends on mode
            if args.mode == "one_to_one":
                # keep only features that appear in match_map with status matched
                matched_anns = [a for a in all_anns if match_map.get(a.uid, {}).get("status") == "matched"]
                write_combined_geojson(case_out / "matched_AB.geojson", matched_anns, match_map, case_key=key)
            elif args.mode == "components":
                # keep only TP-components (component_tp)
                matched_anns = [a for a in all_anns if match_map.get(a.uid, {}).get("status") == "component_tp"]
                write_combined_geojson(case_out / "matched_AB.geojson", matched_anns, match_map, case_key=key)
            # union_per_class: matched_AB is not meaningful

        # Global row
        row = {
            "case_key": key,
            "mode": args.mode,
            "n_a": len(A_anns),
            "n_b": len(B_anns),
            "micro_precision": summary.get("micro_precision"),
            "micro_recall": summary.get("micro_recall"),
            "micro_f1": summary.get("micro_f1"),
            "macro_f1": summary.get("macro_f1"),
            "sigma": summary.get("sigma", (float(sigma) if math.isfinite(sigma) else "inf")),
        }

        if args.mode in ("one_to_one", "components"):
            row.update({
                "TP": summary.get("TP"),
                "FP": summary.get("FP"),
                "FN": summary.get("FN"),
                "cas_threshold": summary.get("cas_threshold"),
            })
            total_TP += int(summary.get("TP") or 0)
            total_FP += int(summary.get("FP") or 0)
            total_FN += int(summary.get("FN") or 0)
        else:
            row.update({
                "TP": "",
                "FP": "",
                "FN": "",
                "total_area_a": summary.get("total_area_a"),
                "total_area_b": summary.get("total_area_b"),
                "total_area_intersection": summary.get("total_area_intersection"),
            })
            total_area_a += float(summary.get("total_area_a") or 0.0)
            total_area_b += float(summary.get("total_area_b") or 0.0)
            total_area_i += float(summary.get("total_area_intersection") or 0.0)

        global_rows.append(row)

    # Write global summary CSV
    global_csv = out_dir / "cases_summary.csv"
    fieldnames = list(global_rows[0].keys()) if global_rows else []
    with open(global_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in global_rows:
            w.writerow(r)

    # Overall.txt
    if args.mode in ("one_to_one", "components"):
        micro_p = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0.0
        micro_r = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0
        overall_txt = (
            f"matched_cases: {len(global_rows)}\n"
            f"mode: {args.mode}\n"
            f"total_TP: {total_TP}\n"
            f"total_FP: {total_FP}\n"
            f"total_FN: {total_FN}\n"
            f"micro_precision: {micro_p:.6f}\n"
            f"micro_recall: {micro_r:.6f}\n"
            f"micro_f1: {micro_f1:.6f}\n"
        )
    else:
        micro_p = total_area_i / total_area_b if total_area_b else (1.0 if total_area_a == 0 else 0.0)
        micro_r = total_area_i / total_area_a if total_area_a else (1.0 if total_area_b == 0 else 0.0)
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0
        overall_txt = (
            f"matched_cases: {len(global_rows)}\n"
            f"mode: union_per_class\n"
            f"total_area_a: {total_area_a}\n"
            f"total_area_b: {total_area_b}\n"
            f"total_area_intersection: {total_area_i}\n"
            f"micro_precision_area: {micro_p:.6f}\n"
            f"micro_recall_area: {micro_r:.6f}\n"
            f"micro_f1_area: {micro_f1:.6f}\n"
        )

    (out_dir / "overall.txt").write_text(overall_txt, encoding="utf-8")
    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
