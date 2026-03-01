from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union

try:
    from shapely.strtree import STRtree

    _HAS_STRTREE = True
except Exception:
    _HAS_STRTREE = False


# -----------------------------
# Utilities
# -----------------------------
@dataclass
class Ann:
    geom: object  # shapely geometry
    cls: str
    src_file: str


def _extract_class(props: dict) -> str:
    """
    QuPath commonly stores class name in:
      properties.classification.name
    but variants exist. We try a few.
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

    name = props.get("name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()

    return "unknown"


def load_qupath_geojson(
    path: Path, keep_classes: Optional[set[str]] = None
) -> List[Ann]:
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features", [])
    out: List[Ann] = []

    for f in feats:
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

        out.append(Ann(geom=g, cls=cls, src_file=str(path)))

    return out


def infer_case_id(geojson_path: Path) -> str:
    """
    Heuristics for your folder layouts:
    - prefer parent folder name (common: .../<case_id>/<case_id>.geojson)
    - strip trailing '.' and '_project'
    - fallback to file stem
    """
    parent = geojson_path.parent.name.rstrip(".")
    if parent.endswith("_project"):
        parent = parent[: -len("_project")].rstrip("_").rstrip(".")
    stem = geojson_path.stem.rstrip(".")
    # if parent looks like the case, use it
    if parent and parent != geojson_path.parent.parent.name:
        return parent
    return stem


def collect_geojson_by_case(root: Path) -> Dict[str, List[Path]]:
    """
    Recursively find all *.geojson under root, group by inferred case_id.
    """
    root = root.expanduser().resolve()
    geojsons = list(root.rglob("*.geojson"))
    grouped: Dict[str, List[Path]] = {}
    for p in geojsons:
        cid = infer_case_id(p)
        grouped.setdefault(cid, []).append(p)
    return grouped


def iou(a, b) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def _build_tree(geoms: List[object]):
    """
    STRtree compatibility:
    - shapely 2: query returns indices
    - shapely 1.8: query returns geometries
    """
    if not _HAS_STRTREE:
        return None, None
    tree = STRtree(geoms)
    # mapping for shapely 1.8
    id2idx = {id(g): i for i, g in enumerate(geoms)}
    return tree, id2idx


def _tree_query_indices(tree, id2idx, geom) -> List[int]:
    res = tree.query(geom)
    if len(res) == 0:
        return []
    # shapely 2 returns numpy array of indices
    first = res[0]
    if isinstance(first, (int, np.integer)):
        return [int(x) for x in res]
    # shapely 1.8 returns geometries
    return [id2idx[id(g)] for g in res]


def greedy_match_iou(
    A: List[Ann], B: List[Ann], iou_thr: float
) -> List[Tuple[int, int, float]]:
    """
    Match boxes between annotators using greedy max-IoU assignment.
    """
    if len(A) == 0 or len(B) == 0:
        return []

    geomsB = [x.geom for x in B]
    tree, id2idx = _build_tree(geomsB)

    pairs: List[Tuple[float, int, int]] = []
    for i, a in enumerate(A):
        if tree is None:
            cand = range(len(B))
        else:
            cand = _tree_query_indices(tree, id2idx, a.geom)

        for j in cand:
            s = iou(a.geom, B[j].geom)
            if s > 0:
                pairs.append((s, i, j))

    pairs.sort(reverse=True, key=lambda x: x[0])

    usedA, usedB = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for s, i, j in pairs:
        if s < iou_thr:
            break
        if i in usedA or j in usedB:
            continue
        usedA.add(i)
        usedB.add(j)
        matches.append((i, j, float(s)))
    return matches


def merge_overlapping(
    polys: List[Ann], iou_merge_thr: float, bbox_output: bool
) -> List[object]:
    """
    Merge overlapping boxes by building connected components using IoU threshold,
    then unary_union each component. Optionally output as bounding boxes.
    """
    n = len(polys)
    if n == 0:
        return []

    geoms = [p.geom for p in polys]
    uf = UnionFind(n)

    if _HAS_STRTREE:
        tree = STRtree(geoms)
        id2idx = {id(g): i for i, g in enumerate(geoms)}
        for i in range(n):
            cand = tree.query(geoms[i])
            # shapely2 indices vs shapely1 geometries
            if len(cand) == 0:
                continue
            if isinstance(cand[0], (int, np.integer)):
                cand_idxs = [int(x) for x in cand]
            else:
                cand_idxs = [id2idx[id(g)] for g in cand]

            for j in cand_idxs:
                if j <= i:
                    continue
                s = iou(geoms[i], geoms[j])
                if s >= iou_merge_thr:
                    uf.union(i, j)
    else:
        # fallback O(N^2)
        for i in range(n):
            for j in range(i + 1, n):
                s = iou(geoms[i], geoms[j])
                if s >= iou_merge_thr:
                    uf.union(i, j)

    comps: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        comps.setdefault(r, []).append(i)

    merged = []
    for idxs in comps.values():
        g = unary_union([geoms[i] for i in idxs])
        if bbox_output:
            g = box(*g.bounds)
        merged.append(g)
    return merged


def write_geojson(out_path: Path, geoms: List[object], case_id: str) -> None:
    feats = []
    for k, g in enumerate(geoms):
        feats.append(
            {
                "type": "Feature",
                "geometry": mapping(g),
                "properties": {
                    "case_id": case_id,
                    "merged_id": k,
                },
            }
        )
    fc = {"type": "FeatureCollection", "features": feats}
    out_path.write_text(json.dumps(fc, indent=2), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--a_root",
        type=str,
        required=True,
        help="Root folder for annotator A (AIIMS_G...)",
    )
    ap.add_argument(
        "--b_root",
        type=str,
        required=True,
        help="Root folder for annotator B (AIIMSD_all_annotations...)",
    )
    ap.add_argument("--out", type=str, default="agreement_out", help="Output folder")
    ap.add_argument(
        "--iou_match",
        type=float,
        default=0.5,
        help="IoU threshold to count a match (agreement)",
    )
    ap.add_argument(
        "--iou_merge",
        type=float,
        default=0.3,
        help="IoU threshold to merge boxes into one",
    )
    ap.add_argument(
        "--classes",
        type=str,
        default="",
        help="Comma-separated class names to keep (optional)",
    )
    ap.add_argument(
        "--bbox_output",
        action="store_true",
        help="Export merged results as rectangles (bbox) instead of unions",
    )
    args = ap.parse_args()

    a_root = Path(args.a_root).expanduser().resolve()
    b_root = Path(args.b_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_classes = None
    if args.classes.strip():
        keep_classes = {c.strip() for c in args.classes.split(",") if c.strip()}

    A_cases = collect_geojson_by_case(a_root)
    B_cases = collect_geojson_by_case(b_root)
    common = sorted(set(A_cases.keys()) & set(B_cases.keys()))

    if not common:
        raise SystemExit(
            "No common case IDs found between the two roots.\n"
            "Tip: verify both sides have *.geojson and that case-id inference matches your folder names."
        )

    rows = []
    for cid in tqdm(common, desc="Cases"):
        A_anns: List[Ann] = []
        for p in A_cases[cid]:
            A_anns.extend(load_qupath_geojson(p, keep_classes=keep_classes))

        B_anns: List[Ann] = []
        for p in B_cases[cid]:
            B_anns.extend(load_qupath_geojson(p, keep_classes=keep_classes))

        matches = greedy_match_iou(A_anns, B_anns, iou_thr=args.iou_match)

        nA, nB, nM = len(A_anns), len(B_anns), len(matches)
        prec = (nM / nB) if nB else (1.0 if nA == 0 else 0.0)
        rec = (nM / nA) if nA else (1.0 if nB == 0 else 0.0)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        miou = float(np.mean([m[2] for m in matches])) if matches else 0.0

        # Save match details
        match_df = pd.DataFrame(
            [
                {
                    "case_id": cid,
                    "a_idx": i,
                    "b_idx": j,
                    "iou": s,
                    "a_class": A_anns[i].cls if i < nA else "unknown",
                    "b_class": B_anns[j].cls if j < nB else "unknown",
                }
                for (i, j, s) in matches
            ]
        )
        match_df.to_csv(out_dir / f"{cid}_matches.csv", index=False)

        # Merge all boxes (A + B) into a single set
        merged = merge_overlapping(
            A_anns + B_anns, iou_merge_thr=args.iou_merge, bbox_output=args.bbox_output
        )
        write_geojson(out_dir / f"{cid}_merged.geojson", merged, case_id=cid)

        rows.append(
            {
                "case_id": cid,
                "n_a": nA,
                "n_b": nB,
                "n_matches": nM,
                "precision_b_vs_a": prec,
                "recall_b_vs_a": rec,
                "f1": f1,
                "mean_iou_on_matches": miou,
                "merged_count": len(merged),
            }
        )

    summary = pd.DataFrame(rows).sort_values("case_id")
    summary.to_csv(out_dir / "agreement_summary.csv", index=False)

    # overall (micro)
    totA = summary["n_a"].sum()
    totB = summary["n_b"].sum()
    totM = summary["n_matches"].sum()
    micro_prec = totM / totB if totB else 0.0
    micro_rec = totM / totA if totA else 0.0
    micro_f1 = (
        (2 * micro_prec * micro_rec / (micro_prec + micro_rec))
        if (micro_prec + micro_rec)
        else 0.0
    )

    (out_dir / "agreement_overall.txt").write_text(
        f"cases: {len(summary)}\n"
        f"total_A: {int(totA)}\n"
        f"total_B: {int(totB)}\n"
        f"total_matches: {int(totM)}\n"
        f"micro_precision: {micro_prec:.4f}\n"
        f"micro_recall: {micro_rec:.4f}\n"
        f"micro_f1: {micro_f1:.4f}\n",
        encoding="utf-8",
    )

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
