#!/usr/bin/env python3
"""
GeoJSON bounding-box evaluation with class-wise NMS + IoU matching.

Assumptions:
- Each annotation is a GeoJSON Feature with geometry type Polygon (rectangle as 5-point closed ring)
  OR MultiPolygon.
- Class label is stored at: properties.classification.name
- Optional confidence can be stored at: properties.score / confidence / probability (or nested under classification).

Metrics:
- Per-class: TP/FP/FN, precision, recall, F1 at a fixed IoU threshold.
- Overall: micro & macro averages.
- Optional: class-agnostic localization match + classification accuracy on matched pairs.
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

def _flatten_points_from_polygon_coords(coords: Any) -> List[Tuple[float,float]]:
    # Polygon coords: [ring][point][2]
    pts: List[Tuple[float,float]] = []
    if not coords:
        return pts
    for ring in coords:
        for xy in ring:
            if len(xy) >= 2:
                pts.append((float(xy[0]), float(xy[1])))
    return pts

def _flatten_points(geom_type: str, coords: Any) -> List[Tuple[float,float]]:
    if geom_type == "Polygon":
        return _flatten_points_from_polygon_coords(coords)
    if geom_type == "MultiPolygon":
        pts: List[Tuple[float,float]] = []
        for poly in coords:
            pts.extend(_flatten_points_from_polygon_coords(poly))
        return pts
    return []

def polygon_to_bbox(geom: Dict[str, Any]) -> Optional[Tuple[float,float,float,float]]:
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    pts = _flatten_points(gtype, coords)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))

def bbox_area(b: Tuple[float,float,float,float]) -> float:
    x1,y1,x2,y2 = b
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0.0 else 0.0

def get_score(props: Dict[str, Any]) -> Optional[float]:
    for k in ("score","confidence","prob","probability","conf"):
        if k in props:
            try: return float(props[k])
            except: pass
    cls = props.get("classification")
    if isinstance(cls, dict):
        for k in ("score","confidence","prob","probability","conf"):
            if k in cls:
                try: return float(cls[k])
                except: pass
    return None

def get_class(props: Dict[str, Any]) -> str:
    cls = props.get("classification")
    if isinstance(cls, dict):
        name = cls.get("name")
        if name is not None:
            return str(name)
    return "__none__"

@dataclass
class Box:
    id: str
    cls: str
    bbox: Tuple[float,float,float,float]
    score: float
    raw_score: Optional[float]

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

def load_boxes(path: str) -> List[Box]:
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    out: List[Box] = []
    for i, feat in enumerate(feats):
        geom = feat.get("geometry", {})
        bb = polygon_to_bbox(geom)
        if bb is None:
            continue
        props = feat.get("properties", {}) or {}
        cls = get_class(props)
        raw = get_score(props)
        score = raw if raw is not None else 1.0
        fid = feat.get("id", i)
        out.append(Box(id=str(fid), cls=cls, bbox=bb, score=float(score), raw_score=raw))
    return out

def nms_classwise(boxes: List[Box], nms_iou: float) -> Tuple[List[Box], int]:
    by_cls: Dict[str, List[Box]] = defaultdict(list)
    for b in boxes:
        by_cls[b.cls].append(b)

    kept_all: List[Box] = []
    suppressed = 0

    for cls, lst in by_cls.items():
        # highest score first; tie-breaker: larger area first
        lst = sorted(lst, key=lambda x: (x.score, x.area), reverse=True)
        kept: List[Box] = []
        for b in lst:
            if any(iou(b.bbox, k.bbox) > nms_iou for k in kept):
                suppressed += 1
                continue
            kept.append(b)
        kept_all.extend(kept)
    return kept_all, suppressed

def filter_small_unmatched(
    pred: List[Box],
    gt: List[Box],
    match_iou: float,
    area_threshold: float
) -> Tuple[List[Box], List[Box]]:
    """
    Remove predictions that:
      (a) do not overlap any GT with IoU >= match_iou, AND
      (b) have area < area_threshold

    Note: if area_threshold <= 0, nothing is removed.
    """
    if area_threshold <= 0:
        return pred, []

    removed: List[Box] = []
    kept: List[Box] = []
    for p in pred:
        best = 0.0
        for g in gt:
            best = max(best, iou(p.bbox, g.bbox))
        if best < match_iou and p.area < area_threshold:
            removed.append(p)
        else:
            kept.append(p)
    return kept, removed

def match_per_class(pred: List[Box], gt: List[Box], match_iou: float) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    pred_by_cls: Dict[str, List[Box]] = defaultdict(list)
    gt_by_cls: Dict[str, List[Box]] = defaultdict(list)
    classes = set()

    for p in pred:
        pred_by_cls[p.cls].append(p); classes.add(p.cls)
    for g in gt:
        gt_by_cls[g.cls].append(g); classes.add(g.cls)

    rows: List[Dict[str, Any]] = []
    tot_tp = tot_fp = tot_fn = 0

    for cls in sorted(classes):
        preds = sorted(pred_by_cls.get(cls, []), key=lambda x: x.score, reverse=True)
        gts = gt_by_cls.get(cls, [])

        matched_gt = set()
        tp = fp = 0

        for p in preds:
            best_iou = -1.0
            best_j = None
            for j, g in enumerate(gts):
                if j in matched_gt:
                    continue
                val = iou(p.bbox, g.bbox)
                if val > best_iou:
                    best_iou = val
                    best_j = j
            if best_j is not None and best_iou >= match_iou:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1

        fn = len(gts) - len(matched_gt)
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) else float("nan")
        f1   = (2*prec*rec/(prec+rec)) if (prec == prec and rec == rec and (prec+rec)) else float("nan")

        rows.append({
            "class": cls,
            "gt": len(gts),
            "pred": len(preds),
            "TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec, "f1": f1
        })

        tot_tp += tp; tot_fp += fp; tot_fn += fn

    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else float("nan")
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else float("nan")
    micro_f1 = (2*micro_p*micro_r/(micro_p+micro_r)) if (micro_p == micro_p and micro_r == micro_r and (micro_p+micro_r)) else float("nan")

    # "detection accuracy" (sometimes used in assignment-style questions)
    det_acc = tot_tp / (tot_tp + tot_fp + tot_fn) if (tot_tp + tot_fp + tot_fn) else float("nan")

    # macro averages
    import math
    ps = [r["precision"] for r in rows if r["precision"] == r["precision"]]
    rs = [r["recall"] for r in rows if r["recall"] == r["recall"]]
    fs = [r["f1"] for r in rows if r["f1"] == r["f1"]]
    macro_p = sum(ps)/len(ps) if ps else float("nan")
    macro_r = sum(rs)/len(rs) if rs else float("nan")
    macro_f1 = sum(fs)/len(fs) if fs else float("nan")

    summary = {
        "TP": float(tot_tp), "FP": float(tot_fp), "FN": float(tot_fn),
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
        "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1,
        "detection_accuracy": det_acc,
    }
    return rows, summary

def match_class_agnostic(pred: List[Box], gt: List[Box], match_iou: float) -> Dict[str, float]:
    """
    Optional: localization match ignoring class label.
    Reports:
      - loc_TP/FP/FN
      - loc_precision/recall/F1
      - mean IoU of matched pairs
      - classification_accuracy among matched pairs (pred.cls == gt.cls)
    """
    preds = sorted(pred, key=lambda x: x.score, reverse=True)
    matched_gt = set()
    matched_pairs: List[Tuple[Box, Box, float]] = []

    fp = 0
    for p in preds:
        best_iou = -1.0
        best_j = None
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            val = iou(p.bbox, g.bbox)
            if val > best_iou:
                best_iou = val
                best_j = j
        if best_j is not None and best_iou >= match_iou:
            matched_gt.add(best_j)
            matched_pairs.append((p, gt[best_j], best_iou))
        else:
            fp += 1

    tp = len(matched_pairs)
    fn = len(gt) - len(matched_gt)

    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) else float("nan")
    f1   = (2*prec*rec/(prec+rec)) if (prec == prec and rec == rec and (prec+rec)) else float("nan")
    mean_iou = sum(v for _,_,v in matched_pairs)/tp if tp else float("nan")
    cls_acc = sum(1 for p,g,_ in matched_pairs if p.cls == g.cls) / tp if tp else float("nan")

    return {
        "loc_TP": float(tp), "loc_FP": float(fp), "loc_FN": float(fn),
        "loc_precision": prec, "loc_recall": rec, "loc_f1": f1,
        "mean_iou_matched": mean_iou,
        "classification_accuracy_on_matched": cls_acc
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="ground truth geojson")
    ap.add_argument("--pred", required=True, help="prediction geojson")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching (default 0.5)")
    ap.add_argument("--nms", type=float, default=0.5, help="IoU threshold for NMS on predictions (default 0.5)")
    ap.add_argument("--ignore_area", type=float, default=0.0,
                    help="ignore unmatched predictions with area < this (default 0 = ignore none)")
    ap.add_argument("--no_loc_report", action="store_true", help="disable class-agnostic localization report")
    ap.add_argument("--out_csv", default=None, help="optional path to write per-class metrics CSV")
    args = ap.parse_args()

    gt = load_boxes(args.gt)
    pred_raw = load_boxes(args.pred)

    pred_nms, suppressed = nms_classwise(pred_raw, args.nms)
    pred, removed = filter_small_unmatched(pred_nms, gt, args.iou, args.ignore_area)

    rows, summary = match_per_class(pred, gt, args.iou)

    print("\n=== Counts ===")
    print(f"GT boxes: {len(gt)}")
    print(f"Pred boxes (raw): {len(pred_raw)}")
    print(f"Pred boxes after NMS: {len(pred_nms)} (suppressed {suppressed})")
    print(f"Pred boxes ignored by area rule: {len(removed)} (area<thr & no IoU>={args.iou})")
    print(f"Pred boxes evaluated: {len(pred)}")

    print("\n=== Overall (micro/macro) ===")
    for k in ("TP","FP","FN","micro_precision","micro_recall","micro_f1","macro_precision","macro_recall","macro_f1","detection_accuracy"):
        print(f"{k}: {summary[k]}")

    print("\n=== Per-class ===")
    # pretty print table
    header = ["class","gt","pred","TP","FP","FN","precision","recall","f1"]
    print("\t".join(header))
    for r in rows:
        print("\t".join([
            str(r["class"]), str(r["gt"]), str(r["pred"]), str(r["TP"]), str(r["FP"]), str(r["FN"]),
            f'{r["precision"]:.6f}' if r["precision"]==r["precision"] else "nan",
            f'{r["recall"]:.6f}' if r["recall"]==r["recall"] else "nan",
            f'{r["f1"]:.6f}' if r["f1"]==r["f1"] else "nan",
        ]))

    if not args.no_loc_report:
        loc = match_class_agnostic(pred, gt, args.iou)
        print("\n=== Class-agnostic localization (optional) ===")
        for k,v in loc.items():
            print(f"{k}: {v}")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in header})
        print(f"\nWrote per-class metrics to: {args.out_csv}")

if __name__ == "__main__":
    main()
