#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_drawing_no_vector_pdf.py
Test script: extract absolute pixel bounding boxes for Drawing Number from a VECTOR PDF using PyMuPDF (no OCR).

Usage:
  python pdf_vector_extractor.py --folder '/Users/ppwcr/Desktop/Tam Tossawee/print_pages/Test' --dpi 300 --save-viz out_dir

Output:
  - Prints per-page detection and absolute pixel bbox (x0,y0,x1,y1) to console
  - Saves JSONL (optional via --out-jsonl)
  - If --save-viz given, saves page PNG with a red rectangle around the detected code

Notes:
  - Works only when the drawing number text is true vector text (selectable) in the PDF
  - Coordinates are computed in page pixels based on --dpi
"""

import re
import json
import csv
import argparse
from pathlib import Path
import fitz  # PyMuPDF

# Regex candidates (tweak as needed)
REGEXES = [
    r"\b[A-Z]{1,4}-?\d{1,3}(?:-\d{1,3})?\b",   # A1-01, A-01, ST-02, AR-000, A101-02, S10-002
    r"\b[SWMAPCRLETBHIV]{1,4}-?\d{2,4}\b",     # discipline-focused: S-12, W-005, M-102, AR-1203
    r"\b[A-Z]{1,4}\d{1,3}(?:-\d{1,3})?\b",     # A101, A101-02 (no dash after letter)
]

LABEL_WORDS = {"DRAWING", "NO", "NUMBER", "SHEET", "เลขที่แบบ"}

def compile_patterns(user_regex: str | None):
    pats = []
    if user_regex:
        pats.append(re.compile(user_regex, re.IGNORECASE))
    pats.extend([re.compile(rx, re.IGNORECASE) for rx in REGEXES])
    return pats

def words_on_page(page: fitz.Page):
    # Returns list of tuples: (x0, y0, x1, y1, text, block_no, line_no, word_no)
    # Coordinates are in points (1/72 inch)
    return page.get_text("words")  # list[tuple]

def points_to_pixels(rect_points: tuple[float, float, float, float], dpi: int) -> tuple[int, int, int, int]:
    # Convert PDF points to pixels at given dpi
    x0, y0, x1, y1 = rect_points
    scale = dpi / 72.0
    return (int(round(x0 * scale)), int(round(y0 * scale)),
            int(round(x1 * scale)), int(round(y1 * scale)))

# --- Title-block region prior & token utilities ---
def in_titleblock_region(x0: float, y0: float, x1: float, y1: float, page_w: float, page_h: float) -> bool:
    """Heuristic prior: title block usually bottom band or right band."""
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    return (cy >= 0.75 * page_h) or (cx >= 0.80 * page_w)  # bottom 25% or rightmost 20%

DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"  # various unicode dashes
_dash_re = re.compile(f"[{DASHES}]")
_spaces_re = re.compile(r"\s+")

def normalize_token(s: str) -> str:
    """Collapse spaces and normalize every dash variant to '-'."""
    s = s.strip()
    s = _dash_re.sub("-", s)
    s = _spaces_re.sub("", s)
    return s

def group_tokens_and_match(words_roi, dpi, compiled_patterns, y_tol_pt: float = 3.0, gap_tol_pt: float = 6.0):
    """
    Group neighboring tokens on the same line, merge tiny gaps, normalize, and regex-match.
    words_roi: list of tuples (x0, y0, x1, y1, text, ...)
    Returns list of dict {text, bbox_points, bbox_pixels}
    """
    if not words_roi:
        return []

    # Sort by (line-y, then x)
    words_sorted = sorted(words_roi, key=lambda w: (round((w[1] + w[3]) * 0.5, 1), w[0]))

    # Bucket into lines by y centroid tolerance
    lines = []
    for w in words_sorted:
        x0, y0, x1, y1, t, *_ = w
        cy = 0.5 * (y0 + y1)
        if not lines:
            lines.append([w])
            continue
        cy_last = 0.5 * (lines[-1][-1][1] + lines[-1][-1][3])
        if abs(cy - cy_last) <= y_tol_pt:
            lines[-1].append(w)
        else:
            lines.append([w])

    results = []
    for ln in lines:
        ln.sort(key=lambda w: w[0])  # left to right
        # Merge tokens across tiny x gaps
        cur = list(ln[0])
        merged = []
        for w in ln[1:]:
            gap = w[0] - cur[2]
            if gap <= gap_tol_pt:
                # extend bbox
                cur[2] = max(cur[2], w[2])
                cur[1] = min(cur[1], w[1])
                cur[3] = max(cur[3], w[3])
                cur[4] = f"{cur[4]} {w[4]}"  # keep a space; we'll normalize later
            else:
                merged.append(cur)
                cur = list(w)
        merged.append(cur)

        # Normalize and match
        for x0, y0, x1, y1, t, *_ in merged:
            t_norm = normalize_token(str(t))
            if any(p.search(t_norm) for p in compiled_patterns):
                x0i, y0i, x1i, y1i = points_to_pixels((x0, y0, x1, y1), dpi)
                results.append({
                    "text": t_norm,
                    "bbox_points": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "bbox_pixels": {"x0": x0i, "y0": y0i, "x1": x1i, "y1": y1i},
                })
    return results

def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a["bbox_pixels"]["x0"], a["bbox_pixels"]["y0"], a["bbox_pixels"]["x1"], a["bbox_pixels"]["y1"]
    bx0, by0, bx1, by1 = b["bbox_pixels"]["x0"], b["bbox_pixels"]["y0"], b["bbox_pixels"]["x1"], b["bbox_pixels"]["y1"]
    inter_w = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def dedup_candidates(cands, iou_thresh: float = 0.6):
    """Merge duplicates that overlap heavily."""
    out = []
    for c in cands:
        keep = True
        for o in out:
            if _iou(c, o) >= iou_thresh and c["text"] == o["text"]:
                keep = False
                break
        if keep:
            out.append(c)
    return out

def find_label_candidates(words):
    # Return centroids of words that look like label hints (DRAWING / NO / NUMBER / SHEET / เลขที่แบบ)
    hints = []
    for (x0, y0, x1, y1, w, *_rest) in words:
        t = w.strip().upper()
        if t in LABEL_WORDS:
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            hints.append((cx, cy))
    return hints

def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def pick_best_match(matches, label_points):
    """Choose best match:
       1) If any label points present, choose the match whose center is closest to a label
       2) Else choose the bottom-right-most match (largest y, then largest x) — heuristics for title-block area
    """
    if not matches:
        return None
    if label_points:
        # Min distance to nearest label point
        ranked = []
        for m in matches:
            x0, y0, x1, y1, text = m
            c = ((x0+x1)/2.0, (y0+y1)/2.0)
            best_d2 = min(dist2(c, lp) for lp in label_points)
            ranked.append((best_d2, m))
        ranked.sort(key=lambda z: z[0])
        return ranked[0][1]
    # fallback: bottom-right preference
    matches.sort(key=lambda m: (m[3], m[2]))  # sort by y1 then x1
    return matches[-1]

def detect_all_on_page(page: fitz.Page, dpi: int, patterns):
    words = words_on_page(page)
    W, H = page.rect.width, page.rect.height

    # 1) Heuristic: keep only title-block-ish words
    words_roi = [(x0, y0, x1, y1, w, *_rest) for (x0, y0, x1, y1, w, *_rest) in words if in_titleblock_region(x0, y0, x1, y1, W, H)]

    # A) Diagnostic: if there is almost no text in the ROI, warn (likely outlined/rasterized)
    if len(words_roi) < 3:
        print("[titleblock] little/no vector text in ROI -> likely outlined/rasterized; consider OCR fallback", flush=True)

    results = []

    # 2) Direct word-level matches in ROI (with normalization)
    for (x0, y0, x1, y1, w, *_rest) in words_roi:
        text_norm = normalize_token(str(w))
        if text_norm and any(p.search(text_norm) for p in patterns):
            x0i, y0i, x1i, y1i = points_to_pixels((x0, y0, x1, y1), dpi)
            results.append({
                "text": text_norm,
                "bbox_points": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "bbox_pixels": {"x0": x0i, "y0": y0i, "x1": x1i, "y1": y1i},
            })

    # 3) Line-grouping + merge tiny gaps, then match (handles split tokens like 'A 1 - 01')
    results += group_tokens_and_match(words_roi, dpi, patterns)

    # 4) Deduplicate overlapping results
    results = dedup_candidates(results, iou_thresh=0.6)
    return results

def save_viz(page: fitz.Page, dpi: int, bbox_px: dict, out_path: Path):
    # Render page bitmap and draw rectangle (simple via PIL would need extra dep;
    # here we use PyMuPDF to draw vector overlay then rasterize)
    # Approach: create a temporary shape on a copy of the page via display list? Simpler: rasterize then pillow draw.
    # To avoid extra deps, we do a quick fitz pixmap then draw using PIL if available.
    from PIL import Image, ImageDraw  # lightweight
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox_px["x0"], bbox_px["y0"], bbox_px["x1"], bbox_px["y1"]
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=4)
    img.save(out_path)

def main():
    ap = argparse.ArgumentParser(description="Detect Drawing Number from vector PDFs in a folder using PyMuPDF (absolute pixel bbox).")
    ap.add_argument("--folder", required=True, help="Path to folder containing PDF files")
    ap.add_argument("--dpi", type=int, default=300, help="Rasterization DPI for pixel bbox")
    ap.add_argument("--regex", type=str, default=None, help="Custom regex for drawing number")
    ap.add_argument("--out-jsonl", type=str, default=None, help="Write detections to JSONL")
    ap.add_argument("--out-csv", type=str, default=None, help="Write detections to CSV")
    ap.add_argument("--save-viz", type=str, default=None, help="Directory to save page images with bbox")
    args = ap.parse_args()

    folder_path = Path(args.folder).expanduser().resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise SystemExit(f"Folder not found or not a directory: {folder_path}")

    out_jsonl = Path(args.out_jsonl).expanduser().resolve() if args.out_jsonl else None
    viz_dir = Path(args.save_viz).expanduser().resolve() if args.save_viz else None

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else None
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cf = open(out_csv, "w", newline="", encoding="utf-8")
        cw = csv.writer(cf)
        cw.writerow(["pdf", "page_index", "candidate_index", "dpi", "text", "x0_px", "y0_px", "x1_px", "y1_px"])
    else:
        cf = None
        cw = None

    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jf = open(out_jsonl, "w", encoding="utf-8")
    else:
        jf = None

    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    patterns = compile_patterns(args.regex)

    try:
        for pdf_path in sorted(folder_path.glob("*.pdf")):
            doc = fitz.open(pdf_path)
            try:
                for i, page in enumerate(doc):
                    results = detect_all_on_page(page, args.dpi, patterns)
                    print(f"[vector] {pdf_path.name} p{i}: {len(results)} candidates")
                    if results:
                        for j, res in enumerate(results):
                            record = {
                                "pdf": str(pdf_path),
                                "page_index": i,
                                "candidate_index": j,
                                "dpi": args.dpi,
                                "detection": res,
                            }
                            line = json.dumps(record, ensure_ascii=False)
                            print(line)
                            if jf:
                                jf.write(line + "\n")
                            if cw:
                                cw.writerow([
                                    str(pdf_path),
                                    i,
                                    j,
                                    args.dpi,
                                    res["text"],
                                    res["bbox_pixels"]["x0"],
                                    res["bbox_pixels"]["y0"],
                                    res["bbox_pixels"]["x1"],
                                    res["bbox_pixels"]["y1"],
                                ])
                            if viz_dir:
                                out_img = viz_dir / f"{pdf_path.stem}_p{i:03d}_cand{j:02d}.png"
                                save_viz(page, args.dpi, res["bbox_pixels"], out_img)
                    else:
                        record = {
                            "pdf": str(pdf_path),
                            "page_index": i,
                            "candidate_index": None,
                            "dpi": args.dpi,
                            "detection": None,
                        }
                        line = json.dumps(record, ensure_ascii=False)
                        print(line)
                        if jf:
                            jf.write(line + "\n")
            finally:
                doc.close()
    finally:
        if jf:
            jf.close()
        if cf:
            cf.close()

if __name__ == "__main__":
    main()