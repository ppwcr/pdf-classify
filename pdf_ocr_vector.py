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
    """
    Return a list of all candidates that match the drawing number patterns on this page.
    Each item is a dict with keys: text, bbox_points, bbox_pixels
    """
    words = words_on_page(page)
    results = []
    for (x0, y0, x1, y1, w, *_rest) in words:
        text = w.strip()
        if not text:
            continue
        if any(p.search(text) for p in patterns):
            pix = points_to_pixels((x0, y0, x1, y1), dpi)
            results.append({
                "text": text,
                "bbox_points": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "bbox_pixels": {"x0": pix[0], "y0": pix[1], "x1": pix[2], "y1": pix[3]},
            })
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