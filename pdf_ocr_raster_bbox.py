#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_ocr_raster.py

Raster-only OCR pipeline for construction drawings.

What it does (no MLX, no VLM):
- Renders PDF pages to images (PyMuPDF) at a chosen DPI.
- (Optional) Auto-detects a title-block crop purely via OCR heuristics on the
  first N pages, then reuses the crop for the rest for speed.
- Falls back to a fixed bottom-right heuristic if auto-detect fails.
- Extracts drawing number(s) like A1-01, S1-01, E1-01, SN1-01 via regex.
- Writes JSONL (per page) and CSV summary. Can save debug images.

Requirements (install in your venv):
  pip install PyMuPDF Pillow pytesseract regex pandas

Note:
- For best results, install Tesseract binary on macOS (e.g. `brew install tesseract`).
- If pytesseract import fails, the script will exit with a friendly error.


source .venv/bin/activate
python pdf_ocr_raster.py \
  --folder "/Users/ppwcr/Desktop/print_pages/Test" \
  --out-jsonl ./out/raster_results.jsonl \
  --out-csv   ./out/raster_results.csv \
  --dpi 200 --detect-pages 3 --debug --debug-dir ./out/debug
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import List, Optional, Tuple, Dict

import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageFilter, ImageDraw

try:
    import pytesseract
except Exception as e:  # pragma: no cover
    print("[ERROR] pytesseract not available. Install with: pip install pytesseract", file=sys.stderr)
    print("        And ensure the Tesseract binary is installed (brew install tesseract)", file=sys.stderr)
    raise

DRAWING_NO_RE = re.compile(r"(?i)(?<![A-Z0-9])(A|S|E|SN)\s*\d{1,2}\s*-\s*\d{2}(?![A-Z0-9])")

@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float
    def to_abs(self, width: int, height: int, pad: float = 0.0) -> Tuple[int, int, int, int]:
        x0 = max(0, int((self.x - pad) * width))
        y0 = max(0, int((self.y - pad) * height))
        x1 = min(width, int((self.x + self.w + pad) * width))
        y1 = min(height, int((self.y + self.h + pad) * height))
        return x0, y0, x1, y1
    def iou(self, other: "BBox") -> float:
        ax0, ay0, ax1, ay1 = self.x, self.y, self.x + self.w, self.y + self.h
        bx0, by0, bx1, by1 = other.x, other.y, other.x + other.w, other.y + other.h
        iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        ih = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = iw * ih
        if inter <= 0:
            return 0.0
        a = self.w * self.h
        b = other.w * other.h
        return inter / (a + b - inter)

# ----------------------------- Heuristics ---------------------------------

def bottom_right_heuristic() -> BBox:
    """Conservative default: bottom-right 35% width × 28% height."""
    return BBox(x=0.65, y=0.72, w=0.34, h=0.27)

KEYWORDS = (
    "DRAWING", "TITLE", "PROJECT", "ARCHITECT", "ENGINEER", "SHEET", "NO", "NUMBER"
)

def ocr_text(img: Image.Image) -> str:
    # Light preproc for OCR
    g = ImageOps.grayscale(img)
    g = g.filter(ImageFilter.MedianFilter(3))
    # Tesseract config: psm 6 (Assume a uniform block of text), oem 3 (default)
    try:
        return pytesseract.image_to_string(g, lang="tha+eng", config="--oem 3 --psm 6")
    except Exception:
        return ""

def ocr_text_and_boxes(img: Image.Image) -> Tuple[str, List[Dict]]:
    """Run Tesseract and return (full_text, word_boxes).
    word_boxes are dicts with keys: text, conf, x, y, w, h (coordinates are in the
    given image's pixel space).
    """
    g = ImageOps.grayscale(img)
    g = g.filter(ImageFilter.MedianFilter(3))
    try:
        data = pytesseract.image_to_data(
            g,
            lang="tha+eng",
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return "", []

    words: List[Dict] = []
    texts: List[str] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf_raw = data.get("conf", ["-1"])[i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0
        if txt and conf >= 0:
            x = int(data.get("left", [0])[i])
            y = int(data.get("top", [0])[i])
            w = int(data.get("width", [0])[i])
            h = int(data.get("height", [0])[i])
            words.append({"text": txt, "conf": conf, "x": x, "y": y, "w": w, "h": h})
            texts.append(txt)
    full_text = " ".join(texts)
    return full_text, words


def _y_overlap_ratio(a: Dict, b: Dict) -> float:
    ay0, ay1 = a["y0"], a["y1"]
    by0, by1 = b["y0"], b["y1"]
    inter = max(0, min(ay1, by1) - max(ay0, by0))
    denom = max(1, min(ay1 - ay0, by1 - by0))
    return inter / float(denom)


def merge_nearby_word_boxes(words_abs: List[Dict], gap_px: int = 12, min_y_overlap: float = 0.5) -> List[Dict]:
    """Group words that lie on the same text line and are horizontally close.
    Returns regions with keys: text, conf_avg, x0,y0,x1,y1,w,h, words(list).
    Joining uses no spaces to handle Thai where words are not whitespace-separated.
    """
    if not words_abs:
        return []

    # 1) Sort by y then x
    words = sorted(words_abs, key=lambda w: (w["y0"], w["x0"]))

    # 2) Build provisional lines using vertical overlap criterion
    lines: List[List[Dict]] = []
    for w in words:
        placed = False
        for line in lines:
            ref = line[len(line)//2]  # median element as representative
            if _y_overlap_ratio(w, ref) >= min_y_overlap:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    # 3) Within each line, merge neighbors when gap <= gap_px
    regions: List[Dict] = []
    for line in lines:
        line.sort(key=lambda w: w["x0"])  # left-to-right
        cur: List[Dict] = []
        for w in line:
            if not cur:
                cur = [w]
                continue
            prev = cur[-1]
            gap = w["x0"] - prev["x1"]
            same_row = _y_overlap_ratio(w, prev) >= min_y_overlap
            if gap <= gap_px and same_row:
                cur.append(w)
            else:
                xs0 = min(t["x0"] for t in cur)
                ys0 = min(t["y0"] for t in cur)
                xs1 = max(t["x1"] for t in cur)
                ys1 = max(t["y1"] for t in cur)
                text_join = "".join(t["text"] for t in cur)  # tight join (Thai-friendly)
                conf_avg = sum(float(t.get("conf", 0)) for t in cur) / max(1, len(cur))
                regions.append({
                    "text": text_join,
                    "conf_avg": conf_avg,
                    "x0": xs0, "y0": ys0, "x1": xs1, "y1": ys1,
                    "w": xs1 - xs0, "h": ys1 - ys0,
                    "words": cur,
                })
                cur = [w]
        if cur:
            xs0 = min(t["x0"] for t in cur)
            ys0 = min(t["y0"] for t in cur)
            xs1 = max(t["x1"] for t in cur)
            ys1 = max(t["y1"] for t in cur)
            text_join = "".join(t["text"] for t in cur)
            conf_avg = sum(float(t.get("conf", 0)) for t in cur) / max(1, len(cur))
            regions.append({
                "text": text_join,
                "conf_avg": conf_avg,
                "x0": xs0, "y0": ys0, "x1": xs1, "y1": ys1,
                "w": xs1 - xs0, "h": ys1 - ys0,
                "words": cur,
            })

    regions.sort(key=lambda r: (r["y0"], r["x0"]))
    return regions

def find_title_block_bbox_by_ocr_probe(img: Image.Image, grid: int = 4) -> Optional[BBox]:
    """Probe the page by splitting into a grid; pick the cell with the
    highest keyword hits as the title block candidate. Returns normalized bbox
    or None if low confidence.
    """
    W, H = img.size
    best = None
    best_score = 0
    for gy in range(grid):
        for gx in range(grid):
            x0 = int(gx * W / grid)
            y0 = int(gy * H / grid)
            x1 = int((gx + 1) * W / grid)
            y1 = int((gy + 1) * H / grid)
            crop = img.crop((x0, y0, x1, y1))
            text = ocr_text(crop)
            score = sum(1 for k in KEYWORDS if k.lower() in text.lower())
            # Bonus for being in bottom-right quadrant
            if gx >= grid // 2 and gy >= grid // 2:
                score += 1
            if score > best_score:
                best_score = score
                best = (x0, y0, x1, y1)
    if not best or best_score < 2:
        return None
    x0, y0, x1, y1 = best
    return BBox(x=x0 / W, y=y0 / H, w=(x1 - x0) / W, h=(y1 - y0) / H)

# ----------------------------- Core Logic ---------------------------------

def render_page(doc: fitz.Document, index: int, dpi: int) -> Image.Image:
    page = doc.load_page(index)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def extract_drawing_numbers(text: str) -> List[str]:
    cands = [m.group(0).strip() for m in DRAWING_NO_RE.finditer(text or "")]
    # normalize whitespace around hyphen
    cands = [re.sub(r"\s*-\s*", "-", c) for c in cands]
    # dedupe preserving order
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def process_pdf(pdf_path: Path,
                dpi: int,
                detect_pages: int,
                crop_pad: float,
                iou_thresh: float,
                debug_dir: Optional[Path]) -> List[Dict]:
    records: List[Dict] = []
    with fitz.open(pdf_path) as doc:
        n = doc.page_count
        print(f"[INFO] PDF: {pdf_path.name} | pages: {n}")
        # Ensure debug dir exists when --debug is enabled
        if debug_dir is not None:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
        # Fixed bottom-right title block region only (no probing)
        print("[INFO] Using fixed bottom-right title-block region.", flush=True)
        global_bbox: BBox = BBox(x=0.75, y=0.75, w=0.25, h=0.25)

        # Process all pages with chosen bbox
        for i in range(n):
            img = render_page(doc, i, dpi)
            # Progress log for multi-page PDFs
            print(f"[PAGE] {pdf_path.name} p{i+1}/{n}")
            W, H = img.size
            # Crop bottom-right title block area from fixed bbox
            x0, y0, x1, y1 = global_bbox.to_abs(W, H, pad=crop_pad)
            crop = img.crop((x0, y0, x1, y1))

            # OCR full text + word boxes in crop-local coordinates
            text, words_local = ocr_text_and_boxes(crop)

            # Map crop-local boxes -> page-absolute pixels and normalized page coords
            words_abs: List[Dict] = []
            for wbox in words_local:
                abs_x0 = x0 + int(wbox["x"])  # page-absolute px
                abs_y0 = y0 + int(wbox["y"])  # page-absolute px
                abs_w = int(wbox["w"])        # px width
                abs_h = int(wbox["h"])        # px height
                words_abs.append({
                    "text": wbox["text"],
                    "conf": wbox["conf"],
                    "x0": abs_x0,
                    "y0": abs_y0,
                    "x1": abs_x0 + abs_w,
                    "y1": abs_y0 + abs_h,
                    "w": abs_w,
                    "h": abs_h,
                    # Normalized to full page size [0,1]
                    "x": abs_x0 / float(W),
                    "y": abs_y0 / float(H),
                    "w_norm": abs_w / float(W),
                    "h_norm": abs_h / float(H),
                })

            # Merge nearby word boxes into line regions
            regions_abs = merge_nearby_word_boxes(words_abs, gap_px=12, min_y_overlap=0.5)
            for rg in regions_abs:
                rg["x"] = rg["x0"] / float(W)
                rg["y"] = rg["y0"] / float(H)
                rg["w_norm"] = rg["w"] / float(W)
                rg["h_norm"] = rg["h"] / float(H)

            # --- Visualization: draw OCR word boxes and title-block bbox ---
            if debug_dir is not None:
                # Full-page viz with page-absolute word boxes
                page_viz = img.copy()
                d = ImageDraw.Draw(page_viz)
                # Draw global bbox in blue
                d.rectangle([(x0, y0), (x1, y1)], outline="blue", width=3)
                # Draw each word box in red
                for wb in words_abs:
                    d.rectangle([(wb["x0"], wb["y0"]), (wb["x1"], wb["y1"])], outline="red", width=2)
                # Draw merged regions in green
                for rg in regions_abs:
                    d.rectangle([(rg["x0"], rg["y0"]), (rg["x1"], rg["y1"])], outline="green", width=3)
                page_viz_path = debug_dir / f"{pdf_path.stem}_p{i:03d}_viz.png"
                page_viz.save(page_viz_path)

                # Crop viz using crop-local coordinates
                crop_viz = crop.copy()
                dc = ImageDraw.Draw(crop_viz)
                for wloc in words_local:
                    cx0 = int(wloc["x"])  # crop-local
                    cy0 = int(wloc["y"])  # crop-local
                    cx1 = cx0 + int(wloc["w"])  # crop-local
                    cy1 = cy0 + int(wloc["h"])  # crop-local
                    dc.rectangle([(cx0, cy0), (cx1, cy1)], outline="red", width=2)
                for rg in regions_abs:
                    cx0 = max(0, rg["x0"] - x0)
                    cy0 = max(0, rg["y0"] - y0)
                    cx1 = max(0, rg["x1"] - x0)
                    cy1 = max(0, rg["y1"] - y0)
                    dc.rectangle([(cx0, cy0), (cx1, cy1)], outline="green", width=3)
                crop_viz_path = debug_dir / f"{pdf_path.stem}_p{i:03d}_crop_viz.png"
                crop_viz.save(crop_viz_path)

            numbers = extract_drawing_numbers(text)
            rec = {
                "pdf_path": str(pdf_path),
                "page_index": i,
                "page_width": W,
                "page_height": H,
                "bbox_norm": {"x": global_bbox.x, "y": global_bbox.y, "w": global_bbox.w, "h": global_bbox.h},
                "bbox_abs": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "ocr_text": text,
                "ocr_words": words_abs,
                "ocr_regions": regions_abs,
                "ocr_text_snippet": (text[:200] if text else ""),
                "drawing_no": (numbers[0] if numbers else ""),
                "all_drawing_no_candidates": numbers,
                "notes": "raster_ocr",
            }
            records.append(rec)
            if debug_dir:
                crop.save(debug_dir / f"{pdf_path.stem}_p{i:03d}_crop.png")
    return records

# ----------------------------- CLI ---------------------------------------

def write_jsonl(jsonl_path: Path, rows: List[Dict]):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(csv_path: Path, rows: List[Dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "pdf_path", "page_index", "page_width", "page_height",
        "drawing_no", "bbox_norm", "bbox_abs", "ocr_text", "ocr_text_snippet", "ocr_words_json", "ocr_regions_json", "notes"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in cols}
            out["ocr_words_json"] = json.dumps(r.get("ocr_words", []), ensure_ascii=False)
            out["ocr_regions_json"] = json.dumps(r.get("ocr_regions", []), ensure_ascii=False)
            w.writerow(out)


def main():
    ap = argparse.ArgumentParser(description="Raster OCR classifier for construction drawings (no MLX/VLM)")
    ap.add_argument("--folder", type=str, required=True, help="Folder containing PDFs (no recursion)")
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--detect-pages", type=int, default=0)
    ap.add_argument("--crop-pad", type=float, default=0.04, help="Normalized padding added to bbox")
    ap.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for bbox consensus")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-dir", type=str, default="./out/debug")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder}", file=sys.stderr)
        sys.exit(2)

    pdfs = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdfs:
        print(f"[WARN] No PDFs in {folder}")

    debug_dir = Path(args.debug_dir) if args.debug else None

    all_rows: List[Dict] = []
    for pdf in pdfs:
        rows = process_pdf(
            pdf_path=pdf,
            dpi=args.dpi,
            detect_pages=args.detect_pages,
            crop_pad=args.crop_pad,
            iou_thresh=args.iou_thresh,
            debug_dir=debug_dir,
        )
        all_rows.extend(rows)

    write_jsonl(Path(args.out_jsonl), all_rows)
    write_csv(Path(args.out_csv), all_rows)
    print(f"Done. JSONL: {args.out_jsonl} | CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
