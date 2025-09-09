#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm_ocr.py

OCR pipeline for construction drawings using MLX-VLM model (Qwen2.5-VL or similar)
instead of Tesseract. Works like pdf_ocr_raster.py but delegates text extraction
to a Vision-Language Model via mlx_vlm.

Requirements:
  pip install mlx-vlm PyMuPDF Pillow
  pip install tqdm

Example:
  python vlm_ocr.py \
    --folder "/path/to/pdfs" \
    --out-jsonl ./out/vlm_results.jsonl \
    --out-csv   ./out/vlm_results.csv \
    --dpi 200 --debug --debug-dir ./out/debug
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import os
import signal
import contextlib
import time
from pathlib import Path
from typing import List, Dict, Optional
import functools

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

# Optional: Tesseract fallback (not on by default, but available if needed)
try:
    import pytesseract
    from PIL import ImageOps, ImageFilter
    def _ocr_tesseract_text(img: Image.Image) -> str:
        g = ImageOps.grayscale(img)
        g = g.filter(ImageFilter.MedianFilter(3))
        try:
            return pytesseract.image_to_string(g, lang="tha+eng", config="--oem 3 --psm 6")
        except Exception:
            return ""
except Exception:
    pytesseract = None

try:
    from mlx_vlm import load as vlm_load, generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
except ImportError:
    print("[ERROR] mlx-vlm is not installed. Please run: pip install mlx-vlm", file=sys.stderr)
    sys.exit(1)

import re as _re

def _parse_vlm_json_block(raw_text: str):
    """Try to extract a JSON object from VLM output that may be wrapped in ```json ... ``` fences
    or preceded/followed by extra text. Returns a dict or {}.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {}
    s = raw_text.strip()
    # 1) Fenced code block first
    m = _re.search(r"```json\s*(\{[\s\S]*?\})\s*```", s, flags=_re.I)
    if not m:
        # also accept any fenced block without explicit json language
        m = _re.search(r"```\s*(\{[\s\S]*?\})\s*```", s)
    if m:
        try:
            import json as _json
            return _json.loads(m.group(1))
        except Exception:
            pass
    # 2) Try to find the first top-level object {...}
    m = _re.search(r"(\{[\s\S]*\})", s)
    if m:
        try:
            import json as _json
            return _json.loads(m.group(1))
        except Exception:
            pass
    return {}

@contextlib.contextmanager
def time_limit(seconds: float):
    def _raise_timeout(signum, frame):
        raise TimeoutError("VLM generation timed out")
    if seconds and seconds > 0:
        old = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        if seconds and seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


def render_page(doc: fitz.Document, index: int, dpi: int) -> Image.Image:
    page = doc.load_page(index)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def vlm_ocr(img: Image.Image, model: str, processor, config, model_obj, temperature: float = 0.0, max_tokens: int = 32, timeout_s: float = 5.0):
    """
    Use VLM to extract title-block fields. Returns a tuple:
    (best_match: str, candidates: List[str], raw_text: str, extras: dict)

    Fields:
      - best_match: primary drawing number
      - candidates: other drawing number candidates
      - raw_text: raw model output
      - extras: {
          "drawing_title": str or "",
          "scale": str or "",           # e.g. "1:100"
          "page_x": str or "",          # e.g. "1" from "1/12"
          "page_total": str or ""       # e.g. "12" from "1/12"
        }
    """
    import re, json
    sys_prompt = (
        "Extract ONLY these title-block fields as strict JSON: "
        "{\"drawing_no\": \"<best or empty>\", "
        " \"drawing_title\": \"<title or empty>\", "
        " \"scale\": \"<like 1:100 or empty>\", "
        " \"candidates\": [\"...\"], "
        " \"page\": {\"x\": \"<page number or empty>\", \"total\": \"<total pages or empty>\"} } "
        "Return UTF-8 text. Thai labels and Thai content are allowed."
    )
    user_prompt = (
        "Look only at the title block area. Return STRICT JSON only. "
        "Examples: drawing_no A-01, A1-01, E-101; scale 1:100, 1:75; page 1/12, SHEET 1 OF 12."
    )
    formatted = apply_chat_template(
        processor, config, f"<|system|>\n{sys_prompt}\n<|user|>\n{user_prompt}\n", num_images=1
    )
    try:
        with time_limit(timeout_s):
            out = vlm_generate(
                model_obj, processor, formatted, [img],
                max_tokens=max_tokens, temperature=0.0, top_p=0.0, seed=0
            )
    except TimeoutError:
        return "", [], "__TIMEOUT__", {"drawing_title": "", "scale": "", "page_x": "", "page_total": ""}
    if isinstance(out, (list, tuple)):
        out = out[0] if out else ""
    raw_text = str(out).strip()

    best, candidates = "", []
    extras = {"drawing_title": "", "scale": "", "page_x": "", "page_total": ""}
    # Try strict JSON first
    try:
        data = json.loads(raw_text)
    except Exception:
        # Try to extract JSON from fenced code block or mixed text
        data = _parse_vlm_json_block(raw_text)

    def _clean(s):
        return str(s).strip() if isinstance(s, (str, int, float)) else ""

    if isinstance(data, dict):
        best = _clean(data.get("drawing_no"))
        candidates = [_clean(c) for c in (data.get("candidates") or []) if _clean(c)]
        # drawing_title
        extras["drawing_title"] = _clean(data.get("drawing_title") or data.get("title"))
        # scale
        extras["scale"] = _clean(data.get("scale"))
        # page object
        page_obj = data.get("page") or {}
        if isinstance(page_obj, dict):
            extras["page_x"] = _clean(page_obj.get("x"))
            extras["page_total"] = _clean(page_obj.get("total"))

    # Regex fallback/augment (broader):
    # - A101, S201, E301, AR000
    # - A1-01, S1-01, E1-01, SN1-01, E-101, A-01
    pat = re.compile(r"(?i)(?<![A-Z0-9])(?:AR|SN|A|S|E)\s*\d{1,3}(?:\s*-\s*\d{2,3})?(?![A-Z0-9])")
    found = [re.sub(r"\s*-\s*", "-", m.group(0).strip()) for m in pat.finditer(raw_text)]
    for c in found:
        if c not in candidates:
            candidates.append(c)
    # --- Fallbacks for title, scale, and page x/xx ---
    # Scale patterns: "SCALE 1:100", "Scale: 1:50", or bare "1:100"
    scale_pat1 = re.search(r"(?i)\bSCALE\b\s*[:\-]?\s*(\d+\s*:\s*\d+)", raw_text)
    scale_pat2 = re.search(r"\b(\d+\s*:\s*\d+)\b", raw_text) if not scale_pat1 else None
    if not extras["scale"]:
        if scale_pat1:
            extras["scale"] = scale_pat1.group(1).replace(" ", "")
        elif scale_pat2:
            extras["scale"] = scale_pat2.group(1).replace(" ", "")

    # Page x/xx patterns: "1/12", "SHEET 1 OF 12"
    page_pat = re.search(r"(?i)\b(\d{1,3})\s*/\s*(\d{1,3})\b", raw_text)
    if not page_pat:
        page_pat = re.search(r"(?i)\bSHEET\s*(\d{1,3})\s*OF\s*(\d{1,3})\b", raw_text)
    if (not extras["page_x"]) and page_pat:
        extras["page_x"], extras["page_total"] = page_pat.group(1), page_pat.group(2)

    if not best and candidates:
        best = candidates[0]

    return best, candidates, raw_text, extras


def process_pdf(pdf_path: Path, dpi: int, model_id: str, debug_dir: Optional[Path], model_obj, processor, config, max_tokens: int, timeout_s: float) -> List[Dict]:
    records: List[Dict] = []
    # Read fast-path tunables from environment (fallback if not provided by CLI)
    max_side = int(os.environ.get("VLM_MAX_SIDE", os.environ.get("VLM_MAX_SIDE_CROP", "960")))
    single_crop = os.environ.get("VLM_SINGLE_CROP", "0") == "1"
    no_debug_save = os.environ.get("VLM_NO_DEBUG", "0") == "1"

    # model_obj, processor = vlm_load(model_id)
    # config = load_config(model_id)

    with fitz.open(pdf_path) as doc:
        n = doc.page_count
        print(f"[INFO] PDF: {pdf_path.name} | pages: {n}")
        for i in tqdm(range(n), desc=f"Processing {pdf_path.name}", unit="page"):
            img = render_page(doc, i, dpi)
            # Get full-resolution page size (before any resizing)
            W_full, H_full = img.size

            # --- CROP FIRST (title block area) on full-res page ---
            # Define crop regions using full-res dims
            br_x0 = int(W_full * 0.70); br_y0 = int(H_full * 0.70)
            bottom_strip_y = int(H_full * 0.85)
            crops = [("br_large", img.crop((br_x0, br_y0, W_full, H_full)))]
            if not single_crop:
                crops.append(("bottom_strip", img.crop((int(W_full * 0.50), bottom_strip_y, W_full, H_full))) )

            # --- THEN resize the crop only (avoid shrinking the whole page) ---
            def _resize_crop(cimg: Image.Image, max_side: int) -> Image.Image:
                w, h = cimg.size
                long_side = max(w, h)
                if long_side > max_side:
                    s = max_side / float(long_side)
                    cimg = cimg.resize((int(w * s), int(h * s)))
                return cimg

            best, candidates, raw_text, extras = "", [], "", {"drawing_title": "", "scale": "", "page_x": "", "page_total": ""}
            chosen = ""; img_to_save = None
            used_crops = []
            for tag, cimg in crops:
                cimg_r = _resize_crop(cimg, max_side=max_side)
                b, cand, raw, extra = vlm_ocr(cimg_r, model_id, processor, config, model_obj, max_tokens=max_tokens, timeout_s=timeout_s)
                contributed = False
                if b and not best:
                    best = b; contributed = True
                for c in (cand or []):
                    if c and c not in candidates:
                        candidates.append(c); contributed = True
                for k in ("drawing_title", "scale", "page_x", "page_total"):
                    if not extras.get(k) and (extra or {}).get(k):
                        extras[k] = (extra or {}).get(k); contributed = True
                if contributed:
                    if img_to_save is None:
                        img_to_save = cimg_r
                    if not raw_text:
                        raw_text = raw
                    used_crops.append(tag)
            if used_crops:
                chosen = ",".join(used_crops)
            else:
                # fallback to the first crop result (even if empty) for reproducibility
                cimg_r = _resize_crop(crops[0][1], max_side=max_side)
                b, cand, raw, extra = vlm_ocr(cimg_r, model_id, processor, config, model_obj, max_tokens=max_tokens, timeout_s=timeout_s)
                best, candidates, raw_text, extras = b, cand, raw, extra
                img_to_save = cimg_r
                chosen = crops[0][0]

            rec = {
                "pdf_path": str(pdf_path),
                "page_index": i,
                "page_width": W_full,
                "page_height": H_full,
                "drawing_no": best,
                "all_drawing_no_candidates": candidates,
                "vlm_raw": raw_text,
                "notes": f"vlm_ocr_drawing_no_only|crop={chosen}",
                "drawing_title": extras.get("drawing_title", ""),
                "scale": extras.get("scale", ""),
                "page_x": extras.get("page_x", ""),
                "page_total": extras.get("page_total", ""),
            }
            records.append(rec)
            if debug_dir and not no_debug_save and img_to_save is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                img_to_save.save(debug_dir / f"{pdf_path.stem}_p{i:03d}_{chosen}.png")
    return records


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict], mode: str = "minimal"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "full":
        cols = [
            "pdf_path", "page_index", "page_width", "page_height",
            "drawing_no", "drawing_title", "scale", "page_x", "page_total", "notes"
        ]
    else:
        cols = [
            "pdf_path", "page_index", "page_width", "page_height",
            "drawing_no", "notes"
        ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            dn = (r.get("drawing_no") or "").strip()
            if not dn:
                cands = r.get("all_drawing_no_candidates") or []
                if isinstance(cands, (list, tuple)) and cands:
                    dn = str(cands[0]).strip()
            base = {
                "pdf_path": r.get("pdf_path", ""),
                "page_index": r.get("page_index", ""),
                "page_width": r.get("page_width", ""),
                "page_height": r.get("page_height", ""),
                "drawing_no": dn,
                "notes": r.get("notes", ""),
            }
            if mode == "full":
                base.update({
                    "drawing_title": r.get("drawing_title", ""),
                    "scale": r.get("scale", ""),
                    "page_x": r.get("page_x", ""),
                    "page_total": r.get("page_total", ""),
                })
            w.writerow(base)


def main():
    ap = argparse.ArgumentParser(description="OCR for PDFs using MLX-VLM model")
    ap.add_argument("--folder", type=str, required=True, help="Folder with PDFs")
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--model", type=str, default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-dir", type=str, default="./out/debug")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout-s", type=float, default=5.0)
    ap.add_argument("--max-side", type=int, default=960, help="Max long side of crop sent to VLM")
    ap.add_argument("--single-crop", action="store_true", help="Use only one bottom-right crop (skip bottom strip)")
    ap.add_argument("--no-debug-save", action="store_true", help="Skip saving debug crop images to reduce I/O")
    ap.add_argument("--csv-mode", choices=["minimal", "full"], default="minimal", help="CSV column set: minimal=drawing_no only, full=include title/scale/page")
    args = ap.parse_args()

    # Prefer offline/local model to avoid repeated fetching
    try:
        if Path(args.model).exists():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
    except Exception:
        pass
    # Make Metal errors synchronous for clearer failures
    os.environ.setdefault("MX_FORCE_SYNC", "0")

    model_id = args.model
    config = load_config(model_id)
    model_obj, processor = vlm_load(model_id)

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder}", file=sys.stderr)
        sys.exit(2)

    pdfs = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdfs:
        print(f"[WARN] No PDFs found in {folder}")

    debug_dir = Path(args.debug_dir) if args.debug else None
    max_side = args.max_side
    single_crop = args.single_crop
    no_debug_save = args.no_debug_save

    all_rows: List[Dict] = []
    for pdf in tqdm(pdfs, desc="PDF files", unit="file"):
        rows = process_pdf(
            pdf,
            dpi=args.dpi,
            model_id=model_id,
            debug_dir=debug_dir,
            model_obj=model_obj,
            processor=processor,
            config=config,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
            # new params below
        )
        all_rows.extend(rows)

    write_jsonl(Path(args.out_jsonl), all_rows)
    write_csv(Path(args.out_csv), all_rows, mode=args.csv_mode)
    print(f"Done. JSONL: {args.out_jsonl} | CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
