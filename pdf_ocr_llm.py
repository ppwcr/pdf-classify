#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_ocr_llm.py

Raster-only OCR pipeline for construction drawings.

What it does (no MLX, no VLM):
- Renders PDF pages to images (PyMuPDF) at a chosen DPI.
- (Optional) Auto-detects a title-block crop purely via OCR heuristics on the
  first N pages, then reuses the crop for the rest for speed.
- Falls back to a fixed bottom-right heuristic if auto-detect fails.
- Extracts drawing number(s) like A1-01, S1-01, E1-01, SN1-01 via regex.
- Writes JSONL (per page) and CSV summary. Can save debug images.

Requirements (install in your venv):
  pip install PyMuPDF Pillow pytesseract regex pandas requests

Note:
- For best results, install Tesseract binary on macOS (e.g. `brew install tesseract`).
- If pytesseract import fails, the script will exit with a friendly error.


source .venv/bin/activate
python pdf_ocr_raster.py \
  --folder "/Users/ppwcr/Desktop/print_pages/Test" \
  --out-jsonl ./out/raster_results.jsonl \
  --out-csv   ./out/raster_results.csv \
  --dpi 200 --detect-pages 3 --debug --debug-dir ./out/debug

# With LLM classification (OpenAI-compatible):
# export OPENAI_API_KEY=sk-...
# python pdf_ocr_llm.py \
#   --folder "/Users/ppwcr/Desktop/print_pages/Test" \
#   --out-jsonl ./out/raster_llm.jsonl \
#   --out-csv   ./out/raster_llm.csv \
#   --dpi 200 --llm --llm-model gpt-4o-mini

# With a local OpenAI-compatible server (e.g., LM Studio/Ollama gateway):
# python pdf_ocr_llm.py --folder ./print_pages --out-jsonl ./out.jsonl --out-csv ./out.csv --dpi 200 \
#   --llm --llm-base-url http://localhost:1234/v1 --llm-model qwen2.5:latest --llm-api-key-env ""

# With Ollama (local Qwen2.5‑VL‑7B):
#   ollama pull qwen2.5-vl:7b-instruct
#   python pdf_ocr_llm.py \
#     --folder "/Users/ppwcr/Desktop/print_pages/Test" \
#     --out-jsonl ./out/raster_llm_ollama.jsonl \
#     --out-csv   ./out/raster_llm_ollama.csv \
#     --dpi 300 --llm \
#     --llm-provider ollama \
#     --llm-base-url http://localhost:11434 \
#     --llm-model qwen2.5-vl:7b-instruct

# With MLX (local text LLM, classification on OCR text only):
#   pip install mlx-lm
#   python pdf_ocr_llm.py \
#     --folder "/Users/ppwcr/Desktop/print_pages/Test" \
#     --out-jsonl ./out/raster_llm_mlx.jsonl \
#     --out-csv   ./out/raster_llm_mlx.csv \
#     --dpi 300 --llm \
#     --llm-provider mlx \
#     --llm-model mlx-community/Qwen2.5-7B-Instruct-4bit
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
import time
import requests

import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageFilter, ImageDraw
try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

try:
    import pytesseract
except Exception as e:  # pragma: no cover
    print("[ERROR] pytesseract not available. Install with: pip install pytesseract", file=sys.stderr)
    print("        And ensure the Tesseract binary is installed (brew install tesseract)", file=sys.stderr)
    raise

DRAWING_NO_RE = re.compile(r"(?i)(?<![A-Z0-9])(A|S|E|SN)\s*\d{1,2}\s*-\s*\d{2}(?![A-Z0-9])")

def getenv_str(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val)

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

def clean_ocr_text(words: List[Dict]) -> str:
    """Group OCR words into readable lines and clean spacing.
    - Filters very low-confidence tokens
    - Sorts by y then x, groups into lines using median word height
    - Normalizes spaces and hyphen spacing
    """
    if not words:
        return ""
    # Filter extremely low-confidence tokens and blanks
    toks = [w for w in words if isinstance(w.get("text"), str) and w.get("text").strip()]
    if not toks:
        return ""
    # Confidence filter (keep if conf unknown or >= 50)
    kept = []
    for w in toks:
        try:
            if float(w.get("conf", 0)) < 50:
                continue
        except Exception:
            pass
        kept.append(w)
    if not kept:
        kept = toks
    # Sort by line (y center), then x
    for w in kept:
        w["_yc"] = int(w.get("y", 0)) + int(w.get("h", 0)) / 2.0
    kept.sort(key=lambda w: (w.get("_yc", 0), int(w.get("x", 0))))
    # Median height for line threshold
    try:
        import statistics as _st
        med_h = float(_st.median([int(w.get("h", 0)) or 0 for w in kept if int(w.get("h", 0)) > 0]) or 12.0)
    except Exception:
        med_h = 12.0
    line_thresh = max(6.0, 0.7 * med_h)
    # Group into lines
    lines: List[List[str]] = []
    cur_y = None
    cur: List[str] = []
    for w in kept:
        t = str(w.get("text", "")).strip()
        if not t:
            continue
        yc = float(w.get("_yc", 0.0))
        if cur_y is None:
            cur_y = yc
        # New line if vertical gap is big
        if abs(yc - cur_y) > line_thresh and cur:
            lines.append(cur)
            cur = [t]
            cur_y = yc
        else:
            cur.append(t)
            # track slowly varying baseline
            cur_y = (cur_y * 0.8) + (yc * 0.2)
    if cur:
        lines.append(cur)
    # Join words per line and normalize hyphen spacing
    out_lines: List[str] = []
    for ws in lines:
        s = " ".join(ws)
        s = re.sub(r"\s*-\s*", "-", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            out_lines.append(s)
    # Deduplicate consecutive identical lines
    final_lines: List[str] = []
    prev = None
    for s in out_lines:
        if s != prev:
            final_lines.append(s)
            prev = s
    return "\n".join(final_lines)

# ----------------------------- LLM Post-process --------------------------

LLM_SYSTEM = (
    "You are an assistant that extracts structured fields from OCR text of architectural/engineering title blocks. "
    "Return strict JSON with keys: drawing_no (string), sheet_name (string), drawing_title (string), "
    "project (string), discipline (string), revision (string), scale (string), date (string). "
    "If a field is not found, use an empty string. Never add extra keys. Prefer explicit labels near the bottom-right area."
)

def build_user_prompt(ocr_snippet: str) -> str:
    return (
        "OCR TEXT:\n"
        + ocr_snippet[:4000]  # keep prompt size manageable
    )

def call_openai_chat(base_url: str, api_key: str, model: str, temperature: float, system: str, user: str, timeout_s: int = 60) -> Dict:
    """
    Minimal OpenAI-compatible Chat Completions caller that expects a JSON object in the response.
    Compatible with OpenAI and many local servers that emulate the API (set --llm-base-url).
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": float(temperature),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print(f"[WARN] LLM classification failed: {e}", file=sys.stderr)
        return {
            "drawing_no": "",
            "sheet_name": "",
            "drawing_title": "",
            "project": "",
            "discipline": "",
            "revision": "",
            "scale": "",
            "date": "",
            "_error": str(e),
        }

def call_ollama_chat(base_url: str, model: str, temperature: float, system: str, user: str, timeout_s: int = 60) -> Dict:
    """
    Minimal Ollama /api/chat caller that expects a JSON object in the message content.
    Requires a local model (e.g., qwen2.5-vl:7b-instruct) and Ollama running at base_url.
    """
    url = base_url.rstrip("/") + "/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "format": "json",  # ask Ollama to produce valid JSON
        "options": {"temperature": float(temperature)},
        "stream": False,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "{}")
        return json.loads(content)
    except Exception as e:
        print(f"[WARN] Ollama classification failed: {e}", file=sys.stderr)
        return {
            "drawing_no": "",
            "sheet_name": "",
            "drawing_title": "",
            "project": "",
            "discipline": "",
            "revision": "",
            "scale": "",
            "date": "",
            "_error": str(e),
        }

def call_mlx_text_chat(model, tokenizer, temperature: float, system: str, user: str, max_tokens: int = 256) -> Dict:
    """
    Minimal MLX text LLM caller for local classification over OCR text.
    Requires `mlx-lm` installed and a locally available model.
    """
    try:
        from mlx_lm import generate as lm_generate
    except Exception as e:
        print(f"[ERROR] mlx-lm not available: {e}", file=sys.stderr)
        return {
            "drawing_no": "",
            "sheet_name": "",
            "drawing_title": "",
            "project": "",
            "discipline": "",
            "revision": "",
            "scale": "",
            "date": "",
            "_error": "mlx-lm not available",
        }
    # Simple instruction-style prompt; the system prompt already requests strict JSON.
    prompt = f"{system}\n\n{user}\n"
    try:
        try:
            out = lm_generate(model, tokenizer, prompt, max_tokens=int(max_tokens), temperature=float(temperature), top_p=0.0)
        except TypeError:
            # Fallback for older mlx-lm versions that expect 'temp'
            try:
                out = lm_generate(model, tokenizer, prompt, max_tokens=int(max_tokens), temp=float(temperature), top_p=0.0)
            except TypeError:
                # Final fallback: no temperature args
                out = lm_generate(model, tokenizer, prompt, max_tokens=int(max_tokens))
        content = out if isinstance(out, str) else str(out)
        try:
            return json.loads(content)
        except Exception:
            # Fallback: extract first JSON object
            m = re.search(r"(\{[\s\S]*?\})", content or "")
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
        return {
            "drawing_no": "",
            "sheet_name": "",
            "drawing_title": "",
            "project": "",
            "discipline": "",
            "revision": "",
            "scale": "",
            "date": "",
            "_error": "mlx-lm JSON parse failed",
        }
    except Exception as e:
        print(f"[WARN] MLX text classification failed: {e}", file=sys.stderr)
        return {
            "drawing_no": "",
            "sheet_name": "",
            "drawing_title": "",
            "project": "",
            "discipline": "",
            "revision": "",
            "scale": "",
            "date": "",
            "_error": str(e),
        }

    

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
                debug_dir: Optional[Path],
                llm_params: Optional[Dict]) -> List[Dict]:
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

        # Progress bar (especially useful for MLX text classification)
        use_pbar = bool(llm_params and (str(llm_params.get("provider") or "").lower() == "mlx") and _tqdm is not None)
        pbar = _tqdm(total=n, desc=f"{pdf_path.name}", unit="page") if use_pbar else None

        # Process all pages with chosen bbox
        for i in range(n):
            img = render_page(doc, i, dpi)
            # Progress log for multi-page PDFs
            if not pbar:
                print(f"[PAGE] {pdf_path.name} p{i+1}/{n}")
            W, H = img.size
            # Crop bottom-right title block area from fixed bbox
            x0, y0, x1, y1 = global_bbox.to_abs(W, H, pad=crop_pad)
            crop = img.crop((x0, y0, x1, y1))

            # OCR full text + word boxes in crop-local coordinates
            raw_text, words_local = ocr_text_and_boxes(crop)
            # Clean and group OCR into readable lines
            text = clean_ocr_text(words_local) or (raw_text or "")

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
                crop_viz_path = debug_dir / f"{pdf_path.stem}_p{i:03d}_crop_viz.png"
                crop_viz.save(crop_viz_path)

            numbers = extract_drawing_numbers(text)
            llm_class = {}
            if llm_params is not None:
                # Build prompt from the crop OCR text (prefer short, relevant snippet)
                user_prompt = build_user_prompt(text)
                provider = (llm_params.get("provider") or "openai").lower()
                mlx_dt_s: Optional[float] = None
                if provider == "ollama":
                    llm_class = call_ollama_chat(
                        base_url=llm_params.get("base_url", "http://localhost:11434"),
                        model=llm_params.get("model", "qwen2.5-vl:7b-instruct"),
                        temperature=float(llm_params.get("temperature", 0.0)),
                        system=llm_params.get("system", LLM_SYSTEM),
                        user=user_prompt,
                        timeout_s=int(llm_params.get("timeout_s", 60)),
                    )
                elif provider == "mlx":
                    t0 = time.time()
                    llm_class = call_mlx_text_chat(
                        model=llm_params.get("_mlx_model"),
                        tokenizer=llm_params.get("_mlx_tokenizer"),
                        temperature=float(llm_params.get("temperature", 0.0)),
                        system=llm_params.get("system", LLM_SYSTEM),
                        user=user_prompt,
                        max_tokens=int(llm_params.get("_mlx_max_tokens", 256)),
                    )
                    mlx_dt_s = time.time() - t0
                else:
                    llm_class = call_openai_chat(
                        base_url=llm_params.get("base_url", "https://api.openai.com/v1"),
                        api_key=llm_params.get("api_key", ""),
                        model=llm_params.get("model", "gpt-4o-mini"),
                        temperature=float(llm_params.get("temperature", 0.0)),
                        system=llm_params.get("system", LLM_SYSTEM),
                        user=user_prompt,
                        timeout_s=int(llm_params.get("timeout_s", 60)),
                    )
                if pbar and mlx_dt_s is not None:
                    try:
                        pbar.set_postfix({"mlx_ms": int(mlx_dt_s * 1000)})
                    except Exception:
                        pass
            rec = {
                "pdf_path": str(pdf_path),
                "page_index": i,
                "page_width": W,
                "page_height": H,
                "bbox_norm": {"x": global_bbox.x, "y": global_bbox.y, "w": global_bbox.w, "h": global_bbox.h},
                "bbox_abs": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "ocr_text": text,
                "ocr_words": words_abs,
                "ocr_text_snippet": (text[:200] if text else ""),
                "drawing_no": (numbers[0] if numbers else ""),
                "all_drawing_no_candidates": numbers,
                "notes": "raster_ocr",
                "llm_class": llm_class,
                "llm_used": bool(llm_params is not None),
            }
            records.append(rec)
            if debug_dir:
                crop.save(debug_dir / f"{pdf_path.stem}_p{i:03d}_crop.png")
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
    return records

# ----------------------------- CLI ---------------------------------------

def write_jsonl(jsonl_path: Path, rows: List[Dict]):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            out = dict(r)
            # Do not include bounding boxes in JSONL output
            out.pop("bbox_norm", None)
            out.pop("bbox_abs", None)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def write_csv(csv_path: Path, rows: List[Dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "pdf_path",
        "llm_json",
        "ocr_text_snippet",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            llm = r.get("llm_class") or {}
            out = {
                "pdf_path": r.get("pdf_path", ""),
                "llm_json": json.dumps(llm, ensure_ascii=False),
                "ocr_text_snippet": r.get("ocr_text_snippet", ""),
            }
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
    ap.add_argument("--llm", action="store_true", help="Enable LLM postprocess classification over OCR text")
    ap.add_argument("--llm-base-url", type=str, default=getenv_str("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="OpenAI-compatible base URL")
    ap.add_argument("--llm-model", type=str, default=getenv_str("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--llm-api-key-env", type=str, default="OPENAI_API_KEY", help="Env var name holding API key (ignored if server does not require)")
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--llm-timeout-s", type=int, default=60)
    ap.add_argument("--llm-system", type=str, default=LLM_SYSTEM)
    ap.add_argument("--llm-provider", type=str, default="openai", choices=["openai", "ollama", "mlx"], help="Which local/remote LLM API to use")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder}", file=sys.stderr)
        sys.exit(2)

    pdfs = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdfs:
        print(f"[WARN] No PDFs in {folder}")

    debug_dir = Path(args.debug_dir) if args.debug else None

    llm_params = None
    if args.llm:
        api_key = getenv_str(args.llm_api_key_env, "")
        # Defaults for Ollama/MLX if not overridden
        base_url = args.llm_base_url
        model_name = args.llm_model
        if args.llm_provider == "ollama":
            if args.llm_base_url == "https://api.openai.com/v1":
                base_url = "http://localhost:11434"
            if args.llm_model == getenv_str("OPENAI_MODEL", "gpt-4o-mini"):
                model_name = "qwen2.5-vl:7b-instruct"
        elif args.llm_provider == "mlx":
            # If default OpenAI model string, choose a local MLX-friendly model
            if args.llm_model == getenv_str("OPENAI_MODEL", "gpt-4o-mini"):
                model_name = "mlx-community/Qwen2.5-7B-Instruct-4bit"
        llm_params = {
            "provider": args.llm_provider,
            "base_url": base_url,
            "api_key": api_key,
            "model": model_name,
            "temperature": args.llm_temperature,
            "timeout_s": args.llm_timeout_s,
            "system": args.llm_system,
        }
        if args.llm_provider == "mlx":
            try:
                # Try to keep HF offline if model path is local
                try:
                    if Path(model_name).exists():
                        os.environ.setdefault("HF_HUB_OFFLINE", "1")
                except Exception:
                    pass
                from mlx_lm import load as lm_load
                mlx_model, mlx_tokenizer = lm_load(model_name)
                llm_params.update({
                    "_mlx_model": mlx_model,
                    "_mlx_tokenizer": mlx_tokenizer,
                    "_mlx_max_tokens": 256,
                })
            except Exception as e:
                print(f"[ERROR] Failed to load MLX text model '{model_name}': {e}", file=sys.stderr)
                sys.exit(3)
        print(f"[INFO] LLM postprocess enabled | provider={args.llm_provider} model={model_name} base={base_url}")

    all_rows: List[Dict] = []
    for pdf in pdfs:
        rows = process_pdf(
            pdf_path=pdf,
            dpi=args.dpi,
            detect_pages=args.detect_pages,
            crop_pad=args.crop_pad,
            iou_thresh=args.iou_thresh,
            debug_dir=debug_dir,
            llm_params=llm_params,
        )
        all_rows.extend(rows)

    write_jsonl(Path(args.out_jsonl), all_rows)
    write_csv(Path(args.out_csv), all_rows)
    print(f"Done. JSONL: {args.out_jsonl} | CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
