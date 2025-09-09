#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm_ocr_fast_bbox.py

Fast 3-step pipeline to extract drawing number (sheet no.) and drawing name/title
from construction drawings by focusing only on the bottom-right title block:

1) Fixed bottom-right crop per page using PyMuPDF clip (avoids full-page raster).
2) Run OCR to get word bounding boxes, locate label regions (e.g., DRAWING NO, TITLE),
   then ask a local VLM (e.g., Qwen2.5‑VL 7B via Ollama) to read only those small boxes
   for higher-accuracy text at low cost.
3) Classify the combined text to produce fields: drawing_no, drawing_title, sheet_name.

Usage examples:

  # Local Ollama (recommended):
  #   ollama pull qwen2.5vl:7b
  python vlm_ocr_fast_bbox.py \
    --folder '/Users/ppwcr/Desktop/print_pages' \
    --out-jsonl ./out/fast_bbox.jsonl \
    --out-csv   ./out/fast_bbox.csv \
    --dpi 250 \
    --llm-provider ollama \
    --llm-base-url http://localhost:11434 \
    --llm-model qwen2.5vl:7b \
    --debug

  # OpenAI-compatible local server (LM Studio / OpenRouter-compatible):
  # export OPENAI_API_KEY=...
  python vlm_ocr_fast_bbox.py \
    --folder ./samples \
    --out-jsonl ./out/fast_bbox.jsonl \
    --out-csv   ./out/fast_bbox.csv \
    --dpi 250 \
    --llm-provider openai \
    --llm-base-url http://localhost:1234/v1 \
    --llm-model qwen2.5vl:7b

Notes:
- Requires: PyMuPDF, Pillow, pytesseract, requests. Install Tesseract binary (e.g., macOS: brew install tesseract).
- This script focuses on speed and accuracy for just the key fields; it does not depend on mlx-vlm.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageFilter

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


# ----------------------------- Utils -------------------------------------


def getenv_str(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return default if v is None else str(v)


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


def render_crop(doc: fitz.Document, index: int, dpi: int, bbox: BBox, pad: float) -> Image.Image:
    page = doc.load_page(index)
    page_rect = page.rect
    x0 = max(0.0, (bbox.x - pad) * page_rect.width)
    y0 = max(0.0, (bbox.y - pad) * page_rect.height)
    x1 = min(page_rect.width, (bbox.x + bbox.w + pad) * page_rect.width)
    y1 = min(page_rect.height, (bbox.y + bbox.h + pad) * page_rect.height)
    clip = fitz.Rect(x0, y0, x1, y1)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def image_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Qwen2.5-VL visual tokenizer wants dimensions to be multiples of 28
def _round_up_to_multiple(x: int, m: int) -> int:
    if m <= 0:
        return x
    r = x % m
    return x if r == 0 else x + (m - r)


# ----------------------------- OCR (words + lines) -----------------------


def ocr_words(img: Image.Image) -> List[Dict]:
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
        return []

    out: List[Dict] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data.get("conf", ["-1"])[i])
        except Exception:
            conf = -1.0
        if conf < 0:
            continue
        out.append(
            {
                "text": txt,
                "conf": conf,
                "x": int(data.get("left", [0])[i]),
                "y": int(data.get("top", [0])[i]),
                "w": int(data.get("width", [0])[i]),
                "h": int(data.get("height", [0])[i]),
            }
        )
    return out


def group_words_to_lines(words: List[Dict]) -> List[Dict]:
    if not words:
        return []
    toks = [w for w in words if str(w.get("text", "")).strip()]
    if not toks:
        return []
    # Sort by center y then x
    for w in toks:
        w["_yc"] = int(w.get("y", 0)) + (int(w.get("h", 0)) / 2.0)
    toks.sort(key=lambda w: (w.get("_yc", 0.0), int(w.get("x", 0))))
    # Estimate line threshold from median height
    try:
        import statistics as _st

        med_h = float(_st.median([int(w.get("h", 0)) or 0 for w in toks if int(w.get("h", 0)) > 0]) or 12.0)
    except Exception:
        med_h = 12.0
    line_thresh = max(6.0, 0.7 * med_h)

    lines: List[Dict] = []
    cur_y: Optional[float] = None
    cur_words: List[Dict] = []
    for w in toks:
        yc = float(w.get("_yc", 0.0))
        if cur_y is None:
            cur_y = yc
        # New line if vertical gap is big
        if cur_words and abs(yc - cur_y) > line_thresh:
            # flush
            xs = [int(t.get("x", 0)) for t in cur_words]
            ys = [int(t.get("y", 0)) for t in cur_words]
            xe = [int(t.get("x", 0)) + int(t.get("w", 0)) for t in cur_words]
            ye = [int(t.get("y", 0)) + int(t.get("h", 0)) for t in cur_words]
            line = {
                "text": re.sub(r"\s+", " ", " ".join([str(t.get("text", "")).strip() for t in cur_words])).strip(),
                "x0": min(xs) if xs else 0,
                "y0": min(ys) if ys else 0,
                "x1": max(xe) if xe else 0,
                "y1": max(ye) if ye else 0,
                "words": cur_words[:],
            }
            lines.append(line)
            cur_words = [w]
            cur_y = yc
        else:
            cur_words.append(w)
            cur_y = (cur_y * 0.8) + (yc * 0.2)
    if cur_words:
        xs = [int(t.get("x", 0)) for t in cur_words]
        ys = [int(t.get("y", 0)) for t in cur_words]
        xe = [int(t.get("x", 0)) + int(t.get("w", 0)) for t in cur_words]
        ye = [int(t.get("y", 0)) + int(t.get("h", 0)) for t in cur_words]
        lines.append(
            {
                "text": re.sub(r"\s+", " ", " ".join([str(t.get("text", "")).strip() for t in cur_words])).strip(),
                "x0": min(xs) if xs else 0,
                "y0": min(ys) if ys else 0,
                "x1": max(xe) if xe else 0,
                "y1": max(ye) if ye else 0,
                "words": cur_words[:],
            }
        )
    return lines


# ----------------------------- VLM calls ---------------------------------


def call_openai_vision_text(base_url: str, api_key: str, model: str, prompt: str, image: Image.Image, temperature: float = 0.0, timeout_s: int = 30) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    b64 = image_to_base64_png(image)
    payload = {
        "model": model,
        "temperature": float(temperature),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return str(content or "").strip()
    except Exception as e:
        print(f"[WARN] OpenAI vision call failed: {e}", file=sys.stderr)
        return ""


def call_ollama_vision_text(base_url: str, model: str, prompt: str, image: Image.Image, temperature: float = 0.0, timeout_s: int = 30) -> str:
    # Ollama vision chat: POST /api/chat, messages[].images[] hold base64 images
    url = base_url.rstrip("/") + "/api/chat"
    headers = {"Content-Type": "application/json"}
    b64 = image_to_base64_png(image)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
        "options": {"temperature": float(temperature)},
        "stream": False,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        return str(data.get("message", {}).get("content", "")).strip()
    except Exception as e:
        print(f"[WARN] Ollama vision call failed: {e}", file=sys.stderr)
        return ""


def vlm_read_text(img: Image.Image, provider: str, llm_params: Dict, prompt: str, max_side: int = 640) -> str:
    # Resize for speed and to satisfy Qwen2.5-VL visual patch multiple (28)
    if isinstance(max_side, int) and max_side > 0:
        w, h = img.size
        s = max(w, h)
        if s > max_side:
            ratio = max_side / float(s)
            img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.BICUBIC)
            w, h = img.size

    # Enforce a floor on the short side to avoid Ollama qwen2.5vl panic (width/height must be >= 28 and multiples of 28)
    min_side = int(llm_params.get("min_side", 56))
    w, h = img.size
    short = min(w, h)
    if short < min_side:
        scale = float(min_side) / float(short)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        w, h = img.size

    # Round up to multiples of 28 (model expects dims multiple of 28)
    new_w = _round_up_to_multiple(w, 28)
    new_h = _round_up_to_multiple(h, 28)
    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.BICUBIC)

    if provider == "ollama":
        return call_ollama_vision_text(
            base_url=llm_params.get("base_url", "http://localhost:11434"),
            model=llm_params.get("model", "qwen2.5vl:7b"),
            prompt=prompt,
            image=img,
            temperature=float(llm_params.get("temperature", 0.0)),
            timeout_s=int(llm_params.get("timeout_s", 30)),
        )
    else:
        # default: OpenAI-compatible vision chat
        return call_openai_vision_text(
            base_url=llm_params.get("base_url", getenv_str("OPENAI_BASE_URL", "https://api.openai.com/v1")),
            api_key=llm_params.get("api_key", getenv_str(llm_params.get("api_key_env", "OPENAI_API_KEY"), "")),
            model=llm_params.get("model", getenv_str("OPENAI_MODEL", "gpt-4o-mini")),
            prompt=prompt,
            image=img,
            temperature=float(llm_params.get("temperature", 0.0)),
            timeout_s=int(llm_params.get("timeout_s", 30)),
        )


def _cleanup_vlm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s2 = s.strip()
    # Remove common code fences
    s2 = re.sub(r"^```[a-zA-Z]*\n|```$", "", s2).strip()
    # Prefer a single line of text
    s2 = re.sub(r"\s+", " ", s2)
    # Drop generic captions often returned by VLMs when no text is detected
    if re.search(r"(?i)\b(picture|image|photo)\b", s2):
        return ""
    return s2


# ----------------------------- Region finding ----------------------------


DRAWING_LABELS = [
    # English variations
    "DRAWING NO", "DRAWING NO.", "DWG NO", "DWG NO.", "DRG NO", "DRG NO.",
    "SHEET NO", "SHEET NO.", "SHEET NUMBER", "SHEET", "DWG", "DRG",
    # Thai
    "เลขที่แบบ", "เลขที่",
]
TITLE_LABELS = [
    # English variations
    "TITLE", "DRAWING TITLE", "DRAWING NAME", "SHEET TITLE", "SHEET NAME", "SUBJECT", "DESCRIPTION",
    # Thai
    "ชื่อแบบ", "หัวข้อ", "ชื่อเรื่อง", "ชื่อแบบ/หัวข้อ",
]
SCALE_LABELS = [
    "SCALE",  # EN
    "มาตราส่วน",  # TH
]
PAGE_LABELS = [
    # EN variants commonly near X/Y or X of Y
    "SHEET", "SHEET NO", "SHEET NO.", "SHEET:", "PAGE", "PAGE NO", "PAGE NO.",
    # TH
    "หน้าที่", "แผ่นที่",
]


def find_label_line(lines: List[Dict], labels: List[str]) -> Optional[Dict]:
    if not lines:
        return None
    labs = [l.upper() for l in labels]
    for ln in lines:
        t = str(ln.get("text", ""))
        up = t.upper()
        for lab in labs:
            if lab in up:
                ln2 = dict(ln)
                ln2["_matched_label"] = lab
                return ln2
    return None


def region_right_of_label(line: Dict, crop_w: int, crop_h: int, extra_lines: int = 1) -> Tuple[int, int, int, int]:
    # Determine the right edge of the label tokens on this line
    x_right = int(line.get("x0", 0))
    for w in line.get("words", []):
        up = str(w.get("text", "")).upper()
        if any(lbl in up for lbl in ["DRAWING", "DWG", "DRG", "NO", "SHEET", "NUMBER", "TITLE", "NAME", "DESCRIPTION", "เลขที่แบบ", "ชื่อแบบ", "หัวข้อ"]):
            xr = int(w.get("x", 0)) + int(w.get("w", 0))
            x_right = max(x_right, xr)
    ly0, ly1 = int(line.get("y0", 0)), int(line.get("y1", 0))
    lh = max(8, ly1 - ly0)
    x0 = min(crop_w - 1, max(0, x_right + 4))
    x1 = min(crop_w, max(x0 + 10, int(line.get("x1", crop_w))))
    # Extend downwards to include extra lines (wrap titles)
    y0 = max(0, ly0 - int(0.2 * lh))
    y1 = min(crop_h, ly1 + int((0.9 + extra_lines) * lh))
    return x0, y0, x1, y1


# ----------------------------- Classification ---------------------------


DRAWING_NO_PAT = re.compile(
    r"(?i)(?<![A-Z0-9])(?:AR|SN|A|S|E|M|P|MEP|T|C|ID)\s*-?\s*\d{1,4}(?:\s*-\s*\d{1,4})?(?![A-Z0-9])"
)

SHEET_KEYWORDS = {
    "PLAN": ["PLAN", "แปลน"],
    "SECTION": ["SECTION", "รูปตัด"],
    "ELEVATION": ["ELEVATION", "รูปด้าน"],
    "DETAIL": ["DETAIL", "รายละเอียด"],
    "LAYOUT": ["LAYOUT", "LAY-OUT"],
    "SCHEDULE": ["SCHEDULE"],
    "NOTES": ["NOTES"],
}


def classify_fields(text_blob: str, title_hint: str = "", prefer_lines: Optional[List[str]] = None, pdf_total_pages_hint: Optional[int] = None) -> Dict[str, str]:
    s = (text_blob or "").strip()
    s_norm = s.replace("—", "-").replace("–", "-")
    # drawing_no
    drawing_no = ""
    m = DRAWING_NO_PAT.search(s_norm)
    if m:
        drawing_no = re.sub(r"\s*-\s*", "-", m.group(0).strip())
    # drawing_title
    drawing_title = title_hint.strip() if title_hint else ""
    if not drawing_title:
        # crude heuristic: longest non-anchor line
        lines = [ln.strip() for ln in re.split(r"[\r\n]+", s_norm) if ln.strip()]
        non_anchor = [
            ln for ln in lines
            if not any(k in ln.upper() for k in ["DRAWING", "DWG", "DRG", "SCALE", "DATE", "PROJECT", "REV", "SHEET"]) and
               not any(k in ln for k in ["เลขที่แบบ", "ชื่อแบบ", "หัวข้อ"])
        ]
        if non_anchor:
            drawing_title = max(non_anchor, key=len)[:200]
    # sheet_name
    sheet_name = ""
    blob_up = s_norm.upper()
    for k, kws in SHEET_KEYWORDS.items():
        if any(kw.upper() in blob_up for kw in kws):
            sheet_name = k
            break
    # scale (prefer lines mentioning SCALE)
    scale = ""
    scale_re = re.compile(r"\b(1\s*[:：]\s*\d{1,4})\b")
    if prefer_lines:
        for ln in prefer_lines:
            if any(lbl in ln.upper() for lbl in ["SCALE"]) or ("มาตราส่วน" in ln):
                m = scale_re.search(ln)
                if m:
                    scale = m.group(1).replace(" ", "")
                    break
    if not scale:
        m = scale_re.search(s_norm)
        if m:
            scale = m.group(1).replace(" ", "")
    # total pages: prefer X/Y or X of Y near PAGE/SHEET lines; avoid scale (uses colon)
    total_page = ""
    if prefer_lines:
        for ln in prefer_lines:
            up = ln.upper()
            if any(k in up for k in ["SHEET", "PAGE", "หน้าที่", "แผ่นที่"]):
                m = re.search(r"\b(\d{1,3})\s*(?:OF|of)\s*(\d{1,4})\b", ln)
                if m:
                    total_page = m.group(2)
                    break
                m = re.search(r"\b(\d{1,3})\s*/\s*(\d{1,4})\b", ln)
                if m:
                    total_page = m.group(2)
                    break
    if not total_page:
        # fallback: global 'of' pattern in blob
        m = re.search(r"\b(\d{1,3})\s*(?:OF|of)\s*(\d{1,4})\b", s_norm)
        if m:
            total_page = m.group(2)
    if not total_page:
        # last resort: global slash pattern (may collide with dates; used only if nothing else worked)
        m = re.search(r"\b(\d{1,3})\s*/\s*(\d{1,4})\b", s_norm)
        if m:
            total_page = m.group(2)
    # If still empty, and a plausible hint is provided, fallback to PDF count
    if not total_page and isinstance(pdf_total_pages_hint, int) and pdf_total_pages_hint > 0:
        total_page = str(pdf_total_pages_hint)
    return {
        "drawing_no": drawing_no,
        "drawing_title": drawing_title,
        "sheet_name": sheet_name,
        "scale": scale,
        "total_page": total_page,
    }


# ----------- OCR-only fallback classification (rule-based) -----------

SCALE_RE = re.compile(r"\b(1\s*[:：]\s*\d{1,4})\b")
DATE_RE = re.compile(r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b")


def _normalize_hyphen(s: str) -> str:
    return s.replace("—", "-").replace("–", "-").replace("_", "-")


def _rule_classify_from_lines(lines: List[str]) -> Dict[str, str]:
    text = "\n".join(lines)
    text_norm = _normalize_hyphen(text)
    # drawing_no (prefer near anchors)
    drawing_no = ""
    for ln in lines:
        ln_n = _normalize_hyphen(ln)
        if any(a in ln_n.upper() for a in ["DRAWING", "DWG", "DRG", "SHEET", "NO"]) or any(a in ln_n for a in ["เลขที่แบบ", "เลขที่"]):
            m = DRAWING_NO_PAT.search(ln_n)
            if m:
                drawing_no = re.sub(r"\s*-\s*", "-", m.group(0).strip())
                break
    if not drawing_no:
        m = DRAWING_NO_PAT.search(text_norm)
        if m:
            drawing_no = re.sub(r"\s*-\s*", "-", m.group(0).strip())

    # title: choose longest non-anchor line
    non_anchor = [
        ln for ln in lines
        if not any(k in ln.upper() for k in ["DRAWING", "DWG", "DRG", "SCALE", "DATE", "PROJECT", "REV", "SHEET"]) and
           not any(k in ln for k in ["เลขที่แบบ", "ชื่อแบบ", "หัวข้อ"])
    ]
    drawing_title = max(non_anchor, key=len).strip() if non_anchor else ""

    # sheet_name via keywords
    sheet_name = ""
    s_up = text_norm.upper()
    for k, kws in SHEET_KEYWORDS.items():
        if any(kw.upper() in s_up for kw in kws):
            sheet_name = k
            break
    # scale from lines with SCALE label
    scale = ""
    for ln in lines:
        if ("SCALE" in ln.upper()) or ("มาตราส่วน" in ln):
            m = SCALE_RE.search(ln)
            if m:
                scale = m.group(1).replace(" ", "")
                break
    # total_page: prefer lines containing PAGE/SHEET indicators
    total_page = ""
    for ln in lines:
        up = ln.upper()
        if any(k in up for k in ["SHEET", "PAGE", "หน้าที่", "แผ่นที่"]):
            m = re.search(r"\b(\d{1,3})\s*(?:OF|of)\s*(\d{1,4})\b", ln)
            if m:
                total_page = m.group(2)
                break
            m = re.search(r"\b(\d{1,3})\s*/\s*(\d{1,4})\b", ln)
            if m:
                total_page = m.group(2)
                break
    return {
        "drawing_no": drawing_no,
        "drawing_title": drawing_title,
        "sheet_name": sheet_name,
        "scale": scale,
        "total_page": total_page,
    }


# ----------------------------- Pipeline ----------------------------------

# Optional quick search across a few bottom-right variants
def candidate_bboxes() -> List[BBox]:
    return [
        BBox(0.72, 0.72, 0.28, 0.28),  # default
        BBox(0.68, 0.72, 0.30, 0.26),
        BBox(0.70, 0.66, 0.28, 0.34),
        BBox(0.60, 0.80, 0.38, 0.18),  # bottom strip right half
        BBox(0.72, 0.40, 0.26, 0.56),  # right-side vertical
    ]


def try_multi_crop(doc: fitz.Document, index: int, dpi: int, pad: float, max_tries: int = 3) -> Tuple[Image.Image, List[Dict], List[Dict], BBox]:
    tried = 0
    best: Optional[Tuple[Image.Image, List[Dict], List[Dict], BBox]] = None
    for bb in candidate_bboxes():
        crop = render_crop(doc, index, dpi, bb, pad)
        words = ocr_words(crop)
        lines = group_words_to_lines(words)
        ln_no = find_label_line(lines, DRAWING_LABELS)
        ln_title = find_label_line(lines, TITLE_LABELS)
        if ln_no or ln_title:
            return crop, words, lines, bb
        if best is None:
            best = (crop, words, lines, bb)
        tried += 1
        if tried >= max_tries:
            break
    assert best is not None
    return best

# --- Anchor-based scoring to auto-pick the best bbox per PDF ---
ANCHOR_MORE = [
    "SCALE", "DATE", "PROJECT", "REV", "REVISION",
    "มาตราส่วน", "วันที่", "โครงการ", "แก้ไข",
]


def _anchor_score(lines: List[Dict]) -> int:
    if not lines:
        return 0
    labs = set([*(l.upper() for l in DRAWING_LABELS), *(l.upper() for l in TITLE_LABELS), *(l.upper() for l in ANCHOR_MORE)])
    score = 0
    seen: set[str] = set()
    for ln in lines:
        up = str(ln.get("text", "")).upper()
        for lab in labs:
            if lab and lab in up:
                score += 1
                seen.add(lab)
    score += len(seen)
    return score


def _auto_detect_bbox(doc: fitz.Document, page_index: int, detect_dpi: int, pad: float, tries: int) -> Tuple[BBox, int]:
    best = candidate_bboxes()[:]
    best_score = -1
    best_bbox = best[0]
    tested = 0
    for bb in best:
        crop = render_crop(doc, page_index, detect_dpi, bb, pad)
        lines = group_words_to_lines(ocr_words(crop))
        sc = _anchor_score(lines)
        if sc > best_score:
            best_score = sc
            best_bbox = bb
        tested += 1
        if tested >= max(3, tries):
            break
    return best_bbox, best_score


def process_pdf(pdf_path: Path, dpi: int, bbox: BBox, crop_pad: float, llm_params: Dict, provider: str, max_side: int, debug_dir: Optional[Path], auto_detect: bool = True, detect_dpi: int = 144, detect_tries: int = 6) -> List[Dict]:
    rows: List[Dict] = []
    with fitz.open(pdf_path) as doc:
        n = doc.page_count
        print(f"[INFO] PDF: {pdf_path.name} | pages: {n}")
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
        pbar = _tqdm(total=n, desc=f"{pdf_path.name}", unit="page") if _tqdm else None
        # Auto-detect bbox from the first page to improve region choice
        chosen_bbox = bbox
        if auto_detect and n > 0:
            try:
                bb, sc = _auto_detect_bbox(doc, 0, detect_dpi, crop_pad, detect_tries)
                if sc > 0:
                    chosen_bbox = bb
                    if debug_dir is not None:
                        print(f"[INFO] Auto-detected bbox for {pdf_path.name}: {bb} (score={sc})")
            except Exception as e:
                print(f"[WARN] auto-detect failed: {e}", file=sys.stderr)
        for i in range(n):
            crop = render_crop(doc, i, dpi, chosen_bbox, crop_pad)
            W, H = crop.size
            words = ocr_words(crop)
            lines = group_words_to_lines(words)

            # locate label lines
            ln_no = find_label_line(lines, DRAWING_LABELS)
            ln_title = find_label_line(lines, TITLE_LABELS)

            text_from_regions: List[str] = []
            title_text: str = ""
            # drawing number region
            if ln_no:
                x0, y0, x1, y1 = region_right_of_label(ln_no, W, H, extra_lines=1)
                sub = crop.crop((x0, y0, x1, y1))
                if debug_dir is not None:
                    print(f"[DEBUG] region size: {sub.size}")
                t_no = vlm_read_text(
                    sub,
                    provider=provider,
                    llm_params=llm_params,
                    prompt="Read exactly the text in this image. Return only the text.",
                    max_side=max_side,
                )
                t_no = _cleanup_vlm_text(t_no)
                if t_no:
                    text_from_regions.append(t_no)
            # title region
            if ln_title:
                x0, y0, x1, y1 = region_right_of_label(ln_title, W, H, extra_lines=2)
                sub = crop.crop((x0, y0, x1, y1))
                if debug_dir is not None:
                    print(f"[DEBUG] region size: {sub.size}")
                t_title = vlm_read_text(
                    sub,
                    provider=provider,
                    llm_params=llm_params,
                    prompt="Read the drawing title text in this image. Return only the text (no labels).",
                    max_side=max_side,
                )
                t_title = _cleanup_vlm_text(t_title)
                if t_title:
                    title_text = t_title
                    text_from_regions.append(t_title)
            # scale region (optional)
            ln_scale = find_label_line(lines, SCALE_LABELS)
            scale_text = ""
            if ln_scale:
                x0, y0, x1, y1 = region_right_of_label(ln_scale, W, H, extra_lines=0)
                sub = crop.crop((x0, y0, x1, y1))
                t_scale = vlm_read_text(
                    sub,
                    provider=provider,
                    llm_params=llm_params,
                    prompt="Read the scale value (e.g., 1:100). Return only the value.",
                    max_side=max_side,
                )
                t_scale = _cleanup_vlm_text(t_scale)
                if t_scale:
                    scale_text = t_scale
                    text_from_regions.append(t_scale)
            # page/total region (optional) — may contain X/Y or X of Y
            ln_page = find_label_line(lines, PAGE_LABELS)
            page_text = ""
            if ln_page:
                x0, y0, x1, y1 = region_right_of_label(ln_page, W, H, extra_lines=0)
                sub = crop.crop((x0, y0, x1, y1))
                t_page = vlm_read_text(
                    sub,
                    provider=provider,
                    llm_params=llm_params,
                    prompt="Read the sheet/page indicator (like 1/10 or 1 of 10). Return only that.",
                    max_side=max_side,
                )
                t_page = _cleanup_vlm_text(t_page)
                if t_page:
                    page_text = t_page
                    text_from_regions.append(t_page)

            # fallback: if nothing extracted, try alternate bottom-right variants quickly
            if not text_from_regions:
                try:
                    crop2, words2, lines2, _bbox_used = try_multi_crop(doc, i, dpi, crop_pad, max_tries=3)
                    crop, words, lines = crop2, words2, lines2
                    W, H = crop.size
                    ln_no = find_label_line(lines, DRAWING_LABELS)
                    ln_title = find_label_line(lines, TITLE_LABELS)
                    if ln_no:
                        x0, y0, x1, y1 = region_right_of_label(ln_no, W, H, extra_lines=1)
                        sub = crop.crop((x0, y0, x1, y1))
                        t_no = vlm_read_text(sub, provider, llm_params, "Read the text; return only the text.", max_side)
                        t_no = _cleanup_vlm_text(t_no)
                        if t_no:
                            text_from_regions.append(t_no)
                    if ln_title:
                        x0, y0, x1, y1 = region_right_of_label(ln_title, W, H, extra_lines=2)
                        sub = crop.crop((x0, y0, x1, y1))
                        t_title = vlm_read_text(sub, provider, llm_params, "Read the drawing title text only.", max_side)
                        t_title = _cleanup_vlm_text(t_title)
                        if t_title:
                            title_text = t_title
                            text_from_regions.append(t_title)
                except Exception:
                    pass

            # final fallback: read the whole crop once
            if not text_from_regions:
                t_all = vlm_read_text(
                    crop,
                    provider=provider,
                    llm_params=llm_params,
                    prompt=(
                        "Read the text in this title block region. Return only the text as a single line."
                    ),
                    max_side=max_side,
                )
                t_all = _cleanup_vlm_text(t_all)
                if t_all:
                    text_from_regions.append(t_all)

            # Join all VLM reads and classify
            vlm_text_blob = " \n ".join([t for t in text_from_regions if t])
            prefer_lines = [ln.get("text", "") for ln in lines if str(ln.get("text", "")).strip()]
            fields_vlm = classify_fields(
                vlm_text_blob,
                title_hint=title_text,
                prefer_lines=prefer_lines,
                pdf_total_pages_hint=n,
            )

            # OCR-only rule-based fallback merge
            ocr_lines = [ln.get("text", "") for ln in lines if str(ln.get("text", "")).strip()]
            fields_rule = _rule_classify_from_lines(ocr_lines)
            fields = {
                "drawing_no": fields_vlm.get("drawing_no") or fields_rule.get("drawing_no") or "",
                "drawing_title": fields_vlm.get("drawing_title") or fields_rule.get("drawing_title") or "",
                "sheet_name": fields_vlm.get("sheet_name") or fields_rule.get("sheet_name") or "",
                "scale": fields_vlm.get("scale") or "",
                "total_page": fields_vlm.get("total_page") or str(n),
            }

            rec = {
                "pdf_path": str(pdf_path),
                "page_index": i,
                "crop_size": {"w": W, "h": H},
                "regions_text": text_from_regions,
                "llm_class": fields,
                "llm_used": True,
                "notes": "vlm_ocr_fast_bbox",
            }
            rows.append(rec)

            if debug_dir is not None:
                crop.save(debug_dir / f"{pdf_path.stem}_p{i:03d}_crop.png")
                # Optional: save annotated regions (if found)
                try:
                    from PIL import ImageDraw

                    dbg = crop.copy()
                    draw = ImageDraw.Draw(dbg)
                    if ln_no:
                        x0, y0, x1, y1 = region_right_of_label(ln_no, W, H, extra_lines=1)
                        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
                    if ln_title:
                        x0, y0, x1, y1 = region_right_of_label(ln_title, W, H, extra_lines=2)
                        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
                    ln_scale = find_label_line(lines, SCALE_LABELS)
                    if ln_scale:
                        x0, y0, x1, y1 = region_right_of_label(ln_scale, W, H, extra_lines=0)
                        draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 255), width=2)
                    ln_page = find_label_line(lines, PAGE_LABELS)
                    if ln_page:
                        x0, y0, x1, y1 = region_right_of_label(ln_page, W, H, extra_lines=0)
                        draw.rectangle([x0, y0, x1, y1], outline=(255, 165, 0), width=2)
                    dbg.save(debug_dir / f"{pdf_path.stem}_p{i:03d}_regions.png")
                except Exception:
                    pass

            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
    return rows


# ----------------------------- I/O ---------------------------------------


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict]):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["pdf_path", "drawing_no", "drawing_title", "scale", "total_page"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            llm = r.get("llm_class") or {}
            w.writerow({
                "pdf_path": r.get("pdf_path", ""),
                "drawing_no": (llm.get("drawing_no") or ""),
                "drawing_title": (llm.get("drawing_title") or ""),
                "scale": (llm.get("scale") or ""),
                "total_page": (llm.get("total_page") or ""),
            })


# ----------------------------- CLI ---------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Fast bottom-right VLM+OCR pipeline for drawing number and name")
    ap.add_argument("--folder", type=str, required=True, help="Folder containing PDFs (no recursion)")
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--bbox-x", type=float, default=0.72)
    ap.add_argument("--bbox-y", type=float, default=0.72)
    ap.add_argument("--bbox-w", type=float, default=0.28)
    ap.add_argument("--bbox-h", type=float, default=0.28)
    ap.add_argument("--crop-pad", type=float, default=0.03, help="Normalized padding added to bbox")
    ap.add_argument("--max-side", type=int, default=640, help="Resize crops for VLM to this long side (0=no resize)")
    ap.add_argument("--min-side", type=int, default=56, help="Ensure crops are at least this on the short side (rounded up to multiple of 28)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-dir", type=str, default="./out/debug_fast_bbox")
    ap.add_argument("--auto-detect", action="store_true", help="Auto-detect best bottom-right bbox on first page (recommended)")
    ap.add_argument("--detect-dpi", type=int, default=144, help="DPI used for bbox auto-detection clips")
    ap.add_argument("--detect-tries", type=int, default=6, help="How many candidate bboxes to try for detection")
    ap.set_defaults(auto_detect=True)

    ap.add_argument("--llm-provider", type=str, default="ollama", choices=["ollama", "openai"], help="VLM API provider")
    ap.add_argument("--llm-base-url", type=str, default="http://localhost:11434", help="Base URL for provider")
    ap.add_argument("--llm-model", type=str, default="qwen2.5vl:7b")
    ap.add_argument("--llm-api-key-env", type=str, default="OPENAI_API_KEY", help="Env var for API key (OpenAI-compatible only)")
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--llm-timeout-s", type=int, default=30)

    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder}", file=sys.stderr)
        sys.exit(2)

    pdfs = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdfs:
        print(f"[WARN] No PDFs in {folder}")

    bbox = BBox(x=args.bbox_x, y=args.bbox_y, w=args.bbox_w, h=args.bbox_h)
    debug_dir = Path(args.debug_dir) if args.debug else None

    llm_params = {
        "provider": args.llm_provider,
        "base_url": args.llm_base_url,
        "model": args.llm_model,
        "api_key_env": args.llm_api_key_env,
        "api_key": getenv_str(args.llm_api_key_env, ""),
        "temperature": args.llm_temperature,
        "timeout_s": args.llm_timeout_s,
        "min_side": args.min_side,
    }

    all_rows: List[Dict] = []
    for pdf in pdfs:
        rows = process_pdf(
            pdf_path=pdf,
            dpi=args.dpi,
            bbox=bbox,
            crop_pad=args.crop_pad,
            llm_params=llm_params,
            provider=args.llm_provider,
            max_side=int(args.max_side),
            debug_dir=debug_dir,
            auto_detect=bool(args.auto_detect),
            detect_dpi=int(args.detect_dpi),
            detect_tries=int(args.detect_tries),
        )
        all_rows.extend(rows)

    write_jsonl(Path(args.out_jsonl), all_rows)
    write_csv(Path(args.out_csv), all_rows)
    print(f"Done. JSONL: {args.out_jsonl} | CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
