#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm_ocr_fixed_bottomright_boxes.py

Fixed bottom-right pipeline + OCR word grouping and VLM classification.

What it does (fast + accurate):
- Renders a fixed bottom-right crop per page (same as the fast baseline).
- Extracts text boxes from the PDF itself using PyMuPDF (vector text) inside
  that crop for very fast and precise boxes. Falls back to Tesseract OCR only
  if requested (for scanned PDFs) to avoid slowing down typical vector cases.
- Sends only the crop image to the VLM with the same strict prompt as the
  baseline – preserving quality and speed (about the same as the baseline).
- Optionally, you can include the box metadata in the prompt (crop-only) to get
  VLM-labeled boxes; disabled by default to avoid any quality regression.
- Exports debug overlays that combine the crop image, bounding boxes, and the
  VLM’s extracted fields; optionally also includes per-box field labels.

Example (Ollama):
  # ollama pull qwen2.5vl:7b
  python vlm_ocr_fixed_bottomright_boxes.py \
    --folder ./samples \
    --out-jsonl ./out/fixed_br_boxes.jsonl \
    --out-csv   ./out/fixed_br_boxes.csv \
    --dpi 250 \
    --llm-provider ollama \
    --llm-base-url http://localhost:11434 \
    --llm-model qwen2.5vl:7b \
    --debug

OpenAI-compatible vision (local gateway / LM Studio):
  export OPENAI_API_KEY=...
  python vlm_ocr_fixed_bottomright_boxes.py \
    --folder ./samples \
    --out-jsonl ./out/fixed_br_boxes.jsonl \
    --out-csv   ./out/fixed_br_boxes.csv \
    --dpi 300 \
    --llm-provider openai --llm-base-url http://localhost:1234/v1 --llm-model gpt-4o-mini
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
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

# Optional OCR fallback (for scanned PDFs); disabled unless --ocr-fallback
try:  # pragma: no cover
    import pytesseract
except Exception:
    pytesseract = None


# ----------------------------- Utils -------------------------------------


def getenv_str(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return default if v is None else str(v)


@dataclass
class BBox:
    x: float  # normalized [0..1]
    y: float
    w: float
    h: float


def render_page(doc: fitz.Document, index: int, dpi: int) -> Image.Image:
    page = doc.load_page(index)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def render_crop(doc: fitz.Document, index: int, dpi: int, bbox: BBox, pad: float) -> Tuple[Image.Image, Tuple[float, float], fitz.Rect]:
    """Render only the bbox region (with pad) at the given DPI using PyMuPDF clip.

    Returns (PIL.Image, (clip_x0_px, clip_y0_px)) so that crop-relative coords
    can be mapped back to full-page pixels.
    """
    page = doc.load_page(index)
    rect = page.rect
    x0 = max(0.0, (bbox.x - pad) * rect.width)
    y0 = max(0.0, (bbox.y - pad) * rect.height)
    x1 = min(rect.width, (bbox.x + bbox.w + pad) * rect.width)
    y1 = min(rect.height, (bbox.y + bbox.h + pad) * rect.height)
    clip = fitz.Rect(x0, y0, x1, y1)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    # Return clip origin in page pixels at this DPI and the clip rect (in points)
    scale = dpi / 72.0
    return img, (x0 * scale, y0 * scale), clip


def image_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if not isinstance(max_side, int) or max_side <= 0:
        return img
    w, h = img.size
    s = max(w, h)
    if s <= max_side:
        return img
    r = max_side / float(s)
    return img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BICUBIC)


def _parse_relaxed_json(s: str) -> Dict:
    if not isinstance(s, str) or not s.strip():
        return {}
    txt = s.strip()
    # Try direct JSON
    try:
        return json.loads(txt)
    except Exception:
        pass
    # Try fenced code block
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", txt, flags=re.I)
    if not m:
        m = re.search(r"```\s*(\{[\s\S]*?\})\s*```", txt)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try first {...}
    m = re.search(r"(\{[\s\S]*\})", txt)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return {}


# ----------------------------- PDF Words (fast, precise) -----------------


def _pdf_words_in_clip(page: fitz.Page, clip: fitz.Rect) -> List[Tuple[float, float, float, float, str, int, int, int]]:
    """Return page words within clip: tuples (x0, y0, x1, y1, text, block, line, word). Coordinates in points.
    Uses PyMuPDF's very fast text extraction (vector PDFs)."""
    try:
        return page.get_text("words", clip=clip) or []
    except Exception:
        return []


def _group_pdf_words_to_lines(words: List[Tuple[float, float, float, float, str, int, int, int]]) -> List[Dict]:
    if not words:
        return []
    # Sort by (block, line, word)
    words_sorted = sorted(words, key=lambda t: (t[5], t[6], t[7]))
    out: List[Dict] = []
    cur_key = None
    cur_items: List[Tuple[float, float, float, float, str]] = []
    for (x0, y0, x1, y1, text, block, line, word) in words_sorted:
        key = (block, line)
        if cur_key is None:
            cur_key = key
        if key != cur_key:
            # flush line
            xs0 = [it[0] for it in cur_items]
            ys0 = [it[1] for it in cur_items]
            xs1 = [it[2] for it in cur_items]
            ys1 = [it[3] for it in cur_items]
            txt = " ".join([str(it[4]).strip() for it in cur_items if str(it[4]).strip()])
            if txt:
                out.append({
                    "text": txt,
                    "x0_pt": min(xs0), "y0_pt": min(ys0),
                    "x1_pt": max(xs1), "y1_pt": max(ys1),
                })
            cur_items = [(x0, y0, x1, y1, text)]
            cur_key = key
        else:
            cur_items.append((x0, y0, x1, y1, text))
    # flush last
    if cur_items:
        xs0 = [it[0] for it in cur_items]
        ys0 = [it[1] for it in cur_items]
        xs1 = [it[2] for it in cur_items]
        ys1 = [it[3] for it in cur_items]
        txt = " ".join([str(it[4]).strip() for it in cur_items if str(it[4]).strip()])
        if txt:
            out.append({
                "text": txt,
                "x0_pt": min(xs0), "y0_pt": min(ys0),
                "x1_pt": max(xs1), "y1_pt": max(ys1),
            })
    return out


# ----------------------------- OCR Fallback (optional) -------------------


def _prep_for_tesseract(img: Image.Image) -> Image.Image:
    # Light preprocessing that tends to help printed title blocks
    g = ImageOps.grayscale(img)
    try:
        g = g.filter(ImageFilter.MedianFilter(size=3))
    except Exception:
        pass
    return g


def ocr_words(img: Image.Image, lang: str = "tha+eng", psm: int = 6) -> List[Dict]:
    if pytesseract is None:
        return []
    g = _prep_for_tesseract(img)
    try:
        data = pytesseract.image_to_data(
            g,
            lang=lang,
            config=f"--oem 3 --psm {int(psm)}",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []

    out: List[Dict] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        try:
            conf = float(data.get("conf", ["-1"])[i])
        except Exception:
            conf = -1.0
        if not txt or conf < 0:
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
    """Group words into line-level boxes (Thai-friendly: vertical overlap grouping).

    This is robust for Thai where intra-word spaces are sparse: we group by
    similar center-y and then merge horizontally.
    """
    if not words:
        return []
    toks = [w for w in words if str(w.get("text", "")).strip()]
    if not toks:
        return []
    for w in toks:
        w["_yc"] = float(int(w.get("y", 0)) + (int(w.get("h", 0)) / 2.0))
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
        if cur_words and abs(yc - cur_y) > line_thresh:
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
            # EMA of y to keep gradual drift
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


# ----------------------------- Prompts -----------------------------------


SYSTEM = (
    "You are a strict OCR+information-extraction assistant for construction drawing title blocks. "
    "You will receive a cropped image from the bottom-right region of a page. "
    "Return ONLY valid JSON with keys: drawing_no, drawing_title, scale, total_page. No extra keys, no explanations. "
    "Field rules: "
    "- drawing_no: Prefer the official number near labels like 'DRAWING NO'/'DWG NO' (Thai: 'เลขที่แบบ'). Normalize spaces/dashes (e.g., A-101, SN-08, E-05). "
    "- drawing_title: The sheet/drawing name to the right of or immediately under the 'TITLE' label (Thai: 'ชื่อแบบ', 'หัวข้อ'). If multiple lines, choose the most descriptive title text. "
    "  Avoid putting page counts (e.g., '1/10', 'TOTAL PAGE'), sheet numbers, scale, revision, date, project, or material/spec strings (e.g., 'RB9mm @ 0.15m', 'Ø12', 'dia 12', '@0.20m') into drawing_title. "
    "  If drawing_no suggests a discipline (A/S/E/SN/MEP), prefer a title consistent with that (e.g., A: PLAN/ELEVATION/SECTION; E: LINE DIAGRAM/RISER/SYMBOLS; SN: STRUCTURAL DETAIL/SECTION). "
    "- scale: Prefer colon format like '1:100' or '1:75'; convert '1/100' to '1:100'. If the page explicitly says 'NOT TO SCALE' or 'NTS', leave scale as an empty string. "
    "- total_page: Format as '<current>/<total>' (e.g., '1/10' or '17/71')."
)


def build_user_prompt() -> str:
    return (
        "Extract fields from this title-block crop using labels near the bottom-right area. "
        "Anchors: 'DRAWING NO'/'DWG NO', 'TITLE', 'SCALE', 'SHEET'/'PAGE' (Thai: 'เลขที่แบบ', 'ชื่อแบบ', 'หัวข้อ', 'มาตราส่วน', 'หน้าที่'). "
        "Return ONLY JSON with keys: drawing_no, drawing_title, scale, total_page. "
        "Ensure drawing_title is a proper name (not page/scale/spec). Format total_page as '<current>/<total>'."
    )


def build_user_prompt_for_boxes(page_w: int, page_h: int, boxes: List[Dict]) -> str:
    """Create a concise instruction + box metadata list.

    Boxes should contain keys: id, x, y, w, h in [0..1] (normalized to whole page),
    and optional text.
    """
    lines = []
    lines.append(
        (
            "Task: From this page image, classify which candidate boxes correspond to these fields: "
            "drawing_no, drawing_title, scale, total_page."
        )
    )
    lines.append(
        "Guidelines: Prefer labels/anchors near the bottom-right title block. Thai labels include 'เลขที่แบบ' (DRAWING NO), 'ชื่อแบบ/หัวข้อ' (TITLE), 'มาตราส่วน' (SCALE), 'หน้าที่' (PAGE)."
    )
    lines.append(
        "If OCR metadata text conflicts with the actual image, trust the image."
    )
    lines.append(f"Image size (px): {page_w}x{page_h}. Candidate boxes (normalized coords):")
    for b in boxes:
        bid = b.get("id", "?")
        x = b.get("x", 0.0)
        y = b.get("y", 0.0)
        w = b.get("w", 0.0)
        h = b.get("h", 0.0)
        t = str(b.get("text", "")).strip()
        lines.append(f"- {bid}: x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}, text='{t}'")
    lines.append(
        (
            "Return JSON with keys: drawing_no, drawing_title, scale, total_page, box_class. "
            "box_class must be a list of {id, field, text}, where field is one of: drawing_no, drawing_title, scale, total_page, other."
        )
    )
    # Thai hint
    lines.append(
        "คำอธิบายภาษาไทย: กรุณาดูภาพทั้งหน้าและ metadata ของกล่อง A, B, C, ... แล้วระบุว่าแต่ละกล่องเป็นฟิลด์ใด (เลขที่แบบ, ชื่อแบบ/หัวข้อ, มาตราส่วน, หน้าที่). ส่งคืนเฉพาะ JSON เท่านั้น."
    )
    return "\n".join(lines)


# ----------------------------- VLM calls ---------------------------------


def call_openai_vision_json(base_url: str, api_key: str, model: str, system: str, user: str, image: Image.Image, temperature: float = 0.0, timeout_s: int = 30) -> Dict:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    b64 = image_to_base64_png(image)
    payload = {
        "model": model,
        "temperature": float(temperature),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return json.loads(content)
    except Exception as e:
        sys.stderr.write(f"[WARN] OpenAI vision JSON failed: {e}\n")
        return {}


def call_ollama_vision_json(base_url: str, model: str, system: str, user: str, image: Image.Image, temperature: float = 0.0, timeout_s: int = 30) -> Dict:
    url = base_url.rstrip("/") + "/api/chat"
    headers = {"Content-Type": "application/json"}
    b64 = image_to_base64_png(image)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user, "images": [b64]},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "format": "json",
        "options": {"temperature": float(temperature)},
        "stream": False,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "")
        obj = _parse_relaxed_json(content)
        return obj
    except Exception as e:
        sys.stderr.write(f"[WARN] Ollama vision JSON failed: {e}\n")
        return {}


# ----------------------------- Post-processing ---------------------------


def clean_title(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip()
    if not t:
        return ""
    t = re.sub(r"(?i)\b(TOTAL\s+PAGES?|SHEET\s*NO\.?|SHEET:?|PAGE\s*NO\.?|PAGE:?|SCALE|REV(?:ISION)?|DATE|PROJECT)\b.*", "", t).strip()
    parts = [p.strip() for p in re.split(r"[,;\n]", t) if p.strip()]
    if len(parts) > 1:
        spec_pat = re.compile(r"(?i)(\bRB\d|Ø\d|D\d|dia\b|@\s?\d|\bmm\b|\bkg\b|\bm\.?\b|\d+\.\d+|\d+\s*[xX]\s*\d+)")
        keep = [p for p in parts if not spec_pat.search(p)]
        if keep:
            t = max(keep, key=len)
        else:
            t = parts[0]
    t = re.sub(r"\s+", " ", t)
    if len(t) < 4:
        return ""
    if re.fullmatch(r"[\W_]+", t):
        return ""
    return t


def normalize_scale(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip()
    if not t:
        return ""
    if re.search(r"(?i)\b(NOT\s*TO\s*SCALE|NTS)\b", t):
        return ""
    t = re.sub(r"\s*/\s*", ":", t)
    t = re.sub(r"\s+", "", t)
    m = re.match(r"^(\d{1,3}):(\d{1,4})$", t)
    if m:
        return f"{int(m.group(1))}:{int(m.group(2))}"
    return ""


def _label_ids(n: int) -> List[str]:
    """Yield Excel-like labels: A..Z, AA..AZ, BA.. etc. until n."""
    labels: List[str] = []
    i = 0
    while len(labels) < n:
        q = i
        s = ""
        while True:
            q, r = divmod(q, 26)
            s = chr(ord('A') + r) + s
            if q == 0:
                break
            q -= 1
        labels.append(s)
        i += 1
    return labels


def _draw_overlay(page_img: Image.Image, boxes_abs: List[Dict], box_class: List[Dict], out_path: Optional[Path] = None, header_text: Optional[str] = None) -> Image.Image:
    """Draw rectangles + IDs and, if available, the predicted field per box."""
    img = page_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Build a map from id -> field
    field_by_id: Dict[str, str] = {}
    if isinstance(box_class, list):
        for bc in box_class:
            try:
                bid = str(bc.get("id", "")).strip()
                fld = str(bc.get("field", "")).strip()
                if bid:
                    field_by_id[bid] = fld
            except Exception:
                pass

    for b in boxes_abs:
        x0 = int(b.get("x0", 0))
        y0 = int(b.get("y0", 0))
        x1 = int(b.get("x1", 0))
        y1 = int(b.get("y1", 0))
        bid = str(b.get("id", ""))
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        label = bid
        if bid in field_by_id:
            label = f"{bid}:{field_by_id[bid]}"
        # Background for text label
        try:
            if font is not None:
                # Prefer textbbox for compatibility
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            else:
                tw, th = len(label) * 6, 10
        except Exception:
            # Fallback worst-case approximation
            tw, th = len(label) * 7, 12
        bx0, by0 = x0, max(0, y0 - (th + 6))
        draw.rectangle([bx0, by0, bx0 + int(tw) + 8, by0 + th + 6], fill=(255, 255, 0))
        draw.text((bx0 + 4, by0 + 3), label, fill=(0, 0, 0), font=font)

    if header_text:
        # Draw a semi-transparent header box at top-left
        try:
            header = header_text.strip()
            if header:
                pad = 6
                if font is not None:
                    # Split into lines to avoid very long lines
                    lines = header.split("\n")
                    widths = [draw.textbbox((0, 0), ln, font=font)[2] for ln in lines]
                    tw = max(widths) if widths else 0
                    th = sum([draw.textbbox((0, 0), ln, font=font)[3] for ln in lines]) + (len(lines) - 1) * 2
                else:
                    lines = header.split("\n")
                    tw = max(len(ln) for ln in lines) * 7
                    th = len(lines) * 12
                draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
                y = pad
                for ln in lines:
                    draw.text((pad, y), ln, fill=(255, 255, 255), font=font)
                    if font is not None:
                        y += draw.textbbox((0, 0), ln, font=font)[3] + 2
                    else:
                        y += 12
        except Exception:
            pass

    if out_path is not None:
        try:
            img.save(out_path)
        except Exception:
            pass
    return img


# ----------------------------- Core --------------------------------------


def process_pdf(
    pdf_path: Path,
    dpi: int,
    bbox: BBox,
    crop_pad: float,
    provider: str,
    llm_params: Dict,
    ocr_lang: str,
    ocr_psm: int,
    max_side: int,
    max_boxes: int,
    debug_dir: Optional[Path],
    include_boxes_in_prompt: bool,
    ocr_fallback: bool,
) -> List[Dict]:
    rows: List[Dict] = []
    with fitz.open(pdf_path) as doc:
        n = doc.page_count
        print(f"[INFO] PDF: {pdf_path.name} | pages: {n}", flush=True)
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
        pbar = _tqdm(total=n, desc=f"{pdf_path.name}", unit="page") if _tqdm else None
        for i in range(n):
            # Render crop at the given DPI and get clip origin + clip rect
            crop_img, (clip_x0_px, clip_y0_px), clip_rect = render_crop(doc, i, dpi, bbox, crop_pad)
            orig_w, orig_h = crop_img.size
            crop_vis = resize_max_side(crop_img, max_side)
            rw = crop_vis.size[0] / float(max(1, orig_w))
            rh = crop_vis.size[1] / float(max(1, orig_h))

            # Discover boxes via PDF words (fast) inside clip
            page = doc.load_page(i)
            words = _pdf_words_in_clip(page, clip_rect)
            lines_pdf = _group_pdf_words_to_lines(words)

            # Convert PDF point coords into crop pixel coords
            scale = dpi / 72.0
            boxes_crop: List[Dict] = []
            for ln in lines_pdf[:max_boxes]:
                x0c = int(ln["x0_pt"] * scale - (clip_rect.x0 * scale))
                y0c = int(ln["y0_pt"] * scale - (clip_rect.y0 * scale))
                x1c = int(ln["x1_pt"] * scale - (clip_rect.x0 * scale))
                y1c = int(ln["y1_pt"] * scale - (clip_rect.y0 * scale))
                # Clamp and scale to the potentially resized crop_vis
                x0cs = int(max(0, min(crop_vis.size[0] - 1, x0c * rw)))
                y0cs = int(max(0, min(crop_vis.size[1] - 1, y0c * rh)))
                x1cs = int(max(0, min(crop_vis.size[0], x1c * rw)))
                y1cs = int(max(0, min(crop_vis.size[1], y1c * rh)))
                boxes_crop.append({
                    "id": None,
                    "x0": x0cs, "y0": y0cs, "x1": x1cs, "y1": y1cs,
                    "text": str(ln.get("text", "")).strip(),
                })

            # If no PDF text extracted and OCR fallback is enabled, use Tesseract on crop
            if not boxes_crop and ocr_fallback:
                words_ocr = ocr_words(crop_img, lang=ocr_lang, psm=ocr_psm)
                lines_ocr = group_words_to_lines(words_ocr)
                for ln in lines_ocr[:max_boxes]:
                    x0cs = int(max(0, min(crop_vis.size[0] - 1, ln.get("x0", 0) * rw)))
                    y0cs = int(max(0, min(crop_vis.size[1] - 1, ln.get("y0", 0) * rh)))
                    x1cs = int(max(0, min(crop_vis.size[0], ln.get("x1", 0) * rw)))
                    y1cs = int(max(0, min(crop_vis.size[1], ln.get("y1", 0) * rh)))
                    boxes_crop.append({
                        "id": None,
                        "x0": x0cs, "y0": y0cs, "x1": x1cs, "y1": y1cs,
                        "text": str(ln.get("text", "")).strip(),
                    })

            # Assign IDs and make normalized list (relative to crop_vis)
            ids = _label_ids(len(boxes_crop))
            for k, b in enumerate(boxes_crop):
                b["id"] = ids[k]
            Wc, Hc = crop_vis.size
            boxes_norm: List[Dict] = []
            for b in boxes_crop:
                x = max(0.0, min(1.0, b["x0"] / float(Wc)))
                y = max(0.0, min(1.0, b["y0"] / float(Hc)))
                w = max(0.0, min(1.0, (b["x1"] - b["x0"]) / float(Wc)))
                h = max(0.0, min(1.0, (b["y1"] - b["y0"]) / float(Hc)))
                boxes_norm.append({"id": b["id"], "x": x, "y": y, "w": w, "h": h, "text": b.get("text", "")})

            # Debug: save crop and boxes
            if debug_dir is not None:
                try:
                    crop_vis.save(debug_dir / f"{pdf_path.stem}_p{i:03d}_crop.png")
                except Exception:
                    pass
                _draw_overlay(crop_vis, boxes_crop, box_class=[], out_path=debug_dir / f"{pdf_path.stem}_p{i:03d}_crop_boxes.png")
                with (debug_dir / f"{pdf_path.stem}_p{i:03d}_crop_boxes.json").open("w", encoding="utf-8") as f:
                    json.dump({"W": Wc, "H": Hc, "boxes": boxes_norm}, f, ensure_ascii=False, indent=2)

            # Build VLM input: default is the same prompt + crop image as baseline
            if include_boxes_in_prompt and boxes_norm:
                user = build_user_prompt_for_boxes(Wc, Hc, boxes_norm)
            else:
                user = build_user_prompt()

            if provider == "ollama":
                obj = call_ollama_vision_json(
                    base_url=llm_params.get("base_url", "http://localhost:11434"),
                    model=llm_params.get("model", "qwen2.5vl:7b"),
                    system=llm_params.get("system", SYSTEM),
                    user=user,
                    image=crop_vis,
                    temperature=float(llm_params.get("temperature", 0.0)),
                    timeout_s=int(llm_params.get("timeout_s", 30)),
                )
            else:
                obj = call_openai_vision_json(
                    base_url=llm_params.get("base_url", getenv_str("OPENAI_BASE_URL", "https://api.openai.com/v1")),
                    api_key=llm_params.get("api_key", getenv_str(llm_params.get("api_key_env", "OPENAI_API_KEY"), "")),
                    model=llm_params.get("model", getenv_str("OPENAI_MODEL", "gpt-4o-mini")),
                    system=llm_params.get("system", SYSTEM),
                    user=user,
                    image=crop_vis,
                    temperature=float(llm_params.get("temperature", 0.0)),
                    timeout_s=int(llm_params.get("timeout_s", 30)),
                )

            # Normalize outputs
            drawing_no = str(obj.get("drawing_no") or "").strip()
            raw_title = str(obj.get("drawing_title") or "").strip()
            scale = normalize_scale(str(obj.get("scale") or ""))
            raw_total = str(obj.get("total_page") or "").strip()

            # Normalize total_page to 'current/total'
            cp, tp = None, None
            m = re.search(r"\b(\d{1,3})\s*(?:OF|of)\s*(\d{1,4})\b", raw_total)
            if m:
                cp, tp = m.group(1), m.group(2)
            else:
                m = re.search(r"\b(\d{1,3})\s*/\s*(\d{1,4})\b", raw_total)
                if m:
                    cp, tp = m.group(1), m.group(2)
            if cp is None and tp is None:
                m = re.search(r"(?i)TOTAL\s+PAGES?\s*(\d{1,4})\s*/\s*(\d{1,4})", raw_total)
                if m:
                    cp, tp = m.group(1), m.group(2)
            if cp is None and tp is None:
                m = re.search(r"\b(\d{1,4})\b", raw_total)
                if m:
                    tp = m.group(1)
                    cp = str(i + 1)
            if cp is None or tp is None:
                cp, tp = str(i + 1), str(n)
            total_page = f"{cp}/{tp}"

            title = clean_title(raw_title)

            out = {
                "drawing_no": drawing_no,
                "drawing_title": title,
                "scale": scale,
                "total_page": total_page,
            }

            rec = {
                "pdf_path": str(pdf_path),
                "page_index": i,
                "llm_class": out,
                "llm_used": True,
                "notes": "fixed_bottom_right_vlm_boxes",
            }
            # Attach raw box classifications if present
            if isinstance(obj.get("box_class"), list):
                rec["box_class"] = obj.get("box_class")

            rows.append(rec)

            # Debug overlay with classifications on crop
            if debug_dir is not None:
                header = f"NO: {out['drawing_no']}\nTITLE: {out['drawing_title']}\nSCALE: {out['scale']}\nPAGE: {out['total_page']}"
                _draw_overlay(crop_vis, boxes_crop, box_class=rec.get("box_class", []), out_path=debug_dir / f"{pdf_path.stem}_p{i:03d}_crop_boxes_class.png", header_text=header)

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
    ap = argparse.ArgumentParser(description="Fixed bottom-right + fast boxes (PDF text) + VLM classification on crop")
    ap.add_argument("--folder", type=str, required=True)
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--dpi", type=int, default=250, help="DPI for rendering page + crop (kept the same for coord mapping)")
    ap.add_argument("--bbox-x", type=float, default=0.72)
    ap.add_argument("--bbox-y", type=float, default=0.72)
    ap.add_argument("--bbox-w", type=float, default=0.28)
    ap.add_argument("--bbox-h", type=float, default=0.28)
    ap.add_argument("--crop-pad", type=float, default=0.02)
    ap.add_argument("--max-side", type=int, default=896, help="Resize crop so max(W,H) <= this for VLM (0 = no resize)")
    ap.add_argument("--max-boxes", type=int, default=60, help="Max candidate boxes to include in prompt/overlay")
    ap.add_argument("--ocr-lang", type=str, default="tha+eng", help="Tesseract languages (fallback only)")
    ap.add_argument("--ocr-psm", type=int, default=6, help="Tesseract PSM mode (fallback only)")
    ap.add_argument("--include-boxes-in-prompt", action="store_true", help="Include crop box metadata in the prompt (may affect accuracy). Default off.")
    ap.add_argument("--ocr-fallback", action="store_true", help="Use Tesseract if no PDF text found in crop (slower). Default off.")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-dir", type=str, default="./out/debug_fixed_br_boxes")

    ap.add_argument("--llm-provider", type=str, default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--llm-base-url", type=str, default="http://localhost:11434")
    ap.add_argument("--llm-model", type=str, default="qwen2.5vl:7b")
    ap.add_argument("--llm-api-key-env", type=str, default="OPENAI_API_KEY")
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--llm-timeout-s", type=int, default=30)

    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        print(f"[ERROR] Path not found: {folder}", file=sys.stderr)
        sys.exit(2)

    pdfs: List[Path] = []
    if folder.is_file() and folder.suffix.lower() == ".pdf":
        pdfs = [folder]
    elif folder.is_dir():
        pdfs = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
        if not pdfs:
            pdfs = sorted(folder.rglob("*.pdf"))
            if pdfs:
                print(f"[INFO] No top-level PDFs; using recursive scan: found {len(pdfs)} PDFs", flush=True)
    else:
        print(f"[ERROR] Path is neither a directory nor a PDF: {folder}", file=sys.stderr)
        sys.exit(2)

    if not pdfs:
        print(f"[WARN] No PDFs found under {folder}", file=sys.stderr)
        sys.exit(0)
    else:
        print(f"[INFO] Found {len(pdfs)} PDF(s) to process", flush=True)

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
        "system": SYSTEM,
    }

    all_rows: List[Dict] = []
    for pdf in pdfs:
        try:
            rows = process_pdf(
                pdf_path=pdf,
                dpi=args.dpi,
                bbox=bbox,
                crop_pad=args.crop_pad,
                provider=args.llm_provider,
                llm_params=llm_params,
                ocr_lang=str(args.ocr_lang),
                ocr_psm=int(args.ocr_psm),
                max_side=int(args.max_side),
                max_boxes=int(args.max_boxes),
                debug_dir=debug_dir,
                include_boxes_in_prompt=bool(args.include_boxes_in_prompt),
                ocr_fallback=bool(args.ocr_fallback),
            )
            all_rows.extend(rows)
        except Exception as e:
            print(f"[WARN] Skipping {pdf.name} due to error: {e}", file=sys.stderr)

    write_jsonl(Path(args.out_jsonl), all_rows)
    write_csv(Path(args.out_csv), all_rows)
    print(f"Done. JSONL: {args.out_jsonl} | CSV: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
