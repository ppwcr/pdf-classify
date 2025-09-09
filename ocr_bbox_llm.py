#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_bbox_llm.py

Crop a fixed title-block bbox from each PDF page and send that small image
to an AI model (OpenAI, Ollama, or MLX-VLM) to perform OCR + classification.

This script does not use Tesseract. The model is expected to both read text
from the provided crop and return strict JSON fields.

Requirements:
  pip install PyMuPDF Pillow requests
  pip install tqdm
  For Ollama (local VLM): install and run Ollama, e.g. `ollama pull qwen2.5-vl:7b-instruct`
  For MLX-VLM (local): `pip install mlx-vlm`

Examples:

- With MLX-VLM (local, no network):
  pip install mlx-vlm
  python ocr_bbox_llm.py \
    --folder "/Users/ppwcr/Desktop/print_pages" \
    --out-jsonl ./out/bbox_llm_mlx.jsonl \
    --out-csv   ./out/bbox_llm_mlx.csv \
    --dpi 250 \
    --llm-model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
    --max-side 896 \
    --max-tokens 96 \
    --debug
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
import time

import fitz  # PyMuPDF
from PIL import Image

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


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


def render_page(doc: fitz.Document, index: int, dpi: int) -> Image.Image:
    page = doc.load_page(index)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

# --- Efficient region rendering using PyMuPDF clip ---
def render_crop(doc: fitz.Document, index: int, dpi: int, bbox: BBox, pad: float) -> Image.Image:
    """Render only the bbox region (with pad) at the given DPI using PyMuPDF clip rect.
    This avoids rendering the full page, which is much faster.
    """
    page = doc.load_page(index)
    page_rect = page.rect  # in PDF points
    # compute fraction -> absolute points, with padding on both sides
    x0 = max(0.0, (bbox.x - pad) * page_rect.width)
    y0 = max(0.0, (bbox.y - pad) * page_rect.height)
    x1 = min(page_rect.width, (bbox.x + bbox.w + pad) * page_rect.width)
    y1 = min(page_rect.height, (bbox.y + bbox.h + pad) * page_rect.height)
    clip = fitz.Rect(x0, y0, x1, y1)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def image_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


LLM_SYSTEM = (
    "You are a **strict OCR+information-extraction** assistant for architectural/engineering title blocks. "
    "You must read a small cropped image from the lower title area and return **only** valid JSON (UTF-8, double quotes, no trailing commas). "
    "Do not include any explanations or extra text. If a field is not found, use an empty string. Follow these rules:\n\n"
    "Required JSON keys: drawing_no, sheet_name, drawing_title, project, discipline, revision, scale, date, ocr_snippet.\n\n"
    "General rules:\n"
    "- Read both English and Thai (e.g., 'DRAWING NO', 'TITLE', 'SCALE', 'DATE', 'PROJECT', 'REVISION' and Thai equivalents: 'เลขที่แบบ', 'ชื่อแบบ/หัวข้อ', 'มาตราส่วน', 'วันที่', 'โครงการ', 'แก้ไข').\n"
    "- Prefer text inside the title block table. Ignore watermarks, stamps, signatures, plot settings, or boilerplate outside the block.\n"
    "- Merge broken text lines that belong together; normalize consecutive spaces to a single space; trim leading/trailing punctuation.\n"
    "- Normalize long dashes and underscores to a simple hyphen '-' when part of IDs (e.g., 'SN—08' -> 'SN-08').\n"
    "- Do **not** invent values. Return empty strings if unsure.\n\n"
    "Field-specific rules:\n"
    "- drawing_no: Prefer the official drawing number near labels like 'DRAWING NO', 'DWG NO', 'เลขที่แบบ'. Typical patterns include letters+digits with optional hyphen/space (e.g., 'SN-08', 'A-101', 'S02', 'MEP-01'). If multiple candidates, choose the one closest to its label.\n"
    "- sheet_name: A short category such as 'PLAN', 'SECTION', 'ELEVATION', 'DETAIL', or Thai equivalents ('แปลน', 'รูปตัด', 'รูปด้าน', 'รายละเอียด').\n"
    "- drawing_title: The long free-text title next to/under 'TITLE' or 'DRAWING TITLE' (Thai: 'ชื่อแบบ', 'หัวข้อ'). Keep as a single line, remove trailing dots/ellipses.\n"
    "- project: Project name near 'PROJECT', 'PROJECT NAME', or Thai 'โครงการ'.\n"
    "- discipline: If an explicit discipline is written (e.g., 'STRUCTURAL', 'ARCHITECTURAL', 'ELECTRICAL', Thai 'โครงสร้าง', 'สถาปัตยกรรม'), return it. Otherwise, infer from clear prefixes in drawing_no (A, S, E, M, P, T) **only if obvious**; else empty.\n"
    "- revision: From 'REV', 'REVISION', 'แก้ไข'. Use the code or index only (e.g., 'A', '01', 'A1').\n"
    "- scale: Extract the ratio only (e.g., '1:75', '1:100'). If multiple scales, choose the primary one next to 'SCALE/มาตราส่วน'.\n"
    "- date: Prefer ISO 'YYYY-MM-DD' when possible. If the block shows 'DD/MM/YY' or 'DD/MM/YYYY', convert to 'YYYY-MM-DD'. For Thai Buddhist years (25xx), convert to Gregorian by subtracting 543 (e.g., '18/07/2567' -> '2024-07-18'). If conversion is uncertain, return the original string.\n"
    "- ocr_snippet: A short snippet (max 120 chars) containing the most relevant raw text you used, joined by spaces.\n\n"
    "Output format example (no extra keys):\n"
    "{\"drawing_no\": \"SN-08\", \"sheet_name\": \"DETAIL\", \"drawing_title\": \"รายละเอียดถังบำบัดน้ำเสีย\", \"project\": \"\", \"discipline\": \"STRUCTURAL\", \"revision\": \"\", \"scale\": \"1:75\", \"date\": \"\", \"ocr_snippet\": \"DRAWING NO SN-08 TITLE รายละเอียดถังบำบัดน้ำเสีย SCALE 1:75\"}"
)


def build_user_prompt() -> str:
    return (
        "Extract the fields from this title-block crop and return **only** a JSON object with keys: "
        "drawing_no, sheet_name, drawing_title, project, discipline, revision, scale, date, ocr_snippet. "
        "Anchors you may see include: 'DRAWING NO', 'TITLE', 'SCALE', 'DATE', 'PROJECT', 'REV', and Thai equivalents 'เลขที่แบบ', 'ชื่อแบบ', 'มาตราส่วน', 'วันที่', 'โครงการ', 'แก้ไข'. "
        "Normalize hyphens, merge broken text, prefer data inside the block, and leave unknown values as empty strings."
    )

ANCHORS_EN = ["DRAWING", "TITLE", "SCALE", "DATE", "PROJECT", "REV", "DWG"]
ANCHORS_TH = ["เลขที่แบบ", "ชื่อแบบ", "หัวข้อ", "มาตราส่วน", "วันที่", "โครงการ", "แก้ไข"]

def build_user_prompt_from_text(ocr_text: str) -> str:
    return (
        "You are given raw OCR text lines taken from the title-block area. "
        "Extract and return **only** a JSON object with keys: drawing_no, sheet_name, drawing_title, project, discipline, revision, scale, date, ocr_snippet. "
        "Use both English/Thai anchors when present. Normalize hyphens (e.g., SN—08 → SN-08). Unknowns → empty strings.\n\n"
        f"Raw OCR text (keep line breaks):\n<<<\n{ocr_text}\n>>>\n"
    )
import csv as _csv
from collections import defaultdict

def process_csv_text(csv_path: Path, max_lines: int = 200, only_anchors: bool = False) -> List[Dict]:
    rows_by_page: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for r in reader:
            pdf = r.get("pdf") or r.get("pdf_path") or ""
            page = str(r.get("page_index") or r.get("page") or "0")
            t = (r.get("text") or r.get("ocr_text") or "").strip()
            if not t:
                continue
            if only_anchors:
                keep = any(a.lower() in t.lower() for a in ANCHORS_EN) or any(a in t for a in ANCHORS_TH)
                if not keep:
                    continue
            rows_by_page[(pdf, page)].append(t)
    out_records: List[Dict] = []
    for (pdf, page), lines in rows_by_page.items():
        lines = lines[:max_lines]
        llm_class = classify_from_text_rules(lines)
        out_records.append({
            "pdf_path": pdf,
            "page_index": int(page) if page.isdigit() else page,
            "llm_class": llm_class,
            "llm_used": False,
            "notes": "ocr_post_classify_from_csv_rule_based",
        })
    return out_records


import datetime

DRAWING_NO_RE = re.compile(r"\b([A-Z]{1,4}[-_ ]?\d{1,4}[A-Z0-9]?)\b")
SCALE_RE = re.compile(r"\b(1\s*[:：]\s*\d{1,4})\b")
# Date patterns: DD/MM/YYYY, DD/MM/YY, DD-MM-YYYY, etc.
DATE_RE = re.compile(r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b")

SHEET_KEYWORDS = {
    "PLAN": ["PLAN", "แปลน"],
    "SECTION": ["SECTION", "รูปตัด"],
    "ELEVATION": ["ELEVATION", "รูปด้าน"],
    "DETAIL": ["DETAIL", "รายละเอียด"],
}
DISCIPLINE_FROM_PREFIX = {
    "A": "ARCHITECTURAL",
    "S": "STRUCTURAL",
    "E": "ELECTRICAL",
    "M": "MECHANICAL",
    "P": "PLUMBING",
    "T": "TELECOM",
}

def _normalize_hyphen(s: str) -> str:
    return s.replace("—", "-").replace("–", "-").replace("_", "-")

def _convert_thai_date(d: int, m: int, y: int) -> str:
    # If year is Buddhist Era (25xx), convert to Gregorian
    if y >= 2400:
        y -= 543
    # Expand 2-digit year: assume >= 70 -> 1900s, else 2000s
    if y < 100:
        y = 1900 + y if y >= 70 else 2000 + y
    try:
        return datetime.date(y, m, d).isoformat()
    except Exception:
        return f"{d:02d}/{m:02d}/{y}"

def classify_from_text_rules(lines: List[str]) -> Dict:
    text = "\n".join(lines)
    text_norm = _normalize_hyphen(text)

    # drawing_no
    drawing_no = ""
    # Prefer candidates near anchors
    for ln in lines:
        ln_n = _normalize_hyphen(ln)
        if any(a.lower() in ln_n.lower() for a in ANCHORS_EN) or any(a in ln_n for a in ANCHORS_TH):
            m = DRAWING_NO_RE.search(ln_n)
            if m:
                drawing_no = m.group(1).strip()
                break
    if not drawing_no:
        m = DRAWING_NO_RE.search(text_norm)
        if m:
            drawing_no = m.group(1).strip()

    # scale
    scale = ""
    for ln in lines:
        if "SCALE" in ln.upper() or "มาตราส่วน" in ln:
            m = SCALE_RE.search(ln)
            if m:
                scale = m.group(1).replace(" ", "")
                break
    if not scale:
        m = SCALE_RE.search(text_norm)
        if m:
            scale = m.group(1).replace(" ", "")

    # date
    date_out = ""
    for ln in lines:
        if "DATE" in ln.upper() or "วันที่" in ln:
            m = DATE_RE.search(ln)
            if m:
                d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                date_out = _convert_thai_date(d, mth, y)
                break
    if not date_out:
        m = DATE_RE.search(text_norm)
        if m:
            d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            date_out = _convert_thai_date(d, mth, y)

    # sheet_name and drawing_title (heuristic)
    sheet_name = ""
    for k, kws in SHEET_KEYWORDS.items():
        if any(kw.lower() in text_norm.lower() for kw in kws):
            sheet_name = k
            break

    drawing_title = ""
    # Pick longest line near TITLE anchors
    title_candidates: List[str] = []
    for ln in lines:
        if any(a in ln for a in ("TITLE", "ชื่อแบบ", "หัวข้อ")):
            title_candidates.append(ln)
    if not title_candidates:
        # fallback: pick the longest non-anchor line
        non_anchor = [ln for ln in lines if not any(a.lower() in ln.lower() for a in ANCHORS_EN) and not any(a in ln for a in ANCHORS_TH)]
        if non_anchor:
            drawing_title = max(non_anchor, key=len).strip()
    else:
        drawing_title = max(title_candidates, key=len).strip()

    # project
    project = ""
    for ln in lines:
        if "PROJECT" in ln.upper() or "โครงการ" in ln:
            project = ln.split(":")[-1].strip()
            if not project:
                project = ln.strip()
            break

    # revision
    revision = ""
    for ln in lines:
        if "REV" in ln.upper() or "แก้ไข" in ln:
            m = re.search(r"\b([A-Z]?\d{1,2})\b", ln.upper())
            if m:
                revision = m.group(1)
                break

    # discipline (explicit word or infer from drawing_no leading letter)
    discipline = ""
    for word in ["STRUCTURAL", "ARCHITECTURAL", "ELECTRICAL", "MECHANICAL", "PLUMBING"]:
        if word in text_norm.upper():
            discipline = word
            break
    if not discipline and drawing_no:
        prefix = drawing_no[0].upper()
        discipline = DISCIPLINE_FROM_PREFIX.get(prefix, "")

    # snippet
    snippet_lines = []
    for ln in lines:
        if any(a.lower() in ln.lower() for a in ANCHORS_EN) or any(a in ln for a in ANCHORS_TH):
            snippet_lines.append(ln.strip())
    ocr_snippet = (" ".join(snippet_lines))[:120]

    return {
        "drawing_no": drawing_no,
        "sheet_name": sheet_name,
        "drawing_title": drawing_title.strip().strip("….").strip(),
        "project": project,
        "discipline": discipline,
        "revision": revision,
        "scale": scale,
        "date": date_out,
        "ocr_snippet": ocr_snippet,
    }


def call_mlx_vlm_chat(model_obj, processor, config, image: Image.Image, temperature: float, system: str, user_text: str, timeout_s: int = 10, max_tokens: int = 128) -> Dict:
    try:
        from mlx_vlm import generate as vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template
    except Exception as e:
        print(f"[ERROR] mlx-vlm not available: {e}", file=sys.stderr)
        return {
            "drawing_no": "", "sheet_name": "", "drawing_title": "", "project": "",
            "discipline": "", "revision": "", "scale": "", "date": "", "ocr_snippet": "",
            "_error": "mlx-vlm not available",
        }

    formatted = apply_chat_template(
        processor, config, f"<|system|>\n{system}\n<|user|>\n{user_text}\n", num_images=1
    )
    try:
        out = vlm_generate(
            model_obj, processor, formatted, [image],
            max_tokens=int(max_tokens), temperature=float(temperature), top_p=0.0, seed=0
        )
        if isinstance(out, (list, tuple)):
            out = out[0] if out else ""
        raw = str(out).strip()
        # Try direct JSON, then fenced block extraction
        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw, flags=re.I)
            if not m:
                m = re.search(r"(\{[\s\S]*?\})", raw)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
        return {
            "drawing_no": "", "sheet_name": "", "drawing_title": "", "project": "",
            "discipline": "", "revision": "", "scale": "", "date": "", "ocr_snippet": "",
            "_error": "mlx-vlm JSON parse failed",
        }
    except Exception as e:
        print(f"[WARN] MLX-VLM classification failed: {e}", file=sys.stderr)
        return {
            "drawing_no": "", "sheet_name": "", "drawing_title": "", "project": "",
            "discipline": "", "revision": "", "scale": "", "date": "", "ocr_snippet": "",
            "_error": str(e),
        }


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["pdf_path", "llm_json", "ocr_snippet"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            llm = r.get("llm_class") or {}
            out = {
                "pdf_path": r.get("pdf_path", ""),
                "llm_json": json.dumps(llm, ensure_ascii=False),
                "ocr_snippet": llm.get("ocr_snippet", ""),
            }
            w.writerow(out)


def process_pdf(pdf_path: Path,
                dpi: int,
                bbox: BBox,
                crop_pad: float,
                debug_dir: Optional[Path],
                llm_params: Dict) -> List[Dict]:
    records: List[Dict] = []
    with fitz.open(pdf_path) as doc:
        n = doc.page_count
        print(f"[INFO] PDF: {pdf_path.name} | pages: {n}")
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)

        use_pbar = bool(_tqdm is not None)
        pbar = _tqdm(total=n, desc=f"{pdf_path.name}", unit="page") if use_pbar else None

        for i in range(n):
            # Render only the bbox region at DPI using PyMuPDF clip (fast)
            crop = render_crop(doc, i, dpi, bbox, crop_pad)
            if not pbar:
                print(f"[PAGE] {pdf_path.name} p{i+1}/{n}")
            if debug_dir is not None:
                crop.save(debug_dir / f"{pdf_path.stem}_p{i:03d}_crop.png")

            # Build prompt
            user_prompt = build_user_prompt()

            # Optional downscale for speed
            crop_for_infer = crop
            try:
                from math import ceil
                ms = int(globals().get("args_max_side", 0))
            except Exception:
                ms = 0
            # Fallback if globals not set; we will set it in main()
            if ms is None:
                ms = 0
            if isinstance(ms, int) and ms > 0:
                w, h = crop.size
                s = max(w, h)
                if s > ms:
                    scale = ms / float(s)
                    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                    crop_for_infer = crop.resize((nw, nh), Image.BICUBIC)

            # MLX-only vision classification
            model = llm_params.get("model")
            temperature = float(llm_params.get("temperature", 0.0))
            system = llm_params.get("system", LLM_SYSTEM)
            timeout_s = int(llm_params.get("timeout_s", 60))
            t0 = time.time()
            llm_class = call_mlx_vlm_chat(
                model_obj=llm_params.get("_mlx_model_obj"),
                processor=llm_params.get("_mlx_processor"),
                config=llm_params.get("_mlx_config"),
                image=crop_for_infer,
                temperature=temperature,
                system=system,
                user_text=user_prompt,
                timeout_s=timeout_s,
                max_tokens=int(llm_params.get("_mlx_max_tokens", 128)),
            )
            dt = time.time() - t0
            if pbar:
                try:
                    pbar.set_postfix({"ms": int(dt * 1000)})
                except Exception:
                    pass

            rec = {
                "pdf_path": str(pdf_path),
                "page_index": i,
                "llm_class": llm_class,
                "llm_used": True,
                "notes": "ocr_bbox_llm",
            }
            records.append(rec)
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
    return records


def main():
    import csv
    ap = argparse.ArgumentParser(description="Crop bbox and classify via AI (OCR+classify)")
    ap.add_argument("--folder", type=str, required=True, help="Folder with PDFs")
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--bbox-x", type=float, default=0.75)
    ap.add_argument("--bbox-y", type=float, default=0.75)
    ap.add_argument("--bbox-w", type=float, default=0.25)
    ap.add_argument("--bbox-h", type=float, default=0.25)
    ap.add_argument("--crop-pad", type=float, default=0.04)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-dir", type=str, default="./out/debug_bbox")

    ap.add_argument("--llm-model", type=str, default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--llm-timeout-s", type=int, default=60)
    ap.add_argument("--llm-system", type=str, default=LLM_SYSTEM)
    # --- Additional CLI args for fast post-processing ---
    ap.add_argument("--csv-input", type=str, default="", help="Path to OCR CSV (text bboxes). If provided, skip PDF rendering and classify from OCR text only.")
    ap.add_argument("--max-lines", type=int, default=200, help="Max OCR lines per page to feed to LLM")
    ap.add_argument("--only-anchors", action="store_true", help="Keep only lines containing anchor keywords (DRAWING/TITLE/SCALE/DATE/PROJECT/REV + Thai eqv.)")
    ap.add_argument("--max-side", type=int, default=896, help="Resize the cropped title-block so max(W,H) <= this. 0 = no resize")
    ap.add_argument("--max-tokens", type=int, default=96, help="Max tokens for MLX generation (smaller = faster)")
    args = ap.parse_args()

    # make max_side visible in process_pdf without refactor churn
    globals()["args_max_side"] = int(args.max_side)

    model_name = args.llm_model
    llm_params: Dict = {
        "model": model_name,
        "temperature": args.llm_temperature,
        "timeout_s": args.llm_timeout_s,
        "system": args.llm_system,
    }
    if not args.csv_input:
        try:
            from mlx_vlm import load as vlm_load
            from mlx_vlm.utils import load_config
            config = load_config(model_name)
            model_obj, processor = vlm_load(model_name)
            llm_params.update({
                "_mlx_model_obj": model_obj,
                "_mlx_processor": processor,
                "_mlx_config": config,
                "_mlx_max_tokens": int(args.max_tokens),
            })
        except Exception as e:
            print(f"[ERROR] Failed to load MLX VLM '{model_name}': {e}", file=sys.stderr)
            sys.exit(3)

    # --- CSV text-only post-processing path (MLX-only, rule-based) ---
    if args.csv_input:
        csv_in = Path(args.csv_input).expanduser().resolve()
        if not csv_in.exists():
            print(f"[ERROR] CSV not found: {csv_in}", file=sys.stderr)
            sys.exit(2)
        print(f"[INFO] CSV post-processing (rule-based) | csv={csv_in}")
        all_rows = process_csv_text(csv_in, max_lines=args.max_lines, only_anchors=bool(args.only_anchors))
        write_jsonl(Path(args.out_jsonl), all_rows)
        # Minimal CSV same as below
        try:
            import csv as _csv
        except Exception:
            pass
        csv_path = Path(args.out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=["pdf_path", "llm_json", "ocr_snippet"])
            w.writeheader()
            for r in all_rows:
                llm = r.get("llm_class") or {}
                w.writerow({
                    "pdf_path": r.get("pdf_path", ""),
                    "llm_json": json.dumps(llm, ensure_ascii=False),
                    "ocr_snippet": llm.get("ocr_snippet", ""),
                })
        print(f"Done. JSONL: {args.out_jsonl} | CSV: {args.out_csv}")
        return

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder}", file=sys.stderr)
        sys.exit(2)

    pdfs = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdfs:
        print(f"[WARN] No PDFs in {folder}")

    debug_dir = Path(args.debug_dir) if args.debug else None
    bbox = BBox(x=args.bbox_x, y=args.bbox_y, w=args.bbox_w, h=args.bbox_h)

    print(f"[INFO] BBox VLM OCR/classify | MLX model={model_name}")

    all_rows: List[Dict] = []
    for pdf in pdfs:
        rows = process_pdf(
            pdf_path=pdf,
            dpi=args.dpi,
            bbox=bbox,
            crop_pad=args.crop_pad,
            debug_dir=debug_dir,
            llm_params=llm_params,
        )
        all_rows.extend(rows)

    write_jsonl(Path(args.out_jsonl), all_rows)
    # CSV with only: pdf_path, llm_json, ocr_snippet
    try:
        import csv  # already imported in main top, but ensure available
    except Exception:
        pass
    # Reuse write_csv-like logic here for minimal CSV
    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pdf_path", "llm_json", "ocr_snippet"])
        w.writeheader()
        for r in all_rows:
            llm = r.get("llm_class") or {}
            w.writerow({
                "pdf_path": r.get("pdf_path", ""),
                "llm_json": json.dumps(llm, ensure_ascii=False),
                "ocr_snippet": llm.get("ocr_snippet", ""),
            })

    print(f"Done. JSONL: {args.out_jsonl} | CSV: {args.out_csv}")


if __name__ == "__main__":
    main()

