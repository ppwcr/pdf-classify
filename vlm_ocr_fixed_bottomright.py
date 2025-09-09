#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm_ocr_fixed_bottomright.py

Simple, fast pipeline:
- Fixed bottom-right crop per page (no bbox detection, no OCR heuristics)
- Send the crop directly to a VLM (e.g., Qwen2.5VL 7B via Ollama)
- Extract 4 fields: drawing_no, drawing_title, scale, total_page

Example (Ollama):
  # ollama pull qwen2.5vl:7b
  python vlm_ocr_fixed_bottomright.py \
    --folder '/Users/ppwcr/Desktop/คุณTao/01 CAD Submission/01 PDF' \
    --out-jsonl ./out/tao_fixed_bl.jsonl \
    --out-csv   ./out/ta0_fixed_bl.csv \
    --dpi 96 \
    --llm-provider ollama \
    --llm-base-url http://localhost:11434 \
    --llm-model qwen2.5vl:7b \

OpenAI-compatible vision:
  export OPENAI_API_KEY=...
  python vlm_ocr_fixed_bottomright.py --folder ./samples \
    --out-jsonl ./out/fixed_br.jsonl --out-csv ./out/fixed_br.csv --dpi 300 \
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
    x: float  # normalized [0..1]
    y: float
    w: float
    h: float


def render_crop(doc: fitz.Document, index: int, dpi: int, bbox: BBox, pad: float) -> Image.Image:
    page = doc.load_page(index)
    rect = page.rect
    x0 = max(0.0, (bbox.x - pad) * rect.width)
    y0 = max(0.0, (bbox.y - pad) * rect.height)
    x1 = min(rect.width, (bbox.x + bbox.w + pad) * rect.width)
    y1 = min(rect.height, (bbox.y + bbox.h + pad) * rect.height)
    clip = fitz.Rect(x0, y0, x1, y1)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def image_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def crop_rel(img: Image.Image, x0r: float, y0r: float, x1r: float, y1r: float) -> Image.Image:
    """Crop an image using relative coords in [0,1]. Clamps to edges."""
    w, h = img.size
    x0 = max(0, min(w - 1, int(x0r * w)))
    y0 = max(0, min(h - 1, int(y0r * h)))
    x1 = max(x0 + 1, min(w, int(x1r * w)))
    y1 = max(y0 + 1, min(h, int(y1r * h)))
    return img.crop((x0, y0, x1, y1))


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
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}
        ],
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return str(content or "").strip()
    except Exception as e:
        sys.stderr.write(f"[WARN] OpenAI vision text failed: {e}\n")
        return ""


def call_ollama_vision_text(base_url: str, model: str, prompt: str, image: Image.Image, temperature: float = 0.0, timeout_s: int = 30) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    headers = {"Content-Type": "application/json"}
    b64 = image_to_base64_png(image)
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt, "images": [b64]},
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
        sys.stderr.write(f"[WARN] Ollama vision text failed: {e}\n")
        return ""


def clean_title(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip()
    if not t:
        return ""
    # Remove anchors and trailing metadata (TOTAL PAGE, PAGE, SHEET, SCALE, REV, DATE, PROJECT)
    t = re.sub(r"(?i)\b(TOTAL\s+PAGES?|SHEET\s*NO\.?|SHEET:?|PAGE\s*NO\.?|PAGE:?|SCALE|REV(?:ISION)?|DATE|PROJECT)\b.*", "", t).strip()
    # If comma-separated, drop segments that look like material/spec lines
    parts = [p.strip() for p in re.split(r"[,;\n]", t) if p.strip()]
    if len(parts) > 1:
        spec_pat = re.compile(r"(?i)(\bRB\d|Ø\d|D\d|dia\b|@\s?\d|\bmm\b|\bkg\b|\bm\.?\b|\d+\.\d+|\d+\s*[xX]\s*\d+)")
        keep = [p for p in parts if not spec_pat.search(p)]
        if keep:
            t = max(keep, key=len)
        else:
            t = parts[0]
    # Collapse spaces
    t = re.sub(r"\s+", " ", t)
    # Drop too-short or obviously wrong
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
    # replace slash with colon if likely a scale
    t = re.sub(r"\s*/\s*", ":", t)
    t = re.sub(r"\s+", "", t)
    # keep only patterns like 1:100 or 1:1 or 1:75
    m = re.match(r"^(\d{1,3}):(\d{1,4})$", t)
    if m:
        return f"{int(m.group(1))}:{int(m.group(2))}"
    return ""


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if not isinstance(max_side, int) or max_side <= 0:
        return img
    w, h = img.size
    s = max(w, h)
    if s <= max_side:
        return img
    r = max_side / float(s)
    return img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BICUBIC)


def process_pdf(pdf_path: Path, dpi: int, bbox: BBox, crop_pad: float, max_side: int, provider: str, llm_params: Dict, debug_dir: Optional[Path]) -> List[Dict]:
    rows: List[Dict] = []
    with fitz.open(pdf_path) as doc:
        n = doc.page_count
        print(f"[INFO] PDF: {pdf_path.name} | pages: {n}", flush=True)
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
        pbar = _tqdm(total=n, desc=f"{pdf_path.name}", unit="page") if _tqdm else None
        for i in range(n):
            crop = render_crop(doc, i, dpi, bbox, crop_pad)
            crop = resize_max_side(crop, max_side)
            if debug_dir is not None:
                crop.save(debug_dir / f"{pdf_path.stem}_p{i:03d}_brcrop.png")
            if pbar is None:
                # Fallback progress output when tqdm is unavailable
                print(f"  - {pdf_path.name}: page {i+1}/{n}", flush=True)
            user = build_user_prompt()
            if provider == "ollama":
                obj = call_ollama_vision_json(
                    base_url=llm_params.get("base_url", "http://localhost:11434"),
                    model=llm_params.get("model", "qwen2.5vl:7b"),
                    system=llm_params.get("system", SYSTEM),
                    user=user,
                    image=crop,
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
                    image=crop,
                    temperature=float(llm_params.get("temperature", 0.0)),
                    timeout_s=int(llm_params.get("timeout_s", 30)),
                )
            # normalize result
            drawing_no = str(obj.get("drawing_no") or "").strip()
            raw_title = str(obj.get("drawing_title") or "").strip()
            scale = normalize_scale(str(obj.get("scale") or ""))
            raw_total = str(obj.get("total_page") or "").strip()
            # Normalize to '<current>/<total>'
            cp, tp = None, None
            m = re.search(r"\b(\d{1,3})\s*(?:OF|of)\s*(\d{1,4})\b", raw_total)
            if m:
                cp, tp = m.group(1), m.group(2)
            else:
                m = re.search(r"\b(\d{1,3})\s*/\s*(\d{1,4})\b", raw_total)
                if m:
                    cp, tp = m.group(1), m.group(2)
            if cp is None and tp is None:
                # 'TOTAL PAGES 46/57' style
                m = re.search(r"(?i)TOTAL\s+PAGES?\s*(\d{1,4})\s*/\s*(\d{1,4})", raw_total)
                if m:
                    cp, tp = m.group(1), m.group(2)
            if cp is None and tp is None:
                # single number: assume it's total; use current page index + 1
                m = re.search(r"\b(\d{1,4})\b", raw_total)
                if m:
                    tp = m.group(1)
                    cp = str(i + 1)
            if cp is None or tp is None:
                cp, tp = str(i + 1), str(n)
            total_page = f"{cp}/{tp}"

            # Clean title; keep fast: no extra VLM pass by default
            title = clean_title(raw_title)
            # No extra VLM calls by default to keep things fast.

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
                "notes": "fixed_bottom_right_vlm",
            }
            rows.append(rec)
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
    return rows


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


def main():
    ap = argparse.ArgumentParser(description="Fixed bottom-right crop + VLM: extract drawing_no, drawing_title, scale, total_page")
    ap.add_argument("--folder", type=str, required=True)
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--dpi", type=int, default=300)
    # Fixed bottom-right crop (tweak if your title block is larger)
    ap.add_argument("--bbox-x", type=float, default=0.72)
    ap.add_argument("--bbox-y", type=float, default=0.72)
    ap.add_argument("--bbox-w", type=float, default=0.28)
    ap.add_argument("--bbox-h", type=float, default=0.28)
    ap.add_argument("--crop-pad", type=float, default=0.02)
    ap.add_argument("--max-side", type=int, default=896)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-dir", type=str, default="./out/debug_fixed_br")

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
        # Top-level search first
        pdfs = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
        # Fallback to recursive search if none found at top-level
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
                max_side=args.max_side,
                provider=args.llm_provider,
                llm_params=llm_params,
                debug_dir=debug_dir,
            )
            all_rows.extend(rows)
        except Exception as e:
            print(f"[WARN] Skipping {pdf.name} due to error: {e}", file=sys.stderr)

    write_jsonl(Path(args.out_jsonl), all_rows)
    write_csv(Path(args.out_csv), all_rows)
    print(f"Done. JSONL: {args.out_jsonl} | CSV: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
