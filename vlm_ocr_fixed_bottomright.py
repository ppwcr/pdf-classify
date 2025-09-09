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
    --folder "/Users/ppwcr/Desktop/print_pages" \
    --out-jsonl ./out/fixed_br.jsonl \
    --out-csv   ./out/fixed_br.csv \
    --dpi 300 \
    --llm-provider ollama \
    --llm-base-url http://localhost:11434 \
    --llm-model qwen2.5vl:7b \
    --debug

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
    "You are a strict OCR+IE assistant for construction drawing title blocks. "
    "You will receive a cropped image from the bottom-right region of a page. "
    "Return only valid JSON with keys: drawing_no, drawing_title, scale, total_page. "
    "Rules: If a field is not visible, return an empty string; prefer exact text near labels; normalize spaces; no extra keys; no explanations. "
    "drawing_title must be the descriptive sheet/drawing name (e.g., PLAN, SECTION, ELEVATION, or a longer Thai/English title). "
    "Do NOT place page counts (e.g., 'TOTAL PAGE 10/10', '1/10', 'PAGE 1') or labels like SHEET/PAGE/SCALE/REV/DATE/PROJECT into drawing_title."
)


def build_user_prompt() -> str:
    return (
        "Extract fields from this title-block crop. "
        "Preferred anchors: 'DRAWING NO', 'DWG NO', 'TITLE', 'SCALE', 'SHEET', 'PAGE' (and Thai equivalents). "
        "Return only JSON with keys: drawing_no, drawing_title, scale, total_page. "
        "Format total_page as '<current>/<total>' (e.g., '1/10')."
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
    # Remove common anchors and tail after them (e.g., TOTAL PAGE 10/10)
    t = re.sub(r"(?i)\b(TOTAL\s+PAGES?|SHEET\s*NO\.?|SHEET:?|PAGE\s*NO\.?|PAGE:?|SCALE|REV(?:ISION)?|DATE|PROJECT)\b.*", "", t).strip()
    # Collapse spaces
    t = re.sub(r"\s+", " ", t)
    # Drop too-short or obviously wrong
    if len(t) < 4:
        return ""
    if re.fullmatch(r"[\W_]+", t):
        return ""
    return t


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
        print(f"[INFO] PDF: {pdf_path.name} | pages: {n}")
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
        pbar = _tqdm(total=n, desc=f"{pdf_path.name}", unit="page") if _tqdm else None
        for i in range(n):
            crop = render_crop(doc, i, dpi, bbox, crop_pad)
            crop = resize_max_side(crop, max_side)
            if debug_dir is not None:
                crop.save(debug_dir / f"{pdf_path.stem}_p{i:03d}_brcrop.png")
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
            scale = str(obj.get("scale") or "").replace(" ", "").strip()
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

            # Clean title; if missing or contaminated by PAGE/SHEET/SCALE, do a targeted second pass
            title = clean_title(raw_title)
            need_title_fallback = not title or re.search(r"(?i)\b(TOTAL\s+PAGES?|PAGE|SHEET|SCALE)\b", raw_title)
            # First try a position-aware sub-crop for title region (center band of title block)
            title_sub = crop_rel(crop, 0.08, 0.18, 0.96, 0.70)
            if provider == "ollama":
                t_band = call_ollama_vision_text(
                    base_url=llm_params.get("base_url", "http://localhost:11434"),
                    model=llm_params.get("model", "qwen2.5vl:7b"),
                    prompt=(
                        "Return ONLY the drawing title (the descriptive sheet/drawing name) from this region. "
                        "It is the text to the right of or immediately below the label 'TITLE' (Thai: 'ชื่อแบบ', 'หัวข้อ'). "
                        "Do NOT include page counts (e.g., 1/10, TOTAL PAGE), sheet numbers, scale, date, revision, or project info."
                    ),
                    image=title_sub,
                    temperature=float(llm_params.get("temperature", 0.0)),
                    timeout_s=int(llm_params.get("timeout_s", 30)),
                )
            else:
                t_band = call_openai_vision_text(
                    base_url=llm_params.get("base_url", getenv_str("OPENAI_BASE_URL", "https://api.openai.com/v1")),
                    api_key=llm_params.get("api_key", getenv_str(llm_params.get("api_key_env", "OPENAI_API_KEY"), "")),
                    model=llm_params.get("model", getenv_str("OPENAI_MODEL", "gpt-4o-mini")),
                    prompt=(
                        "Return ONLY the drawing title (the descriptive sheet/drawing name) from this region. "
                        "It is the text to the right of or immediately below the label 'TITLE' (Thai: 'ชื่อแบบ', 'หัวข้อ'). "
                        "Do NOT include page counts (e.g., 1/10, TOTAL PAGE), sheet numbers, scale, date, revision, or project info."
                    ),
                    image=title_sub,
                    temperature=float(llm_params.get("temperature", 0.0)),
                    timeout_s=int(llm_params.get("timeout_s", 30)),
                )
            t_band = clean_title(t_band)
            if t_band:
                title = t_band
            if need_title_fallback and not title:
                prompt_title = (
                    "Return ONLY the drawing title (the descriptive sheet/drawing name). "
                    "Do NOT return page counts (e.g., 1/10, TOTAL PAGE), sheet numbers, scale, date, revision, or project info."
                )
                if provider == "ollama":
                    t_only = call_ollama_vision_text(
                        base_url=llm_params.get("base_url", "http://localhost:11434"),
                        model=llm_params.get("model", "qwen2.5vl:7b"),
                        prompt=prompt_title,
                        image=crop,
                        temperature=float(llm_params.get("temperature", 0.0)),
                        timeout_s=int(llm_params.get("timeout_s", 30)),
                    )
                else:
                    t_only = call_openai_vision_text(
                        base_url=llm_params.get("base_url", getenv_str("OPENAI_BASE_URL", "https://api.openai.com/v1")),
                        api_key=llm_params.get("api_key", getenv_str(llm_params.get("api_key_env", "OPENAI_API_KEY"), "")),
                        model=llm_params.get("model", getenv_str("OPENAI_MODEL", "gpt-4o-mini")),
                        prompt=prompt_title,
                        image=crop,
                        temperature=float(llm_params.get("temperature", 0.0)),
                        timeout_s=int(llm_params.get("timeout_s", 30)),
                    )
                t_only = clean_title(t_only)
                if t_only:
                    title = t_only
            if debug_dir is not None:
                title_sub.save(debug_dir / f"{pdf_path.stem}_p{i:03d}_titleband.png")

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
        "system": SYSTEM,
    }

    all_rows: List[Dict] = []
    for pdf in pdfs:
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

    write_jsonl(Path(args.out_jsonl), all_rows)
    write_csv(Path(args.out_csv), all_rows)
    print(f"Done. JSONL: {args.out_jsonl} | CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
