#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm_ocr_typhoon_ocr.py

OCR via OpenAI-compatible endpoint using the Typhoon OCR model.

Behavior:
- Renders each PDF page as an image at the requested DPI (full page).
- Sends the image to an OpenAI-compatible VLM endpoint with a Typhoon-specific prompt.
- Expects the model to return JSON with a single key: `natural_text`.
- Writes JSONL and CSV containing only raw OCR text (no classification fields).

Usage (LM Studio / OpenAI-compatible):
  # LM Studio example: http://localhost:1234/v1
  python vlm_ocr_typhoon_ocr.py \
    --folder ./samples \
    --out-jsonl ./out/typhoon.jsonl \
    --out-csv   ./out/typhoon.csv \
    --dpi 300 \
    --llm-base-url http://localhost:1234/v1 \
    --llm-model hf.co/mradermacher/typhoon-ocr-7b-GGUF:Q4_K_M

Usage (Ollama):
  # ollama serve (ensure your gateway maps the model name accordingly)
  python vlm_ocr_typhoon_ocr.py \
    --folder ./samples \
    --out-jsonl ./out/typhoon.jsonl \
    --out-csv   ./out/typhoon.csv \
    --dpi 300 \
    --llm-provider ollama \
    --llm-base-url http://localhost:11434 \
    --llm-model hf.co/mradermacher/typhoon-ocr-7b-GGUF:Q4_K_M

Notes:
- This script is specialized for the model `hf.co/mradermacher/typhoon-ocr-7b-GGUF:Q4_K_M`.
- If you pass a different model, the same prompt is used but behavior is not guaranteed.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

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


PROMPTS = {
    "default": lambda base_text: (f"Below is an image of a document page along with its dimensions. "
        f"Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
        f"If the document contains images, use a placeholder like dummy.png for each image.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"),
    "structure": lambda base_text: (
        f"Below is an image of a document page, along with its dimensions and possibly some raw textual content previously extracted from it. "
        f"Note that the text extraction may be incomplete or partially missing. Carefully consider both the layout and any available text to reconstruct the document accurately.\n"
        f"Your task is to return the markdown representation of this document, presenting tables in HTML format as they naturally appear.\n"
        f"If the document contains images or figures, analyze them and include the tag <figure>IMAGE_ANALYSIS</figure> in the appropriate location.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    ),
}


def image_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_relaxed_json(s: str) -> Dict:
    """Extract the first JSON object from a string, tolerant of extra text/fences."""
    if not isinstance(s, str) or not s.strip():
        return {}
    t = s.strip()
    # Try direct JSON
    try:
        return json.loads(t)
    except Exception:
        pass
    # Try fenced code block
    import re as _re
    m = _re.search(r"```json\s*(\{[\s\S]*?\})\s*```", t, flags=_re.I)
    if not m:
        m = _re.search(r"```\s*(\{[\s\S]*?\})\s*```", t)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # First {...}
    m = _re.search(r"(\{[\s\S]*\})", t)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return {}


def call_openai_vision_text(base_url: str, api_key: str, model: str, prompt: str, image: Image.Image, temperature: float = 0.0, timeout_s: int = 60) -> str:
    """Minimal OpenAI-compatible vision chat call returning raw text.
    Sends the prompt + image and returns the assistant message content.
    """
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
        return str(data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    except Exception as e:
        sys.stderr.write(f"[WARN] OpenAI vision text failed: {e}\n")
        return ""


def call_ollama_vision_text(base_url: str, model: str, prompt: str, image: Image.Image, temperature: float = 0.0, timeout_s: int = 60) -> str:
    """Ollama /api/chat call returning raw text."""
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


def render_page(doc: fitz.Document, index: int, dpi: int) -> Image.Image:
    page = doc.load_page(index)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if not isinstance(max_side, int) or max_side <= 0:
        return img
    w, h = img.size
    s = max(w, h)
    if s <= max_side:
        return img
    r = max_side / float(s)
    return img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BICUBIC)


def process_pdf(pdf_path: Path, dpi: int, max_side: int, llm_params: Dict, prompt_style: str) -> List[Dict]:
    rows: List[Dict] = []
    base_url = llm_params.get("base_url", getenv_str("OPENAI_BASE_URL", "http://localhost:1234/v1")).strip()
    api_key = llm_params.get("api_key", getenv_str(llm_params.get("api_key_env", "OPENAI_API_KEY"), "")).strip()
    model = llm_params.get("model", "hf.co/mradermacher/typhoon-ocr-7b-GGUF:Q4_K_M").strip()
    temperature = float(llm_params.get("temperature", 0.0))
    timeout_s = int(llm_params.get("timeout_s", 60))
    provider = (llm_params.get("provider") or "").lower().strip()

    # Validate and normalize base URL per provider
    def _normalize_base_url(u: str, prov: str) -> str:
        if not u:
            return u
        b = u.strip()
        if "://" not in b:
            b = "http://" + b
        b = b.strip()
        # Remove trailing /v1 for Ollama (/api usage)
        if prov == "ollama":
            bb = b.rstrip("/")
            if bb.endswith("/v1"):
                bb = bb[:-3]
            b = bb
        # Basic validation
        parsed = urlparse(b)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid base URL for provider '{prov}': {u}")
        return b
    try:
        base_url = _normalize_base_url(base_url, provider)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return []

    use_typhoon_prompt = (model.strip() == "hf.co/mradermacher/typhoon-ocr-7b-GGUF:Q4_K_M")
    if not use_typhoon_prompt:
        sys.stderr.write(f"[INFO] Model '{model}' is not the Typhoon OCR model; proceeding anyway with the same prompt format.\n")

    with fitz.open(pdf_path) as doc:
        n = doc.page_count
        print(f"[INFO] PDF: {pdf_path.name} | pages: {n}", flush=True)
        pbar = _tqdm(total=n, desc=f"{pdf_path.name}", unit="page") if _tqdm else None
        for i in range(n):
            page_img = render_page(doc, i, dpi)
            page_img = resize_max_side(page_img, max_side)
            # No pre-OCR (base_text) by default; model is expected to OCR the image
            base_text = ""
            prompt = PROMPTS.get(prompt_style, PROMPTS["default"])(base_text)
            if provider == "ollama":
                raw = call_ollama_vision_text(
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    image=page_img,
                    temperature=temperature,
                    timeout_s=timeout_s,
                )
            else:
                raw = call_openai_vision_text(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    image=page_img,
                    temperature=temperature,
                    timeout_s=timeout_s,
                )
            obj = _parse_relaxed_json(raw)
            natural_text = ""
            if isinstance(obj, dict):
                natural_text = str(obj.get("natural_text") or "").strip()
            if not natural_text:
                # Fallback to using raw content if JSON failed
                natural_text = raw.strip()
            rec = {
                "pdf_path": str(pdf_path),
                "page_index": i,
                "natural_text": natural_text,
                "llm_used": True,
                "model": model,
                "notes": f"typhoon_ocr|prompt={prompt_style}",
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
    # CSV contains only pdf_path, page_index, natural_text
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["pdf_path", "page_index", "natural_text"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                "pdf_path": r.get("pdf_path", ""),
                "page_index": r.get("page_index", ""),
                "natural_text": r.get("natural_text", ""),
            })


def main():
    ap = argparse.ArgumentParser(description="OCR with Typhoon OCR model (raw natural_text only; no classification)")
    ap.add_argument("--folder", type=str, required=True)
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max-side", type=int, default=1280, help="Max long side of page image sent to model")
    ap.add_argument("--prompt-style", type=str, default="default", choices=["default", "structure"], help="Choose Typhoon prompt variant")

    ap.add_argument("--llm-provider", type=str, default="", choices=["", "openai", "ollama"], help="Force provider; blank = auto-detect")
    ap.add_argument("--llm-base-url", type=str, default=getenv_str("OPENAI_BASE_URL", "http://localhost:1234/v1"))
    ap.add_argument("--llm-model", type=str, default="hf.co/mradermacher/typhoon-ocr-7b-GGUF:Q4_K_M")
    ap.add_argument("--llm-api-key-env", type=str, default="OPENAI_API_KEY")
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--llm-timeout-s", type=int, default=60)

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

    # Detect provider if not explicitly given
    provider = (args.llm_provider or "").lower()
    if not provider:
        base = (args.llm_base_url or "").lower()
        if ":11434" in base or base.endswith(":11434/") or "/api" in base:
            provider = "ollama"
        else:
            provider = "openai"
    if provider == "ollama" and args.llm_base_url.strip().rstrip("/").endswith("/v1"):
        # Common gotcha: user points Ollama but leaves /v1
        print("[INFO] Adjusting Ollama base URL: removing trailing /v1 for /api usage", flush=True)
        args.llm_base_url = args.llm_base_url.strip().rstrip("/")[:-3]

    llm_params = {
        "provider": provider,
        "base_url": args.llm_base_url,
        "model": args.llm_model,
        "api_key_env": args.llm_api_key_env,
        "api_key": getenv_str(args.llm_api_key_env, ""),
        "temperature": args.llm_temperature,
        "timeout_s": args.llm_timeout_s,
    }

    print(f"[INFO] Provider={provider} | base={llm_params['base_url']} | model={llm_params['model']}", flush=True)

    all_rows: List[Dict] = []
    for pdf in pdfs:
        try:
            rows = process_pdf(
                pdf_path=pdf,
                dpi=args.dpi,
                max_side=args.max_side,
                llm_params=llm_params,
                prompt_style=args.prompt_style,
            )
            all_rows.extend(rows)
        except Exception as e:
            print(f"[WARN] Skipping {pdf.name} due to error: {e}", file=sys.stderr)

    write_jsonl(Path(args.out_jsonl), all_rows)
    write_csv(Path(args.out_csv), all_rows)
    print(f"Done. JSONL: {args.out_jsonl} | CSV: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
