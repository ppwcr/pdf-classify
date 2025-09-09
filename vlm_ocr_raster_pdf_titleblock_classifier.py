#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures as futures
import csv
import io
import json
import logging
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Optional deps
try:
    import fitz  # PyMuPDF
except Exception as e:  # pragma: no cover
    fitz = None

try:
    from PIL import Image, ImageDraw
except Exception as e:  # pragma: no cover
    Image = None
    ImageDraw = None

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None

try:
    import easyocr  # type: ignore
except Exception:
    easyocr = None

VLM_AVAILABLE = True
try:
    from mlx_vlm import load as vlm_load, generate as vlm_generate  # type: ignore
except Exception:
    vlm_load = None  # type: ignore
    vlm_generate = None  # type: ignore
    VLM_AVAILABLE = False

try:
    # Preferred location per user instruction
    from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore
except Exception:
    try:
        # Possible alternative in different versions
        from mlx_vlm.prompts import apply_chat_template  # type: ignore
    except Exception:
        apply_chat_template = None  # type: ignore

try:
    from mlx_vlm.utils import load_config  # type: ignore
except Exception:
    try:
        # Some versions expose load_config at top-level
        from mlx_vlm import load_config  # type: ignore
    except Exception:
        load_config = None  # type: ignore


DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

# Reduce tokenizer fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------------
# Utility data structures
# -----------------------------

@dataclass
class NormBBox:
    x: float
    y: float
    w: float
    h: float

    def as_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class AbsBBox:
    x0: int
    y0: int
    x1: int
    y1: int

    def as_dict(self) -> Dict[str, int]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


# -----------------------------
# Argument parsing
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect title-block bbox in construction drawing PDFs, crop, OCR, and extract drawing numbers."
    )
    p.add_argument("--folder", type=str, required=True, help="Directory containing PDFs (no recursion)")
    p.add_argument("--out-jsonl", type=str, required=True, help="Path to write JSONL output")
    p.add_argument("--out-csv", type=str, required=True, help="Path to write CSV output")
    p.add_argument("--detect-pages", type=int, default=3, help="Initial pages to detect bbox (1-5)")
    p.add_argument("--bbox-iou-thresh", type=float, default=0.5, help="IoU threshold for bbox consistency")
    p.add_argument(
        "--consensus-ratio",
        type=float,
        default=0.6,
        help="Fraction of detected pages that must agree (IoU >= thresh) to reuse global bbox",
    )
    p.add_argument(
        "--no-fallback-per-page",
        action="store_true",
        help="Do not run VLM on pages beyond --detect-pages; reuse initial median bbox (or single detection)",
    )
    p.add_argument(
        "--ocr-fullpage-when-no-bbox",
        action="store_true",
        help="If a page has no bbox, OCR the full page instead of skipping",
    )
    p.add_argument("--crop-pad", type=float, default=0.04, help="Extra normalized padding around bbox")
    p.add_argument("--dpi", type=int, default=200, help="Render DPI for PDF-to-image")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="MLX model id")
    p.add_argument("--workers", type=int, default=2, help="Parallel workers for OCR")
    p.add_argument("--debug", action="store_true", help="Save debug artifacts (overlays and crops)")
    p.add_argument("--debug-dir", type=str, default=None, help="Directory to write debug images")
    p.add_argument("--max-pages", type=int, default=None, help="If set, limit processed pages per PDF")
    p.add_argument(
        "--detect-scope",
        type=str,
        default="per-pdf",
        choices=["per-pdf", "global"],
        help="Apply detection either per PDF (default) or only on the first N pages across the entire folder (global)",
    )
    return p.parse_args()


# -----------------------------
# Logging setup
# -----------------------------


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


# -----------------------------
# File discovery and rendering
# -----------------------------


def discover_pdfs(folder: str) -> List[str]:
    pdfs = []
    for name in os.listdir(folder):
        if name.lower().endswith(".pdf"):
            pdfs.append(os.path.abspath(os.path.join(folder, name)))
    pdfs.sort()
    return pdfs


def render_pdf_pages(pdf_path: str, dpi: int, max_pages: Optional[int] = None) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required to render PDFs.")
    if Image is None:
        raise RuntimeError("Pillow is required to handle images.")

    images: List[Image.Image] = []
    dims: List[Tuple[int, int]] = []
    doc = fitz.open(pdf_path)
    try:
        page_count = len(doc)
        if max_pages is not None:
            page_count = min(page_count, max_pages)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        for i in range(page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat)
            # Convert to PIL Image via PNG bytes for simplicity
            pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
            pil_img.load()  # ensure fully loaded
            images.append(pil_img)
            dims.append((pil_img.width, pil_img.height))
    finally:
        doc.close()
    return images, dims


# -----------------------------
# Geometry utilities
# -----------------------------


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def clamp_bbox_norm(b: NormBBox) -> NormBBox:
    x = clamp(b.x, 0.0, 1.0)
    y = clamp(b.y, 0.0, 1.0)
    w = clamp(b.w, 0.0, 1.0)
    h = clamp(b.h, 0.0, 1.0)
    # Ensure x+w <= 1, y+h <= 1
    if x + w > 1.0:
        w = max(0.0, 1.0 - x)
    if y + h > 1.0:
        h = max(0.0, 1.0 - y)
    return NormBBox(x, y, w, h)


def iou_bbox_norm(a: NormBBox, b: NormBBox) -> float:
    ax0, ay0 = a.x, a.y
    ax1, ay1 = a.x + a.w, a.y + a.h
    bx0, by0 = b.x, b.y
    bx1, by1 = b.x + b.w, b.y + b.h
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    a_area = a.w * a.h
    b_area = b.w * b.h
    denom = a_area + b_area - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def consensus_from_bboxes(bboxes: List[NormBBox], iou_thresh: float, ratio: float) -> Tuple[bool, Optional[NormBBox], Dict[str, Any]]:
    meta: Dict[str, Any] = {"hits": 0, "required": 0, "pair_ok": False, "ious": []}
    if not bboxes:
        return False, None, meta
    if len(bboxes) == 1:
        meta.update({"hits": 1, "required": 1, "pair_ok": False, "ious": [1.0]})
        return True, bboxes[0], meta
    med = median_bbox(bboxes)
    ious = [iou_bbox_norm(med, b) for b in bboxes]
    hits = sum(1 for v in ious if v >= iou_thresh)
    required = max(1, int(math.ceil(ratio * len(bboxes))))
    pair_ok = False
    if len(bboxes) == 2:
        pair_ok = iou_bbox_norm(bboxes[0], bboxes[1]) >= iou_thresh
    ok = hits >= required or pair_ok
    meta.update({"hits": hits, "required": required, "pair_ok": pair_ok, "ious": ious})
    return ok, med if ok else med, meta


def median_bbox(bboxes: List[NormBBox]) -> NormBBox:
    xs = [b.x for b in bboxes]
    ys = [b.y for b in bboxes]
    ws = [b.w for b in bboxes]
    hs = [b.h for b in bboxes]
    return NormBBox(
        x=float(statistics.median(xs)),
        y=float(statistics.median(ys)),
        w=float(statistics.median(ws)),
        h=float(statistics.median(hs)),
    )


def norm_to_abs_bbox(b: NormBBox, width: int, height: int, pad: float) -> AbsBBox:
    # Apply padding as a fraction of width/height
    px = int(round(pad * width))
    py = int(round(pad * height))
    x0 = int(round(b.x * width)) - px
    y0 = int(round(b.y * height)) - py
    x1 = int(round((b.x + b.w) * width)) + px
    y1 = int(round((b.y + b.h) * height)) + py
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return AbsBBox(x0=x0, y0=y0, x1=x1, y1=y1)


# -----------------------------
# OCR handling
# -----------------------------


class OCRBackend:
    def __init__(self) -> None:
        self.backend: Optional[str] = None
        self.reader: Any = None
        if pytesseract is not None:
            self.backend = "pytesseract"
        elif easyocr is not None:
            try:
                # Initialize EasyOCR with English by default
                self.reader = easyocr.Reader(["en"], gpu=False)  # type: ignore
                self.backend = "easyocr"
            except Exception:
                self.reader = None
                self.backend = None
        else:
            self.backend = None

    def name(self) -> Optional[str]:
        return self.backend

    def run(self, image: Image.Image) -> str:
        if self.backend == "pytesseract":
            try:
                # Preserve hyphenation; OEM defaults ok
                text = pytesseract.image_to_string(image)  # type: ignore
            except Exception:
                text = ""
            return normalize_text(text)
        elif self.backend == "easyocr" and self.reader is not None:
            try:
                result = self.reader.readtext(np_image_from_pil(image))  # type: ignore
                # result: list of (bbox, text, conf)
                text = "\n".join([r[1] for r in result])
            except Exception:
                text = ""
            return normalize_text(text)
        else:
            return ""


def np_image_from_pil(img: Image.Image):
    try:
        import numpy as np  # type: ignore
    except Exception:
        # If numpy not available, convert via bytes to RGB and back – EasyOCR requires ndarray; return empty
        return None
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def normalize_text(text: str) -> str:
    t = text.replace("\r", "\n")
    t = re.sub(r"[\t\x0b\x0c]+", " ", t)
    t = re.sub(r"\n+", "\n", t)
    t = re.sub(r"[ ]+", " ", t)
    return t.strip()


# -----------------------------
# Drawing number extraction
# -----------------------------


DRAWING_NO_REGEX = re.compile(
    r"(^|[^A-Z0-9])((?:A|S|E|SN)[0-9]{1,2})\s*-\s*([0-9]{2})([^0-9A-Z]|$)",
    re.IGNORECASE,
)


def extract_drawing_numbers(text: str) -> Tuple[str, List[str]]:
    candidates: List[str] = []
    for m in DRAWING_NO_REGEX.finditer(text):
        prefix = m.group(2)
        suffix = m.group(3)
        cand = f"{prefix}-{suffix}"
        cand = cand.upper()
        if cand not in candidates:
            candidates.append(cand)
    best = candidates[0] if candidates else ""
    return best, candidates


# -----------------------------
# VLM wrapper (MLX Qwen2.5-VL)
# -----------------------------


class TitleBlockDetector:
    def __init__(self, model: str, temperature: float = 0.1, timeout_s: int = 60) -> None:
        self.model_id = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.model = None
        self.processor = None
        self.config = None
        self._ok = False

        if not VLM_AVAILABLE:
            logging.warning(
                "mlx_vlm not available. Install in this Python env: 'python -m pip install mlx-vlm'. "
                f"Interpreter: {sys.executable}"
            )
            self._ok = False
        else:
            try:
                # Load model/processor/config once
                self.model, self.processor = vlm_load(self.model_id)
                self.config = load_config(self.model_id) if load_config is not None else None
                self._ok = True
                logging.info(f"mlx_vlm initialized with model: {self.model_id}")
            except Exception as e:
                logging.warning(
                    f"Failed to load VLM model '{self.model_id}': {e}. "
                    "Ensure the model is available locally or accessible to mlx_vlm."
                )
                self._ok = False

    def _detect_in_region(self, image: Image.Image, region_norm: Tuple[float, float, float, float], system_prompt: str, user_prompt: str) -> Optional[NormBBox]:
        """
        Run VLM on a cropped region specified in normalized page coords and map
        the returned bbox back to page-normalized coordinates.

        region_norm = (rx, ry, rw, rh) with values in [0..1]
        """
        try:
            rx, ry, rw, rh = region_norm
            rx = clamp(rx, 0.0, 1.0); ry = clamp(ry, 0.0, 1.0)
            rw = clamp(rw, 0.0, 1.0); rh = clamp(rh, 0.0, 1.0)
            # Ensure valid
            if rw <= 0.0 or rh <= 0.0:
                return None
            W, H = image.width, image.height
            x0 = int(round(rx * W))
            y0 = int(round(ry * H))
            x1 = int(round((rx + rw) * W))
            y1 = int(round((ry + rh) * H))
            x0 = max(0, min(W - 1, x0)); x1 = max(1, min(W, x1))
            y0 = max(0, min(H - 1, y0)); y1 = max(1, min(H, y1))
            if x1 <= x0 or y1 <= y0:
                return None
            crop = image.crop((x0, y0, x1, y1))

            # Inform the model this is a cropped strip of the title block area
            user_prompt_cropped = user_prompt + "\nThe image you see is a cropped strip near the page border (title block region)."

            resp = self._vlm_generate_with_image(system_prompt=system_prompt, user_prompt=user_prompt_cropped, image=crop)
            b = self._parse_bbox_json(resp)
            if b is None:
                # Retry once with stricter JSON reminder
                resp2 = self._vlm_generate_with_image(system_prompt=system_prompt, user_prompt=user_prompt_cropped + "\nReturn STRICT JSON only.", image=crop)
                b = self._parse_bbox_json(resp2)
            if b is None:
                return None

            # Map back to page-normalized coordinates
            gx = rx + b.x * rw
            gy = ry + b.y * rh
            gw = b.w * rw
            gh = b.h * rh
            return clamp_bbox_norm(NormBBox(gx, gy, gw, gh))
        except Exception:
            return None

    def detect_bbox(self, image: Image.Image) -> Optional[NormBBox]:
        prompt_system = (
            "You are a vision-language expert. Return a TIGHT bbox around the DRAWING NUMBER code that appears INSIDE the title block band. "
            "Return JSON only. Never include explanations."
        )
        prompt_user = (
            "TASK: Return ONE NORMALIZED bbox tightly around the DRAWING NUMBER code (examples: A1-01, A-01, S2-03, AR-000) that lies INSIDE the title block.\n"
            "\n"
            "SEARCH REGION (soft prior):\n"
            "- Look ONLY in the bottom 25% of the page OR the rightmost 20% strip (title blocks live there).\n"
            "- The code must be within the bordered admin strip (title block), not in the plan/viewport.\n"
            "\n"
            "SELECTION:\n"
            "- Target a short alphanumeric with optional dash (2–7 chars).\n"
            "- Prefer the code adjacent to labels: \"DRAWING NO\", \"DRAWING NUMBER\", \"SHEET NO\", or Thai \"เลขที่แบบ\".\n"
            "- Return bbox for TEXT ONLY (minimal margin), not the whole cell.\n"
            "\n"
            "AVOID:\n"
            "- Any text in the plan area (room/viewport titles, callouts).\n"
            "- \"DRAWING TITLE\" and other long phrases (PROJECT, OWNER, DATE, SCALE, approvals, logos).\n"
            "\n"
            "TIE-BREAKERS:\n"
            "1) Closest to labels DRAWING NO / SHEET NO / เลขที่แบบ.\n"
            "2) Bottom-right within the title block band.\n"
            "3) Most legible bold dashed short code.\n"
            "\n"
            "OUTPUT (STRICT JSON ONLY):\n"
            "{\"bbox\": {\"x\": <0..1>, \"y\": <0..1>, \"w\": <0..1>, \"h\": <0..1>}}\n"
            "If not found: {\"bbox\": null}\n"
            "No explanations, no extra keys, no extra text."
        )

        # Try bottom and right strips first (geometry prior)
        bottom_region = (0.0, 0.75, 1.0, 0.25)  # bottom 25%
        right_region  = (0.80, 0.0, 0.20, 1.0)  # rightmost 20%

        # 1) Bottom strip
        bbox = self._detect_in_region(image, bottom_region, prompt_system, prompt_user)
        if bbox is not None and bbox.w > 0 and bbox.h > 0:
            return bbox

        # 2) Right strip
        bbox = self._detect_in_region(image, right_region, prompt_system, prompt_user)
        if bbox is not None and bbox.w > 0 and bbox.h > 0:
            return bbox

        # 3) Fallback: full page (least preferred)
        resp = self._vlm_generate_with_image(system_prompt=prompt_system, user_prompt=prompt_user, image=image)
        bbox = self._parse_bbox_json(resp)
        if bbox is not None:
            return bbox

        # 4) Retry with explicit strict JSON reminder
        retry_user = prompt_user + "\nIMPORTANT: Return STRICT JSON only. Respond with exactly one object: {\"bbox\": {..}} or {\"bbox\": null}."
        resp2 = self._vlm_generate_with_image(system_prompt=prompt_system, user_prompt=retry_user, image=image)
        bbox2 = self._parse_bbox_json(resp2)
        return bbox2

    def _vlm_generate_with_image(self, system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        if not self._ok:
            logging.warning("mlx_vlm not initialized. Skipping detection.")
            return ""
        try:
            # Ensure RGB for processor
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")
            # Downscale large page images for VLM robustness/memory
            image = _resize_for_vlm(image)
            # Compose a simple chat-style prompt
            prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}"
            if apply_chat_template is not None and self.config is not None:
                formatted = apply_chat_template(self.processor, self.config, prompt, num_images=1)
                out = vlm_generate(
                    self.model,
                    self.processor,
                    formatted,
                    [image],
                    max_tokens=120,
                    temperature=float(self.temperature),
                )
            else:
                # Fallback path for older/newer mlx_vlm versions that accept direct prompt+image
                out = vlm_generate(
                    self.model,
                    self.processor,
                    prompt=prompt + "\n<image>",
                    image=image,
                    max_tokens=120,
                    temperature=float(self.temperature),
                )
            if isinstance(out, (list, tuple)):
                out = "".join(map(str, out))
            return str(out).strip()
        except Exception as e:
            logging.warning(f"VLM generation failed: {e}")
            return ""

    def _parse_bbox_json(self, text: str) -> Optional[NormBBox]:
        if not text:
            return None
        # Extract first JSON object from text
        try:
            first = text.find("{")
            last = text.rfind("}")
            if first == -1 or last == -1 or last <= first:
                return None
            snippet = text[first : last + 1]
            data = json.loads(snippet)
            if not isinstance(data, dict) or "bbox" not in data:
                return None
            if data["bbox"] is None:
                return None
            bb = data["bbox"]
            x = float(bb.get("x", 0.0))
            y = float(bb.get("y", 0.0))
            w = float(bb.get("w", 0.0))
            h = float(bb.get("h", 0.0))
            b = clamp_bbox_norm(NormBBox(x, y, w, h))
            # Very small or zero area considered invalid
            if b.w <= 0.0 or b.h <= 0.0:
                return None
            return b
        except Exception:
            return None


# -----------------------------
# Debug drawing
# -----------------------------


def draw_bbox_overlay(image: Image.Image, ab: AbsBBox) -> Image.Image:
    if ImageDraw is None:
        return image
    img = image.copy()
    draw = ImageDraw.Draw(img)
    # Red rectangle with width 3
    try:
        draw.rectangle([ab.x0, ab.y0, ab.x1, ab.y1], outline=(255, 0, 0), width=3)
    except Exception:
        # Fallback without width
        draw.rectangle([ab.x0, ab.y0, ab.x1, ab.y1], outline=(255, 0, 0))
    return img


# -----------------------------
# Image preprocessing for VLM
# -----------------------------


def _resize_for_vlm(img: Image.Image, max_side: int = 1400) -> Image.Image:
    try:
        w, h = img.size
        m = max(w, h)
        if m <= max_side:
            return img
        scale = max_side / float(m)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        # Use high-quality resample
        resample = getattr(Image, "LANCZOS", Image.BICUBIC)
        return img.resize((nw, nh), resample)
    except Exception:
        return img


# -----------------------------
# Core pipeline per PDF
# -----------------------------


def process_pdf(
    pdf_path: str,
    detector: TitleBlockDetector,
    ocr: OCRBackend,
    detect_pages: int,
    bbox_iou_thresh: float,
    consensus_ratio: float,
    no_fallback_per_page: bool,
    ocr_fullpage_when_no_bbox: bool,
    crop_pad: float,
    dpi: int,
    workers: int,
    debug: bool,
    debug_dir: Optional[str],
    max_pages: Optional[int],
    jsonl_fh,
    detect_scope: str = "per-pdf",
    global_state: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    # Render pages once
    images, dims = render_pdf_pages(pdf_path, dpi=dpi, max_pages=max_pages)
    page_count = len(images)
    logging.info(f"PDF: {os.path.basename(pdf_path)} | pages: {page_count}")
    if page_count == 0:
        return []

    detect_n = max(1, min(5, min(detect_pages, page_count)))

    use_global = False
    global_bbox: Optional[NormBBox] = None
    decision_note = ""

    if detect_scope == "global":
        # Use only first N pages across entire folder
        if global_state is None:
            global_state = {
                "attempted": 0,
                "bboxes": [],
                "established": False,
                "bbox": None,
                "note": "",
            }
        # If already established, reuse without any detection
        if global_state.get("established"):
            use_global = True
            global_bbox = global_state.get("bbox")
            decision_note = global_state.get("note", "global_scope_reuse")
        else:
            allowed = max(0, detect_pages - int(global_state.get("attempted", 0)))
            det_results: List[Optional[NormBBox]] = [None] * page_count
            # Try to detect on up to 'allowed' pages in this PDF (starting from page 0)
            for i in range(min(detect_n, page_count)):
                if allowed <= 0:
                    break
                b = detector.detect_bbox(images[i])
                det_results[i] = b
                global_state["attempted"] = int(global_state.get("attempted", 0)) + 1
                allowed -= 1
                if b is not None:
                    lst = list(global_state.get("bboxes", []))
                    lst.append(b)
                    global_state["bboxes"] = lst
                    ok, med, meta = consensus_from_bboxes(lst, bbox_iou_thresh, consensus_ratio)
                    logging.debug(
                        f"Global consistency: ious={['%.3f'%v for v in meta['ious']]}, hits={meta['hits']}, required={meta['required']}, pair_ok={meta['pair_ok']}"
                    )
                    if ok:
                        global_state["established"] = True
                        global_state["bbox"] = med
                        global_state["note"] = "global_scope_bbox_reuse"
                        break
            # Decide reuse
            if global_state.get("established"):
                use_global = True
                global_bbox = global_state.get("bbox")
                decision_note = global_state.get("note", "global_scope_bbox_reuse")
            else:
                # Not established, optionally force or not
                if no_fallback_per_page and global_state.get("bboxes"):
                    # Force median of whatever we have
                    _, med, _ = consensus_from_bboxes(global_state.get("bboxes"), bbox_iou_thresh, consensus_ratio)
                    use_global = True
                    global_bbox = med
                    decision_note = "global_scope_forced_median"
                else:
                    use_global = False
                    decision_note = "global_scope_not_ready"
    else:
        # Original per-PDF behavior on first N pages
        det_results: List[Optional[NormBBox]] = []
        for i in range(detect_n):
            b = detector.detect_bbox(images[i])
            det_results.append(b)
        present_bboxes = [b for b in det_results if b is not None]
        if len(present_bboxes) >= 2:
            ok, med, meta = consensus_from_bboxes(present_bboxes, bbox_iou_thresh, consensus_ratio)
            logging.debug(
                f"Consistency check: ious={['%.3f'%v for v in meta['ious']]}, hits={meta['hits']}, required={meta['required']}, pair_ok={meta['pair_ok']}"
            )
            if ok:
                use_global = True
                global_bbox = med
                decision_note = "global_bbox_reuse"
            else:
                if no_fallback_per_page:
                    use_global = True
                    global_bbox = med
                    decision_note = "global_bbox_forced"
        elif len(present_bboxes) == 1:
            use_global = True
            global_bbox = present_bboxes[0]
            decision_note = "global_bbox_single"

    if use_global and global_bbox is not None:
        if decision_note == "global_bbox_reuse":
            logging.info("Detection: consistent bbox across initial pages -> reusing global bbox.")
        elif decision_note == "global_bbox_forced":
            logging.info("Detection: forcing reuse of initial median bbox for all pages.")
        elif decision_note == "global_bbox_single":
            logging.info("Detection: single initial bbox -> reusing it for all pages.")
        elif decision_note == "global_scope_bbox_reuse":
            logging.info("Detection: global-scope reuse of bbox for all pages.")
        elif decision_note == "global_scope_forced_median":
            logging.info("Detection: global-scope forced median bbox for all pages.")
        # Prepare per-page bboxes
        per_page_bboxes: List[Optional[NormBBox]] = [global_bbox] * page_count  # type: ignore
        notes_baseline = decision_note or "global_bbox_reuse"
    else:
        if no_fallback_per_page:
            logging.info("Detection: no initial consensus and fallback disabled -> no bbox for remaining pages.")
            per_page_bboxes = [None] * page_count
            # Keep any existing bboxes for initial pages if present
            if detect_scope == "per-pdf":
                for i in range(min(detect_n, page_count)):
                    per_page_bboxes[i] = det_results[i]
            notes_baseline = "no_fallback_initial_only"
        else:
            if detect_scope == "per-pdf":
                logging.info("Detection: inconsistent -> per-page detection for all pages.")
                per_page_bboxes = []
                for i in range(page_count):
                    # If already detected for initial pages, reuse; else detect now
                    if i < detect_n:
                        per_page_bboxes.append(det_results[i])
                    else:
                        per_page_bboxes.append(detector.detect_bbox(images[i]))
                notes_baseline = "per_page_detection"
            else:
                # Global scope: do not run further detections beyond the global budget
                logging.info("Detection: global scope not established; using only initial sample detections (no further VLM calls).")
                per_page_bboxes = det_results
                notes_baseline = "global_scope_partial"

    # Create debug output subdir for this PDF
    pdf_debug_dir = None
    if debug and debug_dir is not None:
        base = Path(debug_dir).expanduser().resolve()
        pdf_base = Path(pdf_path).stem
        pdf_debug_dir = base / pdf_base
        pdf_debug_dir.mkdir(parents=True, exist_ok=True)

    # Prepare OCR tasks
    page_tasks: List[Tuple[int, Optional[AbsBBox], Optional[Image.Image]]]= []
    for idx, (img, bnorm) in enumerate(zip(images, per_page_bboxes)):
        w, h = img.width, img.height
        if bnorm is None:
            if ocr_fullpage_when_no_bbox:
                ab = AbsBBox(0, 0, w, h)
                crop = img
                page_tasks.append((idx, ab, crop))
            else:
                page_tasks.append((idx, None, None))
        else:
            ab = norm_to_abs_bbox(bnorm, w, h, pad=crop_pad)
            crop = img.crop((ab.x0, ab.y0, ab.x1, ab.y1))
            page_tasks.append((idx, ab, crop))

    # OCR in parallel where applicable
    results_text: Dict[int, str] = {}

    def _ocr_task(t) -> Tuple[int, str]:
        idx, ab, crop = t
        if crop is None:
            return idx, ""
        return idx, ocr.run(crop)

    max_workers = max(1, min(workers, os.cpu_count() or 1))
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, text in ex.map(_ocr_task, page_tasks):
            results_text[idx] = text

    # Write JSONL and compute CSV rows
    csv_rows: List[Dict[str, Any]] = []
    for idx, (img, bnorm) in enumerate(zip(images, per_page_bboxes)):
        w, h = img.width, img.height
        if bnorm is None:
            if ocr_fullpage_when_no_bbox:
                ab = AbsBBox(0, 0, w, h)
            else:
                ab = None
        else:
            ab = norm_to_abs_bbox(bnorm, w, h, pad=crop_pad)
        text = results_text.get(idx, "")
        best_dwg, all_dwgs = extract_drawing_numbers(text)

        notes = notes_baseline
        if bnorm is None:
            notes = (notes + ";cover_no_title_block") if notes else "cover_no_title_block"

        record = {
            "pdf_path": os.path.abspath(pdf_path),
            "page_index": idx,
            "page_width": w,
            "page_height": h,
            "bbox_norm": bnorm.as_dict() if bnorm is not None else None,
            "bbox_abs": ab.as_dict() if ab is not None else None,
            "ocr_backend": ocr.name(),
            "ocr_text": text,
            "drawing_no": best_dwg,
            "all_drawing_no_candidates": all_dwgs,
            "notes": notes,
        }
        jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        csv_rows.append(
            {
                "pdf_file": os.path.basename(pdf_path),
                "page_index": idx,
                "drawing_no": best_dwg,
                "bbox_norm": json.dumps(bnorm.as_dict()) if bnorm is not None else "null",
                "ocr_text_snippet": record["ocr_text"][:200],
                "notes": notes,
            }
        )

        # Debug artifacts
        if debug and pdf_debug_dir is not None:
            # Full with bbox overlay
            full_out = pdf_debug_dir / f"page_{idx:04d}_full.png"
            if not full_out.exists():
                try:
                    # Save full image once
                    if img.mode != "RGB":
                        img_rgb = img.convert("RGB")
                    else:
                        img_rgb = img
                    img_rgb.save(full_out)
                    logging.debug(f"Saved: {full_out}")
                except Exception:
                    pass
            if ab is not None:
                over = draw_bbox_overlay(img, ab)
                over_out = pdf_debug_dir / f"page_{idx:04d}_full_with_bbox.png"
                try:
                    if over.mode != "RGB":
                        over = over.convert("RGB")
                    over.save(over_out)
                    logging.debug(f"Saved: {over_out}")
                except Exception:
                    pass
                crop_out = pdf_debug_dir / f"page_{idx:04d}_crop.png"
                try:
                    crop_img = img.crop((ab.x0, ab.y0, ab.x1, ab.y1))
                    if crop_img.mode != "RGB":
                        crop_img = crop_img.convert("RGB")
                    crop_img.save(crop_out)
                    logging.debug(f"Saved: {crop_out}")
                except Exception:
                    pass

    # Summary logging
    det_summary = sum(1 for b in per_page_bboxes if b is not None)
    logging.info(f"Detection summary: {det_summary}/{page_count} pages with bbox.")
    return csv_rows


# -----------------------------
# Main entry
# -----------------------------


def main() -> int:
    args = parse_args()
    setup_logging(args.debug)

    folder = os.path.abspath(os.path.expanduser(args.folder))
    if not os.path.isdir(folder):
        logging.error(f"Folder not found: {folder}")
        return 2

    # Prepare outputs
    out_jsonl = os.path.abspath(os.path.expanduser(args.out_jsonl))
    out_csv = os.path.abspath(os.path.expanduser(args.out_csv))

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if args.debug and args.debug_dir:
        os.makedirs(os.path.abspath(os.path.expanduser(args.debug_dir)), exist_ok=True)

    pdfs = discover_pdfs(folder)
    if not pdfs:
        logging.warning("No PDFs found in folder.")
        # Still create empty outputs
        with open(out_jsonl, "w", encoding="utf-8") as fjsonl:
            pass
        with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(
                fcsv,
                fieldnames=[
                    "pdf_file",
                    "page_index",
                    "drawing_no",
                    "bbox_norm",
                    "ocr_text_snippet",
                    "notes",
                ],
            )
            writer.writeheader()
        print(f"Done. JSONL written to: {out_jsonl}; CSV written to: {out_csv}")
        return 0

    # Initialize components
    detector = TitleBlockDetector(model=args.model, temperature=0.1)
    ocr_backend = OCRBackend()
    if ocr_backend.name() is None:
        logging.warning("No OCR backend available (pytesseract/easyocr). Proceeding without OCR.")

    all_csv_rows: List[Dict[str, Any]] = []

    with open(out_jsonl, "w", encoding="utf-8") as fjsonl:
        # Global detection state across folder (only used when detect-scope=global)
        global_state = {
            "attempted": 0,
            "bboxes": [],
            "established": False,
            "bbox": None,
            "note": "",
        }
        for pdf_path in pdfs:
            try:
                rows = process_pdf(
                    pdf_path=pdf_path,
                    detector=detector,
                    ocr=ocr_backend,
                    detect_pages=args.detect_pages,
                    bbox_iou_thresh=args.bbox_iou_thresh,
                    consensus_ratio=args.consensus_ratio,
                    no_fallback_per_page=args.no_fallback_per_page,
                    ocr_fullpage_when_no_bbox=args.ocr_fullpage_when_no_bbox,
                    crop_pad=args.crop_pad,
                    dpi=args.dpi,
                    workers=args.workers,
                    debug=args.debug,
                    debug_dir=args.debug_dir,
                    max_pages=args.max_pages,
                    jsonl_fh=fjsonl,
                    detect_scope=args.detect_scope,
                    global_state=global_state if args.detect_scope == "global" else None,
                )
                all_csv_rows.extend(rows)
            except Exception as e:
                logging.error(f"Failed to process {os.path.basename(pdf_path)}: {e}")

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "pdf_file",
                "page_index",
                "drawing_no",
                "bbox_norm",
                "ocr_text_snippet",
                "notes",
            ],
        )
        writer.writeheader()
        for row in all_csv_rows:
            writer.writerow(row)

    print(f"Done. JSONL written to: {out_jsonl}; CSV written to: {out_csv}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
