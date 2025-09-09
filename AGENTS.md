# Repository Guidelines

## Project Structure & Module Organization
- Root-level Python scripts provide CLI pipelines: `pdf_ocr_*.py` for OCR-only flows and `vlm_ocr*.py` for VLM/LLM-assisted flows. `ocr_bbox_llm.py` includes bbox heuristics + LLM classification.
- `out/` and `vlm_ocr_out/` contain generated artifacts (JSONL, CSV, debug images). They are git-ignored.
- `requirement.txt` lists Python dependencies; `.venv/` is a local virtualenv (ignored).
- No package layout yet; scripts are intended to be run directly.

## Build, Test, and Development Commands
- Setup environment:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirement.txt`
- Run OCR-only example:
  - `python pdf_ocr_raster.py --folder ./samples --out-jsonl ./out/raster.jsonl --out-csv ./out/raster.csv --dpi 200`
- Run with LLM classification (remote OpenAI-compatible):
  - `export OPENAI_API_KEY=...`
  - `python pdf_ocr_llm.py --folder ./samples --out-jsonl ./out/llm.jsonl --out-csv ./out/llm.csv --dpi 300 --llm --llm-model gpt-4o-mini`
- Run with local server (e.g., LM Studio/Ollama gateway):
  - `python pdf_ocr_llm.py --folder ./samples --llm --llm-base-url http://localhost:11434 --llm-model qwen2.5-vl:7b-instruct`
- Help for any script: `python <script>.py -h`

## Coding Style & Naming Conventions
- Python 3.10+ recommended. Use 4‑space indentation and snake_case for files, modules, and functions; constants in UPPER_CASE.
- Prefer type hints and dataclasses where appropriate.
- Formatting and linting (recommended): `black .` and `ruff check .` before commits.

## Testing Guidelines
- No test suite yet. New contributions should add `pytest` tests under `tests/` (files named `test_*.py`).
- Include focused unit tests for helpers and small end‑to‑end samples using fixture PDFs. Tests should write only to `out/` or temporary dirs.
- Run: `pytest -q` (aim for ~80% coverage on changed code).

## Commit & Pull Request Guidelines
- History is informal; adopt Conventional Commits going forward: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`.
- PRs should include: clear description, related issue links, example command lines used for validation, and (if applicable) snippets of JSON/CSV output or debug images paths.

## Security & Configuration Tips
- Never commit API keys or secrets. Use environment variables (e.g., `OPENAI_API_KEY`).
- For OCR, install the Tesseract binary (e.g., macOS: `brew install tesseract`).
- Large outputs belong in `out/` and should not be tracked.

