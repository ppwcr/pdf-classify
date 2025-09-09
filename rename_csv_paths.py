#!/usr/bin/env python3
"""
Quick utility to rewrite the `pdf_path` column of a CSV by appending
page info and drawing number in the format:

    <pdf_path> page<total_page with '/' -> '_'>_<drawing_no>

Defaults align with out/fixed_bl.csv schema:
- path col: pdf_path
- page col: total_page (e.g., '57/57')
- drawing col: drawing_no (e.g., 'SN-08')

Writes to a new CSV by default to avoid clobbering input.
Use --inplace to overwrite the path column instead of adding a new one.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_new_path(original: str, page: str | None, drawing: str | None, prefix: str = "page") -> str:
    page = (page or "").strip()
    drawing = (drawing or "").strip()

    # Normalize page like '57/57' -> '57_57'; handle blanks safely
    page_norm = page.replace("/", "_") if page else ""

    parts = [original]
    suffix_bits = []
    if page_norm:
        suffix_bits.append(f"{prefix}{page_norm}")
    if drawing:
        suffix_bits.append(drawing)

    if suffix_bits:
        return f"{original} {'_'.join(suffix_bits)}"
    return original


def process_csv(
    in_csv: Path,
    out_csv: Path,
    path_col: str,
    page_col: str,
    drawing_col: str,
    inplace: bool,
    prefix: str,
) -> None:
    with in_csv.open("r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        if path_col not in fieldnames:
            raise SystemExit(f"Missing required path column '{path_col}' in {in_csv}")
        # Decide output header
        if inplace:
            out_fieldnames = fieldnames
        else:
            new_col = "renamed_path"
            if new_col in fieldnames:
                # avoid collision
                base = new_col
                i = 1
                while new_col in fieldnames:
                    new_col = f"{base}_{i}"
                    i += 1
            out_fieldnames = fieldnames + [new_col]

        with out_csv.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
            writer.writeheader()

            for row in reader:
                original = row.get(path_col, "")
                page = row.get(page_col, "") if page_col in row else ""
                drawing = row.get(drawing_col, "") if drawing_col in row else ""
                new_value = build_new_path(original, page, drawing, prefix=prefix)

                if inplace:
                    row[path_col] = new_value
                    writer.writerow(row)
                else:
                    # append to row as the last column
                    complete = {**row}
                    complete[out_fieldnames[-1]] = new_value
                    writer.writerow(complete)


def main() -> None:
    ap = argparse.ArgumentParser(description="Append page and drawing to CSV path column")
    ap.add_argument("--csv", dest="in_csv", type=Path, default=Path("out/fixed_bl.csv"), help="Input CSV path")
    ap.add_argument("--out-csv", dest="out_csv", type=Path, default=Path("out/fixed_bl_renamed.csv"), help="Output CSV path")
    ap.add_argument("--path-col", default="pdf_path", help="Column containing original path")
    ap.add_argument("--page-col", default="total_page", help="Column containing 'current/total' page text")
    ap.add_argument("--drawing-col", default="drawing_no", help="Column containing drawing number")
    ap.add_argument("--prefix", default="page", help="Prefix before page text (default: 'page')")
    ap.add_argument("--inplace", action="store_true", help="Overwrite the path column instead of adding a new column")

    args = ap.parse_args()

    if not args.in_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.in_csv}")

    # Ensure output directory exists
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    process_csv(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        path_col=args.path_col,
        page_col=args.page_col,
        drawing_col=args.drawing_col,
        inplace=args.inplace,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()

