"""Stage A: PDF -> cleaned pages -> chunks with citation metadata.

Parser decision: pypdfium2 for all text (fast, correct reading order, tolerant of the pages
that crash pypdf/pdfplumber) + pdfplumber only for pages that carry a "Table x.y" caption.
"""
import json
import re
import unicodedata
from pathlib import Path

import pypdfium2 as pdfium

from . import config

NUTRI_LABELS = ("Energy", "Protein", "Carbohydrates", "Fat")
RUNNING_HEADER = re.compile(r"^\s*(\d+\s+)?ICMR-National Institute of Nutrition Dietary Guidelines for Indians(\s+\d+)?\s*$")
TABLE_CHARS = 900
TABLE_CAPTION = re.compile(r"^\s*(Table\s+\d+\.\d+\.?.{0,120})$", re.M)


def doc_key(path: Path) -> str:
    """Map a file name to a short key ('healthy_gut' or 'dgi'); unknown files get their stem."""
    name = path.name.lower()
    if "healthy_gut" in name or "healthy gut" in name:
        return "healthy_gut"
    if "dgi" in name or "dietary" in name:
        return "dgi"
    return re.sub(r"\W+", "_", path.stem.lower())[:40]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("­", "").replace("￾", "").replace("﻿", "")
    text = re.sub(r"[•●▪◦]\s*\t?\s*", "- ", text)
    text = text.replace("\t", " ")
    lines = [re.sub(r"[ ]{2,}", " ", l).rstrip() for l in text.split("\n")]
    lines = [l for l in lines if not RUNNING_HEADER.match(l)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def is_readable(text: str) -> bool:
    """Drop cover pages whose legacy fonts come out as gibberish, and near-empty pages."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 60:
        return False
    ok = sum(1 for c in chars if c.isascii() and (c.isalnum() or c in ".,;:()%-/'\"&+="))
    return ok / len(chars) >= 0.9


def load_pages(path: Path) -> list[dict]:
    """One dict per readable PDF page (1-based page numbers = what a PDF viewer shows)."""
    key = doc_key(path)
    pdf = pdfium.PdfDocument(str(path))
    pages = []
    for i in range(len(pdf)):
        try:
            raw = pdf[i].get_textpage().get_text_range()
        except Exception:
            continue  # one bad page must not stop ingestion
        text = clean_text(raw)
        if is_readable(text):
            pages.append({"doc": key, "source": path.name, "page": i + 1, "text": text})
    pdf.close()
    return pages


# ---------- tables (pdfplumber, only on pages with a caption) ----------

def _md_row(cells):
    return "| " + " | ".join((c or "").replace("\n", " ").strip() for c in cells) + " |"


def tables_as_markdown(path: Path, page_numbers: list[int]) -> list[dict]:
    """Extract tables from the given pages; failures return nothing for that page."""
    import pdfplumber

    out = []
    try:
        pdf = pdfplumber.open(str(path))
    except Exception:
        return out
    with pdf:
        for pn in page_numbers:
            try:
                page = pdf.pages[pn - 1]
                caption = None
                m = TABLE_CAPTION.search(page.extract_text() or "")
                if m:
                    caption = re.sub(r"\s+", " ", m.group(1)).strip()
                for t in page.extract_tables():
                    t = [r for r in t if r and any((c or "").strip() for c in r)]
                    if len(t) < 2 or max(len(r) for r in t) < 2:
                        continue
                    header = t[0]
                    width = len(header)
                    head = _md_row(header) + "\n| " + " | ".join(["---"] * width) + " |"
                    batch, size = [], len(head)
                    for r in t[1:] + [None]:
                        line = _md_row(r) if r else ""
                        if r is None or (batch and size + len(line) > TABLE_CHARS):
                            out.append({"page": pn, "section": caption or f"Table on page {pn}",
                                        "text": head + "\n" + "\n".join(batch)})
                            batch, size = [], len(head)
                        if r is not None:
                            batch.append(line)
                            size += len(line) + 1
            except Exception:
                continue
    return out


# ---------- splitting ----------

def split_text(text: str, size: int = config.CHUNK_CHARS, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Greedy sentence/line-aware splitter with character overlap (no external dependency)."""
    flat = re.sub(r"(?<![\n.!?:;])\n(?!\n|- )", " ", text)  # re-join wrapped lines, keep bullets
    units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n+", flat) if u.strip()]
    pieces = []
    for u in units:  # hard-split any unit longer than a chunk
        while len(u) > size:
            cut = u.rfind(" ", 0, size) or size
            pieces.append(u[:cut])
            u = u[cut:].strip()
        pieces.append(u)
    chunks, cur = [], []
    cur_len = 0
    for p in pieces:
        if cur and cur_len + len(p) + 1 > size:
            chunks.append(" ".join(cur))
            keep, kl = [], 0
            for q in reversed(cur):  # carry the tail as overlap
                if kl + len(q) > overlap:
                    break
                keep.insert(0, q)
                kl += len(q) + 1
            cur, cur_len = keep, kl
        cur.append(p)
        cur_len += len(p) + 1
    if cur:
        chunks.append(" ".join(cur))
    # merge a tiny tail into the previous chunk
    if len(chunks) > 1 and len(chunks[-1]) < config.MIN_CHUNK_CHARS:
        chunks[-2] += " " + chunks.pop()
    return chunks


# ---------- recipe-specific logic (Healthy Gut) ----------

def _good_title(t: str) -> bool:
    t = t.strip()
    return 2 < len(t) <= 80 and not t.startswith("-") and not t.endswith(".") and not re.search(r"\b(time|serving)\b", t, re.I)


def find_title(lines: list[str], energy_idx: int | None):
    """Titles sit either on the first line (followed by 'Preparation time') or just above the author line
    that precedes the nutrition box; a title may wrap onto a second '(...)' line."""
    if len(lines) > 1 and _good_title(lines[0]) and lines[1].lower().startswith("preparation"):
        return lines[0].strip()
    if energy_idx == 1 and _good_title(lines[0]):
        return lines[0].strip()
    if energy_idx is not None and energy_idx >= 2:
        t = lines[energy_idx - 2].strip()
        if t.startswith("(") and energy_idx >= 3:
            t = lines[energy_idx - 3].strip() + " " + t
        if _good_title(t):
            return t
    return None


def parse_nutrition(lines: list[str]):
    """Return (energy_idx, nutrition_string, remaining_lines) for the Energy/Protein/Carbohydrates/Fat block."""
    for j, l in enumerate(lines):
        if l.strip() == "Energy":
            vals, k = {}, j
            while k + 1 < len(lines) and lines[k].strip() in NUTRI_LABELS:
                vals[lines[k].strip()] = lines[k + 1].strip()
                k += 2
            if len(vals) == 4:
                nutr = "Energy {Energy}, Protein {Protein}, Carbohydrates {Carbohydrates}, Fat {Fat} (per serving)".format(**vals)
                rest = [r for r in lines[:j] + lines[k:] if r.strip() not in ("Nutritional Value", "(per serving)")]
                return j, nutr, rest
    return None, None, lines


def recipe_chunks(pages: list[dict]) -> list[dict]:
    chunks, current = [], "Introduction"
    for pg in pages:
        lines = pg["text"].split("\n")
        j, nutr, rest = parse_nutrition(lines)
        title = find_title(lines, j)
        if "Recipe Contributor" in pg["text"]:
            title = "Credits and contributors"
        if title:
            current = title
        elif nutr and current == "Introduction":
            current = f"Recipe on page {pg['page']}"
        body = "\n".join(rest)
        if nutr:
            chunks.append({**_meta(pg), "section": current, "type": "nutrition",
                           "text": f"Recipe: {current}. Nutrition: {nutr}."})
        for t in split_text(body):
            chunks.append({**_meta(pg), "section": current, "type": "recipe_text",
                           "text": f"Recipe: {current}. {t}" if current != "Introduction" else t})
    return chunks


def guideline_chunks(pages: list[dict], path: Path) -> list[dict]:
    chunks = []
    caption_pages = [p["page"] for p in pages if TABLE_CAPTION.search(p["text"])]
    for pg in pages:
        m = TABLE_CAPTION.search(pg["text"])
        section = re.sub(r"\s+", " ", m.group(1)).strip() if m else f"Page {pg['page']}"
        for t in split_text(pg["text"]):
            chunks.append({**_meta(pg), "section": section, "type": "text", "text": t})
    for tb in tables_as_markdown(path, caption_pages):
        pg = next(p for p in pages if p["page"] == tb["page"])
        chunks.append({**_meta(pg), "section": tb["section"], "type": "table", "text": f"{tb['section']}\n{tb['text']}"})
    return chunks


def _meta(pg):
    return {"doc": pg["doc"], "source": pg["source"], "doc_name": config.DOC_NAMES.get(pg["doc"], pg["source"]),
            "kind": config.DOC_KIND.get(pg["doc"], "other"), "page": pg["page"]}


def chunk_pdf(path: Path) -> list[dict]:
    pages = load_pages(path)
    key = doc_key(path)
    chunks = recipe_chunks(pages) if key == "healthy_gut" else guideline_chunks(pages, path)
    chunks = [c for c in chunks if len(c["text"]) >= config.MIN_CHUNK_CHARS]
    for i, c in enumerate(chunks):
        c["chunk_index"] = i
        c["id"] = f"{key}-{i}"
    return chunks


def ingest_all(raw_dir: Path = config.RAW_DIR, out_path: Path = config.CHUNKS_PATH) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    allc = []
    for pdf in sorted(raw_dir.glob("*.pdf")):
        allc.extend(chunk_pdf(pdf))
    for i, c in enumerate(allc):
        c["global_index"] = i
    with open(out_path, "w", encoding="utf-8") as f:
        for c in allc:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return allc


if __name__ == "__main__":
    cs = ingest_all()
    print(f"{len(cs)} chunks written to {config.CHUNKS_PATH}")
