import threading
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# `fitz` (PyMuPDF) and `pdfplumber` are imported lazily inside the functions
# that need them, so importing this module stays cheap for callers that only
# touch the lighter helpers.


class TimeoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=15, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message
        self.timer = None

    def _timeout_handler(self):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self._timeout_handler)
        self.timer.daemon = True
        self.timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()
        return False

@dataclass
class ContentBlock:
    text: str; page: int; content_type: str


def table_to_markdown(table: List[List[str]]) -> str:
    """Converts a 2D list representation of a table into Markdown format."""
    if not table or not table[0]: return ""
    # Ensure all cells are strings before joining
    cleaned_table = [[str(cell).strip() if cell is not None else "" for cell in row] for row in table]
    header, *rows = cleaned_table
    md = f"| {' | '.join(header)} |\n| {' | '.join(['---'] * len(header))} |\n"
    for row in rows:
        # Pad rows that are shorter than the header
        while len(row) < len(header): row.append("")
        # Truncate rows that are longer than the header
        md += f"| {' | '.join(row[:len(header)])} |\n"
    return md


def chunk_text(text: str, page_num: int, chunk_size: int, overlap: int) -> List[Dict[str, any]]:
    """Chunks a single block of text with overlap."""
    chunks = []
    start = 0
    text_length = len(text)
    chunk_idx = 0
    while start < text_length:
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'page': page_num,
                    'content_type': 'text',
                    'chunk_id': f"p{page_num}-t-{chunk_idx}"
                }
            })
            chunk_idx += 1
        start = end - overlap if end < text_length else end
    return chunks


def _extract_pdf_blocks(pdf_path: str, table_timeout: int = 15,
                        max_pages: Optional[int] = None
                        ) -> Tuple[List[Tuple[int, str]], List[Dict[str, any]], int]:
    """Shared two-pass PDF extraction.

    Pass 1 (PyMuPDF): fast per-page text + detection of pages with tables.
    Pass 2 (pdfplumber): high-accuracy table extraction on flagged pages only.

    Returns ``(page_texts, table_chunks, n_pages)`` where ``page_texts`` is an
    ordered list of ``(page_number, text)``, ``table_chunks`` are
    ``{'text', 'metadata'}`` dicts, and ``n_pages`` is the true page count.
    When ``max_pages`` is set, Pass 1 stops after that many pages and Pass 2 is
    skipped — a lightweight mode for previews/probes.
    """
    import fitz  # PyMuPDF

    page_texts: List[Tuple[int, str]] = []
    table_chunks: List[Dict[str, any]] = []
    table_page_nums = set()

    # === PASS 1: Fast text extraction + table-page location (PyMuPDF) ===
    doc = fitz.open(pdf_path)
    try:
        n_pages = len(doc)
        last_page = n_pages if max_pages is None else min(n_pages, max_pages)
        for page_num_zero_indexed in range(last_page):
            page = doc[page_num_zero_indexed]
            page_num_one_indexed = page_num_zero_indexed + 1

            text_blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))
            full_page_text = "\n\n".join(
                [block[4].strip() for block in text_blocks if block[4].strip()]
            )
            if full_page_text:
                page_texts.append((page_num_one_indexed, full_page_text))

            if page.find_tables():
                table_page_nums.add(page_num_zero_indexed)
    finally:
        doc.close()

    # === PASS 2: Targeted high-accuracy table extraction (pdfplumber) ===
    if table_page_nums and max_pages is None:
        import pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num_zero_indexed in sorted(list(table_page_nums)):
                    page_num_one_indexed = page_num_zero_indexed + 1
                    try:
                        with timeout(seconds=table_timeout):
                            page = pdf.pages[page_num_zero_indexed]
                            tables = page.extract_tables()
                            for table in (tables or []):
                                if table and len(table) > 1:
                                    table_chunks.append({
                                        'text': table_to_markdown(table),
                                        'metadata': {'page': page_num_one_indexed,
                                                     'content_type': 'table'}
                                    })
                    except TimeoutError:
                        print(f"    - ⚠️  Table extraction on page {page_num_one_indexed} timed out. Skipping.")
                    except Exception as e:
                        print(f"    - ⚠️  Error extracting tables from page {page_num_one_indexed}: {e}")
        except Exception as e:
            print(f"❌ Error during table extraction (pdfplumber): {e}")

    return page_texts, table_chunks, n_pages


def _assemble_flat_text(page_texts: List[Tuple[int, str]],
                        table_chunks: List[Dict[str, any]]) -> str:
    """Join per-page text and per-page table markdown into one flat string,
    in page order — tables placed after the text of their page."""
    tables_by_page: Dict[int, List[str]] = {}
    for tc in table_chunks:
        tables_by_page.setdefault(tc['metadata']['page'], []).append(tc['text'])

    parts: List[str] = []
    seen_pages = set()
    for page_num, text in page_texts:
        parts.append(text)
        seen_pages.add(page_num)
        parts.extend(tables_by_page.get(page_num, []))
    # Tables on pages with no extracted text
    for page_num in sorted(tables_by_page):
        if page_num not in seen_pages:
            parts.extend(tables_by_page[page_num])
    return "\n\n".join(parts)


def extract_pdf_text(pdf_path: str, table_timeout: int = 15,
                     max_pages: Optional[int] = None) -> str:
    """Flat two-pass text extraction (PyMuPDF text + pdfplumber tables as
    markdown), no chunking. For lightweight document reads."""
    page_texts, table_chunks, _ = _extract_pdf_blocks(pdf_path, table_timeout, max_pages)
    return _assemble_flat_text(page_texts, table_chunks)


def extract_pdf_two_pass(pdf_path: str, chunk_size: int = 500, overlap: int = 50,
                         table_timeout: int = 15) -> List[Dict[str, any]]:
    """
    A robust two-pass hybrid extraction pipeline for RAG.
    Pass 1 (PyMuPDF): Fast extraction of all text and identification of pages containing tables.
    Pass 2 (pdfplumber): High-accuracy extraction of tables from only the identified pages.
    """
    print(f"Starting robust two-pass processing for: {pdf_path}")

    try:
        page_texts, table_chunks, _ = _extract_pdf_blocks(pdf_path, table_timeout)
    except Exception as e:
        print(f"❌ Error during PDF extraction: {e}")
        return []

    text_chunks: List[Dict[str, any]] = []
    for page_num, full_page_text in page_texts:
        text_chunks.extend(chunk_text(full_page_text, page_num, chunk_size, overlap))

    print(f"  - Extracted {len(text_chunks)} text chunks; "
          f"found {len(table_chunks)} tables.")

    # === Final Merge and Post-processing ===
    all_content = text_chunks + table_chunks
    all_content.sort(key=lambda x: (x['metadata']['page'], 0 if x['metadata']['content_type'] == 'text' else 1))

    for i, chunk in enumerate(all_content):
        chunk['metadata']['source'] = pdf_path
        chunk['metadata']['chunk_id'] = f"{Path(pdf_path).stem}-{i}"

    print(f"✓ Created {len(all_content)} total chunks ({len(table_chunks)} tables)")
    return all_content
