"""
pdf_loader.py - Loads and splits PDF documents from ALL college data directories.

Key behaviours
--------------
* Recursively walks *every* sub-folder under data/ (handles spaces, mixed case).
* Enriches every chunk with 6 guaranteed metadata fields:
      college_name  - folder name under data/
      folder_name   - same as college_name (alias kept for compatibility)
      pdf_name      - filename of the source PDF
      source        - absolute path string
      page_number   - 1-based page index
      document_type - inferred category (fee, hostel, placement ...)
* Returns a stable SHA-256 fingerprint of the data directory so callers can
  detect when PDFs are added/removed without re-reading all bytes.

FIX (Issue 1 & 2)
-----------------
On Windows the filesystem is case-insensitive: rglob("*.pdf") already returns
BOTH "file.pdf" AND "file.PDF".  The previous code concatenated
  list(rglob("*.pdf")) + list(rglob("*.PDF"))
which produced every file twice.  Fixed by resolving every path to a canonical
absolute path and de-duplicating with an ordered set before processing.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import List, Tuple

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# -- Tuning knobs (overridable via .env) --------------------------------------
#
# Best-practice values for multi-page admission PDFs (dense, tabular content):
#   * CHUNK_SIZE    = 1200  - captures a full fee/eligibility table in one chunk
#   * CHUNK_OVERLAP = 200   - enough overlap to preserve sentence boundaries
#   * DATA_PATH     = data  - root folder containing per-college sub-directories
#
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP",  200))
DATA_PATH     = os.getenv("DATA_PATH", "data")


# -- Document-type inference ----------------------------------------------------

_DOC_TYPE_PATTERNS: dict[str, str] = {
    "fee_structure":     r"fee",
    "hostel":            r"hostel",
    "scholarship":       r"scholarship",
    "placement":         r"placement|recruit|career",
    "eligibility":       r"eligib|criteria",
    "admission_process": r"admiss|brochure|prospectus",
    "seat_matrix":       r"seat|allotment|matrix",
    "calendar":          r"calendar|schedule|date|deadline",
    "faq":               r"faq|frequently",
    "course_details":    r"course|program|syllabus|curriculum|branch",
    "documents":         r"document|checklist|certificate",
    "reservation":       r"reserv|quota|categor|obc|sc|st",
    "cutoff":            r"cutoff|cut.off|rank|jee|gate|merit",
    "contact":           r"contact|address|phone|email|helpdesk",
    "faculty":           r"faculty|professor|staff|department",
}


def _infer_doc_type(filename: str) -> str:
    name = filename.lower()
    for doc_type, pattern in _DOC_TYPE_PATTERNS.items():
        if re.search(pattern, name):
            return doc_type
    return "general"


# -- Single-PDF loader ---------------------------------------------------------

def _load_single_pdf(pdf_path: Path, college_name: str) -> List[Document]:
    """
    Load one PDF and stamp every page with the 6 required metadata fields.
    Returns [] and prints a warning on any error - never raises.
    """
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        doc_type = _infer_doc_type(pdf_path.name)

        for page in pages:
            # page.metadata["page"] is 0-based from PyPDFLoader
            raw_page = page.metadata.get("page", 0)
            page.metadata.update(
                {
                    "college_name":  college_name,
                    "folder_name":   college_name,          # alias
                    "pdf_name":      pdf_path.name,
                    "source":        str(pdf_path.resolve()),
                    "page_number":   int(raw_page) + 1,     # 1-based
                    "document_type": doc_type,
                }
            )
        return pages

    except Exception as exc:
        print(f"  [pdf_loader] [!] Could not load {pdf_path.name}: {exc}")
        return []


# -- Directory scanner ----------------------------------------------------------

def _find_pdfs(base: Path) -> list[Tuple[Path, str]]:
    """
    Return a flat list of (pdf_path, college_name) tuples by walking every
    immediate sub-directory of *base* recursively.

    BUG FIX (Issue 1 & 2):
    -----------------------
    On Windows, Path.rglob() is case-insensitive so rglob("*.pdf") already
    matches "*.PDF".  The previous implementation called rglob() twice with
    both patterns and concatenated the results, causing every file to appear
    twice.  The fix: use a single case-insensitive rglob("*.pdf") and
    de-duplicate by resolved absolute path before returning.

    college_name is derived from the immediate child-directory name so that
    nested PDFs still carry the correct top-level college label.
    """
    pairs: list[Tuple[Path, str]] = []

    if not base.exists():
        print(f"[pdf_loader] [x] Data path does not exist: {base}")
        return pairs

    college_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not college_dirs:
        print(f"[pdf_loader] [x] No sub-directories found in {base}")
        return pairs

    for college_dir in college_dirs:
        college_name = college_dir.name

        # -- FIXED: single rglob + resolved-path dedup -------------------------
        # rglob("*.pdf") is case-insensitive on Windows (finds .pdf AND .PDF).
        # We resolve every path to its canonical absolute path and track seen
        # paths in a set to eliminate any duplicates the OS might return.
        seen_resolved: set[Path] = set()
        raw_pdfs: list[Path] = []
        for p in college_dir.rglob("*.pdf"):
            rp = p.resolve()
            if rp not in seen_resolved:
                seen_resolved.add(rp)
                raw_pdfs.append(p)

        pdfs = sorted(raw_pdfs)
        # -- END FIX -----------------------------------------------------------

        if pdfs:
            for p in pdfs:
                pairs.append((p, college_name))
        else:
            print(f"  [pdf_loader] [i]  No PDFs in {college_name} - skipping.")

    return pairs


# -- Public loaders -------------------------------------------------------------

def load_all_pdfs(data_path: str = DATA_PATH) -> List[Document]:
    """
    Walk every college sub-directory under *data_path* and load all PDFs.

    Prints a per-college summary and a grand total including duplicate-detection.
    Returns a flat list of raw page-level Documents.
    """
    base  = Path(data_path)
    pairs = _find_pdfs(base)

    if not pairs:
        return []

    # -- Duplicate PDF detection (sanity check) ---------------------------------
    pdf_paths = [str(p.resolve()) for p, _ in pairs]
    seen_paths: dict[str, int] = {}
    for pp in pdf_paths:
        seen_paths[pp] = seen_paths.get(pp, 0) + 1
    duplicates = {k: v for k, v in seen_paths.items() if v > 1}
    if duplicates:
        print(f"\n[pdf_loader] [WARN] {len(duplicates)} duplicate PDF path(s) detected "
              f"after dedup - this should not happen. Please report this bug.")
        for dup_path, count in duplicates.items():
            print(f"  DUP ({count}x): {dup_path}")

    # Group by college for clean logging
    college_map: dict[str, list[Path]] = {}
    for pdf_path, college_name in pairs:
        college_map.setdefault(college_name, []).append(pdf_path)

    print(f"\n[pdf_loader] {'-'*50}")
    print(f"[pdf_loader]  Colleges detected  : {len(college_map)}")
    print(f"[pdf_loader]  Unique PDFs found   : {len(pairs)}")
    print(f"[pdf_loader]  Duplicate PDFs found: {len(duplicates)}")
    print(f"[pdf_loader] {'-'*50}")

    all_docs: List[Document] = []

    for college_name, pdfs in sorted(college_map.items()):
        college_total = 0
        for pdf_path in pdfs:
            pages = _load_single_pdf(pdf_path, college_name)
            all_docs.extend(pages)
            college_total += len(pages)
            print(f"  [{college_name}] {pdf_path.name} -> {len(pages)} page(s)")
        print(f"  [{college_name}] Subtotal: {college_total} page(s)")

    print(f"\n[pdf_loader]  Total pages loaded: {len(all_docs)}")
    return all_docs


def split_documents(
    documents:    List[Document],
    chunk_size:    int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split page-level documents into smaller, overlapping chunks.
    All metadata from the parent page is preserved on every child chunk.

    After splitting, duplicate chunks (identical page_content) are removed
    to prevent duplicate vectors in FAISS.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[pdf_loader]  Raw chunks after splitting  : {len(chunks)}")

    # -- Deduplicate chunks by content hash ------------------------------------
    # Even after fixing _find_pdfs(), this guard prevents any future duplicate
    # vectors if a PDF is accidentally placed in two locations.
    seen_hashes: set[int] = set()
    unique_chunks: List[Document] = []
    dup_count = 0
    for chunk in chunks:
        h = hash(chunk.page_content)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_chunks.append(chunk)
        else:
            dup_count += 1

    if dup_count:
        print(f"[pdf_loader]  [!]  Duplicate chunks removed  : {dup_count}")
    print(f"[pdf_loader]  Unique chunks for indexing  : {len(unique_chunks)}")

    # -- Log per-college chunk distribution ------------------------------------
    college_chunk_counts: dict[str, int] = {}
    for chunk in unique_chunks:
        cn = chunk.metadata.get("college_name", "Unknown")
        college_chunk_counts[cn] = college_chunk_counts.get(cn, 0) + 1
    print(f"[pdf_loader]  Chunks per college:")
    for cn, cnt in sorted(college_chunk_counts.items()):
        print(f"    * {cn:<35} {cnt:>5} chunks")

    return unique_chunks


def load_and_split(data_path: str = DATA_PATH) -> List[Document]:
    """Convenience wrapper: load all PDFs and return split, deduplicated chunks."""
    docs   = load_all_pdfs(data_path)
    chunks = split_documents(docs)
    return chunks


# -- Fingerprint helpers --------------------------------------------------------

def compute_data_fingerprint(data_path: str = DATA_PATH) -> str:
    """
    Return a SHA-256 hex string that changes whenever any PDF is
    added, removed, or modified under *data_path*.

    The hash covers: sorted list of resolved absolute paths + their file sizes.
    Using size instead of content keeps this fast even for hundreds of PDFs.
    """
    base  = Path(data_path)
    pairs = _find_pdfs(base)

    if not pairs:
        return "empty"

    h = hashlib.sha256()
    for pdf_path, college_name in sorted(pairs, key=lambda x: str(x[0].resolve())):
        try:
            size = pdf_path.stat().st_size
        except OSError:
            size = 0
        h.update(f"{pdf_path.resolve()}:{size}".encode())

    return h.hexdigest()


def get_data_stats(data_path: str = DATA_PATH) -> dict:
    """Return a dict with college_count, pdf_count, college_names."""
    base  = Path(data_path)
    pairs = _find_pdfs(base)

    college_names = sorted({cn for _, cn in pairs})
    return {
        "college_count": len(college_names),
        "pdf_count":     len(pairs),
        "college_names": college_names,
    }
