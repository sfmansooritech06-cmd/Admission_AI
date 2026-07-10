"""
pdf_loader.py – Loads and splits PDF documents from college data directories.

Each chunk carries metadata:
    - college_name  : folder name under data/
    - pdf_name      : original filename
    - page_number   : page within the PDF
    - document_type : inferred category (brochure, fee, hostel, scholarship …)
    - source_path   : absolute path for reference
"""

import os
import re
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
DATA_PATH     = os.getenv("DATA_PATH", "data")


# ── helpers ──────────────────────────────────────────────────────────────────

_DOC_TYPE_PATTERNS = {
    "fee_structure":     r"fee",
    "hostel":            r"hostel",
    "scholarship":       r"scholarship",
    "placement":         r"placement",
    "eligibility":       r"eligib",
    "admission_process": r"admiss|brochure|prospectus",
    "seat_matrix":       r"seat|allotment",
    "calendar":          r"calendar|schedule|date",
    "faq":               r"faq|frequently",
    "course_details":    r"course|program|syllabus|curriculum",
    "documents":         r"document|checklist",
    "reservation":       r"reserv|quota|categor",
}

def _infer_doc_type(filename: str) -> str:
    name = filename.lower()
    for doc_type, pattern in _DOC_TYPE_PATTERNS.items():
        if re.search(pattern, name):
            return doc_type
    return "general"


def _load_single_pdf(pdf_path: Path, college_name: str) -> List[Document]:
    """Load a single PDF and return raw LangChain Document pages."""
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        doc_type = _infer_doc_type(pdf_path.name)

        for page in pages:
            page.metadata.update(
                {
                    "college_name":  college_name,
                    "pdf_name":      pdf_path.name,
                    "page_number":   page.metadata.get("page", 0) + 1,
                    "document_type": doc_type,
                    "source_path":   str(pdf_path),
                }
            )
        return pages
    except Exception as exc:
        print(f"[pdf_loader] WARNING – could not load {pdf_path}: {exc}")
        return []


def load_all_pdfs(data_path: str = DATA_PATH) -> List[Document]:
    """
    Walk every college sub-directory under *data_path* and load all PDFs.
    Returns a flat list of raw page-level Documents.
    """
    all_docs: List[Document] = []
    base = Path(data_path)

    if not base.exists():
        print(f"[pdf_loader] Data path '{data_path}' does not exist.")
        return all_docs

    college_dirs = [d for d in base.iterdir() if d.is_dir()]

    if not college_dirs:
        print(f"[pdf_loader] No college sub-directories found in '{data_path}'.")
        return all_docs

    for college_dir in sorted(college_dirs):
        college_name = college_dir.name
        pdfs = list(college_dir.rglob("*.pdf")) + list(college_dir.rglob("*.PDF"))

        if not pdfs:
            print(f"[pdf_loader] No PDFs in {college_dir} – skipping.")
            continue

        print(f"[pdf_loader] Loading {len(pdfs)} PDF(s) for {college_name} …")
        for pdf_path in sorted(pdfs):
            pages = _load_single_pdf(pdf_path, college_name)
            all_docs.extend(pages)
            print(f"  → {pdf_path.name}: {len(pages)} page(s)")

    print(f"\n[pdf_loader] Total pages loaded: {len(all_docs)}")
    return all_docs


def split_documents(
    documents: List[Document],
    chunk_size:    int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split page-level documents into smaller, overlapping chunks.
    Metadata from the parent page is preserved on every child chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[pdf_loader] Total chunks after splitting: {len(chunks)}")
    return chunks


def load_and_split(data_path: str = DATA_PATH) -> List[Document]:
    """Convenience wrapper: load all PDFs and return split chunks."""
    docs   = load_all_pdfs(data_path)
    chunks = split_documents(docs)
    return chunks
