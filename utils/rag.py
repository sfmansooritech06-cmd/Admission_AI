"""
rag.py – Core RAG pipeline for AdmitAI.

Responsibilities:
  1. Build / load a FAISS vector store from college PDF chunks.
  2. Retrieve the top-K most relevant chunks for a query.
  3. Format the context string and source citations.
  4. Orchestrate the full QA flow: retrieve → format → call IBM Granite.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils.pdf_loader import load_and_split
from utils.ibm_granite import generate_answer

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/faiss_index")
TOP_K            = int(os.getenv("TOP_K_RESULTS", 5))
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# ── Global singletons ─────────────────────────────────────────────────────────

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: FAISS | None                = None


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print(f"[rag] Loading embedding model: {EMBEDDING_MODEL} …")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[rag] Embedding model loaded.")
    return _embeddings


# ── Vector store management ───────────────────────────────────────────────────

def build_vectorstore(data_path: str = "data") -> FAISS:
    """
    Build a FAISS vector store from all PDFs under *data_path*,
    persist it to disk, and return the store object.
    """
    print("[rag] Building vector store …")
    chunks = load_and_split(data_path)

    if not chunks:
        raise ValueError(
            "No document chunks found. "
            "Please add PDF files to the data/ sub-directories."
        )

    embeddings = _get_embeddings()
    store      = FAISS.from_documents(chunks, embeddings)

    # Persist – create the FULL target directory (not just its parent).
    # Bug was: Path(VECTORSTORE_PATH).parent.mkdir() only created
    # 'vectorstore/' and never 'vectorstore/faiss_index/', so
    # save_local() had nowhere to write and the files were lost.
    save_path = Path(VECTORSTORE_PATH).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(save_path))

    # Verify the files were actually written
    faiss_file = save_path / "index.faiss"
    pkl_file   = save_path / "index.pkl"
    if not faiss_file.exists() or not pkl_file.exists():
        raise RuntimeError(
            f"save_local() did not produce index files in '{save_path}'. "
            f"index.faiss present: {faiss_file.exists()}, "
            f"index.pkl present: {pkl_file.exists()}"
        )

    print(f"[rag] Vector store saved to '{save_path}'.")
    print(f"[rag]   index.faiss : {faiss_file.stat().st_size:,} bytes")
    print(f"[rag]   index.pkl   : {pkl_file.stat().st_size:,} bytes")

    global _vectorstore
    _vectorstore = store
    return store


def load_vectorstore() -> FAISS:
    """
    Load a persisted FAISS vector store from disk.
    Raises FileNotFoundError if it doesn't exist.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    # Use the absolute, OS-normalised path so forward-slash strings from
    # .env work correctly on Windows.
    load_path  = Path(VECTORSTORE_PATH).resolve()
    faiss_file = load_path / "index.faiss"

    if not faiss_file.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{load_path}'. "
            "Please run:  python build_db.py"
        )

    print(f"[rag] Loading vector store from '{load_path}' …")
    embeddings   = _get_embeddings()
    _vectorstore = FAISS.load_local(
        str(load_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("[rag] Vector store loaded.")
    return _vectorstore


def get_vectorstore() -> FAISS:
    """Return the loaded store, or raise a clear error."""
    return load_vectorstore()


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(
    query: str,
    top_k: int = TOP_K,
    college_filter: str | None = None,
) -> List[Document]:
    """
    Retrieve the *top_k* most relevant document chunks for *query*.

    Parameters
    ----------
    query          : str            – Student's question.
    top_k          : int            – Number of chunks to retrieve.
    college_filter : str | None     – Optionally restrict to a specific college.

    Returns
    -------
    List[Document] – Ranked chunks with full metadata.
    """
    store = get_vectorstore()

    if college_filter:
        # Retrieve more then filter by college
        candidates = store.similarity_search(query, k=top_k * 4)
        results = [
            doc for doc in candidates
            if doc.metadata.get("college_name", "").lower() == college_filter.lower()
        ][:top_k]
        # Fall back to unfiltered if nothing matched
        if not results:
            results = candidates[:top_k]
    else:
        results = store.similarity_search(query, k=top_k)

    return results


# ── Context & citation helpers ────────────────────────────────────────────────

def format_context(chunks: List[Document]) -> str:
    """
    Combine retrieved chunks into a single context string fed to the LLM.
    Each chunk is labelled with its college, document name, and page.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta     = chunk.metadata
        college  = meta.get("college_name", "Unknown")
        pdf_name = meta.get("pdf_name", "Unknown")
        page     = meta.get("page_number", "?")
        parts.append(
            f"[Chunk {i} | College: {college} | Document: {pdf_name} | Page: {page}]\n"
            f"{chunk.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def format_sources(chunks: List[Document]) -> List[Dict[str, Any]]:
    """
    Build a deduplicated list of source citations for the frontend.
    Each entry contains: college_name, pdf_name, page_number, document_type.
    """
    seen    = set()
    sources = []
    for chunk in chunks:
        meta     = chunk.metadata
        college  = meta.get("college_name", "Unknown")
        pdf_name = meta.get("pdf_name", "Unknown")
        page     = meta.get("page_number", "?")
        doc_type = meta.get("document_type", "general")

        key = (college, pdf_name, page)
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "college_name":  college,
                    "pdf_name":      pdf_name,
                    "page_number":   page,
                    "document_type": doc_type,
                }
            )
    return sources


# ── Main QA pipeline ──────────────────────────────────────────────────────────

def answer_question(
    question: str,
    college_filter: str | None = None,
    top_k: int = TOP_K,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Full RAG pipeline:
        question → retrieve → format context → IBM Granite → answer + sources

    Parameters
    ----------
    question       : str            – Student's question.
    college_filter : str | None     – Restrict retrieval to one college.
    top_k          : int            – Number of chunks to retrieve.

    Returns
    -------
    (answer: str, sources: List[Dict])
    """
    if not question or not question.strip():
        return "Please enter a valid question.", []

    # 1. Retrieve relevant chunks
    try:
        chunks = retrieve_chunks(question, top_k=top_k, college_filter=college_filter)
    except FileNotFoundError as exc:
        return str(exc), []
    except Exception as exc:
        return f"Retrieval error: {exc}", []

    if not chunks:
        return (
            "No relevant documents found for your question. "
            "Please ensure the vector store is built and PDF documents are available.",
            [],
        )

    # 2. Format context
    context = format_context(chunks)

    # 3. Generate answer via IBM Granite
    answer = generate_answer(question, context)

    # 4. Compile source citations
    sources = format_sources(chunks)

    return answer, sources


# ── Vectorstore status ────────────────────────────────────────────────────────

def vectorstore_exists() -> bool:
    """Return True if the FAISS index file exists on disk."""
    # Check for the actual index file, not just the directory,
    # so an empty directory doesn't report as "exists".
    return (Path(VECTORSTORE_PATH).resolve() / "index.faiss").exists()


def get_vectorstore_stats() -> Dict[str, Any]:
    """Return basic stats about the loaded vector store."""
    try:
        store = load_vectorstore()
        count = store.index.ntotal if hasattr(store, "index") else "unknown"
        return {"status": "loaded", "total_vectors": count, "path": VECTORSTORE_PATH}
    except FileNotFoundError:
        return {"status": "not_found", "total_vectors": 0, "path": VECTORSTORE_PATH}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "path": VECTORSTORE_PATH}
