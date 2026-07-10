"""
build_db.py – One-time script to build the FAISS vector store from college PDFs.

Usage:
    python build_db.py                      # Build from default data/ directory
    python build_db.py --data_path ./data   # Explicit path
    python build_db.py --rebuild            # Force rebuild even if store exists
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Build the FAISS vector store for AdmitAI."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to the root data directory containing college sub-folders.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the vector store even if one already exists.",
    )
    args = parser.parse_args()

    # Late import so dotenv is loaded first
    from utils.rag import build_vectorstore, vectorstore_exists, VECTORSTORE_PATH
    from utils.pdf_loader import DATA_PATH

    data_path = args.data_path or DATA_PATH

    print("=" * 60)
    print("  AdmitAI – Vector Store Builder")
    print("=" * 60)
    print(f"  Data path     : {data_path}")
    print(f"  Store path    : {VECTORSTORE_PATH}")
    print("=" * 60)

    # Check data directory
    if not Path(data_path).exists():
        print(f"\n[ERROR] Data directory '{data_path}' not found.")
        print("Please create it and add PDF files inside college sub-folders.")
        sys.exit(1)

    # Check for existing store
    if vectorstore_exists() and not args.rebuild:
        print(
            f"\n[INFO] Vector store already exists at '{VECTORSTORE_PATH}'."
        )
        print("Use --rebuild flag to force a rebuild.")
        print("\nVector store is ready. You can start the app with:")
        print("  python app.py")
        sys.exit(0)

    # Count PDFs
    pdf_files = list(Path(data_path).rglob("*.pdf")) + list(Path(data_path).rglob("*.PDF"))
    if not pdf_files:
        print(f"\n[WARNING] No PDF files found in '{data_path}'.")
        print("Please add PDF files into the college sub-folders under data/")
        print("Example:")
        print("  data/LNCT/admission_brochure.pdf")
        print("  data/MANIT/fee_structure.pdf")
        sys.exit(1)

    print(f"\n[INFO] Found {len(pdf_files)} PDF file(s) across all colleges.")
    print("\n[STEP 1] Loading and splitting documents …")
    start = time.time()

    try:
        store = build_vectorstore(data_path)
    except ValueError as ve:
        print(f"\n[ERROR] {ve}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Failed to build vector store: {exc}")
        sys.exit(1)

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("  ✓ Vector store built successfully!")
    print(f"  Total vectors : {store.index.ntotal}")
    print(f"  Time taken    : {elapsed:.1f} seconds")
    print(f"  Saved to      : {VECTORSTORE_PATH}")
    print("=" * 60)
    print("\nNext step – start the application:")
    print("  python app.py")
    print("  Then open http://localhost:5000\n")


if __name__ == "__main__":
    main()
