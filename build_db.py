"""
build_db.py – Build or rebuild the FAISS vector store for AdmitAI.

Usage:
    python build_db.py                      # Smart build (skips if up-to-date)
    python build_db.py --data_path ./data   # Explicit data path
    python build_db.py --rebuild            # Force full rebuild
    python build_db.py --check              # Check status without building
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check status and print stats without building anything.",
    )
    args = parser.parse_args()

    # Late import so dotenv is loaded first
    from utils.rag import (
        build_vectorstore,
        vectorstore_exists,
        _needs_rebuild,
        print_startup_banner,
        VECTORSTORE_PATH,
    )
    from utils.pdf_loader import DATA_PATH, get_data_stats, compute_data_fingerprint

    data_path = args.data_path or DATA_PATH

    print("=" * 60)
    print("  AdmitAI – Vector Store Builder")
    print("=" * 60)
    print(f"  Data path     : {data_path}")
    print(f"  Store path    : {VECTORSTORE_PATH}")
    print("=" * 60)

    # ── Check data directory ───────────────────────────────────────────────────
    if not Path(data_path).exists():
        print(f"\n[ERROR] Data directory '{data_path}' not found.")
        print("Please create it and add PDF files inside college sub-folders.")
        sys.exit(1)

    # ── Show data stats ────────────────────────────────────────────────────────
    stats = get_data_stats(data_path)
    print(f"\n  Colleges detected : {stats['college_count']}")
    print(f"  Unique PDFs found  : {stats['pdf_count']}")
    print(f"  (Note: each PDF is counted once, duplicates are removed)")
    if stats["college_names"]:
        print(f"  College list:")
        for cn in stats["college_names"]:
            print(f"    • {cn}")

    if stats["pdf_count"] == 0:
        print(f"\n[WARNING] No PDF files found in '{data_path}'.")
        print("Please add PDF files into the college sub-folders under data/")
        print("Example:")
        print("  data/LNCT/admission_brochure.pdf")
        print("  data/MANIT/fee_structure.pdf")
        sys.exit(1)

    # ── Check-only mode ────────────────────────────────────────────────────────
    if args.check:
        print_startup_banner()
        sys.exit(0)

    # ── Decide whether to build ────────────────────────────────────────────────
    needs_build = args.rebuild or _needs_rebuild(data_path)

    if not needs_build:
        print(
            f"\n[INFO] Vector store is up-to-date at '{VECTORSTORE_PATH}'."
        )
        print("       Data fingerprint unchanged – no rebuild needed.")
        print("       Use --rebuild to force a full rebuild.")
        print_startup_banner()
        sys.exit(0)

    # ── Build ──────────────────────────────────────────────────────────────────
    if args.rebuild:
        print("\n[INFO] --rebuild flag set – forcing full rebuild …")
    else:
        print("\n[INFO] Changes detected in data/ – rebuilding index …")

    print(f"\n[STEP 1] Loading and splitting {stats['pdf_count']} PDFs …")
    start = time.time()

    try:
        store = build_vectorstore(data_path)
    except ValueError as ve:
        print(f"\n[ERROR] {ve}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Failed to build vector store: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("  ✓ Vector store built successfully!")
    print(f"  Total vectors    : {store.index.ntotal:,}")
    print(f"  Unique PDFs      : {stats['pdf_count']}")
    print(f"  Colleges indexed : {stats['college_count']}")
    print(f"  Time taken       : {elapsed:.1f} seconds")
    print(f"  Saved to         : {VECTORSTORE_PATH}")
    print("=" * 60)

    print_startup_banner()

    print("Next step – start the application:")
    print("  python app.py")
    print("  Then open http://localhost:5000\n")


if __name__ == "__main__":
    main()
