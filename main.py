#!/usr/bin/env python3
"""Huishang QCFA Agent — Document Chunking & Fact Extraction Pipeline.

Usage:
    python main.py [--mode chunk] [--top-char N] [--input-dir DIR] [--output PATH]
    python main.py --mode extract-facts [--input PATH] [--output PATH]
"""

import argparse
import glob
import json
import os
import sys

import yaml

from src.document_processor import process_book, write_chunks_jsonl

SETTINGS_PATH = "settings.yaml"
DEFAULT_INPUT_DIR = "data/books"
DEFAULT_CHUNKS_OUTPUT = "outputs/huishang_qcfa/chunks.jsonl"
DEFAULT_FACTS_OUTPUT = "outputs/huishang_qcfa/fact_candidates.jsonl"
DEFAULT_QCFA_OUTPUT = "outputs/huishang_qcfa/qcfa_candidates.jsonl"


def load_settings(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_book_files(input_dir: str) -> list[str]:
    patterns = [os.path.join(input_dir, "*.txt"), os.path.join(input_dir, "*.md")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str) -> str:
    """Write a list of dicts to a JSONL file. Returns the output path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def run_chunk_phase(args, settings):
    """Phase 1: chunk book files into chunks.jsonl."""
    top_char = args.top_char or settings.get("top_char", 1200)

    book_files = find_book_files(args.input_dir)
    if not book_files:
        print(f"No .txt or .md files found in '{args.input_dir}'.", file=sys.stderr)
        sys.exit(1)

    all_chunks = []
    for book_path in book_files:
        print(f"Processing: {book_path}")
        try:
            chunks = process_book(book_path, top_char)
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks generated")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    output_path = write_jsonl(all_chunks, args.output)
    print(f"\nDone. {len(all_chunks)} total chunks written to: {output_path}")


def run_extract_facts_phase(args, settings):
    """Phase 2: extract candidate facts from chunks.jsonl."""
    from src.agents.fact_extraction_agent import extract_facts

    deepseek_cfg = settings.get("deepseek", {})
    api_key = deepseek_cfg.get("api_key", "")
    model = deepseek_cfg.get("model", "deepseek-v4-flash")
    base_url = deepseek_cfg.get("base_url", "https://api.deepseek.com")

    if not api_key:
        print("ERROR: deepseek.api_key not found in settings.yaml", file=sys.stderr)
        sys.exit(1)

    # Read chunks
    chunks = read_jsonl(args.input)
    print(f"Loaded {len(chunks)} chunks from: {args.input}")

    # Extract facts from each chunk
    all_facts: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] extracting facts...", end="", flush=True)
        try:
            facts = extract_facts(
                chunk_record=chunk,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )
            all_facts.extend(facts)
            print(f" → {len(facts)} fact(s)")
        except Exception as e:
            print(f" ERROR: {e}")

    # Write output
    output_path = write_jsonl(all_facts, args.output)
    print(f"\nDone. {len(all_facts)} total candidate facts written to: {output_path}")


def run_generate_qcfa_phase(args, settings):
    """Phase 3: generate QCFA records from fact_candidates.jsonl."""
    from src.agents.qcfa_generation_agent import generate_qcfa

    deepseek_cfg = settings.get("deepseek", {})
    api_key = deepseek_cfg.get("api_key", "")
    model = deepseek_cfg.get("model", "deepseek-v4-flash")
    base_url = deepseek_cfg.get("base_url", "https://api.deepseek.com")

    if not api_key:
        print("ERROR: deepseek.api_key not found in settings.yaml", file=sys.stderr)
        sys.exit(1)

    facts = read_jsonl(args.input)
    print(f"Loaded {len(facts)} facts from: {args.input}")

    qcfa_records: list[dict] = []
    for i, fact in enumerate(facts, 1):
        print(f"  [{i}/{len(facts)}] generating QCFA...", end="", flush=True)
        try:
            record = generate_qcfa(
                fact_record=fact,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )
            if record:
                qcfa_records.append(record)
                print(" ✓")
            else:
                print(" skipped (empty fact)")
        except Exception as e:
            print(f" ERROR: {e}")

    output_path = write_jsonl(qcfa_records, args.output)
    print(f"\nDone. {len(qcfa_records)} QCFA records written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Huishang QCFA Agent Pipeline.")
    parser.add_argument(
        "--mode",
        default="chunk",
        choices=["chunk", "extract-facts", "generate-qcfa"],
        help="Pipeline phase to run (default: chunk)",
    )
    # Chunk-phase args
    parser.add_argument("--top-char", type=int, help="Characters per chunk (overrides settings.yaml)")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help=f"Input directory (default: {DEFAULT_INPUT_DIR})")
    # Shared input/output
    parser.add_argument("--input", help=f"Input path (default varies by mode)")
    parser.add_argument("--output", help=f"Output JSONL path (default varies by mode)")
    args = parser.parse_args()

    settings = load_settings(SETTINGS_PATH)

    if args.mode == "chunk":
        if args.input is None:
            args.input = DEFAULT_INPUT_DIR
        if args.output is None:
            args.output = DEFAULT_CHUNKS_OUTPUT
        args.input_dir = args.input  # map generic --input to --input-dir
        run_chunk_phase(args, settings)
    elif args.mode == "extract-facts":
        if args.input is None:
            args.input = DEFAULT_CHUNKS_OUTPUT
        if args.output is None:
            args.output = DEFAULT_FACTS_OUTPUT
        run_extract_facts_phase(args, settings)
    elif args.mode == "generate-qcfa":
        if args.input is None:
            args.input = DEFAULT_FACTS_OUTPUT
        if args.output is None:
            args.output = DEFAULT_QCFA_OUTPUT
        run_generate_qcfa_phase(args, settings)


if __name__ == "__main__":
    main()
