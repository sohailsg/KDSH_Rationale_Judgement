import pathway as pw
import os
import logging
import argparse
import pandas as pd
import time
from typing import List, Dict

# Import modules (adjusting imports based on refactoring)
from src.ingestion.loader import load_documents, process_documents
from src.processing.chunker import chunk_documents
from src.indexing.vector_db import HybridIndex
from src.reasoning.dossier import format_dossier

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_smoke_test(data_dir, output_file="outputs/smoke_test.md"):
    logging.info("Starting Smoke Test...")
    
    # 1. Ingest and Chunk Data using Pathway
    # To get data out of Pathway into Python memory for the local HybridIndex,
    # we can use `pw.debug.table_to_list` or just iterate if we can run the loop.
    # However, `pw.io.fs.read` is streaming. For a static dataset, we can use `pw.debug.compute_and_print`
    # logic but capture the output. Or `pw.io.python.write` to a list.
    
    logging.info(f"Ingesting documents from {data_dir}...")
    docs = load_documents(data_dir)
    processed = process_documents(docs)
    chunked_table = chunk_documents(processed)
    
    # Collect chunks into a list
    # We use a trick: write to a python list
    collected_chunks = []

    def on_change(key, row, time, is_addition):
        if is_addition:
            collected_chunks.append(row)

    # Use pw.io.python.write to push data to our list
    # Note: in recent Pathway, we might need a custom connector or `pw.io.subscribe`.
    # But let's try a simple approach: running the graph for a bit?
    # No, Pathway 0.28 has `pw.debug.compute_and_print` which runs until input exhaustion for static sources?
    # `pw.io.fs.read` monitors the directory. If mode="streaming", it waits.
    # If we want batch behavior, we assume it reads all current files.

    # Let's use `pw.debug.table_to_list` if available? No.
    # Let's use `pw.io.csv.write` to a temp file then read it back?
    # Or just use `pw.debug.compute_and_print` but that prints.

    # Alternative: Use `pw.run()` with a non-blocking loop?
    # "Provide a simple local runner"

    # Best way for this "minimal baseline" to bridge Pathway -> Local Python is:
    # Use `pw.io.memory.write` (if available) or a custom class.

    logging.info("Materializing chunks (this might take a moment)...")

    # We will use a dedicated run loop that terminates when data is processed?
    # pw.io.fs.read is continuous.
    # We can assume that for the smoke test we just want what's there now.

    # Let's try `pw.debug.compute_and_print(chunked_table)` logic but storing.
    # Or just `pw.io.csv.write` to a temporary file.
    temp_csv = "temp_chunks.csv"
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    pw.io.csv.write(chunked_table, temp_csv)

    # Run the pipeline. Since fs.read is streaming, we need to know when to stop.
    # For a smoke test on static files, usually we run for a fixed time or check file existence.
    # This is hacky but robust enough for a "minimal baseline" smoke test without complex signaling.

    # Use a background thread to run Pathway?
    # Or just run it for X seconds.
    import threading

    def run_pw():
        # run_until_complete is not standard.
        # We can run indefinitely.
        pw.run()

    t = threading.Thread(target=run_pw, daemon=True)
    t.start()
    
    # Wait for file to be created and populated
    logging.info("Waiting for chunks to be materialized...")
    wait_time = 0
    max_wait = 30 # seconds
    chunks_df = pd.DataFrame()

    while wait_time < max_wait:
        time.sleep(2)
        wait_time += 2
        if os.path.exists(temp_csv) and os.path.getsize(temp_csv) > 0:
            try:
                # Try reading. It might be partial write.
                chunks_df = pd.read_csv(temp_csv)
                if len(chunks_df) > 0:
                    logging.info(f"Found {len(chunks_df)} chunks so far...")
                    # For smoke test, if we have chunks from all expected books, we are good.
                    # Let's just wait a bit more to ensure flush.
                    time.sleep(2)
                    chunks_df = pd.read_csv(temp_csv) # Read again
                    break
            except Exception as e:
                pass

    if chunks_df.empty:
        logging.error("No chunks materialized. Exiting.")
        return

    # Convert DataFrame to list of dicts
    chunks = chunks_df.to_dict(orient='records')

    # 2. Build Hybrid Index
    logging.info(f"Building Hybrid Index with {len(chunks)} chunks...")
    index = HybridIndex(chunks)

    # 3. Load Train Data
    train_file = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_file):
        logging.error(f"Train file not found: {train_file}")
        return

    train_df = pd.read_csv(train_file)
    samples = train_df.head(2) # Take first 2 rows

    # 4. Retrieval & Dossier Generation
    full_output = ""

    for i, row in samples.iterrows():
        claim = row.get('backstory', 'No backstory provided')
        # Assuming we can infer book name from row or filename logic?
        # The prompt says: "Map from book_name in CSV -> novel txt filename"
        # Since we ingested all novels, we have `novel_id` in chunks (normalized).
        # We need to normalize the CSV book name to match `novel_id`.
        # For this baseline, we'll try to match loosely or pass None to search all.

        # Check if CSV has a book column.
        book_name_col = [c for c in row.index if 'book' in c.lower() or 'novel' in c.lower()]
        target_book = row[book_name_col[0]] if book_name_col else None

        # Normalize target_book if possible
        normalized_book = None
        if target_book:
            normalized_book = str(target_book).replace("_", " ").rsplit('.', 1)[0]

        logging.info(f"Processing Row {i}: Book='{normalized_book}'")

        results = index.search(claim, book_name=normalized_book, k=5)

        dossier = format_dossier(
            claim=claim,
            evidence_items=results,
            metadata={'row_id': i, 'target_book': normalized_book}
        )

        full_output += dossier + "\n"
        print("---")
        print(dossier)
        print("---")

    # 5. Save Output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(full_output)
    logging.info(f"Smoke test complete. Results saved to {output_file}")

    # Cleanup temp file
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

def main():
    parser = argparse.ArgumentParser(description="KDSH 2026 Track A Runner")
    parser.add_argument("--mode", type=str, default="run", choices=["run", "smoke_test"], help="Mode to run the app")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    if args.mode == "smoke_test":
        run_smoke_test(data_dir)
    else:
        # Placeholder for full server mode
        logging.info("Full server mode not fully implemented for this baseline. Use --mode smoke_test")

if __name__ == "__main__":
    main()
