import pathway as pw
import os
import logging
import argparse
import pandas as pd
import time
from typing import List, Dict

# Import modules
from src.ingestion.loader import load_documents, process_documents
from src.processing.chunker import chunk_documents
from src.indexing.vector_db import HybridIndex
from src.reasoning.dossier import format_dossier
from src.reasoning.validator import EvidenceValidator

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_evaluation(data_dir, output_file="results.csv"):
    logging.info("Starting Full Evaluation...")
    
    # 1. Ingest and Chunk Data
    logging.info(f"Ingesting documents from {data_dir}...")
    docs = load_documents(data_dir)
    processed = process_documents(docs)
    chunked_table = chunk_documents(processed)
    
    # Materialize chunks
    temp_csv = "temp_chunks.csv"
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    pw.io.csv.write(chunked_table, temp_csv)

    # Run Pathway in background
    import threading
    def run_pw():
        pw.run()
    t = threading.Thread(target=run_pw, daemon=True)
    t.start()

    # Wait for data
    logging.info("Waiting for chunks to be materialized...")
    wait_time = 0
    max_wait = 60 # seconds increased for stability check
    chunks_df = pd.DataFrame()
    stability_count = 0

    while wait_time < max_wait:
        time.sleep(2)
        wait_time += 2
        if os.path.exists(temp_csv) and os.path.getsize(temp_csv) > 0:
            try:
                # Read CSV to check progress
                current_df = pd.read_csv(temp_csv)
                current_len = len(current_df)

                if current_len > 0:
                    logging.info(f"Found {current_len} chunks so far...")
                    # If we have data, wait for stability (no new writes for a few seconds)
                    # This is better than breaking immediately.
                    if current_len == len(chunks_df):
                         # Size stable for one iteration (2s)
                         stability_count += 1
                    else:
                         stability_count = 0

                    chunks_df = current_df

                    if stability_count >= 3: # Stable for 6 seconds
                        logging.info("Chunk materialization appears stable. Proceeding.")
                        break
            except Exception as e:
                pass

    if chunks_df.empty:
        logging.error("No chunks materialized. Exiting.")
        return

    # Convert to list of dicts
    chunks = chunks_df.to_dict(orient='records')

    # 2. Build Hybrid Index
    logging.info(f"Building Hybrid Index with {len(chunks)} chunks...")
    index = HybridIndex(chunks)

    # 3. Initialize Validator
    logging.info("Initializing Evidence Validator (NLI model)...")
    validator = EvidenceValidator()

    # 4. Load Train Data
    train_file = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_file):
        logging.error(f"Train file not found: {train_file}")
        return

    train_df = pd.read_csv(train_file)
    logging.info(f"Loaded {len(train_df)} rows from train.csv")

    # 5. Retrieval & Validation Loop
    results_list = []

    for i, row in train_df.iterrows():
        claim = row.get('content', '')
        if not claim or pd.isna(claim):
             claim = row.get('backstory', '') # Fallback

        book_col = [c for c in row.index if 'book' in c.lower() or 'novel' in c.lower()]
        target_book = row[book_col[0]] if book_col else None

        # Normalize book name
        normalized_book = None
        if target_book:
            normalized_book = str(target_book).replace("_", " ").rsplit('.', 1)[0]

        logging.info(f"Processing Row {i}: Book='{normalized_book}'")

        # Retrieve
        evidence_items = index.search(claim, book_name=normalized_book, k=5)

        # Validate
        val_result = validator.validate(claim, evidence_items)

        # Append Result
        results_list.append({
            'id': row.get('id', i),
            'label': val_result['label'],
            'rationale': val_result['rationale'],
            'confidence': val_result['confidence']
        })

        print(f"Row {i} -> Label: {val_result['label']} | Conf: {val_result['confidence']:.4f}")

    # 6. Save Output
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_file, index=False)
    logging.info(f"Evaluation complete. Results saved to {output_file}")

    # Cleanup
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

def main():
    parser = argparse.ArgumentParser(description="KDSH 2026 Track A Runner")
    parser.add_argument("--mode", type=str, default="run", choices=["run", "smoke_test", "evaluate"], help="Mode to run the app")
    args = parser.parse_args()
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    if args.mode == "smoke_test":
        # Keep the old smoke test function if needed, or map to evaluate
        run_evaluation(data_dir, output_file="outputs/smoke_test_results.csv")
    elif args.mode == "evaluate":
        run_evaluation(data_dir, output_file="results.csv")
    else:
        logging.info("Use --mode evaluate to generate results.csv")

if __name__ == "__main__":
    main()
