import pathway as pw
import os
import logging
import argparse
import pandas as pd
import time
import numpy as np
import threading
import re
from typing import List, Dict

# Import modules
from src.ingestion.loader import load_documents, process_documents
from src.processing.chunker import chunk_documents
from src.processing.claim_tools import ClaimExtractor
from src.indexing.vector_db import HybridIndex
from src.reasoning.validator import EvidenceValidator

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def materialize_chunks(data_dir):
    """
    Ingests and processes documents using Pathway, writing chunks to a temporary CSV.
    """
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
    def run_pw():
        try:
            pw.run()
        except Exception:
            pass

    t = threading.Thread(target=run_pw, daemon=True)
    t.start()

    # Wait for data
    logging.info("Waiting for chunks to be materialized...")
    wait_time = 0
    max_wait = 60
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
                    if current_len == len(chunks_df):
                         stability_count += 1
                    else:
                         stability_count = 0

                    chunks_df = current_df

                    if stability_count >= 3:
                        logging.info("Chunk materialization appears stable. Proceeding.")
                        break
            except Exception as e:
                pass

    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    return chunks_df

def generate_shadow_queries(claim, subject_name):
    """
    Generates 'Shadow Queries' searching for the Physical Opposite.
    """
    queries = []
    text_lower = claim.lower()

    state_opposites = {
        'free': ['prison', 'chains', 'cell', 'dungeon', 'arrest', 'sentence', 'captive'],
        'wealthy': ['poor', 'beggar', 'debt', 'starving', 'penniless'],
        'alive': ['dead', 'died', 'killed', 'funeral', 'corpse', 'grave', 'buried'],
        'single': ['married', 'wife', 'husband', 'wedding', 'spouse'],
        'soldier': ['civilian', 'peace'],
        'captain': ['mutiny', 'disgraced', 'demoted'],
        'french': ['english', 'spanish', 'italian'],
        'english': ['french'],
        'only child': ['brother', 'sister', 'sibling']
    }

    for state, opposites in state_opposites.items():
        if state in text_lower:
            # Create a query: "Edmond Dantes prison chains..."
            q = f"{subject_name} {' '.join(opposites)}"
            queries.append(q)

    if "never" in text_lower:
        queries.append(text_lower.replace("never", ""))

    return queries

def check_spatiotemporal_collision(claim_anchors, chunk_text):
    """
    Module C Helper: Check for 100% Date/Location match within the evidence.
    If the evidence discusses the SAME date/location but is contradictory (NLI > 0.45),
    it implies a collision.
    """
    if not claim_anchors:
        return False

    for anchor in claim_anchors:
        date = anchor.get('date')
        locations = anchor.get('locations', [])

        # Check Date Match
        if date and date in chunk_text:
            return True

        # Check Location Match
        for loc in locations:
            if loc in chunk_text:
                return True

    return False

def run_forensic_engine(data_dir):
    # 1. Initialize Pathway Vector Store (Ingest & Index)
    chunks_df = materialize_chunks(data_dir)
    if chunks_df.empty:
        logging.error("No chunks found. Aborting.")
        return

    chunks = chunks_df.to_dict(orient='records')
    logging.info(f"Building Pathway HybridIndex with {len(chunks)} chunks...")
    index = HybridIndex(chunks)

    validator = EvidenceValidator() # Uses cross-encoder/nli-deberta-v3-large
    claim_extractor = ClaimExtractor()

    # 2. Load Input Data
    # Prefer test.csv, fallback to train.csv for demo
    input_file = os.path.join(data_dir, "test.csv")
    if not os.path.exists(input_file):
        logging.warning("test.csv not found, falling back to train.csv")
        input_file = os.path.join(data_dir, "train.csv")

    df = pd.read_csv(input_file)
    results = []

    logging.info("Starting Forensic Engine Iteration (Module A, B, C)...")

    for i, row in df.iterrows():
        row_id = row.get('id', i)
        content = row.get('content', '') or row.get('backstory', '')

        # Subject Extraction (Critical for Identity Guard)
        subject_name = str(row.get('char', '')).strip()
        subject_tokens = [t for t in subject_name.split() if len(t) > 2] if subject_name else []

        # Module A: Atomic SPS Deconstruction
        claims = claim_extractor.extract_claims(content)
        row_pred = 1 # Default Consistent
        rationale = "Consistent with canon."

        book_col = [c for c in row.index if 'book' in c.lower() or 'novel' in c.lower()]
        target_book = row[book_col[0]] if book_col else None
        normalized_book = str(target_book).replace("_", " ").rsplit('.', 1)[0] if target_book else None

        for claim in claims:
            # Filter: Hard Facts only (Pardon Rule)
            if claim_extractor.classify_claim(claim) == 'Soft':
                continue

            # Module B: Adversarial Retrieval
            # Shadow Query
            shadow_queries = generate_shadow_queries(claim, subject_name)
            evidence_items = index.search(claim, book_name=normalized_book, k=5)

            for q in shadow_queries:
                shadow_ev = index.search(q, book_name=normalized_book, k=3)
                for item in shadow_ev:
                    if not any(e['chunk'].get('chunk_id') == item['chunk'].get('chunk_id') for e in evidence_items):
                        evidence_items.append(item)

            # Identity Guard: Discard evidence not mentioning subject
            filtered_evidence = []
            if not subject_tokens:
                filtered_evidence = evidence_items # Risky fallback
            else:
                for item in evidence_items:
                    txt = item['chunk']['text']
                    if any(token in txt for token in subject_tokens):
                        filtered_evidence.append(item)

            if not filtered_evidence:
                continue

            # Verification: NLI
            nli_probs = validator.get_raw_probs(claim, filtered_evidence)
            if not nli_probs: continue

            contra_scores = [p[0] for p in [x['probs'] for x in nli_probs]]
            max_c = max(contra_scores)
            best_idx = contra_scores.index(max_c)
            best_item = filtered_evidence[best_idx]
            best_chunk_id = best_item['chunk'].get('chunk_id', 'Unknown')
            best_chapter = best_item['chunk'].get('chapter_title', 'Unknown Chapter')
            best_text = best_item['chunk']['text']

            # Module C: The Jury Gate (Decision Engine)
            is_collision = False

            # Rule 1: High Confidence (> 0.85)
            if max_c > 0.85:
                is_collision = True

            # Rule 2: Medium Confidence (> 0.45) AND Spatiotemporal Match
            elif max_c > 0.45:
                # Extract anchors specifically for this claim
                # Note: We need a method to extract anchors from a *single* claim string.
                # Re-using the extractor logic roughly:
                claim_anchors = claim_extractor.extract_anchors(claim)
                if check_spatiotemporal_collision(claim_anchors, best_text):
                    is_collision = True

            if is_collision:
                row_pred = 0
                # Format: STATE COLLISION: Claim [X] asserts [State A]. Novel Chapter [Y] documents [State B]. Since [State A] and [State B] are causally exclusive in the narrative timeline, the backstory is disproven.
                # We approximate [State A] as the claim, and [State B] as the evidence excerpt.
                rationale = (f"STATE COLLISION: Claim [{claim[:50]}...] asserts [State A]. "
                             f"Novel Chapter [{best_chapter}] documents [State B] ('{best_text[:100]}...'). "
                             f"Since [State A] and [State B] are causally exclusive in the narrative timeline, "
                             f"the backstory is disproven. (Score: {max_c:.2f})")
                break # Stop at first hard collision

        results.append({
            'id': row_id,
            'prediction': int(row_pred),
            'rationale': rationale
        })

    # Write Final Output
    res_df = pd.DataFrame(results)
    res_df.to_csv("results.csv", index=False)
    logging.info(f"Processing complete. Saved {len(results)} rows to results.csv")

def main():
    parser = argparse.ArgumentParser(description="KDSH 2026 Track A Runner")
    # Argument to allow flexibility, though 'run' is default
    parser.add_argument("--mode", type=str, default="run", help="Execution mode")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    run_forensic_engine(data_dir)

if __name__ == "__main__":
    main()
