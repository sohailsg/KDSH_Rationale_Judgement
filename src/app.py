import pathway as pw
import os
import logging
import argparse
import pandas as pd
import time
import numpy as np
from typing import List, Dict

# Import modules
from src.ingestion.loader import load_documents, process_documents
from src.processing.chunker import chunk_documents
from src.processing.claim_tools import ClaimExtractor
from src.indexing.vector_db import HybridIndex
from src.reasoning.dossier import format_dossier
from src.reasoning.validator import EvidenceValidator
from src.training.trainer import LogicClassifier

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def materialize_chunks(data_dir):
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
        try:
            pw.run()
        except Exception:
            pass

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

def process_single_row(row, index, validator, classifier, claim_extractor):
    content = row.get('content', '')
    if not content or pd.isna(content): content = row.get('backstory', '')

    book_col = [c for c in row.index if 'book' in c.lower() or 'novel' in c.lower()]
    target_book = row[book_col[0]] if book_col else None
    if target_book:
        normalized_book = str(target_book).replace("_", " ").rsplit('.', 1)[0]
    else:
        normalized_book = None

    # Extract Atomic Claims
    claims = claim_extractor.extract_claims(content)

    feature_vectors = []
    evidence_metadata = [] # Store evidence for rationale generation

    # Evaluate each atomic claim
    for claim in claims:
        # Revert K to 5 for stability
        evidence_items = index.search(claim, book_name=normalized_book, k=5)
        nli_probs = validator.get_raw_probs(claim, evidence_items)
        retrieval_scores = [item['score'] for item in evidence_items]

        # Pass claim text for overlap calculation
        feat_vec = classifier.extract_features(nli_probs, retrieval_scores, claim)
        feature_vectors.append(feat_vec)

        # Store metadata for rationale
        if evidence_items:
            evidence_metadata.append({
                'chunk_text': evidence_items[0]['chunk']['text'],
                'chapter_title': evidence_items[0]['chunk'].get('chapter_title', 'Unknown Chapter')
            })
        else:
            evidence_metadata.append(None)

    return feature_vectors, evidence_metadata

def run_training_and_eval(data_dir):
    # 1. Prepare Data
    chunks_df = materialize_chunks(data_dir)
    if chunks_df.empty:
        logging.error("No chunks found. Aborting.")
        return

    chunks = chunks_df.to_dict(orient='records')
    logging.info(f"Building Index with {len(chunks)} chunks...")
    index = HybridIndex(chunks)

    validator = EvidenceValidator()
    classifier = LogicClassifier() # Helper for extraction
    claim_extractor = ClaimExtractor()

    train_file = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_file)

    # 2. Extract Features
    logging.info("Extracting features for training (aggregated per backstory)...")
    X = []
    y = []

    for i, row in train_df.iterrows():
        label_str = row.get('label', 'consistent')
        label = 1 if str(label_str).lower().strip() == 'consistent' else 0

        claim_features_list, _ = process_single_row(row, index, validator, classifier, claim_extractor)

        if not claim_features_list:
            agg_features = [0.0] * 6 # Fallback size
        else:
            matrix = np.array(claim_features_list)
            # Feature def: [max_contra, max_entail, max_neutral, top_retrieval, mean_retrieval, max_overlap]

            global_max_contra = np.max(matrix[:, 0])
            global_max_entail = np.max(matrix[:, 1])
            global_max_retrieval = np.max(matrix[:, 3])
            global_max_overlap = np.max(matrix[:, 5])
            num_bad_claims = np.sum(matrix[:, 0] > 0.35)

            agg_features = [global_max_contra, global_max_entail, global_max_retrieval, global_max_overlap, num_bad_claims, len(claim_features_list)]

        X.append(agg_features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # 3. CV Training
    logging.info("Running Cross-Validation on Aggregated Features...")
    mean_acc, std_acc = classifier.train_cv(X, y, cv=5)
    logging.info(f"Cross-Validation Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")

    # 4. Final Training
    classifier.train(X, y)

    # 5. Predict on Test
    test_file = os.path.join(data_dir, "test.csv")
    if os.path.exists(test_file):
        logging.info("Predicting on test.csv...")
        test_df = pd.read_csv(test_file)
        results = []

        for i, row in test_df.iterrows():
            claim_features_list, evidence_metadata = process_single_row(row, index, validator, classifier, claim_extractor)

            if not claim_features_list:
                agg_features = [0.0] * 6
                final_prob = 0.0 # Unknown
            else:
                matrix = np.array(claim_features_list)
                global_max_contra = np.max(matrix[:, 0])
                global_max_entail = np.max(matrix[:, 1])
                global_max_retrieval = np.max(matrix[:, 3])
                global_max_overlap = np.max(matrix[:, 5])
                num_bad_claims = np.sum(matrix[:, 0] > 0.35)
                agg_features = [global_max_contra, global_max_entail, global_max_retrieval, global_max_overlap, num_bad_claims, len(claim_features_list)]

            # Predict
            pred = classifier.predict([agg_features])[0]

            # Rationale Generation
            rationale_text = "Consistent with canon."

            if pred == 0 and claim_features_list:
                # Find the "Weakest Link" claim
                matrix = np.array(claim_features_list)
                bad_claim_idx = np.argmax(matrix[:, 0])
                score = matrix[bad_claim_idx, 0] # Max Contra Score

                content = row.get('content', '') or row.get('backstory', '')
                claims = claim_extractor.extract_claims(content)
                bad_claim = claims[bad_claim_idx]

                ev_meta = evidence_metadata[bad_claim_idx]
                if ev_meta:
                    chunk_text = ev_meta['chunk_text']
                    chapter = ev_meta['chapter_title']

                    rationale_text = (f"The system identified a high-probability causal break (Score: {score:.2f}). "
                                      f"While the backstory claims '{bad_claim[:50]}...', "
                                      f"the novel's internal state in '{chapter}' establishes '{chunk_text[:100]}...'. "
                                      f"These states are mutually exclusive in a 19th-century physical reality.")
                else:
                    rationale_text = f"Claim '{bad_claim[:50]}...' flagged as contradictory (Score: {score:.2f}) but evidence context is missing."

            results.append({
                'id': row.get('id', i),
                'label': int(pred),
                'rationale': rationale_text
            })

        res_df = pd.DataFrame(results)
        res_df.to_csv("submission.csv", index=False)
        logging.info("Saved predictions to submission.csv")

def main():
    parser = argparse.ArgumentParser(description="KDSH 2026 Track A Runner")
    parser.add_argument("--mode", type=str, default="train_eval", choices=["train_eval", "smoke_test", "evaluate"], help="Mode to run the app")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    if args.mode == "train_eval":
        run_training_and_eval(data_dir)
    elif args.mode == "smoke_test":
        run_training_and_eval(data_dir)
    elif args.mode == "evaluate":
        run_training_and_eval(data_dir)

if __name__ == "__main__":
    main()
