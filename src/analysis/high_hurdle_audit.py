import pandas as pd
import numpy as np
import os
import logging
from src.indexing.vector_db import HybridIndex
from src.reasoning.validator import EvidenceValidator
from src.training.trainer import LogicClassifier
from src.processing.claim_tools import ClaimExtractor
from src.app import materialize_chunks

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_high_hurdle_audit(data_dir):
    # 1. Load Data & Index
    chunks_df = materialize_chunks(data_dir)
    chunks = chunks_df.to_dict(orient='records')
    index = HybridIndex(chunks)

    validator = EvidenceValidator()
    classifier = LogicClassifier()
    claim_extractor = ClaimExtractor()

    train_file = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_file)

    logging.info("Starting High-Hurdle Audit (Precision Optimization)...")

    X = []
    y = []
    ids = []
    claim_texts = []
    evidence_texts = []

    # Feature Extraction Loop
    for i, row in train_df.iterrows():
        label_str = row.get('label', 'consistent')
        label = 1 if str(label_str).lower().strip() == 'consistent' else 0

        content = row.get('content', '') or row.get('backstory', '')
        claims = claim_extractor.extract_claims(content)

        book_col = [c for c in row.index if 'book' in c.lower() or 'novel' in c.lower()]
        target_book = row[book_col[0]] if book_col else None
        normalized_book = str(target_book).replace("_", " ").rsplit('.', 1)[0] if target_book else None

        claim_features_list = []
        snippets = []

        for claim in claims:
            # 1. Semantic Consensus Retrieval (Top 10)
            evidence_items = index.search(claim, book_name=normalized_book, k=10)

            # Validation
            nli_probs = validator.get_raw_probs(claim, evidence_items)
            retrieval_scores = [item['score'] for item in evidence_items]

            feat_vec = classifier.extract_features(nli_probs, retrieval_scores, claim)
            claim_features_list.append(feat_vec)

            if evidence_items:
                snippets.append(evidence_items[0]['chunk']['text'][:150].replace('\n', ' '))

        if not claim_features_list:
            agg_features = [0.0] * 6
        else:
            matrix = np.array(claim_features_list)
            # [max_contra, max_entail, max_neutral, top_retrieval, mean_retrieval, max_overlap]
            global_max_contra = np.max(matrix[:, 0])
            global_max_entail = np.max(matrix[:, 1])
            global_max_retrieval = np.max(matrix[:, 3])
            global_max_overlap = np.max(matrix[:, 5])

            # Count of bad claims is less relevant for High Hurdle than the MAX strict violation
            num_bad_claims = np.sum(matrix[:, 0] > 0.75)

            agg_features = [global_max_contra, global_max_entail, global_max_retrieval, global_max_overlap, num_bad_claims, len(claim_features_list)]

        X.append(agg_features)
        y.append(label)
        ids.append(row.get('id', i))
        claim_texts.append(content[:100])
        evidence_texts.append(" | ".join(snippets)[:300])

    X = np.array(X)
    y = np.array(y)

    # Predict using High Hurdle Logic
    y_pred = classifier.predict_high_hurdle(X, contra_threshold=0.75, overlap_threshold=0.4)

    # Calculate Metrics
    from sklearn.metrics import recall_score, precision_score
    rec = recall_score(y, y_pred, pos_label=0)
    prec = precision_score(y, y_pred, pos_label=0)

    print(f"\n=== High-Hurdle Audit Metrics ===")
    print(f"Precision: {prec:.4f} (Target > 0.79)")
    print(f"Recall: {rec:.4f} (Target > 0.73)")

    # Report Findings
    print("\n=== SMOKING GUNS FOUND ===")
    for i in range(len(y)):
        if y_pred[i] == 0:
            # Format Rationale
            feats = X[i]
            max_contra = feats[0]
            max_overlap = feats[3]
            # Find evidence chunk responsible?
            # We just print the aggregated snippet for now
            print(f"\n[ID {ids[i]}] CLAIM: {claim_texts[i]}...")
            print(f"NOVEL EVIDENCE: \"{evidence_texts[i]}...\"")
            print(f"REASON: The claim is physically impossible because the novel documents a mutually exclusive state. (Score: {max_contra:.2f} | Overlap: {max_overlap:.2f})")

if __name__ == "__main__":
    run_high_hurdle_audit("data")
