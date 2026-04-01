import pandas as pd
import numpy as np
import os
import logging
from src.indexing.vector_db import HybridIndex
from src.reasoning.validator import EvidenceValidator
from src.training.trainer import LogicClassifier
from src.processing.claim_tools import ClaimExtractor
from src.app import materialize_chunks, process_single_row

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_error_analysis(data_dir):
    # 1. Load Data & Index
    chunks_df = materialize_chunks(data_dir)
    chunks = chunks_df.to_dict(orient='records')
    index = HybridIndex(chunks)

    validator = EvidenceValidator()
    classifier = LogicClassifier()
    claim_extractor = ClaimExtractor()

    train_file = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_file)

    # 2. Extract Features & Actuals
    X = []
    y = []
    ids = []
    rows_data = []

    logging.info("Extracting features for error analysis...")

    for i, row in train_df.iterrows():
        label_str = row.get('label', 'consistent')
        label = 1 if str(label_str).lower().strip() == 'consistent' else 0

        # We need to capture the raw evidence for debugging
        # Modifying process_single_row is hard without changing app.py,
        # so we will duplicate logic slightly to capture evidence text.

        content = row.get('content', '') or row.get('backstory', '')
        claims = claim_extractor.extract_claims(content)

        book_col = [c for c in row.index if 'book' in c.lower() or 'novel' in c.lower()]
        target_book = row[book_col[0]] if book_col else None
        normalized_book = str(target_book).replace("_", " ").rsplit('.', 1)[0] if target_book else None

        claim_features_list = []
        evidence_snippets = []

        for claim in claims:
            evidence_items = index.search(claim, book_name=normalized_book, k=5)
            nli_probs = validator.get_raw_probs(claim, evidence_items)
            retrieval_scores = [item['score'] for item in evidence_items]

            feat_vec = classifier.extract_features(nli_probs, retrieval_scores)
            claim_features_list.append(feat_vec)

            if evidence_items:
                evidence_snippets.append(evidence_items[0]['chunk']['text'][:100])

        if not claim_features_list:
            agg_features = [0.0] * 5
        else:
            matrix = np.array(claim_features_list)
            # [max_contra, max_entail, max_neutral, top_retrieval, mean_retrieval]
            global_max_contra = np.max(matrix[:, 0])
            global_max_entail = np.max(matrix[:, 1])
            global_max_retrieval = np.max(matrix[:, 3])
            num_bad_claims = np.sum(matrix[:, 0] > 0.5)
            agg_features = [global_max_contra, global_max_entail, global_max_retrieval, num_bad_claims, len(claim_features_list)]

        X.append(agg_features)
        y.append(label)
        ids.append(row.get('id', i))
        rows_data.append({
            'claim': content[:50],
            'evidence_snippet': " | ".join(evidence_snippets)[:200],
            'max_contra': agg_features[0],
            'max_entail': agg_features[1],
            'retrieval': agg_features[2]
        })

    X = np.array(X)
    y = np.array(y)

    # 3. Leave-One-Out Prediction (Approximation with CV)
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import GradientBoostingClassifier

    # We use the raw model from LogicClassifier logic
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    logging.info("Running CV Predictions...")
    y_pred = cross_val_predict(clf, X, y, cv=5)

    # 4. Identify Errors
    errors = []
    for i in range(len(y)):
        if y[i] != y_pred[i]:
            errors.append({
                'id': ids[i],
                'actual': y[i],
                'predicted': y_pred[i],
                'claim': rows_data[i]['claim'],
                'evidence': rows_data[i]['evidence_snippet'],
                'max_contra': rows_data[i]['max_contra'],
                'max_entail': rows_data[i]['max_entail'],
                'retrieval_score': rows_data[i]['retrieval']
            })

    errors_df = pd.DataFrame(errors)
    errors_df.to_csv("src/analysis/errors.csv", index=False)
    logging.info(f"Analysis complete. Found {len(errors)} errors. Saved to src/analysis/errors.csv")

    # Calculate Accuracy
    acc = 1 - (len(errors) / len(y))
    logging.info(f"Current Accuracy: {acc:.4f}")

if __name__ == "__main__":
    data_dir = "data"
    run_error_analysis(data_dir)
