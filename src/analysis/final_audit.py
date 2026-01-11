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

def generate_shadow_queries(claim):
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
        'captain': ['mutiny', 'disgraced', 'demoted']
    }

    for state, opposites in state_opposites.items():
        if state in text_lower:
            queries.append(" ".join(opposites))

    if "never" in text_lower:
        queries.append(text_lower.replace("never", ""))

    return queries

def run_final_audit(data_dir):
    # 1. Load Data & Index
    chunks_df = materialize_chunks(data_dir)
    chunks = chunks_df.to_dict(orient='records')
    index = HybridIndex(chunks)

    validator = EvidenceValidator()
    classifier = LogicClassifier()
    claim_extractor = ClaimExtractor()

    train_file = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_file)

    logging.info("Starting Chief Forensic Audit (High Precision Logic)...")

    X = []
    y = []
    ids = []
    claim_texts = []
    evidence_texts = []

    for i, row in train_df.iterrows():
        label_str = row.get('label', 'consistent')
        label = 1 if str(label_str).lower().strip() == 'consistent' else 0

        content = row.get('content', '') or row.get('backstory', '')
        claims = claim_extractor.extract_claims(content)
        dates = claim_extractor.extract_dates(content)

        # Entity Check for "Anchor" verification
        row_entities = claim_extractor.extract_entities(content)

        book_col = [c for c in row.index if 'book' in c.lower() or 'novel' in c.lower()]
        target_book = row[book_col[0]] if book_col else None
        normalized_book = str(target_book).replace("_", " ").rsplit('.', 1)[0] if target_book else None

        claim_features_list = []
        snippets = []

        for claim in claims:
            # 1. Standard Retrieval
            evidence_items = index.search(claim, book_name=normalized_book, k=10)

            # 2. Date Search
            for date in dates:
                date_ev = index.search(date, book_name=normalized_book, k=3)
                for item in date_ev:
                    if not any(e['chunk'].get('chunk_id') == item['chunk'].get('chunk_id') for e in evidence_items):
                        evidence_items.append(item)

            # 3. Shadow Query
            shadows = generate_shadow_queries(claim)
            for q in shadows:
                shadow_ev = index.search(q, book_name=normalized_book, k=3)
                for item in shadow_ev:
                    if not any(e['chunk'].get('chunk_id') == item['chunk'].get('chunk_id') for e in evidence_items):
                        evidence_items.append(item)

            # Step 1: Subject-Predicate Alignment Filter
            # Only keep chunks that have Keyword Overlap > 0.1? Or rely on NLI?
            # We filter for NLI input to ensure agency matching
            # Actually, NLI handles "entity mismatch" as Neutral usually.

            nli_probs = validator.get_raw_probs(claim, evidence_items)
            retrieval_scores = [item['score'] for item in evidence_items]

            # Step 3: Pathway Consensus Reranking (2-of-3 Rule)
            # Count how many chunks have Contradiction > 0.7
            contra_scores = [p[0] for p in [x['probs'] for x in nli_probs]]
            high_contra_count = sum(1 for s in contra_scores if s > 0.7)

            # Extract features for classifier (still useful)
            feat_vec = classifier.extract_features(nli_probs, retrieval_scores, claim)

            # Inject "Consensus Count" into features?
            # We'll use custom logic later, but let's store features.
            claim_features_list.append({
                'vec': feat_vec,
                'contra_scores': contra_scores,
                'dates_present': len(dates) > 0,
                'high_contra_count': high_contra_count
            })

            if evidence_items:
                snippets.append(evidence_items[0]['chunk']['text'][:150].replace('\n', ' '))

        # Decision Logic per Row
        final_pred = 1 # Default Consistent
        max_score = 0.0

        for cf in claim_features_list:
            max_c = max(cf['contra_scores']) if cf['contra_scores'] else 0
            if max_c > max_score:
                max_score = max_c

            # Step 4: Final Inference Calibration
            # Hard Threshold: > 0.92
            if max_c > 0.92:
                final_pred = 0
            # Soft Threshold: > 0.70 IF Date Match
            elif max_c > 0.70 and cf['dates_present']:
                final_pred = 0
            # Consensus Rule: At least 2 chunks > 0.7
            elif cf['high_contra_count'] >= 2:
                final_pred = 0

        # Store for Metrics
        # We dummy X here because we used custom logic
        X.append([max_score])
        y.append(label)
        ids.append(row.get('id', i))
        claim_texts.append(content[:100])
        evidence_texts.append(" | ".join(snippets)[:300])

        # Store prediction in a list to calculate metrics manually
        # (We bypass the classifier training for this "Chief Architect" logic run)

    # Metrics Calculation
    # We have y (actual) and we need y_pred
    # Wait, I calculated final_pred inside loop but didn't store it.
    # Let's fix loop to store y_pred

    return y, ids, claim_texts, evidence_texts # Refactor to return data, then we loop again?
    # No, let's rewrite the loop cleanly.

# ... Redefining run_final_audit to be self-contained ...

def run_final_audit_v2(data_dir):
    chunks_df = materialize_chunks(data_dir)
    chunks = chunks_df.to_dict(orient='records')
    index = HybridIndex(chunks)
    validator = EvidenceValidator()
    claim_extractor = ClaimExtractor()
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

    y_true = []
    y_pred = []

    print("\n=== CHIEF FORENSIC SYSTEMS AUDIT (v2) ===")

    for i, row in train_df.iterrows():
        label_str = row.get('label', 'consistent')
        actual = 1 if str(label_str).lower().strip() == 'consistent' else 0
        y_true.append(actual)

        content = row.get('content', '') or row.get('backstory', '')
        claims = claim_extractor.extract_claims(content)
        dates = claim_extractor.extract_dates(content)

        # Entity Check for "Anchor" verification
        row_entities = claim_extractor.extract_entities(content)

        book_col = [c for c in row.index if 'book' in c.lower() or 'novel' in c.lower()]
        target_book = row[book_col[0]] if book_col else None
        normalized_book = str(target_book).replace("_", " ").rsplit('.', 1)[0] if target_book else None

        row_pred = 1 # Default Consistent
        reason = ""

        for claim in claims:
            # Re-extract entities per claim for finer granularity?
            # Or use row_entities (which are whole backstory).
            # Let's use row_entities to ensure broad coverage.

            # Retrieval
            evidence_items = index.search(claim, book_name=normalized_book, k=10)
            for date in dates:
                date_ev = index.search(date, book_name=normalized_book, k=3)
                for item in date_ev:
                    if not any(e['chunk'].get('chunk_id') == item['chunk'].get('chunk_id') for e in evidence_items):
                        evidence_items.append(item)

            # NLI
            nli_probs = validator.get_raw_probs(claim, evidence_items)
            contra_scores = [p[0] for p in [x['probs'] for x in nli_probs]]

            if not contra_scores: continue

            max_c = max(contra_scores)
            best_chunk_idx = contra_scores.index(max_c)
            best_chunk_text = evidence_items[best_chunk_idx]['chunk']['text']

            # Filter consensus by overlap to improve Precision
            # extract features per chunk is hard here, we do it in bulk
            # We will approximate overlap for consensus: require high score
            high_count = sum(1 for s in contra_scores if s > 0.75)

            # ENTITY ANCHOR CHECK
            # Does the contradictory evidence actually talk about the people/places in the claim?
            has_anchor = False
            if not row_entities:
                has_anchor = True # No entities to check, assume valid
            else:
                for ent in row_entities:
                    if ent in best_chunk_text:
                        has_anchor = True
                        break

            # ADAPTIVE LOGIC GATES
            # If we have an entity match, we trust lower NLI scores.
            # If we don't, we require extreme confidence.

            threshold = 0.92 if has_anchor else 0.97

            # 1. Hard Threshold
            if max_c > threshold:
                row_pred = 0
                reason = f"Hard Lock (>{threshold}): {max_c:.2f}" + (" [Anchor]" if has_anchor else " [NoAnchor]")
                break # Weakest Link found

            # 2. Soft Threshold + Date
            if max_c > 0.70 and dates and has_anchor:
                row_pred = 0
                reason = f"Date Conflict (>0.70): {max_c:.2f}"
                break

            # 3. Consensus (3 chunks > 0.75) - Tighter for Precision
            if high_count >= 3:
                row_pred = 0
                reason = f"Consensus ({high_count} chunks > 0.75)"
                break

        y_pred.append(row_pred)

        if row_pred == 0 and actual == 0:
            print(f"[TP] ID {row.get('id')}: {reason}")
        elif row_pred == 0 and actual == 1:
            print(f"[FP] ID {row.get('id')}: {reason}")

    # Metrics
    from sklearn.metrics import precision_score, recall_score
    rec = recall_score(y_true, y_pred, pos_label=0)
    prec = precision_score(y_true, y_pred, pos_label=0)

    print(f"\n=== Final Physics Engine Metrics ===")
    print(f"Recall: {rec:.4f} (Target 0.73-0.78)")
    print(f"Precision: {prec:.4f} (Target > 0.85)")

if __name__ == "__main__":
    run_final_audit_v2("data")
