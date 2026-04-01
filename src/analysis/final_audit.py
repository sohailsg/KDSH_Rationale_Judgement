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

def generate_shadow_queries(claim, subject_name):
    """
    Generates 'Shadow Queries' searching for the Physical Opposite.
    Joins Subject Name + Opposite Keyword for strict retrieval.
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

        # Subject extraction from 'char' column
        subject_name = str(row.get('char', '')).strip()
        # Split subject into tokens for flexible matching (e.g. "Edmond Dantes" -> "Edmond", "Dantes")
        subject_tokens = [t for t in subject_name.split() if len(t) > 2] if subject_name else []

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

        # Subject extraction from 'char' column
        subject_name = str(row.get('char', '')).strip()
        # Split subject into tokens for flexible matching (e.g. "Edmond Dantes" -> "Edmond", "Dantes")
        subject_tokens = [t for t in subject_name.split() if len(t) > 2] if subject_name else []

        book_col = [c for c in row.index if 'book' in c.lower() or 'novel' in c.lower()]
        target_book = row[book_col[0]] if book_col else None
        normalized_book = str(target_book).replace("_", " ").rsplit('.', 1)[0] if target_book else None

        row_pred = 1 # Default Consistent
        reason = ""

        for claim in claims:
            # Step 1: Pardon Rule (Skip Soft Traits)
            claim_type = claim_extractor.classify_claim(claim)
            if claim_type == 'Soft':
                continue # Pardon

            # Step 2: Negative Space Retrieval
            # Strict Shadow Query
            shadow_queries = generate_shadow_queries(claim, subject_name)

            evidence_items = index.search(claim, book_name=normalized_book, k=10)

            # Add Shadow results
            for q in shadow_queries:
                shadow_ev = index.search(q, book_name=normalized_book, k=5)
                for item in shadow_ev:
                    if not any(e['chunk'].get('chunk_id') == item['chunk'].get('chunk_id') for e in evidence_items):
                        evidence_items.append(item)

            # Step 3: The Exclusionary Jury

            # Filter Evidence: Identity Gate
            # We don't strictly discard, but we tag "Anchored" evidence.
            anchored_evidence = []

            # Combine Subject Name tokens and Claim Entities
            # anchor_tokens = set(subject_tokens + row_entities) # row_entities not passed here?
            # We need to re-extract or pass row_entities. Let's re-extract strictly here.

            # Check for Subject Name OR Entities in the text
            anchors = subject_tokens + claim_extractor.extract_entities(claim)

            # NLI
            nli_probs = validator.get_raw_probs(claim, evidence_items)
            if not nli_probs: continue

            contra_scores = [p[0] for p in [x['probs'] for x in nli_probs]]
            max_c = max(contra_scores)
            best_idx = contra_scores.index(max_c)
            best_item = evidence_items[best_idx]
            best_chunk_id = best_item['chunk'].get('chunk_id', 'Unknown')
            best_text = best_item['chunk']['text']

            # Check anchor in best text
            has_anchor = any(t in best_text for t in anchors) if anchors else False

            # Consensus Count
            high_count = sum(1 for s in contra_scores if s > 0.80)

            # LOGIC GATES

            # 1. Consensus Rule (High Recall Safety)
            if high_count >= 3:
                row_pred = 0
                reason = (f"PHYSICAL COLLISION VERIFIED (Consensus {high_count}): Backstory Anchor [{claim[:30]}...] "
                          f"contradicted by multiple sources. Best Chunk {best_chunk_id}. Score: {max_c:.2f}")
                break

            # 2. Adaptive Single-Shot Rule
            threshold = 0.90 if has_anchor else 0.98

            if max_c > threshold:
                row_pred = 0
                anchor_tag = "[Anchor Verified]" if has_anchor else "[No Anchor]"
                reason = (f"PHYSICAL COLLISION VERIFIED {anchor_tag}: Backstory Anchor [{claim[:30]}...] "
                          f"requires State A. However, Novel (Chunk {best_chunk_id}) "
                          f"provides verbatim evidence of State B. Score: {max_c:.2f}")
                break

        y_pred.append(row_pred)

        if row_pred == 0 and actual == 0:
            print(f"[TP] ID {row.get('id')}: {reason}")
        elif row_pred == 0 and actual == 1:
            print(f"[FP] ID {row.get('id')}: {reason}")

    # Metrics
    from sklearn.metrics import precision_score, recall_score, confusion_matrix
    rec = recall_score(y_true, y_pred, pos_label=0)
    prec = precision_score(y_true, y_pred, pos_label=0)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0]) # 1=Consistent (Neg), 0=Inconsistent (Pos)

    print(f"\n=== Final Physics Engine Metrics ===")
    print(f"Confusion Matrix (Labels: [Consistent, Inconsistent]):\n{cm}")
    print(f"Recall (Sensitivity): {rec:.4f}")
    print(f"Precision (PPV): {prec:.4f}")

if __name__ == "__main__":
    run_final_audit_v2("data")
