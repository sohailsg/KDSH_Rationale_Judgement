
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import os
import re

class LogicClassifier:
    def __init__(self):
        # Gradient Boosting
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

    def compute_overlap(self, claim, evidence_text):
        if not claim or not evidence_text:
            return 0.0

        # Simple token set overlap
        # Filter for words > 3 chars to avoid 'the', 'and'
        claim_tokens = set(w.lower() for w in re.findall(r'\w+', claim) if len(w) > 3)
        ev_tokens = set(w.lower() for w in re.findall(r'\w+', evidence_text) if len(w) > 3)

        if not claim_tokens:
            return 0.0

        intersection = claim_tokens.intersection(ev_tokens)
        return len(intersection) / len(claim_tokens) # Recall-like metric

    def extract_features(self, nli_results, retrieval_scores, claim_text):
        if not nli_results:
            return [0.0] * 6

        contra_probs = [r['probs'][0] for r in nli_results]
        entail_probs = [r['probs'][1] for r in nli_results]
        neutral_probs = [r['probs'][2] for r in nli_results]

        max_contra = max(contra_probs) if contra_probs else 0
        max_entail = max(entail_probs) if entail_probs else 0
        max_neutral = max(neutral_probs) if neutral_probs else 0

        # Retrieval Scores
        top_retrieval = retrieval_scores[0] if retrieval_scores else 0
        mean_retrieval = np.mean(retrieval_scores) if retrieval_scores else 0

        # Overlap Feature
        # Compute max overlap among retrieved chunks
        overlaps = [self.compute_overlap(claim_text, r.get('text', '')) for r in nli_results]
        max_overlap = max(overlaps) if overlaps else 0.0

        return [max_contra, max_entail, max_neutral, top_retrieval, mean_retrieval, max_overlap]

    def train_cv(self, X, y, cv=5):
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=cv_splitter, scoring='accuracy')
        return scores.mean(), scores.std()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_strict(self, X, threshold=0.7):
        base_preds = self.model.predict(X)
        final_preds = []
        for i, pred in enumerate(base_preds):
            max_contra = X[i][0]
            if max_contra > threshold:
                final_preds.append(0)
            else:
                final_preds.append(pred)
        return np.array(final_preds)

    def predict_recall_oriented(self, X, contra_threshold=0.35, overlap_threshold=0.4):
        """
        Hybrid Threshold Logic for Balanced Precision/Recall.
        Goal: Precision > 0.79, Recall > 0.73.

        Logic:
        1. "Smoking Gun": If MaxContra > 0.85, force 0 (regardless of overlap).
        2. "Supported Contradiction": If MaxContra > 0.45 AND Overlap > 0.25, force 0.
        3. Else: Use model prediction (which might be 0 or 1).
        """
        base_preds = self.model.predict(X)
        final_preds = []

        for i, pred in enumerate(base_preds):
            max_contra = X[i][0]
            if X.shape[1] == 6:
                max_overlap = X[i][3] # Aggregated
            else:
                max_overlap = X[i][5] # Atomic

            # Tier 1: undeniable contradiction
            if max_contra > 0.85:
                final_preds.append(0)
            # Tier 2: strong contradiction with context
            elif max_contra > 0.45 and max_overlap > 0.25:
                final_preds.append(0)
            else:
                final_preds.append(pred)

        return np.array(final_preds)
