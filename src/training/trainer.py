
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
