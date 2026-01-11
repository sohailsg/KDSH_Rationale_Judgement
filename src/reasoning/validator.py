from sentence_transformers import CrossEncoder
import torch

class EvidenceValidator:
    """
    Validates claims against evidence using an NLI model.
    """
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-large"):
        # Use the requested large model for higher accuracy
        self.model = CrossEncoder(model_name)
        # Deberta-v3-large NLI labels are typically: 0: Contradiction, 1: Entailment, 2: Neutral
        self.label_mapping = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}

    def get_raw_probs(self, claim, evidence_items):
        """
        Returns raw NLI probabilities for all evidence items.
        Returns: List of dicts {'probs': [contra, entail, neutral], 'text': str}
        """
        if not evidence_items:
            return []

        pairs = [[claim, item['chunk']['text']] for item in evidence_items]
        scores = self.model.predict(pairs)
        probs = torch.nn.functional.softmax(torch.tensor(scores), dim=1).numpy()

        results = []
        for i, prob in enumerate(probs):
            results.append({
                'probs': prob.tolist(), # [contra, entail, neutral]
                'text': evidence_items[i]['chunk']['text']
            })
        return results

    def validate(self, claim, evidence_items, threshold=0.5):
        """
        Evaluates the claim against a list of evidence items.
        (Retained for backward compatibility/smoke tests)
        """
        # ... logic unchanged for basic validation ...
        # (For training, we will use get_raw_probs + LogicClassifier)
        # But let's keep the existing logic as a fallback
        if not evidence_items:
            return {
                'label': 1,
                'rationale': "No evidence found to verify or refute.",
                'confidence': 0.0,
                'max_entailment': 0.0,
                'max_contradiction': 0.0
            }

        probs_data = self.get_raw_probs(claim, evidence_items)

        max_contradiction = 0.0
        max_entailment = 0.0
        best_contra_idx = -1
        best_entail_idx = -1

        for i, item in enumerate(probs_data):
            prob = item['probs']
            p_contra = prob[0]
            p_entail = prob[1]
            p_neutral = prob[2]

            if p_contra > max_contradiction:
                max_contradiction = p_contra
                best_contra_idx = i

            if p_entail > max_entailment:
                max_entailment = p_entail
                best_entail_idx = i

        threshold_contra = 0.8
        threshold_entail = 0.5

        label = 1
        rationale = "No strong evidence found."
        confidence = 0.0

        if max_contradiction > threshold_contra:
            label = 0
            chunk_text = evidence_items[best_contra_idx]['chunk']['text']
            rationale = f"Contradicted by evidence: '{chunk_text[:200]}...'"
            confidence = max_contradiction
        elif max_entailment > threshold_entail:
            label = 1
            chunk_text = evidence_items[best_entail_idx]['chunk']['text']
            rationale = f"Supported by evidence: '{chunk_text[:200]}...'"
            confidence = max_entailment
        else:
            label = 1
            if best_entail_idx >= 0:
                chunk_text = evidence_items[best_entail_idx]['chunk']['text']
                rationale = f"Consistent (Neutral/Weak Entailment): '{chunk_text[:200]}...'"
                confidence = probs_data[best_entail_idx]['probs'][2] # Neutral score
            elif best_contra_idx >= 0:
                 chunk_text = evidence_items[best_contra_idx]['chunk']['text']
                 rationale = f"Consistent (Weak Contradiction ignored): '{chunk_text[:200]}...'"

        return {
            'label': label,
            'rationale': rationale,
            'confidence': float(confidence),
            'max_entailment': float(max_entailment),
            'max_contradiction': float(max_contradiction)
        }

if __name__ == "__main__":
    val = EvidenceValidator()
    # Test
    claim = "The sky is green."
    evidence = [{'chunk': {'text': "The sky is blue today."}}]
    print(val.validate(claim, evidence))
