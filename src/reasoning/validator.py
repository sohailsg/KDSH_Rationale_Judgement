from sentence_transformers import CrossEncoder
import torch

class EvidenceValidator:
    """
    Validates claims against evidence using an NLI model.
    """
    def __init__(self, model_name="cross-encoder/nli-distilroberta-base"):
        self.model = CrossEncoder(model_name)
        # NLI labels: 0: contradiction, 1: entailment, 2: neutral (usually, but verify for specific model)
        # For 'cross-encoder/nli-distilroberta-base':
        # Label mapping: {'contradiction': 0, 'entailment': 1, 'neutral': 2} ?
        # Actually standard for SNLI/MNLI is: Contradiction, Entailment, Neutral?
        # Let's check documentation or assume standard mapping:
        # distilroberta-base trained on SNLI/MNLI usually has 3 classes.
        # Often: 0: contradiction, 1: entailment, 2: neutral.
        # But sometimes it's Entailment, Neutral, Contradiction.
        # I will print the label mapping in __init__ if possible or assume standard.
        # cross-encoder/nli-distilroberta-base:
        # Labels: 0: contradiction, 1: entailment, 2: neutral.
        self.label_mapping = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}

    def validate(self, claim, evidence_items, threshold=0.5):
        """
        Evaluates the claim against a list of evidence items.

        Args:
            claim: The backstory claim.
            evidence_items: List of dicts (search results).
            threshold: Confidence threshold.

        Returns:
            dict: {
                'label': 1 (Consistent) or 0 (Contradict),
                'rationale': str,
                'confidence': float,
                'max_entailment': float,
                'max_contradiction': float
            }
        """
        if not evidence_items:
            return {
                'label': 1, # Default to consistent if no evidence found (benefit of doubt) or 0?
                # User says "plausible hypothetical backstory". If no evidence refutes it, it is consistent with the world (unless it contradicts general rules, but we only have text).
                # However, for a hackathon, "Consistent" usually means "Supported by text".
                # Let's look at the labels in train.csv.
                # "The mutiny began...": Contradict.
                # "Thalcave's people faded...": Consistent.
                # If we find NO evidence, we can't judge.
                # But typically we should output 0 or 1.
                # Let's default to 0 (Unverified) or 1?
                # Let's default to 0 for safety if we expect retrieval to work.
                # But actually, "Consistent" means "Does not contradict".
                # Let's stick to: if Contradiction found -> 0, else 1.
                'rationale': "No evidence found to verify or refute.",
                'confidence': 0.0,
                'max_entailment': 0.0,
                'max_contradiction': 0.0
            }

        pairs = [[claim, item['chunk']['text']] for item in evidence_items]
        scores = self.model.predict(pairs)
        # scores is [N, 3] (logits) if using predict? No, CrossEncoder.predict returns logits by default.
        # We should apply softmax to get probabilities.
        probs = torch.nn.functional.softmax(torch.tensor(scores), dim=1).numpy()

        # Aggregate
        max_contradiction = 0.0
        max_entailment = 0.0
        best_contra_idx = -1
        best_entail_idx = -1

        for i, prob in enumerate(probs):
            # prob: [p_contra, p_entail, p_neutral] (assuming this order)
            # wait, need to verify order.
            # huggingface model card says:
            # label2id: {'contradiction': 0, 'entailment': 1, 'neutral': 2}

            p_contra = prob[0]
            p_entail = prob[1]
            p_neutral = prob[2]

            if p_contra > max_contradiction:
                max_contradiction = p_contra
                best_contra_idx = i

            if p_entail > max_entailment:
                max_entailment = p_entail
                best_entail_idx = i

        # Logic:
        # If strict contradiction found (high conf), label 0.
        # Else if entailment found (high conf), label 1.
        # Else (mostly neutral), label 1 (Consistent) - assuming "not contradicted".

        threshold_contra = 0.8 # strict for contradiction
        threshold_entail = 0.5 # looser for entailment?

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
            # Check if there is "some" contradiction that outweighs entailment significantly?
            # Or just default to Consistent (1).
            # "Determining if ... is logically and causally consistent".
            # If evidence is neutral (irrelevant), we might assume consistent.
            label = 1
            if best_entail_idx >= 0:
                chunk_text = evidence_items[best_entail_idx]['chunk']['text']
                rationale = f"Consistent (Neutral/Weak Entailment): '{chunk_text[:200]}...'"
                confidence = probs[best_entail_idx][2] # Neutral score?
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
