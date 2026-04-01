import pandas as pd
import os
import argparse
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report

def evaluate_results(results_path="results.csv", truth_path="data/train.csv"):
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return
    if not os.path.exists(truth_path):
        print(f"Error: Ground truth file not found at {truth_path}")
        return

    print(f"Loading results from {results_path}...")
    pred_df = pd.read_csv(results_path)

    print(f"Loading truth from {truth_path}...")
    truth_df = pd.read_csv(truth_path)

    # Merge on ID to ensure alignment
    # prediction col in results.csv is 'prediction' (0/1)
    # label col in train.csv is 'label' (consistent/contradict)

    merged = pd.merge(truth_df, pred_df, on='id', suffixes=('_true', '_pred'))

    if len(merged) == 0:
        print("Error: No overlapping IDs found between truth and results.")
        return

    # Convert truth labels to binary (1=Consistent, 0=Contradict)
    # Note: src/app.py logic: label = 1 if 'consistent' else 0
    y_true = merged['label'].apply(lambda x: 1 if str(x).lower().strip() == 'consistent' else 0)
    y_pred = merged['prediction']

    # Metrics
    # Pos Label = 0 (Contradiction/Inconsistent) for forensic auditing usually
    # But scikit-learn defaults to 1.
    # Let's align with the report:
    # Precision/Recall usually focus on the "Positive" detection of the anomaly (Contradiction = 0).
    # So pos_label=0.

    print("\n=== Forensic Audit Evaluation ===")
    print(f"Evaluated {len(merged)} samples.\n")

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tn, fp, fn, tp = cm.ravel()

    print("Confusion Matrix (Labels: [Consistent=1, Contradict=0]):")
    print(cm)
    print(f"  True Consistent  (TN): {tn}")
    print(f"  False Contradict (FP): {fp} (Type I Error)")
    print(f"  Missed Contradict(FN): {fn} (Type II Error)")
    print(f"  True Contradict  (TP): {tp}")

    # Calculate metrics for Contradiction (Class 0)
    prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    print("\nKey Performance Indicators (Targeting Contradictions):")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")

    print("\nFull Classification Report:")
    print(classification_report(y_true, y_pred, labels=[1, 0], target_names=['Consistent', 'Contradict'], digits=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Forensic Audit Results")
    parser.add_argument("--results", type=str, default="results.csv", help="Path to predictions CSV")
    parser.add_argument("--truth", type=str, default="data/train.csv", help="Path to ground truth CSV")
    args = parser.parse_args()

    evaluate_results(args.results, args.truth)
