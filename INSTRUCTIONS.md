# Forensic Narrative Auditor - Execution Instructions

This guide details how to set up, run, and evaluate the Forensic Narrative Auditor (Jules) system.

## 1. Prerequisites & Setup

Ensure you have a Python 3.10+ environment.

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Key libraries: `pathway`, `sentence-transformers`, `rank-bm25`, `pandas`, `scikit-learn`)*

2.  **Verify Directory Structure**:
    The system expects the following layout:
    ```
    .
    ├── data/
    │   ├── train.csv       # Backstories (Ground Truth)
    │   ├── test.csv        # Backstories (Evaluation Target - Optional)
    │   └── *.txt           # Full novels (e.g., "The Count of Monte Cristo.txt")
    ├── src/
    │   ├── app.py          # Main Execution Script
    │   ├── evaluate.py     # Scoring Script
    │   └── ...             # Internal modules
    └── results.csv         # Output (Generated)
    ```

## 2. Taking Input & Running the Pipeline

The system is designed as a single-command pipeline. It will ingest documents from `data/`, process the backstories, and generate predictions.

### Command
To run the full forensic audit:

```bash
python3 src/app.py
```

### Execution Logic
1.  **Ingestion**: Streaming reads of `.txt` novels and `.csv` backstories.
2.  **Indexing**: Builds a HybridIndex (Vector + BM25) in memory.
3.  **Forensic Engine**:
    *   Deconstructs backstories into "Hard Facts".
    *   Performs "Shadow Queries" (Negative Space Retrieval).
    *   Validates using `nli-deberta-v3-large`.
    *   Applies the "Jury Gate" (NLI > 0.85 OR Spatiotemporal Lock).
4.  **Output**: Writes findings to `results.csv`.

*Note: The process involves loading large models and indexing texts. It may take 1-2 minutes to initialize.*

## 3. Getting Output

Upon completion, checking `results.csv` will show the forensic determination for each claim.

**Format**:
*   `id`: Unique Identifier.
*   `prediction`: `1` (Consistent) or `0` (Contradiction).
*   `rationale`: A verbatim explanation of the "State Collision" if one was found.

**Example Rationale**:
> "STATE COLLISION: Claim [He died in 1815...] asserts [State A]. Novel Chapter [12] documents [State B] ('He was alive in 1820...'). Since [State A] and [State B] are causally exclusive..."

## 4. Evaluating Performance (Confusion Matrix)

To calculate the Precision, Recall, and Confusion Matrix (comparing `results.csv` against `data/train.csv` labels):

### Command
```bash
python3 src/evaluate.py
```

### Expected Output
This will print a detailed report to the console:

```text
=== Forensic Audit Evaluation ===
Evaluated 60 samples.

Confusion Matrix (Labels: [Consistent=1, Contradict=0]):
[[TN  FP]
 [FN  TP]]

Key Performance Indicators (Targeting Contradictions):
  Precision: 0.XXX
  Recall:    0.XXX
```

*   **Precision**: High precision indicates that when the system flags a contradiction, it is almost certainly a real error in the backstory.
*   **Recall**: High recall indicates the system is successfully catching most "Hard State Collisions."
