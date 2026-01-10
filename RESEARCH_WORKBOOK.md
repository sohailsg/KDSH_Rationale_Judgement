# Research Workbook: KDSH 2026 Track A – Systems Reasoning

**Role**: Jules, Technical Lead & Research Mentor
**Date**: January 2026
**Analogy**: *Think of this system not as a search engine, but as a detective building a case file. The novel is the crime scene; the backstory is a witness statement. Your job is to find the single drop of blood (the evidence) that proves the statement is a lie.*

---

## Section 1: The 1-Page Project Brief

### Problem Definition
We are solving a **binary classification problem** for systems reasoning.
*   **Input**: A full-length novel (100k+ words) and a hypothetical character backstory (text).
*   **Task**: Act as a factual auditor to determine if the backstory is **Consistent (1)** or **Contradictory (0)** with the established canon of the novel.
*   **Constraint**: The system must not guess. If evidence is missing, it is "Consistent" (Insufficient Evidence to Contradict), but explicit contradictions must be flagged with verbatim proof.

### Pathway Integration Procedure (The Pipeline)
We utilize the **Pathway** framework for its robust data processing capabilities:
1.  **Ingestion**:
    *   Use `pw.io.fs.read` to monitor the `data/` directory for `.txt` (novels) and `.csv` (backstories).
    *   This allows for real-time updates and streaming processing of large corpora.
2.  **Indexing**:
    *   Implement **Hierarchical Chunking**:
        *   Target: **400–600 token paragraphs**.
        *   Overlap: **50–100 tokens** to preserve context across boundaries.
        *   Metadata: Tag chunks with `book_name`, `chapter_id`, and detected entities.
3.  **Retrieval**:
    *   **Multi-Stage Retriever**:
        *   **BM25**: For specific entity names, dates, and rare terms (Keyword Search).
        *   **Vector Similarity**: For semantic matching of traits, themes, and paraphrased events (Semantic Search).
    *   The results are merged (Hybrid Search) to ensure high recall.

### Evaluation Focus
*   **Rewarded**:
    *   High Accuracy (>0.85).
    *   **Verbatim Evidence**: Citing the exact sentence from the book.
    *   **Logical Analysis**: Explaining *why* X refutes Y.
*   **Discouraged**:
    *   Black-box "trust me" RAG responses.
    *   Decisions based on single, out-of-context passages without cross-reference.
    *   Vague reasoning ("It feels wrong").

### Contradiction Taxonomy (5 Types)
1.  **Explicit Contradiction**: Direct factual denial (e.g., Claim: "Faria died at home." vs. Novel: "Faria died in the Château d’If.").
2.  **Timeline Inconsistency**: Events occurring in the impossible order (e.g., Claim: "Paganel met Captain Grant in 1860." vs. Novel: "Grant was shipwrecked in 1862.").
3.  **Motive/Character Inconsistency**: Personality traits clashing with established patterns (e.g., A sworn pacifist character suddenly committing cold-blooded murder without arc development).
4.  **World-Rule Violation**: Breaking the story’s internal logic or physics (e.g., Noirtier using a smartphone in 19th-century France).
5.  **Causal Chain Break**: Introducing new causes that render known downstream plot points impossible (e.g., Claim: "Dantès escaped in 1814." implies he was never in prison to meet Faria, breaking the entire plot).

---

## Section 2: The Templates Section

### Claim Extraction Checklist
Split backstories into **8–25 atomic, testable claims**. Use this checklist to categorize them:
*   [ ] **Identity**: Names, Titles, Lineage, Physical appearance.
*   [ ] **Formative Events**: Birthplace, Education, Traumas, Key meetings.
*   [ ] **Beliefs & Values**: Politics, Religion, Personal codes, Phobias.
*   [ ] **Skills & Abilities**: Languages spoken, Combat skills, Technical knowledge.
*   [ ] **World-Context**: Locations visited, Historical events witnessed, Items possessed.

### Evidence Dossier Template
For every finding, produce a Markdown entry following this structure:

```markdown
### Evidence Item [ID]
**Claim**: "[Insert atomic claim from backstory]"
**Verdict**: [Refuted / Consistent]

**Excerpts (Canon)**:
> "Insert direct verbatim passage from the novel here..."
> (Chapter X, Page Y)

**Analysis**:
[Concise explanation of the violation]
*   **Type**: [e.g., Type 1: Explicit Contradiction]
*   **Rationale**: The claim states X, but the text explicitly confirms Y. Because X and Y are mutually exclusive, the claim is false.
```

---

## Section 3: Project Guidance

### Consistency Checklist
To label a backstory as **Consistent (1)**, it must pass all layers:
1.  **Factual Alignment**: No direct fact (name, date, place) is contradicted.
2.  **Chronological Integrity**: The backstory fits within the "white space" of the timeline without overlapping known events.
3.  **Character State Consistency**: The character's knowledge, scars, and possessions at the start of the novel match the backstory's outcome.

### 10 FAQs for the Team
1.  **Do we use Track B's BDH architecture?**
    *   No. Track A focuses on *audit and reasoning*, not generation.
2.  **What if the book is silent on a claim?**
    *   It is **Consistent**. Absence of evidence is not evidence of absence.
3.  **Can we use outside knowledge (Wikipedia, history)?**
    *   No. Only the provided `.txt` novel files are canon.
4.  **How do we handle "fuzzy" matches?**
    *   Use the Logic Classifier. If the semantic distance is close but not exact, check for synonymy. If it changes the meaning (e.g., "blue eyes" vs "green eyes"), it's a contradiction.
5.  **Is a partial contradiction a 0 or 1?**
    *   A **0**. If *any* part of the backstory breaks the canon, the whole backstory is flagged as Contradictory (or at least that claim is invalid).
6.  **Does Pathway handle the PDF parsing?**
    *   We assume `.txt` input for now. If PDFs are provided, we add a parsing step before Ingestion.
7.  **What is the minimum confidence threshold?**
    *   We target **0.7**. Below that, flag as "Unverified" or fallback to Consistent.
8.  **How do we handle metaphor?**
    *   The NLI model should distinguish figurative language. If "He had a heart of stone" is matched against "He died of cardiac arrest", the system must discern the difference.
9.  **Should we optimize for speed or accuracy?**
    *   **Accuracy** is paramount. Evidence-grounding requires precision.
10. **What if the novel itself has contradictions?**
    *   Assume the "Narrator's Truth" or the most frequent mention is canon.

---

## Section 4: Open Questions List
*   **Scoring Weights**: What is the penalty for a False Positive vs. False Negative?
*   **Label Distribution**: Is the test set balanced (50/50) or skewed?
*   **Granularity**: Do we need to output the specific sentence index, or just the chapter?
*   **Logic Types**: We need more examples of "Type 5: Causal Chain Break" to train the NLI model effectively.
*   **Chunking Strategy**: Will 600 tokens be enough to capture long-range dependencies (e.g., a promise made in Chapter 1 kept in Chapter 50)?

---
*Created by Jules, Engineering Lead*
