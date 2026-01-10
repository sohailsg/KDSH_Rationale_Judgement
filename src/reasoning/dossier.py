def format_dossier(claim, evidence_items, metadata=None):
    """
    Formats the dossier with the claim and the supporting evidence.

    Args:
        claim: The claim string (backstory).
        evidence_items: List of dicts containing search results (chunk, score, etc.).
        metadata: Optional dictionary of extra metadata (e.g., query params).

    Returns:
        A Markdown formatted string.
    """
    output = []
    output.append(f"# Dossier Evidence\n")
    if metadata:
        output.append(f"**Metadata**: {metadata}\n")

    output.append(f"## Claim (Backstory)\n")
    output.append(f"{claim}\n")

    output.append(f"## Retrieved Evidence\n")
    if not evidence_items:
        output.append("No evidence found matching the criteria.\n")
    else:
        for i, item in enumerate(evidence_items, 1):
            chunk = item['chunk']
            output.append(f"### Evidence {i}\n")
            output.append(f"- **Source**: {chunk.get('novel_id', 'Unknown')}")
            if chunk.get('chapter_title'):
                output.append(f", {chunk['chapter_title']}")
            output.append(f"\n")
            output.append(f"- **Score**: {item.get('score', 0.0):.4f} (Vector: {item.get('vector_score', 0.0):.4f}, BM25: {item.get('bm25_score', 0.0):.4f})\n")
            output.append(f"- **Excerpt**:\n")
            output.append(f"> {chunk.get('text', '').strip()}\n")
            output.append(f"\n---\n")

    return "".join(output)
