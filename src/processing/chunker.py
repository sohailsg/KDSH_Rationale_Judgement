import pathway as pw
import hashlib
import re

def chunk_documents(documents, chunk_size=2000, overlap=200):
    """
    Chunks the documents using a sliding window strategy.
    Updated to target ~400-600 tokens (approx 2000 chars).

    Args:
        documents: A Pathway Table with columns [novel_id, text, path]
        chunk_size: Target size of each chunk in characters. Default 2000 (~500 tokens).
        overlap: Overlap between chunks in characters.
    """
    
    @pw.udf
    def chunk_text_with_metadata(text):
        chunks = []
        n = len(text)
        if n == 0:
            return []

        step = chunk_size - overlap
        if step <= 0:
            step = 1

        for i in range(0, n, step):
            chunk_content = text[i : i + chunk_size]
            # Simple metadata extraction: check for "Chapter" at start or inside
            # We look for the *last* chapter header seen so far in the text flow, or inside this chunk.
            # Ideally stateful, but for parallel chunks we just look inside.
            chapter_match = re.search(r'(Chapter\s+\d+|[IVXLCDM]+\.)', chunk_content, re.IGNORECASE)
            chapter_title = chapter_match.group(0) if chapter_match else None

            chunks.append((chunk_content, chapter_title))
        return chunks

    chunked = documents.select(
        chunks=chunk_text_with_metadata(pw.this.text),
        novel_id=pw.this.novel_id,
        path=pw.this.path
    )
    
    chunked = chunked.flatten(pw.this.chunks)

    # Unpack the tuple (chunk_text, chapter_title)
    chunked = chunked.select(
        novel_id=pw.this.novel_id,
        path=pw.this.path,
        text=pw.this.chunks[0],
        chapter_title=pw.this.chunks[1]
    )

    @pw.udf
    def compute_hash(text, novel_id):
        return hashlib.md5(f"{novel_id}:{text}".encode()).hexdigest()

    # Add chunk_id
    chunked = chunked.select(
        novel_id=pw.this.novel_id,
        text=pw.this.text,
        path=pw.this.path,
        chapter_title=pw.this.chapter_title,
        chunk_id=compute_hash(pw.this.text, pw.this.novel_id)
    )
    
    return chunked

if __name__ == '__main__':
    import pandas as pd
    # Example Usage
    class Document(pw.Schema):
        novel_id: str
        text: str
        path: str

    # Create a dummy table for demonstration
    documents = pw.Table.from_pandas(
        pd.DataFrame({
            'novel_id': ['Test Book'],
            'text': ['Chapter 1. This is a long text that needs to be chunked. ' * 100],
            'path': ['/fake/path.txt']
        }),
        schema=Document
    )
    
    chunked_data = chunk_documents(documents, chunk_size=2000, overlap=200)
    
    pw.debug.compute_and_print(chunked_data)
