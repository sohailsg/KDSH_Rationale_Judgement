import pathway as pw
import hashlib
import re

def chunk_documents(documents, chunk_size=1000, overlap=200):
    """
    Chunks the documents using a sliding window strategy.

    Args:
        documents: A Pathway Table with columns [novel_id, text, path]
        chunk_size: Target size of each chunk (in characters/words, here roughly characters for simplicity)
        overlap: Overlap between chunks
    """
    
    @pw.udf
    def chunk_text_with_metadata(text):
        # Simple character-based sliding window
        # For "hierarchical" or more advanced chunking, one would parse chapters first.
        # But per requirements: "target ~1000 tokens with ~200 overlap".
        # Assuming ~4 chars per token, 1000 tokens ~ 4000 chars.
        # But let's stick to the parameter interpretation. If it means tokens, we should approximate.
        # Let's assume the input `chunk_size` is in characters for this simple baseline if not specified.
        # "target ~1000 tokens" -> let's default to 4000 chars if not specified.
        # But keeping the default 1000 passed in, if interpreted as tokens, user should pass 4000.
        # I will use the passed parameters directly.

        chunks = []
        n = len(text)
        if n == 0:
            return []

        step = chunk_size - overlap
        if step <= 0:
            step = 1

        for i in range(0, n, step):
            chunk_content = text[i : i + chunk_size]
            # Simple metadata extraction: check for "Chapter"
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
            'text': ['Chapter 1. This is a long text that needs to be chunked. ' * 50],
            'path': ['/fake/path.txt']
        }),
        schema=Document
    )
    
    chunked_data = chunk_documents(documents, chunk_size=100, overlap=20)
    
    pw.debug.compute_and_print(chunked_data)
