import pathway as pw
import hashlib

def chunk_documents(documents, chunk_size=1000, overlap=200):
    """
    Chunks the documents using a sliding window strategy.
    """
    
    def chunk_text(text):
        return [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size - overlap)
        ]

    chunked = documents.select(
        chunks=pw.this.text.apply(chunk_text)
    )
    
    chunked = chunked.flatten(pw.this.chunks)
    chunked = chunked.rename(chunk_text=pw.this.chunks)

    # Add metadata
    chunked = chunked.select(
        novel_id=pw.this.novel_id,
        text=pw.this.chunk_text,
        path=pw.this.path,
        chunk_id=pw.this.chunk_text.apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    )
    
    return chunked

if __name__ == '__main__':
    # Example Usage
    class Document(pw.Schema):
        novel_id: str
        text: str
        path: str

    # Create a dummy table for demonstration
    documents = pw.Table.from_pandas(
        pd.DataFrame({
            'novel_id': ['Test Book'],
            'text': ['This is a long text that needs to be chunked. ' * 200],
            'path': ['/fake/path.txt']
        }),
        schema=Document
    )
    
    chunked_data = chunk_documents(documents)
    
    pw.debug.compute_and_print(chunked_data)
