
import pytest
import pandas as pd
import pathway as pw
from src.processing.chunker import chunk_documents

def test_chunking_logic():
    # Setup dummy data
    text = "Chapter 1. " + "A" * 200 + " Chapter 2. " + "B" * 200
    df = pd.DataFrame({
        'novel_id': ['TestBook'],
        'text': [text],
        'path': ['dummy.txt']
    })

    class Document(pw.Schema):
        novel_id: str
        text: str
        path: str

    # Use pw.debug.table_from_pandas
    table = pw.debug.table_from_pandas(df) # Schema inference usually works, or pass schema

    # Run chunker with small window to force splits
    chunked_table = chunk_documents(table, chunk_size=100, overlap=10)

    assert isinstance(chunked_table, pw.Table)

    print("Chunking definition successful.")

if __name__ == "__main__":
    test_chunking_logic()
