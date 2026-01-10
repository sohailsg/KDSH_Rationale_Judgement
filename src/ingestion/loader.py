import pathway as pw
import pandas as pd
import os
from datetime import datetime

def load_documents(data_dir):
    """
    Reads documents from the specified directory.
    - For .txt files, it reads the raw text content.
    - For .csv files, it reads the structured data.
    """
    return pw.io.fs.read(
        data_dir,
        format="binary",
        mode="streaming",
        with_metadata=True
    )

def normalize_book_name(path):
    """
    Extracts and normalizes the book name from the file path.
    """
    if not path:
        return "Unknown"

    # Handle both slash types just in case
    name = path.replace("\\", "/").split("/")[-1]

    # Remove extension
    name = name.rsplit(".", 1)[0]

    # Normalize: replace underscores with spaces
    return name.replace("_", " ")

@pw.udf
def extract_path(metadata) -> str:
    # Pathway Json objects need to be treated carefully or cast to dict if possible,
    # but pw.io.fs.read metadata is a specific Json column.
    # In newer pathway versions, metadata columns might be accessible directly or via dictionary access if it's a python dict.
    # But the error says 'Json' object has no attribute 'get'.
    # This implies 'metadata' is of type pw.Json.
    # To access fields in pw.Json, we might need to use [] operator or cast.
    # However, UDFs receive the underlying python object. If it receives a wrapper, that's tricky.
    # Usually UDF receives standard python types (dict, str, etc).
    # The error 'Json' object has no attribute 'get' suggests it might be receiving a custom Pathway Json object wrapper that behaves differently than dict.

    if metadata is None:
        return ""
    # Try dictionary access
    try:
        return metadata["path"]
    except:
        return str(metadata)

@pw.udf
def check_endswith(path) -> bool:
    if not path:
        return False
    return path.endswith(".txt")

@pw.udf
def decode_utf8(data) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore")
    return str(data)

def process_documents(documents):
    """
    Process raw document data into a structured format.
    - Decodes text from binary.
    - Handles different file types (.txt, .csv).
    - Creates a unified schema.
    """
    
    # Extract path from _metadata JSON column
    documents = documents.select(
        data=pw.this.data,
        path=pw.this._metadata["path"].as_str() # Correct way to access JSON fields in Pathway expressions
    )

    # Use @pw.udf with type hints to ensure Pathway knows the return type
    documents = documents.select(
        data=pw.this.data,
        path=pw.this.path,
        is_txt=check_endswith(pw.this.path)
    )

    texts = documents.filter(pw.this.is_txt)

    texts = texts.select(
        doc=decode_utf8(pw.this.data),
        path=pw.this.path
    )
    texts = texts.select(
        novel_id=pw.apply(normalize_book_name, pw.this.path),
        text=pw.this.doc,
        path=pw.this.path
    )

    return texts

if __name__ == "__main__":
    # This is an example of how to run the loader
    data_dir = "../data"
    documents = load_documents(data_dir)
    processed_docs = process_documents(documents)

    pw.debug.compute_and_print(processed_docs)
