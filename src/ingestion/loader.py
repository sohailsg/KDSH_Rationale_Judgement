import pathway as pw
import pandas as pd
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

def process_documents(documents):
    """
    Process raw document data into a structured format.
    - Decodes text from binary.
    - Handles different file types (.txt, .csv).
    - Creates a unified schema.
    """
    
    # Process text files
    texts = documents.filter(pw.this.path.endswith(".txt"))
    texts = texts.select(
        doc=pw.this.data.decode("utf-8", errors="ignore"),
        path=pw.this.path
    )
    texts = texts.select(
        novel_id=pw.this.path.split("\\")[-1].replace(".txt", ""),
        text=pw.this.doc,
        path=pw.this.path
    )

    # Process csv files for metadata
    csvs = documents.filter(pw.this.path.endswith(".csv"))
    csvs = csvs.select(
        data=pw.this.data.decode("utf-8", errors="ignore")
    )
    # At this point, you would typically parse the CSV data.
    # For this specific use case, we are more interested in the novels themselves,
    # but this shows how to differentiate file types.

    return texts

if __name__ == "__main__":
    # This is an example of how to run the loader
    data_dir = "../data"
    documents = load_documents(data_dir)
    processed_docs = process_documents(documents)

    pw.debug.compute_and_print(processed_docs)

