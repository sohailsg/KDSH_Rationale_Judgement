import pathway as pw
import os
import logging

from ingestion.loader import load_documents, process_documents
from processing.chunker import chunk_documents
from indexing.vector_db import setup_vector_store, get_retrieval_app

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(data_dir, host='0.0.0.0', port=8080):
    """
    Sets up and runs the entire data processing and retrieval pipeline.
    """
    # 1. Ingestion
    logging.info(f"Starting file ingestion from: {data_dir}")
    raw_documents = load_documents(data_dir)
    processed_docs = process_documents(raw_documents)
    
    # 2. Processing (Chunking)
    logging.info("Chunking documents...")
    chunked_docs = chunk_documents(processed_docs)
    
    # Log the number of chunks created
    def log_chunk_count(df):
        logging.info(f"Created {len(df)} chunks.")
        return df
    
    chunked_docs = chunked_docs.apply(log_chunk_count)

    # 3. Indexing
    logging.info("Setting up vector store...")
    vector_store = setup_vector_store(chunked_docs)

    # 4. Retrieval API
    logging.info("Setting up retrieval API...")
    query_app = get_retrieval_app(vector_store)
    
    # Attach the query app to the same HTTP server
    pw.io.http.write(query_app, "/")

    # 5. Run the pipeline
    logging.info(f"Starting Pathway pipeline. Access the API at http://{host}:{port}")
    pw.run(host=host, port=port)

if __name__ == "__main__":
    # Point to the data directory relative to the script's location
    data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Check if data directory exists
    if not os.path.isdir(data_directory):
        logging.error(f"Data directory not found: {data_directory}")
    else:
        run_pipeline(data_directory)
