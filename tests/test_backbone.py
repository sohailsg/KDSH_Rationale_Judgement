import pytest
import pandas as pd
import pathway as pw

from src.ingestion.loader import process_documents
from src.processing.chunker import chunk_documents
from src.indexing.vector_db import setup_vector_store, get_retrieval_app

# Fixture for creating a dummy documents table
@pytest.fixture
def mock_documents():
    class DocumentInput(pw.Schema):
        data: bytes
        path: str

    # Create a list of dictionaries with the file content and path
    files = [
        {"path": "data/The Count of Monte Cristo.txt", "data": "This is the story of Edmond Dantes.".encode('utf-8')},
        {"path": "data/In search of the castaways.txt", "data": "A story about Captain Grant.".encode('utf-8')},
        {"path": "data/train.csv", "data": "id,book_name,char,content,label\n1,The Count of Monte Cristo,Dantes,some_claim,true".encode('utf-8')}
    ]
    
    # Create a Pathway table from the list of dictionaries
    table = pw.Table.from_list(files, schema=DocumentInput)
    return table

def test_novel_mapping_and_loading(mock_documents):
    """
    Tests that the loader correctly identifies and processes .txt files,
    mapping them to the correct novel_id.
    """
    processed_docs = process_documents(mock_documents)
    
    # Capture the results
    results = pw.debug.compute_and_list(processed_docs)
    
    # Create a dictionary for easy lookup
    results_map = {item['novel_id']: item for item in results}

    assert len(results) == 2
    assert "The Count of Monte Cristo" in results_map
    assert "In search of the castaways" in results_map
    assert results_map["The Count of Monte Cristo"]['text'] == "This is the story of Edmond Dantes."
    assert results_map["In search of the castaways"]['text'] == "A story about Captain Grant."

def test_retrieval_isolation():
    """
    Tests that retrieval for a specific book name returns zero results
    from the other book.
    """
    class Document(pw.Schema):
        novel_id: str
        text: str
        chunk_id: str
        path: str

    documents = pw.Table.from_pandas(
        pd.DataFrame([
            {'novel_id': 'Book A', 'text': 'Passage one from book A', 'chunk_id': 'a1', 'path': '/a'},
            {'novel_id': 'Book A', 'text': 'Passage two from book A', 'chunk_id': 'a2', 'path': '/a'},
            {'novel_id': 'Book B', 'text': 'Passage one from book B', 'chunk_id': 'b1', 'path': '/b'},
        ]),
        schema=Document
    )
    
    vector_store = setup_vector_store(documents, device='cpu')

    class QueryInputSchema(pw.Schema):
        query: str
        book_name: str
        k: int

    # Query for "Book A"
    query_input_A = pw.Table.from_list([{'query': 'passage', 'book_name': 'Book A', 'k': 5}], schema=QueryInputSchema)
    results_A = vector_store.query(query_input_A, vector=pw.this.query.apply(vector_store.embedder), k=pw.this.k, filters=pw.this.book_name == vector_store.table.novel_id)
    results_A_list = pw.debug.compute_and_list(results_A)
    
    # Query for "Book B"
    query_input_B = pw.Table.from_list([{'query': 'passage', 'book_name': 'Book B', 'k': 5}], schema=QueryInputSchema)
    results_B = vector_store.query(query_input_B, vector=pw.this.query.apply(vector_store.embedder), k=pw.this.k, filters=pw.this.book_name == vector_store.table.novel_id)
    results_B_list = pw.debug.compute_and_list(results_B)

    # Unpack the nested results
    retrieved_docs_A = results_A_list[0]['result']
    retrieved_docs_B = results_B_list[0]['result']

    # Check results for Book A
    assert len(retrieved_docs_A) == 2
    assert all(doc['novel_id'] == 'Book A' for doc in retrieved_docs_A)
    
    # Check results for Book B
    assert len(retrieved_docs_B) == 1
    assert all(doc['novel_id'] == 'Book B' for doc in retrieved_docs_B)
