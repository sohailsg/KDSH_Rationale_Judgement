import pathway as pw
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from pathway.xpacks.llm.vector_store import VectorStoreServer

def setup_vector_store(data_source, model_name="all-MiniLM-L6-v2", device="cuda:0"):
    """
    Sets up the vector store for indexing and retrieval.
    """
    embedder = SentenceTransformerEmbedder(model_name, device=device)
    
    # We are expecting a schema with `doc` and `novel_id`
    vector_store = VectorStoreServer(
        data_source,
        embedder=embedder,
        vector_name="vector",
        dimensions=embedder.dimensions,
    )
    
    return vector_store

def get_retrieval_app(vector_store):
    """
    Returns a retrieval app that can be used to query the vector store.
    """
    
    class QueryInputSchema(pw.Schema):
        query: str
        book_name: str
        k: int = 5

    query_input = pw.io.http.read(
        "/",
        schema=QueryInputSchema,
        format="json",
        autocommit_duration_ms=50,
    )

    query_results = vector_store.query(
        query_input,
        vector=pw.this.query.apply(vector_store.embedder),
        k=pw.this.k,
        # Filter by book name BEFORE the nearest neighbor search
        filters=pw.this.book_name == vector_store.table.novel_id
    )

    # The query result is a table with a single row.
    # We want to return the results as a list of dicts.
    response = query_results.select(
        results=pw.this.result
    )

    return response


if __name__ == '__main__':
    # This part is for demonstration and won't be run directly in the main app
    # It shows how you would set up and query the vector store.

    # 1. Sample Data
    class Document(pw.Schema):
        novel_id: str
        text: str
        chunk_id: str
        path: str

    documents = pw.Table.from_pandas(
        pd.DataFrame({
            'novel_id': ['Book A', 'Book B'],
            'text': ['This is a story about A.', 'This is a story about B.'],
            'chunk_id': ['1', '2'],
            'path': ['/path/a', '/path/b']
        }),
        schema=Document
    )
    
    # 2. Setup Vector Store
    vector_store = setup_vector_store(documents)
    
    # 3. Setup Retrieval App
    query_app = get_retrieval_app(vector_store)
    
    # 4. Run the pipeline
    pw.run()

    # To query, you would send a POST request to the HTTP server, e.g.:
    # curl -X POST -H "Content-Type: application/json" -d '{"query": "A", "book_name": "Book A", "k": 1}' http://localhost:8080/
