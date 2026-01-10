import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch

class HybridIndex:
    """
    A local hybrid index supporting Vector Search + BM25.
    This class is designed to run in memory after Pathway has materialized the chunks.
    """
    def __init__(self, chunks, model_name="all-MiniLM-L6-v2"):
        """
        Args:
            chunks: List of dicts, each containing 'text', 'novel_id', 'chunk_id', 'chapter_title', etc.
            model_name: SentenceTransformer model name.
        """
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        # Prepare corpus for BM25
        # Tokenizing: simple whitespace split for baseline (rank_bm25 expects token lists)
        tokenized_corpus = [chunk['text'].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Build Vector Index
        print("Computing embeddings...")
        texts = [chunk['text'] for chunk in self.chunks]
        if texts:
            self.embeddings = self.model.encode(texts, convert_to_tensor=True)
            # Normalize for cosine similarity
            self.embeddings = torch.nn.functional.normalize(self.embeddings, p=2, dim=1)
        else:
            self.embeddings = None
        print(f"Index built with {len(self.chunks)} chunks.")

    def search(self, query, book_name=None, k=5, alpha=0.5):
        """
        Hybrid search merging BM25 and Vector scores.
        Args:
            query: The search query string.
            book_name: Optional book_name filter.
            k: Number of results to return.
            alpha: Weight for vector search (0.0 to 1.0). 1.0 = pure vector, 0.0 = pure BM25.
        """
        if not self.chunks:
            return []

        # 1. Filter indices by book_name if provided
        valid_indices = [i for i, chunk in enumerate(self.chunks)
                         if book_name is None or chunk.get('novel_id') == book_name]

        if not valid_indices:
            return []

        # 2. Vector Search
        query_emb = self.model.encode(query, convert_to_tensor=True)
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=0)

        # Compute cosine similarity for all, then select valid
        # Optimization: encode is fast, but computing scores for all is safer if small dataset
        if self.embeddings is not None:
            vector_scores = torch.matmul(self.embeddings, query_emb).cpu().numpy()
        else:
            vector_scores = np.zeros(len(self.chunks))

        # 3. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores (simple min-max scaling across valid indices)
        valid_bm25 = [bm25_scores[i] for i in valid_indices]
        if valid_bm25 and max(valid_bm25) > min(valid_bm25):
            min_s, max_s = min(valid_bm25), max(valid_bm25)
            # Avoid div/0
            scale = lambda x: (x - min_s) / (max_s - min_s)
        else:
            scale = lambda x: 0.0 if max(valid_bm25) == 0 else 1.0

        # Combine scores
        final_results = []
        for i in valid_indices:
            v_score = vector_scores[i]
            b_score = bm25_scores[i]
            # Ideally we normalize both. For baseline, we trust raw vector cosine (-1 to 1)
            # and normalized BM25 (0 to 1).
            norm_b_score = scale(b_score) if valid_bm25 else 0.0

            # Simple weighted sum
            final_score = alpha * v_score + (1 - alpha) * norm_b_score

            final_results.append({
                'chunk': self.chunks[i],
                'score': final_score,
                'vector_score': v_score,
                'bm25_score': b_score
            })

        # Sort and return top K
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:k]

if __name__ == "__main__":
    # Smoke test for the class
    dummy_chunks = [
        {'text': "The quick brown fox jumps over the lazy dog.", 'novel_id': 'Book1', 'chunk_id': '1'},
        {'text': "The lazy dog slept in the sun.", 'novel_id': 'Book1', 'chunk_id': '2'},
        {'text': "Apples are red and tasty.", 'novel_id': 'Book2', 'chunk_id': '3'}
    ]
    idx = HybridIndex(dummy_chunks)
    res = idx.search("lazy dog", book_name="Book1", k=2)
    print("Top result:", res[0]['chunk']['text'])
