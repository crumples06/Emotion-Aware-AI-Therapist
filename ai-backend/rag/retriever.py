import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Use the exact same model as ingest.py
# If you use a different model, the vectors won't be comparable
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3  # how many chunks to retrieve per query


class Retriever:

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(os.getenv("PINECONE_INDEX"))
        print("Retriever ready.")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list:
        """
        Given a user query, find the most relevant chunks from the knowledge base.

        Args:
            query: the user's message
            top_k: how many chunks to return

        Returns:
            list of dicts, each containing 'text' and 'source'
        """
        # Step 1: Convert query to vector
        query_vector = self.model.encode(query).tolist()

        # Step 2: Search Pinecone for similar vectors
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True  # we need the text back, not just the ID
        )

        # Step 3: Extract the text and source from each match
        chunks = []
        for match in results.matches:
            chunks.append({
                "text": match.metadata["text"],
                "source": match.metadata["source"],
                "score": round(match.score, 3)  # cosine similarity score
            })

        # Filter out chunks below relevance threshold
        RELEVANCE_THRESHOLD = 0.45
        chunks = [chunk for chunk in chunks if chunk["score"] >= RELEVANCE_THRESHOLD]

        return chunks

    def format_context(self, chunks: list) -> str:
        """
        Format retrieved chunks into a single string to inject into the prompt.

        Args:
            chunks: list returned by retrieve()

        Returns:
            formatted string ready to be inserted into the system prompt
        """
        if not chunks:
            return ""

        context = "Relevant information from mental health resources:\n\n"

        for i, chunk in enumerate(chunks):
            context += f"[Source {i + 1}: {chunk['source']}]\n"
            context += f"{chunk['text']}\n\n"

        return context.strip()


# Test the retriever directly from terminal
if __name__ == "__main__":
    retriever = Retriever()

    test_queries = [
        "I can't sleep because of anxiety",
        "I feel hopeless and don't want to do anything",
        "how do I manage stress during exams"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        chunks = retriever.retrieve(query)
        for chunk in chunks:
            print(f"Score: {chunk['score']} | Source: {chunk['source']}")
            print(f"Text preview: {chunk['text'][:150]}...")
            print()