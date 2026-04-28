import os
import time
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
PDF_FOLDER = os.path.join(os.path.dirname(__file__), "data")
CHUNK_SIZE = 500  # words per chunk
CHUNK_OVERLAP = 50  # words shared between consecutive chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 100  # how many vectors to upload at once


def load_pdf(filepath: str) -> str:
    """Extract all text from a PDF file"""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:  # some pages return None
            text += extracted + " "
    return text.strip()


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list:
    """
    Split text into overlapping chunks of roughly chunk_size words.

    Overlap means consecutive chunks share some words at the boundary.
    This prevents a sentence from being cut in half between two chunks,
    losing its meaning.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        # Take chunk_size words starting from current position
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Move forward by chunk_size minus overlap
        # So next chunk starts overlap words before end of current chunk
        start += chunk_size - overlap

    return chunks


def ingest():
    # Step 1: Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Step 2: Connect to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX"))
    print("Connected to Pinecone index")

    # Step 3: Process each PDF
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in {PDF_FOLDER}")
        return

    print(f"Found {len(pdf_files)} PDFs: {pdf_files}")

    all_vectors = []
    chunk_id = 0  # unique ID for each chunk across all PDFs

    for pdf_file in pdf_files:
        filepath = os.path.join(PDF_FOLDER, pdf_file)

        # Extract text
        text = load_pdf(filepath)
        if not text:
            continue

        # Split into chunks
        chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

        # Embed each chunk
        embeddings = model.encode(chunks, show_progress_bar=True)

        # Prepare vectors for Pinecone
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector = {
                "id": f"chunk_{chunk_id}",
                "values": embedding.tolist(),  # convert numpy array to plain list
                "metadata": {
                    "text": chunk,
                    "source": pdf_file,
                    "chunk_index": i
                }
            }
            all_vectors.append(vector)
            chunk_id += 1

    # Step 4: Upload to Pinecone in batches
    print(f"\nUploading {len(all_vectors)} vectors to Pinecone...")

    for i in range(0, len(all_vectors), BATCH_SIZE):
        batch = all_vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)
        time.sleep(0.5)  # small delay to avoid rate limiting

    print(f"\nDone. {len(all_vectors)} chunks uploaded to Pinecone.")


if __name__ == "__main__":
    ingest()