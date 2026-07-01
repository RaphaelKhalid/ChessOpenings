# Verifies that sentence-transformers loads and produces 384-dim embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("What is the Sicilian Defense?")

print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5].tolist()}")
print("EMBEDDINGS OK")
