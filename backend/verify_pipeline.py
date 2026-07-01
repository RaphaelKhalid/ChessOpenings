# Verifies the full RAG pipeline: embed -> store in pgvector -> similarity search -> Gemini
import os
import csv
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv(Path(__file__).parent / ".env")

# Load 10 rows from TSV files
data_dir = os.path.join(os.path.dirname(__file__), "data")
rows = []
with open(os.path.join(data_dir, "a.tsv"), encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)
        if len(rows) == 10:
            break

# Load remaining from other files if needed
for fname in ["b.tsv", "c.tsv", "d.tsv", "e.tsv"]:
    if len(rows) >= 10:
        break
    with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
            if len(rows) == 10:
                break

print(f"Loaded {len(rows)} rows")

# Embed each row
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [f"{r['eco']} {r['name']}: {r['pgn']}" for r in rows]
embeddings = model.encode(texts)
print(f"Generated {len(embeddings)} embeddings")

# Connect to Supabase
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
conn.autocommit = True
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("""
    CREATE TABLE IF NOT EXISTS pipeline_test (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector(384)
    );
""")
print("pipeline_test table created")

# Insert rows
for text, emb in zip(texts, embeddings):
    vec_str = "[" + ",".join(str(x) for x in emb.tolist()) + "]"
    cur.execute(
        "INSERT INTO pipeline_test (content, embedding) VALUES (%s, %s::vector);",
        (text, vec_str)
    )
print("Inserted 10 rows")

# Embed query and similarity search
query = "tell me about the Sicilian Defense"
query_emb = model.encode(query)
query_vec = "[" + ",".join(str(x) for x in query_emb.tolist()) + "]"

cur.execute("""
    SELECT content, 1 - (embedding <=> %s::vector) AS similarity
    FROM pipeline_test
    ORDER BY embedding <=> %s::vector
    LIMIT 3;
""", (query_vec, query_vec))
top3 = cur.fetchall()
print(f"\nTop 3 results for '{query}':")
for content, sim in top3:
    print(f"  [{sim:.3f}] {content[:80]}")

# Drop test table
cur.execute("DROP TABLE pipeline_test;")
cur.close()
conn.close()
print("pipeline_test table dropped")

# Ask Gemini
context = "\n".join(f"- {row[0]}" for row in top3)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-2.5-flash")
prompt = f"Based only on the following chess opening information, answer: what is the Sicilian Defense?\n\nContext:\n{context}"
response = gemini.generate_content(prompt)
print(f"\nGemini response:\n{response.text}")

print("PIPELINE OK")
