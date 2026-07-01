# One-time ingestion script: reads all TSV openings, embeds them, and writes to Supabase pgvector
import os
import csv
import time
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).parent / ".env")

DATA_DIR = Path(__file__).parent / "data"
FILES = ["a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv"]
BATCH_SIZE = 128


def load_openings():
    rows = []
    for fname in FILES:
        with open(DATA_DIR / fname, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append({
                    "eco": row["eco"].strip(),
                    "name": row["name"].strip(),
                    "pgn": row["pgn"].strip(),
                    "content": f"{row['eco'].strip()} {row['name'].strip()}: {row['pgn'].strip()}"
                })
    return rows


def setup_table(cur):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("DROP TABLE IF EXISTS openings;")
    cur.execute("""
        CREATE TABLE openings (
            id        SERIAL PRIMARY KEY,
            eco       TEXT NOT NULL,
            name      TEXT NOT NULL,
            pgn       TEXT NOT NULL,
            content   TEXT NOT NULL,
            embedding vector(384) NOT NULL
        );
    """)
    print("openings table created")


def embed_and_insert(cur, model, rows):
    total = len(rows)
    inserted = 0

    for start in range(0, total, BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        texts = [r["content"] for r in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        records = [
            (
                r["eco"], r["name"], r["pgn"], r["content"],
                "[" + ",".join(str(x) for x in emb.tolist()) + "]"
            )
            for r, emb in zip(batch, embeddings)
        ]

        execute_values(cur, """
            INSERT INTO openings (eco, name, pgn, content, embedding)
            VALUES %s
        """, records, template="(%s, %s, %s, %s, %s::vector)")

        inserted += len(batch)
        print(f"  Inserted {inserted}/{total}", end="\r")

    print(f"\nAll {inserted} rows inserted")


def verify(cur):
    cur.execute("SELECT COUNT(*) FROM openings;")
    count = cur.fetchone()[0]
    cur.execute("SELECT eco, name FROM openings LIMIT 3;")
    samples = cur.fetchall()
    print(f"Verified: {count} rows in openings table")
    for eco, name in samples:
        print(f"  {eco} — {name}")
    return count


def main():
    print("Loading openings from TSV files...")
    rows = load_openings()
    print(f"Loaded {len(rows)} openings")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to Supabase...")
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    conn.autocommit = True
    cur = conn.cursor()

    setup_table(cur)

    print(f"Embedding and inserting in batches of {BATCH_SIZE}...")
    t0 = time.time()
    embed_and_insert(cur, model, rows)
    elapsed = time.time() - t0
    print(f"Ingestion took {elapsed:.1f}s")

    verify(cur)

    cur.close()
    conn.close()
    print("INGEST OK")


if __name__ == "__main__":
    main()
