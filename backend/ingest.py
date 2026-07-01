# One-time ingestion script: reads all TSV openings, embeds with fastembed (ONNX, no torch), writes to Supabase
import os
import csv
import time
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from fastembed import TextEmbedding

load_dotenv(Path(__file__).parent / ".env")

DATA_DIR = Path(__file__).parent / "data"
FILES = ["a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv"]
BATCH_SIZE = 256
EMBEDDING_DIM = 384


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
    cur.execute(f"""
        CREATE TABLE openings (
            id        SERIAL PRIMARY KEY,
            eco       TEXT NOT NULL,
            name      TEXT NOT NULL,
            pgn       TEXT NOT NULL,
            content   TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}) NOT NULL
        );
    """)
    print(f"openings table created (vector dim: {EMBEDDING_DIM})")


def embed_and_insert(cur, model, rows):
    total = len(rows)
    texts = [r["content"] for r in rows]

    print("Generating embeddings...")
    embeddings = list(model.embed(texts))
    print(f"Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    print("Inserting into Supabase...")
    for start in range(0, total, BATCH_SIZE):
        batch_rows = rows[start:start + BATCH_SIZE]
        batch_embs = embeddings[start:start + BATCH_SIZE]

        records = [
            (
                r["eco"], r["name"], r["pgn"], r["content"],
                "[" + ",".join(str(x) for x in emb.tolist()) + "]"
            )
            for r, emb in zip(batch_rows, batch_embs)
        ]

        execute_values(cur, """
            INSERT INTO openings (eco, name, pgn, content, embedding)
            VALUES %s
        """, records, template="(%s, %s, %s, %s, %s::vector)")

        inserted = min(start + BATCH_SIZE, total)
        print(f"  Inserted {inserted}/{total}", end="\r")

    print(f"\nAll {total} rows inserted")


def verify(cur):
    cur.execute("SELECT COUNT(*) FROM openings;")
    count = cur.fetchone()[0]
    cur.execute("SELECT eco, name FROM openings LIMIT 3;")
    samples = cur.fetchall()
    print(f"Verified: {count} rows in openings table")
    for eco, name in samples:
        print(f"  {eco} — {name}")


def main():
    print("Loading openings from TSV files...")
    rows = load_openings()
    print(f"Loaded {len(rows)} openings")

    print("Loading fastembed model (all-MiniLM-L6-v2 via ONNX)...")
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    print("Connecting to Supabase...")
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    conn.autocommit = True
    cur = conn.cursor()

    setup_table(cur)

    t0 = time.time()
    embed_and_insert(cur, model, rows)
    print(f"Ingestion took {time.time() - t0:.1f}s")

    verify(cur)
    cur.close()
    conn.close()
    print("INGEST OK")


if __name__ == "__main__":
    main()
