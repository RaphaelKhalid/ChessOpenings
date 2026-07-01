# Verifies Supabase connection, pgvector extension, and basic CRUD operations
import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

load_dotenv(Path(__file__).parent / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = True
cur = conn.cursor()

# Enable pgvector
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
print("pgvector extension enabled")

# Create test table
cur.execute("""
    CREATE TABLE IF NOT EXISTS verify_test (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector(384)
    );
""")
print("verify_test table created")

# Insert a row (embedding as zeros for this connectivity test)
zeros = "[" + ",".join(["0"] * 384) + "]"
cur.execute(
    "INSERT INTO verify_test (content, embedding) VALUES (%s, %s::vector) RETURNING id;",
    ("King's Indian Defense test row", zeros)
)
row_id = cur.fetchone()[0]
print(f"Inserted row id={row_id}")

# Read it back
cur.execute("SELECT id, content FROM verify_test WHERE id = %s;", (row_id,))
row = cur.fetchone()
print(f"Read back: id={row[0]}, content='{row[1]}'")

# Drop test table
cur.execute("DROP TABLE verify_test;")
print("verify_test table dropped")

cur.close()
conn.close()
print("SUPABASE OK")
