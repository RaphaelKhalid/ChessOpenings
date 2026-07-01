# FastAPI app for Chess Opening Assistant
import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed import TextEmbedding
from google import genai

load_dotenv(Path(__file__).parent / ".env")

app = FastAPI(title="Chess Opening Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GENERATION_MODEL = "gemini-2.5-flash"

# Both loaded once at startup
_embed_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_db():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def embed(text: str) -> list[float]:
    return list(next(iter(_embed_model.embed([text])))).copy()


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "message": "Chess Opening Assistant API is running"}


@app.post("/ask")
def ask(body: Question):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    q_vec = "[" + ",".join(str(x) for x in embed(question)) + "]"

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT eco, name, pgn, 1 - (embedding <=> %s::vector) AS similarity
            FROM openings
            ORDER BY embedding <=> %s::vector
            LIMIT 5;
        """, (q_vec, q_vec))
        results = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    sources = [
        {"eco": eco, "name": name, "pgn": pgn, "similarity": round(float(sim), 3)}
        for eco, name, pgn, sim in results
    ]

    context_lines = "\n".join(
        f"- {eco} {name}: {pgn}" for eco, name, pgn, _ in results
    )
    prompt = (
        "You are a chess opening expert. Answer the user's question using only the "
        "opening information provided below. Be concise and helpful.\n\n"
        f"Opening data:\n{context_lines}\n\n"
        f"Question: {question}"
    )

    response = _gemini.models.generate_content(model=GENERATION_MODEL, contents=prompt)

    return {
        "answer": response.text,
        "question": question,
        "sources": sources,
    }
