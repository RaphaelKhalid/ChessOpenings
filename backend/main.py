# FastAPI app for Chess Opening Assistant
import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv(Path(__file__).parent / ".env")

app = FastAPI(title="Chess Opening Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup — keeps latency low per request
_model = SentenceTransformer("all-MiniLM-L6-v2")
_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_db():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


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

    # Embed the question
    q_emb = _model.encode(question)
    q_vec = "[" + ",".join(str(x) for x in q_emb.tolist()) + "]"

    # Retrieve top 5 most similar openings from Supabase
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

    # Build context for Gemini
    context_lines = "\n".join(
        f"- {eco} {name}: {pgn}" for eco, name, pgn, _ in results
    )
    prompt = (
        "You are a chess opening expert. Answer the user's question using only the "
        "opening information provided below. Be concise and helpful.\n\n"
        f"Opening data:\n{context_lines}\n\n"
        f"Question: {question}"
    )

    response = _gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {
        "answer": response.text,
        "question": question,
        "sources": sources,
    }
