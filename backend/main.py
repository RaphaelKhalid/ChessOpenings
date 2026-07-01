# FastAPI app for Chess Opening Assistant
import os
import re
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


# Matches a chess move token as it appears at the start of a PGN, e.g.
# "e4", "d4", "c4", "Nf3", "g3", "O-O". Used to detect which White first
# move (if any) the user is asking about a reply/response/defense to.
_MOVE_TOKEN = r"(?:[a-h][1-8]|[NBRQK][a-h]?[1-8]?x?[a-h][1-8]|O-O(?:-O)?)"

_REPLY_INTENT = re.compile(
    r"\b(repl(?:y|ies)|respon(?:se|d)|defen[cs]e|answer|counter)\b", re.I
)


def detect_first_move(question: str) -> str | None:
    """If the question asks about replies to a specific White opening move,
    return that move token (e.g. "e4"); otherwise None."""
    if not _REPLY_INTENT.search(question):
        return None
    m = re.search(rf"\b({_MOVE_TOKEN})\b", question)
    return m.group(1) if m else None


@app.get("/health")
def health():
    return {"status": "ok", "message": "Chess Opening Assistant API is running"}


@app.post("/ask")
def ask(body: Question):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    q_vec = "[" + ",".join(str(x) for x in embed(question)) + "]"
    first_move = detect_first_move(question)

    conn = get_db()
    try:
        cur = conn.cursor()
        if first_move:
            # Restrict to openings where that's White's 1st move and a
            # Black reply actually follows, so lines like the Danish
            # Gambit (a White try after 1...e5) can't surface as a
            # "reply to 1.e4".
            cur.execute("""
                SELECT eco, name, pgn, 1 - (embedding <=> %s::vector) AS similarity
                FROM openings
                WHERE pgn ~ %s
                ORDER BY embedding <=> %s::vector
                LIMIT 10;
            """, (q_vec, rf"^1\.\s*{re.escape(first_move)}\s+\S", q_vec))
            results = cur.fetchall()
            if not results:
                # No qualifying opening in the DB; fall back to plain search.
                first_move = None
        if not first_move:
            cur.execute("""
                SELECT eco, name, pgn, 1 - (embedding <=> %s::vector) AS similarity
                FROM openings
                ORDER BY embedding <=> %s::vector
                LIMIT 10;
            """, (q_vec, q_vec))
            results = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    sources = [
        {"eco": eco, "name": name, "pgn": pgn, "similarity": round(float(sim), 3)}
        for eco, name, pgn, sim in results[:5]
    ]

    context_lines = "\n".join(
        f"- {eco} {name}: {pgn}" for eco, name, pgn, _ in results
    )
    prompt = (
        "You are a chess opening expert. Each opening below is labeled with its "
        "full move sequence (PGN), which tells you exactly whose move each ply is: "
        "the 1st, 3rd, 5th... moves listed are White's, the 2nd, 4th, 6th... are "
        "Black's. Some of the openings retrieved below were matched by loose text "
        "similarity and may NOT actually answer the question — for example an "
        "opening starting '1. e4 e5 2. d4' is a White gambit that assumes Black "
        "already played 1...e5, so it is NOT a valid reply for Black to 1.e4. "
        "Before citing an opening, check its move sequence actually matches what "
        "the question asks (correct side to move, correct move number). Silently "
        "ignore retrieved openings that don't fit; only discuss ones that do. If "
        "none fit, say so honestly instead of citing an irrelevant one.\n\n"
        f"Opening data:\n{context_lines}\n\n"
        f"Question: {question}"
    )

    response = _gemini.models.generate_content(model=GENERATION_MODEL, contents=prompt)

    return {
        "answer": response.text,
        "question": question,
        "sources": sources,
    }
