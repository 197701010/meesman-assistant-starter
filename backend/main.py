# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import os, re, requests
from dotenv import load_dotenv

# --- ENV laden (backend en projectroot) ---
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

API_KEY: str = (os.getenv("MISTRAL_API_KEY") or "").strip()
BASE_URL: str = (os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").rstrip("/"))
CHAT_PATH: str = os.getenv("MISTRAL_CHAT_PATH", "/chat/completions")
MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.07"))
CHAT_URL: str = f"{BASE_URL}{CHAT_PATH}"

# CORS
raw_origins = (os.getenv("CORS_ORIGINS") or "").strip()
CORS_ORIGINS: List[str] = [o.strip() for o in raw_origins.split(",") if o.strip()]

from retriever import top_k_meta, reload_index  # lokale RAG

# --- FastAPI app ---
app = FastAPI(title="Meesman RAG API", version="2025-10-02")

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ---------- In/Out modellen ----------
class ChatBodyFlexible(BaseModel):
    # ondersteunt zowel {"question": "..."} als {"messages":[...], "history":[...]}
    question: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None   # [{role, content}, ...]
    history: Optional[List[Dict[str, Any]]] = None    # legacy veld, mag leeg

class SearchBody(BaseModel):
    query: str
    k: Optional[int] = 8

# ---------- Helpers ----------
def _extract_question(body: ChatBodyFlexible) -> str:
    """Haal vraag op uit 'question' of uit laatste user-bericht in messages/history."""
    if body.question and isinstance(body.question, str) and body.question.strip():
        return body.question.strip()

    def from_msgs(msgs: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        if not msgs: 
            return None
        # pak laatste user content die een string is
        for m in reversed(msgs):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
        return None

    q = from_msgs(body.messages) or from_msgs(body.history)
    if q:
        return q

    raise HTTPException(status_code=422, detail="Geen vraag gevonden. Stuur 'question' of 'messages' met een user-bericht.")

def build_messages(question: str, hits: List[Dict]):
    """Bouw systeem-/user-berichten met uitsluitend corpuscontext."""
    ctx_parts: List[str] = []
    for h in hits[:5]:
        fname = Path(h["source"]).name
        ctx_parts.append(f"### {fname}\n{h['text']}")
    context = "\n\n---\n\n".join(ctx_parts) if ctx_parts else "GEEN CONTEXT GEVONDEN"

    system = (
        "Je bent een assistent die uitsluitend antwoordt op basis van de gegeven context. "
        "Gebruik geen buitenkennis. Als het antwoord niet in de context staat of je bent niet zeker, antwoord dan exact: "
        "'Niet gevonden in de documenten.' Antwoord kort en feitelijk. "
        "Voeg onderaan een sectie 'Bronnen:' toe met alleen de bestandsnamen die je gebruikte."
    )
    user = f"Vraag: {question}\n\nContext:\n{context}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def select_hits(question: str, hits: List[Dict], threshold: float) -> List[Dict]:
    """Filter op score; geef voorkeur aan bestandsnaam die in de vraag genoemd wordt."""
    # detecteer expliciet aangevraagde bestandsnaam
    fname = None
    m = re.search(r"([A-Za-z0-9._-]+\.pdf)", question)
    if m:
        fname = m.group(1).lower()

    filtered = [h for h in hits if float(h.get("score", 0.0)) >= threshold]

    if fname:
        prefer = [h for h in filtered if Path(h["source"]).name.lower() == fname]
        if prefer:
            return prefer[:5]
    return filtered[:5]

def call_llm(messages: List[Dict]) -> str:
    """Eerst Authorization Bearer, daarna fallback x-api-key."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 800,
    }

    # 1) Authorization: Bearer
    try:
        r = requests.post(
            CHAT_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        if r.status_code < 400:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code not in (401, 403, 404, 415):
            r.raise_for_status()
    except Exception:
        pass

    # 2) x-api-key
    r2 = requests.post(
        CHAT_URL,
        headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=60
    )
    r2.raise_for_status()
    return r2.json()["choices"][0]["message"]["content"].strip()

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/status", "/llm-health", "/reload", "/search", "/chat"]}

@app.get("/status")
def status():
    return {
        "key_loaded": bool(API_KEY),
        "key_prefix": (API_KEY[:3] + "…") if API_KEY else None,
        "base_url": BASE_URL,
        "chat_path": CHAT_PATH,
        "model": MODEL,
        "score_threshold": SCORE_THRESHOLD,
        "cors": CORS_ORIGINS or None,
    }

@app.get("/llm-health")
def llm_health():
    """Probeert /models met beide headerstijlen."""
    attempts: List[Tuple[str, int, str]] = []
    try:
        r1 = requests.get(f"{BASE_URL}/models",
                          headers={"Authorization": f"Bearer {API_KEY}"},
                          timeout=20)
        attempts.append(("Authorization", r1.status_code, r1.text[:200]))
    except Exception as e:
        attempts.append(("Authorization", -1, f"{e}"))
    try:
        r2 = requests.get(f"{BASE_URL}/models",
                          headers={"x-api-key": API_KEY},
                          timeout=20)
        attempts.append(("x-api-key", r2.status_code, r2.text[:200]))
    except Exception as e:
        attempts.append(("x-api-key", -1, f"{e}"))
    return {"base_url": BASE_URL,
            "attempts": [{"scheme": s, "status": c, "body": b} for (s, c, b) in attempts]}

@app.post("/reload")
def reload():
    reload_index()
    return {"ok": True}

@app.post("/search")
def search(body: SearchBody):
    """Debug/UX: geef ruwe RAG-hits (id, source, score, preview)."""
    k = body.k or 8
    hits = top_k_meta(body.query, k=k)
    return {"results": [
        {"id": h["id"], "source": Path(h["source"]).name, "score": float(h["score"]), "preview": h["preview"]}
        for h in hits
    ]}

@app.post("/chat")
def chat(body: ChatBodyFlexible):
    # 1) Vraag extraheren (werkt met zowel question als met messages/history)
    question = _extract_question(body)

    # 2) RAG ophalen
    raw_hits = top_k_meta(question, k=8)
    hits = select_hits(question, raw_hits, SCORE_THRESHOLD)

    # 3) Geen relevante context → duidelijk antwoord (zonder bronnen)
    if not hits:
        return {"answer": "Niet gevonden in de documenten.", "sources": []}

    # 4) Prompt bouwen met geselecteerde context
    messages = build_messages(question, hits)

    # 5) Local mode (geen key) → geef fragmenten terug
    if not API_KEY:
        previews = "\n\n---\n\n".join(f"{Path(h['source']).name} → {h['preview']}" for h in hits)
        sources = list({Path(h["source"]).name for h in hits})
        return {"answer": f"(Local mode) Relevante fragmenten:\n\n{previews}", "sources": sources}

    # 6) LLM aanroepen met fallback
    try:
        answer = call_llm(messages)
    except Exception as e:
        previews = "\n\n---\n\n".join(f"{Path(h['source']).name} → {h['preview']}" for h in hits)
        answer = f"(LLM-fallback) Kon geen chat-antwoord genereren ({e}). Relevante fragmenten:\n\n{previews}"

    # 7) Bronnen alleen tonen als we context hadden
    sources = list({Path(h["source"]).name for h in hits})
    if answer.strip().lower().startswith("niet gevonden"):
        sources = []

    return {"answer": answer, "sources": sources}
