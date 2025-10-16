# backend/main.py
# FastAPI-backend voor Meesman klantenbot met RAG + nette bronverwijzingen

from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os, re, requests
from dotenv import load_dotenv

# ----------------------------- ENV ---------------------------------

# Laad .env uit backend/ en projectroot
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

API_KEY: str = (os.getenv("MISTRAL_API_KEY") or "").strip()
BASE_URL: str = (os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").rstrip("/"))
CHAT_PATH: str = os.getenv("MISTRAL_CHAT_PATH", "/chat/completions")
MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.05"))

# Corpus map (voor retriever én static serving)
CORPUS_DIR: Path = Path(os.getenv("CORPUS_DIR") or (BASE_DIR / "corpus")).resolve()

# CORS
raw_origins = (os.getenv("CORS_ORIGINS") or "").strip()
CORS_ORIGINS: List[str] = [o.strip() for o in raw_origins.split(",") if o.strip()]

CHAT_URL = f"{BASE_URL}{CHAT_PATH}"

# --------------------------- RETRIEVER ------------------------------

# retriever.top_k_meta(query, k) -> [{id, source, text, preview, score, ...(page?)}]
from retriever import top_k_meta, reload_index

# ----------------------------- APP ---------------------------------

app = FastAPI(title="Meesman Klantenbot API", version="2025-10-06")

# Statics: serveer PDF’s zodat front-end direct naar /files/<naam>#page=N kan linken
if CORPUS_DIR.exists():
    app.mount("/files", StaticFiles(directory=str(CORPUS_DIR)), name="files")

# CORS
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# --------------------------- MODELLEN ------------------------------

class ChatBodyFlexible(BaseModel):
    question: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    history: Optional[List[Dict[str, Any]]] = None

class SearchBody(BaseModel):
    query: str
    k: Optional[int] = 10

# ------------------------- HULPFUNCTIES ----------------------------

# 1) Vraag extraheren (compatibel met jouw front-end)
def _extract_question(body: ChatBodyFlexible) -> str:
    if body.question and body.question.strip():
        return body.question.strip()

    def from_msgs(msgs: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        if not msgs:
            return None
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

# 2) Slimme query-expansie (synoniemen zodat klanten-taal beter matcht)
SYNONYMS = {
    r"\beigen vermogen\b": ["eigen kapitaal", "balans", "eigen vermogen beheerder"],
    r"\bfondsvermogen\b": ["vermogen onder beheer", "AUM", "intrinsieke waarde"],
    r"\bkosten\b": ["beheerkosten", "lopende kosten", "LKF", "TER", "beheervergoeding"],
    r"\bresultaat\b": ["winst", "nettowinst", "jaarresultaat"],
}

def expand_query(q: str) -> str:
    terms = [q]
    lower = q.lower()
    for pat, alts in SYNONYMS.items():
        if re.search(pat, lower):
            terms += alts
    # Jaartal-varianten
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        y = m.group(1)
        terms += [f"boekjaar {y}", f"per 31 december {y}", f"jaarverslag {y}", f"jaarrekening {y}"]
    # Dedup, simpele OR-query voor retriever
    terms = list(dict.fromkeys(terms))
    return " OR ".join(terms)

# 3) Filter hits op score (en geef voorkeur aan expliciet genoemd bestand)
def select_hits(question: str, hits: List[Dict], threshold: float) -> List[Dict]:
    fname = None
    m = re.search(r"([A-Za-z0-9._-]+\.pdf)", question)
    if m:
        fname = m.group(1).lower()

    filtered = [h for h in hits if float(h.get("score", 0.0)) >= threshold]
    if fname:
        preferred = [h for h in filtered if Path(h["source"]).name.lower() == fname]
        if preferred:
            return preferred[:5]
    return filtered[:5]

# 4) Bouw context voor het LLM (puur corpus, geen buitenkennis)
def build_messages(question: str, hits: List[Dict]) -> List[Dict]:
    ctx_parts: List[str] = []
    for h in hits[:5]:
        fname = Path(h["source"]).name
        ctx_parts.append(f"### {fname}\n{h['text']}")
    context = "\n\n---\n\n".join(ctx_parts) if ctx_parts else "GEEN CONTEXT GEVONDEN"

    system = (
        "Je bent een behulpzame klantenassistent. Antwoord uitsluitend op basis van de context. "
        "Gebruik geen buitenkennis. Als het antwoord niet in de context staat of je twijfelt, zeg exact: "
        "'Niet gevonden in de documenten.' Geef een kort, helder antwoord met bullet points waar nuttig. "
        "Sluit af met een sectie 'Bronnen:' met de bestandsnamen en (indien duidelijk) paginanummers."
    )
    user = f"Vraag: {question}\n\nContext:\n{context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# 5) LLM-call (Mistral) – eerst Bearer, eventueel fallback x-api-key
def call_llm(messages: List[Dict]) -> str:
    payload = {"model": MODEL, "messages": messages, "temperature": 0.0, "max_tokens": 700}
    # Authorization: Bearer
    try:
        r = requests.post(
            CHAT_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json=payload, timeout=60
        )
        if r.status_code < 400:
            j = r.json()
            return j["choices"][0]["message"]["content"].strip()
        if r.status_code not in (401, 403, 404, 415):
            r.raise_for_status()
    except Exception:
        pass
    # x-api-key
    r2 = requests.post(
        CHAT_URL,
        headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
        json=payload, timeout=60
    )
    r2.raise_for_status()
    j2 = r2.json()
    return j2["choices"][0]["message"]["content"].strip()

# 6) Citations: haal paginanummers uit id 'file.pdf#p12' of uit 'page' veld
def citations_from_hits(hits: List[Dict]) -> List[Dict]:
    pages_by_file: Dict[str, set] = {}
    for h in hits:
        fname = Path(h["source"]).name
        p = None
        # a) pref: expliciet veld
        if "page" in h and isinstance(h["page"], int):
            p = h["page"]
        # b) parse uit id: something.pdf#p12
        if p is None:
            m = re.search(r"#p(\d+)$", str(h.get("id", "")))
            if m:
                p = int(m.group(1))
        if p is not None:
            pages_by_file.setdefault(fname, set()).add(p)
        else:
            pages_by_file.setdefault(fname, set())

    out = []
    for fname, pages in pages_by_file.items():
        if pages:
            out.append({"source": fname, "pages": sorted(pages)})
        else:
            out.append({"source": fname, "pages": []})
    # sorteer alfabetisch voor consistentie
    out.sort(key=lambda x: x["source"].lower())
    return out

# 7) Follow-up vragen (kort, actiegericht)
def suggest_followups(question: str) -> List[str]:
    flw = []
    if re.search(r"\b(20\d{2})\b", question):
        flw.append("Wil je dat ik de balanspagina voor je open?")
    if re.search(r"\beigen vermogen|balans|eigen kapitaal\b", question.lower()):
        flw.append("Moet ik ook de winst- en verliesrekening erbij pakken?")
    if not flw:
        flw.append("Wil je dat ik de relevante pagina in het document open?")
    return flw[:2]

# --------------------------- ENDPOINTS -----------------------------

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
        "files_mount": "/files" if CORPUS_DIR.exists() else None,
    }

@app.get("/llm-health")
def llm_health():
    attempts: List[Tuple[str, int, str]] = []
    try:
        r1 = requests.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=20)
        attempts.append(("Authorization", r1.status_code, r1.text[:200]))
    except Exception as e:
        attempts.append(("Authorization", -1, f"{e}"))
    try:
        r2 = requests.get(f"{BASE_URL}/models", headers={"x-api-key": API_KEY}, timeout=20)
        attempts.append(("x-api-key", r2.status_code, r2.text[:200]))
    except Exception as e:
        attempts.append(("x-api-key", -1, f"{e}"))
    return {"base_url": BASE_URL, "attempts": [{"scheme": s, "status": c, "body": b} for (s, c, b) in attempts]}

@app.post("/reload")
def reload():
    # retriever zelf bepaalt pad; warm start
    reload_index(str(CORPUS_DIR))
    return {"ok": True}

@app.post("/search")
def search(body: SearchBody):
    q = expand_query(body.query)
    hits = top_k_meta(q, k=body.k or 10)
    return {
        "query": q,
        "results": [
            {
                "id": h["id"],
                "source": Path(h["source"]).name,
                "score": float(h.get("score", 0.0)),
                "page": int(h["page"]) if "page" in h and isinstance(h["page"], int) else None,
                "preview": h.get("preview", ""),
            }
            for h in hits
        ],
    }

@app.post("/chat")
def chat(body: ChatBodyFlexible):
    question = _extract_question(body)
    expanded = expand_query(question)

    raw_hits = top_k_meta(expanded, k=12)
    hits = select_hits(question, raw_hits, SCORE_THRESHOLD)

    if not hits:
        return {
            "answer": "Niet gevonden in de documenten.",
            "sources": [],
            "citations": [],
            "followups": ["Waar wil je dat ik precies naar kijk (bijv. jaar/onderdeel)?", "Zal ik het juiste document voor je openen?"],
        }

    messages = build_messages(question, hits)

    if not API_KEY:
        # Local fallback: toon fragmenten
        previews = "\n\n".join(f"- {Path(h['source']).name}: {h.get('preview','')}" for h in hits)
        sources = sorted({Path(h["source"]).name for h in hits})
        return {
            "answer": f"(Local mode) Relevante fragmenten:\n\n{previews}",
            "sources": sources,
            "citations": citations_from_hits(hits),
            "followups": suggest_followups(question),
        }

    try:
        answer = call_llm(messages)
    except Exception as e:
        previews = "\n\n".join(f"- {Path(h['source']).name}: {h.get('preview','')}" for h in hits)
        answer = f"(LLM-fallback) Kon geen chat-antwoord genereren ({e}). Relevante fragmenten:\n\n{previews}"

    sources = sorted({Path(h["source"]).name for h in hits})
    cits = citations_from_hits(hits)

    # Als het LLM keurig "Niet gevonden..." zegt, verberg bronnen
    if answer.strip().lower().startswith("niet gevonden"):
        sources = []
        cits = []

    return {
        "answer": answer,
        "sources": sources,
        "citations": cits,
        "followups": suggest_followups(question),
    }
