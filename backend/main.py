import os, re, json
from typing import Any, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

VERSION = "rag-only-pdf-v1"

# ---- .env inladen (lokaal handig) ----
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ---- Config / omgeving ----
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
BASE_URL  = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").rstrip("/")
CHAT_PATH = os.getenv("MISTRAL_CHAT_PATH", "/chat/completions")
MODEL     = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
FILES_MOUNT = os.getenv("FILES_MOUNT", "./corpus")

# ---- FastAPI + CORS (localhost & vercel) ----
app = FastAPI(title="Meesman Document Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://.*\.(vercel|netlify)\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static: laat frontend naar https://…/files/… linken
if os.path.isdir(FILES_MOUNT):
    app.mount("/files", StaticFiles(directory=FILES_MOUNT), name="files")

# ---- Datatypes ----
class AskIn(BaseModel):
    question: str

class Citation(BaseModel):
    url: str
    title: str
    page: int

class AskOut(BaseModel):
    answer: str
    citations: Optional[List[Citation]] = None
    sources: Optional[List[Any]] = None
    followups: Optional[List[str]] = None

# ---- PDF corpus inlezen ----
from pypdf import PdfReader  # type: ignore

Page = Tuple[str, int, str]   # (rel_path, page_index, text)
PAGES: List[Page] = []

def _is_pdf(name: str) -> bool:
    return name.lower().endswith(".pdf")

def _iter_pdfs(root: str) -> List[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if _is_pdf(f):
                out.append(os.path.join(dirpath, f))
    return sorted(out)

def _rel(path: str) -> str:
    return os.path.relpath(path, start=FILES_MOUNT).replace("\\", "/")

def _load_corpus() -> None:
    PAGES.clear()
    if not os.path.isdir(FILES_MOUNT):
        return
    for abs_path in _iter_pdfs(FILES_MOUNT):
        try:
            reader = PdfReader(abs_path)
            for i, page in enumerate(reader.pages):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                if txt.strip():
                    PAGES.append((_rel(abs_path), i, _normalize(txt)))
        except Exception:
            continue

def _normalize(t: str) -> str:
    t = t.replace("\x00", " ").replace("\u200b", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    return t.strip()

# ---- Eenvoudige retrieval (woord-score + booleans) ----
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_]+")

def _tokenize(q: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(q)]

def _score(query_tokens: List[str], text: str) -> int:
    if not query_tokens:
        return 0
    text_low = text.lower()
    s = 0
    for w in query_tokens:
        s += len(re.findall(rf"\b{re.escape(w)}\b", text_low))
    if len(query_tokens) >= 2 and " ".join(query_tokens) in text_low:
        s += 2
    return s

def _retrieve(q: str, k: int = 4, min_score: int = 2) -> List[Tuple[str, int, str]]:
    toks = _tokenize(q)
    scored: List[Tuple[int, Page]] = []
    for page in PAGES:
        rel, idx, txt = page
        sc = _score(toks, txt)
        if sc > 0:
            scored.append((sc, page))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = [(rel, idx, txt) for sc, (rel, idx, txt) in scored[:k] if sc >= min_score]
    return top

def _snippet(txt: str, q: str, radius: int = 350) -> str:
    toks = _tokenize(q)
    if not toks:
        return txt[:radius]
    low = txt.lower()
    pos = min((low.find(t) for t in toks if low.find(t) != -1), default=-1)
    if pos == -1:
        return txt[:radius]
    start = max(0, pos - radius // 2)
    end = min(len(txt), start + radius)
    snip = txt[start:end].strip()
    return ("…" if start > 0 else "") + snip + ("…" if end < len(txt) else "")

# ---- Mistral call ----
def _mistral_answer(system_prompt: str, user_prompt: str) -> str:
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY ontbreekt op de server.")
    import httpx
    url = f"{BASE_URL}{CHAT_PATH}"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    try:
        with httpx.Client(timeout=60) as client:
            r = client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Mistral call failed: {type(e).__name__}: {e}")

SYSTEM_PROMPT = (
    "Je bent een strikte RAG-assistent. "
    "Antwoord ALLEEN met informatie die in de aangeleverde PDF-snippets staat. "
    "Als het antwoord niet expliciet in de snippets staat, zeg dan duidelijk dat je het niet uit de documenten kunt halen."
)

def _refuse() -> AskOut:
    msg = (
        "Ik kan dit niet uit de beschikbare PDF-documenten halen. "
        "Specificeer je vraag of verwijs naar een document/pagina."
    )
    return AskOut(answer=msg, citations=[], sources=[], followups=[])

# ---- Endpoints ----
@app.get("/status")
def status():
    return {
        "version": VERSION,
        "key_loaded": bool(MISTRAL_API_KEY),
        "pages_indexed": len(PAGES),
        "files_mount": FILES_MOUNT,
        "model": MODEL,
    }

@app.post("/ask", response_model=AskOut)
def ask(body: AskIn):
    q = (body.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Vraag ontbreekt.")

    if len(PAGES) == 0:
        return _refuse()

    hits = _retrieve(q, k=4, min_score=2)
    if not hits:
        return _refuse()

    lines: List[str] = []
    citations: List[Citation] = []
    for rel, page_i, txt in hits:
        url = f"/files/{rel}"
        title = os.path.basename(rel)
        lines.append(f"[{title} p.{page_i+1}]\n{_snippet(txt, q)}\n")
        citations.append(Citation(url=url, title=title, page=page_i+1))

    context = "\n\n".join(lines)
    user_prompt = (
        f"Vraag:\n{q}\n\n"
        f"Context uit PDF (gebruik ALLEEN dit, citeer niet letterlijk hele stukken):\n{context}\n\n"
        f"Antwoord kort, feitelijk, en geef geen info buiten deze context."
    )
    answer = _mistral_answer(SYSTEM_PROMPT, user_prompt)
    if not answer.strip():
        return _refuse()
    return AskOut(answer=answer, citations=citations, sources=[], followups=[])

# ---- Start: corpus laden bij boot ----
_load_corpus()
