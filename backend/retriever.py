# backend/retriever.py
# --- Simple local retriever over PDF / TXT / MD using TF-IDF ---
_VERSION = "2025-10-02b"  # debug marker

from pathlib import Path
from typing import List, Dict
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Readers ----------

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf niet geïnstalleerd. Run: pip install pypdf")

    reader = PdfReader(str(path))
    parts: List[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            parts.append(f"[pagina {i}]\n{text}")
    return "\n\n".join(parts).strip()

# ---------- Corpus loading & chunking ----------

SUPPORTED_TEXT = {".txt", ".md"}
SUPPORTED_PDF = {".pdf"}

def _chunk_text(text: str, target_len: int = 1200, overlap: int = 200) -> List[str]:
    """Eenvoudige chunker op basis van alinea's met overlap."""
    text = re.sub(r"\s+\n", "\n", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)

    chunks: List[str] = []
    buf: List[str] = []
    cur_len = 0

    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        if cur_len + len(p) > target_len and buf:
            chunks.append("\n\n".join(buf).strip())
            keep = chunks[-1][-overlap:] if overlap and chunks[-1] else ""
            buf = [keep, p] if keep else [p]
            cur_len = len(keep) + len(p)
        else:
            buf.append(p)
            cur_len += len(p)

    if buf:
        chunks.append("\n\n".join(buf).strip())

    if not chunks and text:
        chunks = [text]
    return chunks

def _corpus_base(corpus_dir: str | None) -> Path:
    # Belangrijk: standaard naar map 'corpus' naast dit bestand
    return Path(corpus_dir) if corpus_dir else (Path(__file__).parent / "corpus")

def load_corpus(corpus_dir: str | None = None) -> List[Dict]:
    """
    Leest PDF, TXT en MD uit corpus_dir (of 'corpus' naast dit bestand)
    en geeft een lijst met dicts terug: {id, source, text}
    """
    base = _corpus_base(corpus_dir)
    docs: List[Dict] = []

    if not base.exists():
        print(f"[retriever {_VERSION}] corpus pad bestaat niet: {base.resolve()}")
        return docs

    file_count = 0
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue  # verborgen bestanden overslaan
        file_count += 1
        ext = path.suffix.lower()
        try:
            if ext in SUPPORTED_TEXT:
                text = _read_text(path)
            elif ext in SUPPORTED_PDF:
                text = _read_pdf(path)
            else:
                continue
        except Exception as e:
            print(f"[retriever {_VERSION}] kon {path.name} niet lezen: {e}")
            continue

        if text and text.strip():
            docs.append({"id": path.name, "source": str(path), "text": text.strip()})

    print(f"[retriever {_VERSION}] gevonden bestanden: {file_count}, geladen: {len(docs)} uit {base.resolve()}")
    return docs

# ---------- Index (TF-IDF) ----------

_VECTORIZER: TfidfVectorizer | None = None
_MATRIX = None                   # scipy sparse matrix
_CHUNKS: List[Dict] = []         # [{id, source, text}] op chunk-niveau

def _build_index(docs: List[Dict]):
    """Maakt chunks van documenten en bouwt TF-IDF index."""
    global _VECTORIZER, _MATRIX, _CHUNKS

    chunks: List[Dict] = []
    for d in docs:
        parts = _chunk_text(d["text"], target_len=1200, overlap=200)
        for j, c in enumerate(parts):
            chunks.append({
                "id": f"{d['id']}#c{j+1}",
                "source": d["source"],
                "text": c
            })

    texts = [c["text"] for c in chunks]
    if not texts:
        _VECTORIZER = None
        _MATRIX = None
        _CHUNKS = []
        print(f"[retriever {_VERSION}] geen tekst/chunks gevonden")
        return

    _VECTORIZER = TfidfVectorizer(stop_words=None, max_features=50000)
    _MATRIX = _VECTORIZER.fit_transform(texts)
    _CHUNKS = chunks
    print(f"[retriever {_VERSION}] index gebouwd: {len(_CHUNKS)} chunks uit {len(docs)} documenten")

def reload_index(corpus_dir: str | None = None):
    """Publieke helper om vanaf main.py of de shell opnieuw te laden."""
    docs = load_corpus(corpus_dir)
    _build_index(docs)

# Bouw index direct bij import (standaard pad: 'corpus' naast dit bestand)
reload_index()

# ---------- Query ----------

def _top_k_idxs(query: str, k: int = 5) -> List[int]:
    if not _VECTORIZER or _MATRIX is None or not _CHUNKS:
        return []
    qv = _VECTORIZER.transform([query])
    sims = cosine_similarity(qv, _MATRIX)[0]
    order = sims.argsort()[::-1]
    return [int(i) for i in order[:k]]

def top_k(query: str, k: int = 5) -> List[str]:
    """Retourneert alleen chunk-teksten."""
    idxs = _top_k_idxs(query, k)
    return [_CHUNKS[i]["text"] for i in idxs]

def top_k_meta(query: str, k: int = 5) -> List[Dict]:
    """Retourneert metadata: id/source/score/preview + volledige tekst."""
    if not _VECTORIZER or _MATRIX is None or not _CHUNKS:
        return []
    qv = _VECTORIZER.transform([query])
    sims = cosine_similarity(qv, _MATRIX)[0]
    idxs = sims.argsort()[::-1][:k]

    results: List[Dict] = []
    for i in idxs:
        t = _CHUNKS[i]["text"]
        preview = re.sub(r"\s+", " ", t)[:280]
        results.append({
            "id": _CHUNKS[i]["id"],
            "source": _CHUNKS[i]["source"],
            "score": float(sims[i]),
            "preview": preview,
            "text": t,
        })
    return results

