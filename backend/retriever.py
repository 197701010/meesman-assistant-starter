# backend/retriever.py
# ------------------------------------------------------------
# Eenvoudige lokale retriever (PDF / TXT / MD) met TF-IDF
# - PDF's: indexeer per pagina  → id: <file>#p<nr>
# - TXT/MD: chunk op alinea's   → id: <file>#s<nr>
# - Absoluut pad naar corpus: <dit bestand>/corpus
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_VERSION = "2025-10-03b"

# ------------------------------------------------------------
# Pad-instellingen
# ------------------------------------------------------------
# Default corpus: .../backend/corpus (absoluut)
_DEFAULT_CORPUS = str((Path(__file__).parent / "corpus").resolve())

# ------------------------------------------------------------
# Ondersteunde extensies
# ------------------------------------------------------------
_EXT_TEXT = {".txt", ".md"}
_EXT_PDF = {".pdf"}

# ------------------------------------------------------------
# Globale index
# ------------------------------------------------------------
_VECTORIZER: Optional[TfidfVectorizer] = None
_MATRIX = None  # scipy sparse matrix
_ITEMS: List[Dict] = []  # elk item: {id, source, text}

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def _clean_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _preview(s: str, n: int = 280) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:n]

# ------------------------------------------------------------
# Readers
# ------------------------------------------------------------
def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"kon tekstbestand niet lezen ({path.name}): {e}")

def _read_pdf_pages(path: Path) -> List[str]:
    """Lees PDF per pagina en geef lijst van teksten (één string per pagina)."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        raise RuntimeError("pypdf niet geïnstalleerd. Run: pip install pypdf")

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise RuntimeError(f"kon PDF niet openen ({path.name}): {e}")

    pages: List[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = _clean_ws(txt)
        if txt:
            pages.append(f"[pagina {i}]\n{txt}")
    return pages

def _chunk_text(text: str, target_len: int = 1200, overlap: int = 200) -> List[str]:
    """Heel simpele alinea-chunker voor .txt/.md."""
    text = _clean_ws(text)
    if not text:
        return []

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    cur_len = 0

    for p in paras:
        if cur_len + len(p) > target_len and buf:
            combined = "\n\n".join(buf).strip()
            chunks.append(combined)
            keep = combined[-overlap:] if overlap and combined else ""
            buf = ([keep] if keep else []) + [p]
            cur_len = len(keep) + len(p)
        else:
            buf.append(p)
            cur_len += len(p)

    if buf:
        chunks.append("\n\n".join(buf).strip())

    if not chunks and text:
        chunks = [text]
    return chunks

# ------------------------------------------------------------
# Corpus laden
# ------------------------------------------------------------
def load_corpus(corpus_dir: str) -> List[Dict]:
    """
    Scan 'corpus_dir' en geef een lijst items terug:
    item = { "id": "<file>#p12" of "<file>#s3", "source": "<abs path>", "text": "<inhoud>" }
    """
    base = Path(corpus_dir)
    if not base.is_absolute():
        base = (Path(__file__).parent / base).resolve()

    if not base.exists():
        print(f"[retriever {_VERSION}] corpus map bestaat niet: {base}")
        return []

    files = sorted([p for p in base.rglob("*") if p.is_file()])
    print(f"[retriever {_VERSION}] gevonden bestanden: {len(files)}, map: {base}")

    items: List[Dict] = []
    loaded_files = 0

    for fp in files:
        ext = fp.suffix.lower()
        try:
            if ext in _EXT_PDF:
                pages = _read_pdf_pages(fp)
                for i, page_text in enumerate(pages, start=1):
                    items.append({
                        "id": f"{fp.name}#p{i}",
                        "source": str(fp),
                        "text": page_text,
                    })
                loaded_files += 1

            elif ext in _EXT_TEXT:
                txt = _read_text_file(fp)
                parts = _chunk_text(txt, target_len=1200, overlap=200) or []
                if not parts:
                    continue
                for i, ch in enumerate(parts, start=1):
                    items.append({
                        "id": f"{fp.name}#s{i}",
                        "source": str(fp),
                        "text": ch,
                    })
                loaded_files += 1

            else:
                # onondersteund bestandstype → skip
                continue

        except Exception as e:
            print(f"[retriever {_VERSION}] overslaan {fp.name}: {e}")

    print(f"[retriever {_VERSION}] geladen items: {len(items)} uit {loaded_files} bestanden")
    return items

# ------------------------------------------------------------
# Index bouwen
# ------------------------------------------------------------
def _build_index(items: List[Dict]) -> None:
    global _VECTORIZER, _MATRIX, _ITEMS

    texts = [it["text"] for it in items]
    if not texts:
        _VECTORIZER = None
        _MATRIX = None
        _ITEMS = []
        print(f"[retriever {_VERSION}] geen documenten om te indexeren")
        return

    # TF-IDF met unigrams+bigrams voor iets betere recall
    _VECTORIZER = TfidfVectorizer(
        stop_words=None,          # NL stopwoorden zitten niet standaard in sklearn
        ngram_range=(1, 2),
        max_features=50000
    )
    _MATRIX = _VECTORIZER.fit_transform(texts)
    _ITEMS = items
    print(f"[retriever {_VERSION}] index gebouwd: {len(_ITEMS)} items")

def reload_index(corpus_dir: str = _DEFAULT_CORPUS) -> None:
    items = load_corpus(corpus_dir)
    _build_index(items)

# Bouw index bij import met default pad
reload_index()

# ------------------------------------------------------------
# Zoeken
# ------------------------------------------------------------
def _top_k_idxs(query: str, k: int = 5) -> List[int]:
    if not query or not _VECTORIZER or _MATRIX is None or not _ITEMS:
        return []
    qv = _VECTORIZER.transform([query])
    sims = cosine_similarity(qv, _MATRIX)[0]
    order = sims.argsort()[::-1]
    # filter eventueel helemaal lege/white-space teksten weg (zou niet moeten gebeuren)
    return [int(i) for i in order[:k]]

def top_k(query: str, k: int = 5) -> List[str]:
    """Compat: geef alleen de tekstvelden terug."""
    idxs = _top_k_idxs(query, k)
    return [_ITEMS[i]["text"] for i in idxs]

def top_k_meta(query: str, k: int = 5) -> List[Dict]:
    """
    Rijke resultaten voor UI/back-end:
    { id, source, score, preview, text }
    """
    if not query or not _VECTORIZER or _MATRIX is None or not _ITEMS:
        return []

    qv = _VECTORIZER.transform([query])
    sims = cosine_similarity(qv, _MATRIX)[0]
    idxs = sims.argsort()[::-1][:k]

    results: List[Dict] = []
    for i in idxs:
        item = _ITEMS[int(i)]
        t = item["text"]
        results.append({
            "id": item["id"],
            "source": item["source"],
            "score": float(sims[int(i)]),
            "preview": _preview(t, 280),
            "text": t,
        })
    return results



