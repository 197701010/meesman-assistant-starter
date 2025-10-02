import glob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Laad alle markdown documenten uit backend/corpus
docs = []
paths = sorted(glob.glob("backend/corpus/*.md"))
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        docs.append(f.read())

if docs:
    vectorizer = TfidfVectorizer(stop_words="dutch")
    X = vectorizer.fit_transform(docs)
else:
    vectorizer = None
    X = None

def top_k(query: str, k: int = 5):
    if not docs or not vectorizer or X is None:
        return []
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X)[0]
    idxs = sims.argsort()[::-1][:k]
    return [docs[i] for i in idxs]
