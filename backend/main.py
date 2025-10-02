import os, requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY")
MISTRAL_BASE_URL = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
MODEL            = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
CORS_ORIGINS     = [o.strip() for o in os.getenv("CORS_ORIGINS","*").split(",")]

app = FastAPI(title="Meesman AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = (
    "Je bent de AI-assistent van Meesman Indexbeleggen. "
    "Beantwoord klantvragen uitsluitend op basis van publieke informatie van Meesman en "
    "algemene beleggingsuitleg. "
    "Geef geen persoonlijk advies of aanbevelingen en verwijs bij persoonlijke situaties naar "
    "Meesman support. Antwoord beknopt, feitelijk, in NL. "
    "Als een vraag naar persoonlijk advies neigt: voeg een korte disclaimer toe."
)

class ChatTurn(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatTurn]
    session_id: str | None = None

def call_mistral(messages: list[dict]):
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    payload = {"model": MODEL, "messages": messages, "temperature": 0.2}
    r = requests.post(f"{MISTRAL_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# --- Optional RAG-light ---
try:
    from .retriever import top_k
except Exception:
    top_k = None

@app.post("/chat")
def chat(req: ChatRequest):
    # Geen server-side opslag van geschiedenis
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Voeg optioneel context toe op basis van laatste uservraag
    user_last = ""
    for m in req.messages[::-1]:
        if m.role == "user":
            user_last = m.content
            break
    if top_k and user_last:
        context_snippets = "\n\n".join(top_k(user_last, k=5))
        context_block = (
            "\n\n---\nContext (alleen gebruiken als het relevant en accuraat is; niet verzinnen):\n"
            f"{context_snippets}\n---\n"
        )
        msgs[0]["content"] += context_block

    # Voeg rest van de conversatie toe
    for m in req.messages:
        if m.role in ("user", "assistant"):
            msgs.append({"role": m.role, "content": m.content})

    answer = call_mistral(msgs)
    return {"answer": answer}
