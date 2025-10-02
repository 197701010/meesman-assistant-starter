# Meesman Assistant (Starter)

Dit is een minimalistische startset om de Meesman AI-assistent lokaal te draaien
en als demo te tonen. Volg de stappen hieronder.

## 1) Vereisten
- Python 3.10+
- Een Mistral API key (of pas `MISTRAL_BASE_URL` aan voor Azure)

## 2) Installatie
```bash
cd meesman-assistant
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Environment instellen
Kopieer `.env.example` naar `backend/.env` en vul je sleutel(s) in.

## 4) Starten
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `frontend/index.html` in je browser (of host 'm via `python -m http.server`).
Pas in `frontend/widget.js` de `API`-URL aan naar je backend-URL bij deploy.

## 5) (Optioneel) RAG-light
Voeg markdown-bestanden toe in `backend/corpus` (FAQ, beleidsteksten). De eenvoudige
retriever gebruikt TF-IDF cosine similarity om context mee te geven aan het model.

## 6) Deploy (indicatie)
- Backend: Azure App Service / Render / Fly.io (env vars instellen, HTTPS)
- Frontend: Vercel/Netlify of statisch via je CMS

## 7) Veiligheid en compliance (checklist)
- Server logt geen content; chatgeschiedenis is lokaal (browser) met consent
- Disclaimer in system prompt en in UI
- Rate limiting / timeouts
- Periodieke prompt review met Compliance
