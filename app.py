import uuid
import csv
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# from nlp import HR_QUESTIONS, build_report, score_answer, extract_skills_from_text, extract_name, extract_experience, get_grade
# from storage import save_report  # only needed for classic NLP interview
from stt import transcribe_audio_bytes
from groq_interviewer import GroqSession

# Thread pool for STT (Whisper) and Groq calls
_executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory session store: {session_id: {answers: {}, name: ""}}
SESSIONS: dict = {}

# Groq AI interview sessions: {session_id: GroqSession}
AI_SESSIONS: dict = {}

# ─── Pages ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    candidates = []
    csv_path = "candidates.csv"
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidates.append(row)
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "candidates": candidates,
    })


# ── Classic NLP interview pages — disabled (nlp.py is commented out) ──────────
# @app.get("/interview/{session_id}", response_class=HTMLResponse)
# async def interview_page(request: Request, session_id: str): ...
#
# @app.get("/results/{session_id}", response_class=HTMLResponse)
# async def results_page(request: Request, session_id: str): ...


# ─── API ──────────────────────────────────────────────────────────────────────

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Receive a raw audio blob from the browser (MediaRecorder output) and
    return its Whisper transcription as JSON.
    """
    audio_bytes = await audio.read()
    content_type = audio.content_type or "audio/webm"
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        _executor, transcribe_audio_bytes, audio_bytes, content_type
    )
    return JSONResponse({"transcript": text})


# ── Classic NLP interview API — disabled (nlp.py is commented out) ────────────
# @app.post("/api/start") — start classic interview session
# @app.get("/api/question/{session_id}/{q_index}") — get fixed HR question
# @app.post("/api/answer/{session_id}") — submit + score answer with Word2Vec
# @app.post("/api/finish/{session_id}") — build final NLP report


# ─── Groq AI Interview ───────────────────────────────────────────────────────


@app.get("/ai-voice-interview/{session_id}", response_class=HTMLResponse)
async def ai_voice_interview_page(request: Request, session_id: str):
    if session_id not in AI_SESSIONS:
        return RedirectResponse("/")
    return templates.TemplateResponse("ai_voice_interview.html", {
        "request": request,
        "session_id": session_id,
    })


@app.post("/api/ai/start")
async def ai_start(name: str = Form(...), role: str = Form(default="Software Developer")):
    session_id = str(uuid.uuid4())
    AI_SESSIONS[session_id] = GroqSession(candidate_name=name, job_role=role)
    return JSONResponse({"session_id": session_id})


@app.post("/api/ai/chat/{session_id}")
async def ai_chat(session_id: str, request: Request):
    if session_id not in AI_SESSIONS:
        return JSONResponse({"error": "Invalid session"}, status_code=404)
    body = await request.json()
    user_message = body.get("message", "").strip()
    if not user_message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    session: GroqSession = AI_SESSIONS[session_id]
    loop = asyncio.get_event_loop()
    # Run the blocking Groq streaming call in a thread
    reply = await loop.run_in_executor(_executor, session.chat, user_message)

    return JSONResponse({
        "reply": reply,
        "finished": session.is_finished,
    })


@app.get("/api/ai/session-info/{session_id}")
async def ai_session_info(session_id: str):
    if session_id not in AI_SESSIONS:
        return JSONResponse({"error": "Invalid session"}, status_code=404)
    session: GroqSession = AI_SESSIONS[session_id]
    return JSONResponse({
        "name": session.candidate_name,
        "role": session.job_role,
    })



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
