# server.py (State machine, only 1 LLM call for appointment generation)
# pip install fastapi uvicorn python-multipart requests
# Run:
#   uvicorn server:app --host 127.0.0.1 --port 8000

import os
import re
import csv
import time
import uuid
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# ======================
# Paths / Config
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WHISPER_EXE = os.path.join(BASE_DIR, "bin", "whisper", "whisper-cli.exe")
WHISPER_MODEL = os.path.join(BASE_DIR, "bin", "whisper", "models", "ggml-tiny.en.bin")

PIPER_EXE = os.path.join(BASE_DIR, "bin", "piper", "piper.exe")
PIPER_VOICE = os.path.join(BASE_DIR, "bin", "piper", "en_US-lessac-medium.onnx")

INDEX_HTML = os.path.join(BASE_DIR, "web", "index.html")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

FFMPEG_BIN = "ffmpeg"

# Latency logging
LOG_DIR = Path(BASE_DIR) / "logs"
LOG_DIR.mkdir(exist_ok=True)
LAT_CSV = LOG_DIR / "latency.csv"

# In-memory sessions
SESSIONS = {}

# ======================
# App
# ======================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# Utilities
# ======================
def log_latency_row(row: dict):
    header = ["ts", "session_id", "stage", "ffmpeg_ms", "asr_ms", "llm_ms", "tts_ms", "total_ms"]
    file_exists = LAT_CSV.exists()
    with open(LAT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def run_ffmpeg_to_wav(in_path: str, out_path: str):
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", in_path,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def whisper_asr(wav_path: str) -> str:
    out_txt = wav_path + ".txt"
    if os.path.exists(out_txt):
        os.remove(out_txt)
    cmd = [WHISPER_EXE, "-m", WHISPER_MODEL, "-f", wav_path, "-otxt"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out_txt):
        raise RuntimeError("ASR output not generated")
    with open(out_txt, "r", encoding="utf-8") as f:
        return f.read().strip()

def piper_tts(text: str, out_wav: str):
    cmd = [PIPER_EXE, "-m", PIPER_VOICE, "-f", out_wav]
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
    )
    p.communicate(text)
    if p.returncode != 0:
        raise RuntimeError("Piper TTS failed")

def clean_text(s: str) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def header_safe(s: str, limit: int = 2000) -> str:
    s = clean_text(s)
    s = s.encode("latin-1", "ignore").decode("latin-1")
    return s[:limit]

# ======================
# Extractors
# ======================
PLATE_RE = re.compile(r"\b([A-Z]{2}\d{2}\s?[A-Z]{3})\b")  # UK-like: AB12 CDE

def extract_plate(text: str) -> str | None:
    m = PLATE_RE.search(text.upper())
    return m.group(1).replace(" ", "") if m else None

def extract_mileage(text: str) -> str | None:
    t = text.lower()
    m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{4,6})\b", t)
    if m:
        val = m.group(1).replace(",", "")
        if len(val) >= 4:
            return val
    m2 = re.search(r"\b(\d{1,3})\s*(k|thousand)\b", t)
    if m2:
        return str(int(m2.group(1)) * 1000)
    return None

def likely_vehicle_name(text: str) -> bool:
    t = (text or "").strip()
    if len(t.split()) < 3:
        return False
    low = t.lower()

    if any(p in low for p in ["my vehicle is", "my car is", "it's a", "it is a", "vehicle is", "car is"]):
        return True

    if re.search(r"\b(19\d{2}|20\d{2})\b", low):
        return True

    makes = [
        "toyota","honda","ford","bmw","audi","nissan","hyundai","kia","chevrolet",
        "mercedes","volkswagen","tesla","mazda","subaru","lexus","volvo","porsche",
        "jeep","gmc","ram","peugeot","renault","skoda","seat"
    ]
    if any(m in low for m in makes):
        return True

    return False

# ======================
# LLM (ONLY once)
# ======================
def ollama_generate_appointment(vehicle: str, plate: str, mileage: str, issue: str | None) -> str:
    sys = (
        "You are a car repair service receptionist. "
        "Your goal is to propose ONE appointment quickly.\n"
        "Rules:\n"
        "- Output plain English only.\n"
        "- Maximum 2 sentences.\n"
        "- Include a date within the next 14 days relative to 2026-02-02.\n"
        "- Include a 2-hour time window in 24h format like 15:00-17:00.\n"
        "- Include a realistic address with street number, street name, city, and state.\n"
        "- End with: Does that work?\n"
    )

    user = (
        f"Customer details:\n"
        f"- Vehicle: {vehicle}\n"
        f"- Plate: {plate}\n"
        f"- Mileage: {mileage}\n"
        f"- Issue: {issue or 'N/A'}\n"
        f"Propose an appointment now."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 120,
        },
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    text = r.json()["message"]["content"]
    return clean_text(text)

# ======================
# State machine logic (simplified)
# Stage 0: collect vehicle/plate/mileage
# Stage 1: ask issue description (only once)
# Stage 2: LLM proposes appointment (only once)
# Stage 3: confirm
# ======================
ASK_VEHICLE = "Sure. Please tell me your vehicle full name (make, model, year), your plate number, and the mileage."
ASK_ISSUE_ONE = "Thanks. What issue are you having with the vehicle? Please describe it in one short sentence."
CONFIRM_MSG = "Confirmed. Thank you - see you then."

def get_or_create_session(session_id: str | None):
    if not session_id:
        session_id = uuid.uuid4().hex
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "stage": 0,
            "vehicle": None,
            "plate": None,
            "mileage": None,
            "issue": None,
            "created_ts": time.time(),
        }
    return session_id, SESSIONS[session_id]

def advance_with_user_text(sess: dict, user_text: str):
    t_clean = clean_text(user_text)

    # Stage 0: accept whatever user says as "vehicle info" (no hard checks)
    if sess["stage"] == 0:
        if t_clean:
            # best-effort extraction (optional)
            if likely_vehicle_name(t_clean) and not sess.get("vehicle"):
                sess["vehicle"] = t_clean[:80]
            p = extract_plate(t_clean)
            if p and not sess.get("plate"):
                sess["plate"] = p
            m = extract_mileage(t_clean)
            if m and not sess.get("mileage"):
                sess["mileage"] = m

            # NEW: do NOT require vehicle/plate/mileage to be present
            sess["stage"] = 1
            return (ASK_VEHICLE, sess["stage"])

        # if ASR empty, ask again
        return (ASK_VEHICLE, sess["stage"])

    # Stage 1: collect issue description
    if sess["stage"] == 1:
        if t_clean:
            sess["issue"] = t_clean
            sess["stage"] = 2
            return ("__CALL_LLM_APPOINTMENT__", sess["stage"])
        else:
            return (ASK_ISSUE_ONE, sess["stage"])

    # Stage 2: should immediately call LLM
    if sess["stage"] == 2:
        return ("__CALL_LLM_APPOINTMENT__", sess["stage"])

    # Stage 3: confirm
    if sess["stage"] == 3:
        low = t_clean.lower()
        if any(k in low for k in ["yes", "yeah", "ok", "okay", "works", "that works", "confirm", "book it", "thanks", "thank you"]):
            sess["stage"] = 4
            return (CONFIRM_MSG, sess["stage"])
        return ("No problem. What day and time window would you prefer?", sess["stage"])

    return ("You're welcome.", sess["stage"])

# ======================
# Routes
# ======================
@app.get("/", response_class=HTMLResponse)
def root():
    with open(INDEX_HTML, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health():
    return {
        "ok": True,
        "whisper_exe": os.path.exists(WHISPER_EXE),
        "whisper_model": os.path.exists(WHISPER_MODEL),
        "piper_exe": os.path.exists(PIPER_EXE),
        "piper_voice": os.path.exists(PIPER_VOICE),
        "latency_csv": str(LAT_CSV),
        "sessions": len(SESSIONS),
    }

@app.get("/api/latency")
def get_latency(limit: int = 50):
    if not LAT_CSV.exists():
        return {"rows": []}
    rows = []
    with open(LAT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    limit = max(1, min(int(limit), 500))
    return {"rows": rows[-limit:]}

@app.post("/api/chat-audio-wav")
async def chat_audio_wav(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
):
    session_id, sess = get_or_create_session(session_id)
    t_total0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as td:
        raw_path = os.path.join(td, file.filename or "input.webm")
        with open(raw_path, "wb") as f:
            f.write(await file.read())

        wav_path = os.path.join(td, "input.wav")

        # ffmpeg
        try:
            t0 = time.perf_counter()
            run_ffmpeg_to_wav(raw_path, wav_path)
            t1 = time.perf_counter()
            ffmpeg_ms = int((t1 - t0) * 1000)
        except Exception as e:
            return JSONResponse({"error": "ffmpeg_convert_failed", "detail": str(e)}, status_code=500)

        # ASR
        try:
            t2 = time.perf_counter()
            user_text = whisper_asr(wav_path)
            t3 = time.perf_counter()
            asr_ms = int((t3 - t2) * 1000)
        except Exception as e:
            return JSONResponse({"error": "asr_failed", "detail": str(e)}, status_code=500)

        user_text = clean_text(user_text)

        llm_ms = 0
        tts_ms = 0

        assistant_text, stage = advance_with_user_text(sess, user_text)

        if assistant_text == "__CALL_LLM_APPOINTMENT__":
            try:
                t4 = time.perf_counter()
                assistant_text = ollama_generate_appointment(
                    vehicle=sess["vehicle"] or "Unknown vehicle",
                    plate=sess["plate"] or "Unknown plate",
                    mileage=sess["mileage"] or "Unknown mileage",
                    issue=sess.get("issue"),
                )
                t5 = time.perf_counter()
                llm_ms = int((t5 - t4) * 1000)

                # after LLM proposal, move to confirm stage
                sess["stage"] = 3
                stage = 3
            except Exception as e:
                return JSONResponse({"error": "llm_failed", "detail": str(e)}, status_code=500)

        # TTS
        try:
            t6 = time.perf_counter()
            tts_path = os.path.join(td, "reply.wav")
            piper_tts(assistant_text, tts_path)
            t7 = time.perf_counter()
            tts_ms = int((t7 - t6) * 1000)
        except Exception as e:
            return JSONResponse({"error": "tts_failed", "detail": str(e)}, status_code=500)

        total_ms = int((time.perf_counter() - t_total0) * 1000)

        log_latency_row({
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "stage": stage,
            "ffmpeg_ms": ffmpeg_ms,
            "asr_ms": asr_ms,
            "llm_ms": llm_ms,
            "tts_ms": tts_ms,
            "total_ms": total_ms,
        })

        with open(tts_path, "rb") as af:
            audio_bytes = af.read()

        headers = {
            "X-Session-Id": session_id,
            "X-Stage": str(stage),
            "X-User-Text": header_safe(user_text, 2000),
            "X-Assistant-TTS-Text": header_safe(assistant_text, 2000),
            "X-Latency-FFmpeg-ms": str(ffmpeg_ms),
            "X-Latency-ASR-ms": str(asr_ms),
            "X-Latency-LLM-ms": str(llm_ms),
            "X-Latency-TTS-ms": str(tts_ms),
            "X-Latency-Total-ms": str(total_ms),
        }

        return Response(content=audio_bytes, media_type="audio/wav", headers=headers)