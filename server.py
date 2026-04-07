import os
import re
import csv
import time
import uuid
import random
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
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
FFMPEG_BIN = "ffmpeg"

# Latency logging
LOG_DIR = Path(BASE_DIR) / "logs"
LOG_DIR.mkdir(exist_ok=True)
LAT_CSV = LOG_DIR / "latency.csv"

# Fixed TTS cache
TTS_CACHE_DIR = Path(BASE_DIR) / "cache_tts"
TTS_CACHE_DIR.mkdir(exist_ok=True)

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
def clean_text(s: str) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def header_safe(s: str, limit: int = 2000) -> str:
    s = clean_text(s)
    s = s.encode("latin-1", "ignore").decode("latin-1")
    return s[:limit]


def log_latency_row(row: dict):
    header = ["ts", "session_id", "stage", "ffmpeg_ms", "asr_ms", "tts_ms", "total_ms"]
    file_exists = LAT_CSV.exists()
    with open(LAT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def run_ffmpeg_to_wav(in_path: str, out_path: str):
    cmd = [FFMPEG_BIN, "-y", "-i", in_path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def whisper_asr(wav_path: str) -> str:
    out_txt = wav_path + ".txt"
    if os.path.exists(out_txt): os.remove(out_txt)
    cmd = [WHISPER_EXE, "-m", WHISPER_MODEL, "-f", wav_path, "-otxt", "-nt", "-np"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out_txt): raise RuntimeError("ASR output not generated")
    with open(out_txt, "r", encoding="utf-8") as f:
        return f.read().strip()


# ======================
# Streaming TTS Core
# ======================
def stream_piper_tts(text: str):
    """通过管道流式传输音频块，无需等待全文生成完毕"""
    cmd = [
        PIPER_EXE,
        "-m", PIPER_VOICE,
        "-f", "-",  # 输出到 stdout
        "--output_raw"  # 输出原始 PCM 流提高速度
    ]
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    # 写入文本并触发生成
    p.stdin.write(text.encode("utf-8"))
    p.stdin.close()

    # 持续读取音频块并推送
    while True:
        chunk = p.stdout.read(4096)
        if not chunk:
            print("DEBUG: TTS stream finished")  # 添加这行
            break
        yield chunk
    p.wait()


def get_cached_tts_bytes(fixed_text: str) -> bytes:
    fixed_text = clean_text(fixed_text)
    k = re.sub(r"[^a-zA-Z0-9]+", "_", fixed_text.lower())[:80]
    wav_path = TTS_CACHE_DIR / f"{k}.wav"
    if not wav_path.exists():
        cmd = [PIPER_EXE, "-m", PIPER_VOICE, "-f", str(wav_path)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        p.communicate(fixed_text.encode("utf-8"))
    return wav_path.read_bytes()


# ======================
# Logic / State Machine
# ======================
PLATE_RE = re.compile(r"\b([A-Z]{1,2}\d{2,5}\s?[A-Z]{0,3})\b", re.I)
ASK_VEHICLE = "Sure. Please tell me your vehicle full name (make, model, year), your plate number, and the mileage."
ASK_MISSING_PLATE = "I've got your vehicle model, but I still need your plate number. Could you tell me that?"
ASK_ISSUE = "Thanks. What issue are you having with the vehicle? Please describe it in one short sentence."
CONFIRM_MSG = "Confirmed. Thank you. See you then."


def generate_appointment_text() -> str:
    addr = random.choice(["789 Oak Ave", "1420 Maple Street", "55 Broad Street"])
    date = (datetime.now() + timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d")
    return f"I can book you for {date} at {addr}. Does that work?"


def get_or_create_session(session_id: str | None):
    if not session_id or session_id not in SESSIONS:
        session_id = session_id or uuid.uuid4().hex
        SESSIONS[session_id] = {"stage": 0, "plate": None, "issue": None}
    return session_id, SESSIONS[session_id]


def advance_with_user_text(sess: dict, user_text: str):
    t_clean = clean_text(user_text)
    if sess["stage"] == 0:
        if t_clean:
            sess["stage"] = 1
            return (ASK_VEHICLE, 1)
        return (ASK_VEHICLE, 0)

    if sess["stage"] == 1:
        # 槽位填充：尝试提取车牌
        if not sess["plate"]:
            p_match = PLATE_RE.search(t_clean)
            if p_match: sess["plate"] = p_match.group(1).upper()

        # 校验：如果依然缺车牌，停留并追问
        if not sess["plate"]:
            return (ASK_MISSING_PLATE, 1)

        sess["stage"] = 2
        return (ASK_ISSUE, 2)

    if sess["stage"] == 2:
        if t_clean:
            sess["issue"] = t_clean
            sess["stage"] = 3
            return (generate_appointment_text(), 3)
        return (ASK_ISSUE, 2)

    if sess["stage"] == 3:
        if any(k in t_clean.lower() for k in ["yes", "ok", "works", "confirm"]):
            sess["stage"] = 4
            return (CONFIRM_MSG, 4)
        return ("Does the proposed time work for you?", 3)

    return ("You're welcome.", sess["stage"])


# ======================
# Routes
# ======================
@app.get("/", response_class=HTMLResponse)
def root():
    with open(INDEX_HTML, "r", encoding="utf-8") as f: return f.read()


@app.post("/api/chat-audio-wav")
async def chat_audio_wav(
        file: UploadFile = File(...),
        session_id: str | None = Form(default=None),
):
    session_id, sess = get_or_create_session(session_id)
    t_start = time.perf_counter()

    with tempfile.TemporaryDirectory() as td:
        raw_path = os.path.join(td, "in.webm")
        with open(raw_path, "wb") as f:
            f.write(await file.read())

        wav_path = os.path.join(td, "in.wav")
        t0 = time.perf_counter()
        run_ffmpeg_to_wav(raw_path, wav_path)
        ffmpeg_ms = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        user_text = whisper_asr(wav_path)
        asr_ms = int((time.perf_counter() - t1) * 1000)

        assistant_text, stage = advance_with_user_text(sess, user_text)

        # 判断是否可以使用预生成缓存（固定话术）
        fixed_prompts = (ASK_VEHICLE, ASK_ISSUE, CONFIRM_MSG, ASK_MISSING_PLATE)

        headers = {
            "X-Session-Id": session_id,
            "X-Stage": str(stage),
            "X-Assistant-Text": header_safe(assistant_text),
            "X-Latency-ASR-ms": str(asr_ms),
        }

        if assistant_text in fixed_prompts:
            audio_bytes = get_cached_tts_bytes(assistant_text)
            total_ms = int((time.perf_counter() - t_start) * 1000)
            log_latency_row({"ts": datetime.now().strftime("%H:%M:%S"), "session_id": session_id, "stage": stage,
                             "ffmpeg_ms": ffmpeg_ms, "asr_ms": asr_ms, "tts_ms": 0, "total_ms": total_ms})
            return Response(content=audio_bytes, media_type="audio/wav", headers=headers)
        else:
            # --- 修改这里：动态话术不再使用 StreamingResponse，改回 Response ---
            # 这样前端的 URL.createObjectURL 就能识别它是标准的 WAV 文件了
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tts_path = tf.name

            # 使用你之前定义的管道函数，生成一个临时的完整 wav 文件
            cmd = [PIPER_EXE, "-m", PIPER_VOICE, "-f", tts_path]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
            p.communicate(assistant_text.encode("utf-8"))

            audio_bytes = Path(tts_path).read_bytes()
            os.remove(tts_path)  # 读完记得删除临时文件

            total_ms = int((time.perf_counter() - t_start) * 1000)
            log_latency_row({
                "ts": datetime.now().strftime("%H:%M:%S"),
                "session_id": session_id,
                "stage": stage,
                "ffmpeg_ms": ffmpeg_ms,
                "asr_ms": asr_ms,
                "tts_ms": int((time.perf_counter() - t1) * 1000),  # 估算 TTS 时间
                "total_ms": total_ms
            })

            return Response(content=audio_bytes, media_type="audio/wav", headers=headers)