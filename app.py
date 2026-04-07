import os
import json
import re
import subprocess
from datetime import datetime, timedelta

import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
from jsonschema import validate, ValidationError

# ===== Paths =====
WHISPER_EXE = r".\bin\whisper\whisper-cli.exe"
WHISPER_MODEL = r".\bin\whisper\models\ggml-small.en.bin"

PIPER_EXE = r".\bin\piper\piper.exe"
PIPER_VOICE = r".\bin\piper\en_US-lessac-medium.onnx"

PROMPT_FILE = "repair_prompt.txt"

AUDIO_IN = "input.wav"
AUDIO_TTS = "tts.wav"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

SCHEMA = {
    "type": "object",
    "required": ["summary", "appointment", "questions", "safety_note"],
    "properties": {
        "summary": {"type": "string"},
        "appointment": {
            "type": "object",
            "required": ["date", "time_window", "address"],
            "properties": {
                "date": {"type": "string"},
                "time_window": {"type": "string"},
                "address": {"type": "string"},
            },
        },
        "questions": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "string"}},
        "safety_note": {"type": "string"},
    },
}

def load_system_prompt() -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

def record_wav(path=AUDIO_IN, seconds=6, sr=16000):
    print(f"[REC] Recording {seconds}s... Speak English now.")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype=np.float32)
    sd.wait()
    sf.write(path, audio, sr)
    print("[REC] Saved:", os.path.abspath(path))

def whisper_asr(wav_path=AUDIO_IN) -> str:
    out_txt = wav_path + ".txt"
    if os.path.exists(out_txt):
        os.remove(out_txt)

    cmd = [WHISPER_EXE, "-m", WHISPER_MODEL, "-f", wav_path, "-otxt"]
    subprocess.run(cmd, check=True)

    if not os.path.exists(out_txt):
        raise RuntimeError("ASR output not generated: " + out_txt)

    with open(out_txt, "r", encoding="utf-8") as f:
        return f.read().strip()

def ollama_chat(system_prompt: str, user_text: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {"temperature": 0.3},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]

def extract_json(text: str) -> dict:
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError("No JSON found in model output")
    return json.loads(text[s:e+1])

def clean_strings(x):
    if isinstance(x, dict):
        return {k: clean_strings(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_strings(v) for v in x]
    if isinstance(x, str):
        x = x.replace("\r", " ").replace("\n", " ")
        x = re.sub(r"\s+", " ", x).strip()
        return x
    return x

def check_constraints(obj: dict):
    validate(instance=obj, schema=SCHEMA)

    today = datetime(2026, 2, 2).date()
    d = datetime.strptime(obj["appointment"]["date"], "%Y-%m-%d").date()
    if not ((today + timedelta(days=1)) <= d <= (today + timedelta(days=14))):
        raise ValueError("date out of allowed range")

    tw = obj["appointment"]["time_window"]
    if not re.match(r"^\d{2}:\d{2}-\d{2}:\d{2}$", tw):
        raise ValueError("time_window invalid")

    addr = obj["appointment"]["address"]
    if addr.count(",") < 2 or not re.search(r"\d", addr):
        raise ValueError("address invalid format")

def render_speech(obj: dict) -> str:
    ap = obj["appointment"]
    q = " ".join([f"{i+1}. {s}" for i, s in enumerate(obj["questions"])])
    t = (
        f"{obj['summary']} "
        f"I can book you for {ap['date']} between {ap['time_window']}. "
        f"The address is {ap['address']}. "
        f"Before we proceed: {q} "
        f"Safety note: {obj['safety_note']}"
    )
    return re.sub(r"\s+", " ", t).strip()

def piper_tts(text: str, out_wav=AUDIO_TTS):
    cmd = [PIPER_EXE, "-m", PIPER_VOICE, "-f", out_wav]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True, encoding="utf-8")
    p.communicate(text)
    if p.returncode != 0:
        raise RuntimeError("Piper TTS failed")

def play_wav(path: str):
    data, sr = sf.read(path, dtype=np.float32)
    sd.play(data, sr)
    sd.wait()

def main():
    system_prompt = load_system_prompt()

    record_wav()
    user_text = whisper_asr(AUDIO_IN)
    print("[ASR]", user_text)

    last_err = None
    for attempt in range(3):
        raw = ollama_chat(system_prompt, user_text)
        try:
            obj = clean_strings(extract_json(raw))
            check_constraints(obj)
            print("[JSON OK]", json.dumps(obj, ensure_ascii=False, indent=2))
            speak = render_speech(obj)
            print("[TTS TEXT]", speak)
            piper_tts(speak)
            play_wav(AUDIO_TTS)
            return
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_err = str(e)
            user_text = user_text + f"\nYour output failed validation: {last_err}. Rewrite RAW JSON only."
            print(f"[RETRY {attempt+1}] {last_err}")

    raise RuntimeError("Failed after retries: " + (last_err or "unknown"))

if __name__ == "__main__":
    main()
