"""Kasauti — regression harness for Sarvam speech models.

One request = one test case:
  text --(Bulbul TTS)--> clean audio --(+noise @ SNR)--> test audio
       --(STT baseline model)-->  transcript A --> WER/CER + LLM judge
       --(STT candidate model)--> transcript B --> WER/CER + LLM judge

Both STT models hear the *same* audio, so any difference is the model, not the input.
Stdlib only (no deps) so it runs anywhere: Vercel, CI, a laptop.
"""
import base64
import io
import json
import math
import os
import random
import re
import struct
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

API = "https://api.sarvam.ai"

# ---------------------------------------------------------------- test suite
SUITE = [
    {"id": 1, "lang": "hi-IN", "cat": "code-mix", "text": "कल सुबह 10:30 बजे मीटिंग है, प्लीज़ कन्फर्म कर देना।"},
    {"id": 2, "lang": "hi-IN", "cat": "numbers", "text": "आपका बिल ₹2,450 है और ड्यू डेट 15 नवंबर है।"},
    {"id": 3, "lang": "en-IN", "cat": "entities", "text": "Please book a cab from Gachibowli to Kempegowda International Airport."},
    {"id": 4, "lang": "hi-IN", "cat": "native", "text": "मेरे खाते से पाँच हज़ार रुपये कट गए हैं लेकिन रिचार्ज नहीं हुआ।"},
    {"id": 5, "lang": "ta-IN", "cat": "native", "text": "நாளை காலை ஒன்பது மணிக்கு மருத்துவரை சந்திக்க வேண்டும்."},
    {"id": 6, "lang": "te-IN", "cat": "native", "text": "నా ఆర్డర్ ఇంకా డెలివరీ కాలేదు, దయచేసి చెక్ చేయండి."},
    {"id": 7, "lang": "en-IN", "cat": "numbers", "text": "My OTP is 4 8 2 9 1 7 and my account number ends in 3305."},
    {"id": 8, "lang": "hi-IN", "cat": "code-mix", "text": "मेरा UPI पेमेंट फेल हो गया, ट्रांजैक्शन ID चेक करो।"},
    {"id": 9, "lang": "bn-IN", "cat": "native", "text": "আমি কাল কলকাতা থেকে দিল্লি যাচ্ছি।"},
    {"id": 10, "lang": "mr-IN", "cat": "native", "text": "माझी ट्रेन पुणे स्टेशनवरून सहा वाजता सुटते."},
    {"id": 11, "lang": "kn-IN", "cat": "code-mix", "text": "ನನಗೆ ಬೆಂಗಳೂರಿನಲ್ಲಿ ಒಂದು 2BHK flat ಬೇಕು."},
    {"id": 12, "lang": "en-IN", "cat": "entities", "text": "Dr. Venkataraman prescribed Pantoprazole 40 mg twice daily."},
]


# ---------------------------------------------------------------- HTTP helpers
def _key():
    k = (os.environ.get("SARVAM_API_KEY") or os.environ.get("Sarvam_API_key") or "").strip()
    if not k:
        raise RuntimeError("SARVAM_API_KEY is not set")
    return k


def _post(path, body=None, headers=None, timeout=45):
    h = {"api-subscription-key": _key()}
    h.update(headers or {})
    req = urllib.request.Request(API + path, data=body, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"{path} HTTP {e.code}: {detail}")


def _post_json(path, payload, extra_headers=None, timeout=45):
    h = {"Content-Type": "application/json"}
    h.update(extra_headers or {})
    return _post(path, json.dumps(payload).encode(), h, timeout)


def _post_multipart(path, fields, filename, filedata, mime="audio/wav"):
    b = "----kasauti" + uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
              f"Content-Type: {mime}\r\n\r\n".encode())
    out.write(filedata)
    out.write(f"\r\n--{b}--\r\n".encode())
    return _post(path, out.getvalue(), {"Content-Type": f"multipart/form-data; boundary={b}"})


# ---------------------------------------------------------------- Sarvam calls
def tts(text, lang, model):
    payload = {"text": text, "language_code": lang, "model": model,
               "speech_sample_rate": 16000, "output_audio_codec": "wav"}
    t = time.time()
    r = _post_json("/text-to-speech", payload)
    return base64.b64decode(r["audios"][0]), round(time.time() - t, 2)


def stt(wav_bytes, lang, model):
    t = time.time()
    r = _post_multipart("/speech-to-text", {"model": model, "language_code": lang}, "case.wav", wav_bytes)
    return r.get("transcript", ""), round(time.time() - t, 2)


_JUDGE_MODELS = [m for m in [os.environ.get("JUDGE_MODEL"), "sarvam-105b", "sarvam-m"] if m]
_judge_ok = None


def judge(reference, hypothesis, lang):
    """LLM-as-judge: does the transcript preserve the *meaning*? (WER can't tell ₹500 from 'five hundred rupees')"""
    global _judge_ok
    prompt = (
        "You grade speech-to-text output. Compare the REFERENCE (what was spoken) with the TRANSCRIPT.\n"
        "Ignore punctuation, casing, script (Devanagari vs Latin transliteration) and number formatting "
        "(e.g. '10:30' == 'साढ़े दस' == 'ten thirty'). Penalise wrong numbers, names, amounts, dropped or "
        "changed meaning.\n"
        f"Language: {lang}\nREFERENCE: {reference}\nTRANSCRIPT: {hypothesis}\n\n"
        'Reply with ONLY compact JSON: {"score": <0-10 integer>, "pass": <true if score>=7>, "issue": "<max 12 words>"}'
    )
    models = [_judge_ok] if _judge_ok else _JUDGE_MODELS
    last = None
    for m in models:
        try:
            r = _post_json("/v1/chat/completions",
                           {"model": m, "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0, "max_tokens": 1500},
                           {"Authorization": "Bearer " + _key()}, timeout=24)
            msg = r["choices"][0]["message"]
            txt = (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
            found = re.findall(r"\{[^{}]*\"score\"[^{}]*\}", txt, flags=re.S)
            if not found:
                raise ValueError("no JSON verdict in reply: " + txt.strip()[:80])
            data = json.loads(found[0])
            score = int(float(data.get("score", 0)))
            pv = data.get("pass", score >= 7)
            if isinstance(pv, str):
                pv = pv.strip().lower() in ("true", "yes", "1")
            _judge_ok = m
            return {"score": score, "pass": bool(pv),
                    "issue": str(data.get("issue", ""))[:120], "model": m}
        except Exception as e:  # try next model
            last = str(e)
    return {"score": None, "pass": None, "issue": f"judge error: {last}"[:160], "model": None}


# ---------------------------------------------------------------- audio + metrics
def add_noise(wav_bytes, snr_db, seed=7):
    """Mix white noise at a target SNR (dB). Same seed => identical audio for every model."""
    if snr_db is None:
        return wav_bytes
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            params = w.getparams()
            frames = w.readframes(w.getnframes())
    except Exception:
        return wav_bytes  # unknown container: test on clean audio rather than fail
    if params.sampwidth != 2 or not frames:
        return wav_bytes
    n = len(frames) // 2
    s = struct.unpack(f"<{n}h", frames)
    p_sig = sum(x * x for x in s) / max(n, 1)
    sigma = math.sqrt(p_sig / (10 ** (snr_db / 10))) if p_sig else 0
    rng = random.Random(seed)
    g = rng.gauss
    noisy = [max(-32768, min(32767, int(x + g(0, sigma)))) for x in s]
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params)
        w.writeframes(struct.pack(f"<{n}h", *noisy))
    return out.getvalue()


def _norm(t):
    t = unicodedata.normalize("NFC", t.lower()).replace("\u093c", "")
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return t.split()


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref, hyp):
    r, h = _norm(ref), _norm(hyp)
    return round(_lev(r, h) / max(len(r), 1), 3)


def cer(ref, hyp):
    r, h = "".join(_norm(ref)), "".join(_norm(hyp))
    return round(_lev(list(r), list(h)) / max(len(r), 1), 3)


# ---------------------------------------------------------------- one test case
def run_case(case, tts_model="bulbul:v3", base="saaras:v3", cand="saaras:v4", snr=None, want_audio=True):
    lang, text = case["lang"], case["text"]
    try:
        clean, t_tts = tts(text, lang, tts_model)
    except Exception as e:
        return {"case": case, "error": f"TTS failed: {e}"}
    audio = add_noise(clean, snr)

    def leg(model):
        try:
            hyp, lat = stt(audio, lang, model)
        except Exception as e:
            return {"model": model, "error": str(e)[:200]}
        return {"model": model, "transcript": hyp, "latency_s": lat,
                "wer": wer(text, hyp), "cer": cer(text, hyp), "judge": judge(text, hyp, lang)}

    with ThreadPoolExecutor(2) as ex:
        a, b = ex.map(leg, [base, cand])
    res = {"case": case, "tts_model": tts_model, "tts_latency_s": t_tts,
           "snr_db": snr, "baseline": a, "candidate": b}
    if want_audio:
        res["audio_b64"] = base64.b64encode(audio).decode()
    return res


# ---------------------------------------------------------------- Vercel handler
class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _params(self):
        q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        if self.command == "POST":
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                try:
                    q.update(json.loads(self.rfile.read(n).decode()))
                except Exception:
                    pass
        return q

    def _handle(self):
        p = self._params()
        if p.get("suite") or not (p.get("id") or p.get("text")):
            return self._send(200, {"suite": SUITE})
        if p.get("text"):
            case = {"id": 0, "lang": p.get("lang", "hi-IN"), "cat": "custom", "text": str(p["text"])[:300]}
        else:
            case = next((c for c in SUITE if str(c["id"]) == str(p["id"])), None)
            if not case:
                return self._send(404, {"error": "unknown case id"})
        snr = p.get("snr")
        snr = None if snr in (None, "", "clean", "none") else float(snr)
        try:
            out = run_case(case, p.get("tts", "bulbul:v3"), p.get("base", "saaras:v3"),
                           p.get("cand", "saaras:v4"), snr, str(p.get("audio", "1")) != "0")
            self._send(200, out)
        except Exception as e:
            self._send(500, {"error": str(e)[:300], "case": case})

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
