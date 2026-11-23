"""Speed and accuracy of the speech model on the CPU: today's PyTorch pipeline against CTranslate2 int8, on your recordings.
usage: .venv-train/bin/python bench/asr_cpu.py"""
import json, sys, time, wave
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.common import meaning, wer

HINT = ("Jarvis, set a timer, reminder, sleep, shutdown, lock, focus, stopwatch, notes, do not disturb, hotspot. "
        "Open YouTube, WhatsApp, Classroom, Instagram, Spotify, Telegram.")
rows = [json.loads(l) for l in (ROOT / "data/recordings/labels.jsonl").read_text().splitlines()]
clips = []
for r in rows:
    w = wave.open(str(ROOT / "data/recordings" / r["file"]))
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768 * 4
    clips.append((r["said"], np.clip(x, -1, 1)))


def run(name, fn):
    fn(clips[0][1])
    errs = words = tok = ok = 0; ts = []
    for said, x in clips:
        t = time.perf_counter(); hyp = fn(x); ts.append(time.perf_counter() - t)
        e, n = wer(said, hyp); errs += e; words += n
        m = meaning(said)
        if m:
            tok += 1; ok += meaning(hyp) == m
    print(f"{name:44} WER {100 * errs / words:5.1f}%  times {ok}/{tok}  median {np.median(ts):.2f}s  p90 {np.percentile(ts, 90):.2f}s", flush=True)


import torch
from transformers import pipeline
p = pipeline("automatic-speech-recognition", model=str(ROOT / "models/asr/Whisper-Hindi2Hinglish-Swift"), device=-1, dtype=torch.float32)
ids = torch.tensor(p.tokenizer.get_prompt_ids(HINT))
run("PyTorch fp32 + hint (today)", lambda x: p({"raw": x, "sampling_rate": 16000}, generate_kwargs={"prompt_ids": ids})["text"].replace(HINT, "").strip())
del p
from faster_whisper import WhisperModel
for threads in (4, 6):
    m = WhisperModel(str(ROOT / "models/asr/swift-ct2-int8"), device="cpu", compute_type="int8", cpu_threads=threads)
    for beam, hint, lang, chunk in ((1, True, "en", None), (1, False, "en", None), (5, True, "en", None), (1, True, "en", 10), (1, True, None, None)):
        def fn(x, beam=beam, hint=hint, lang=lang, chunk=chunk):
            segs, _ = m.transcribe(x, beam_size=beam, language=lang, initial_prompt=HINT if hint else None, without_timestamps=True,
                                   condition_on_previous_text=False, temperature=0.0, chunk_length=chunk)
            return " ".join(s.text for s in segs).strip()
        try:
            run(f"CT2 int8 t{threads} beam{beam}{' +hint' if hint else ''} lang={lang} chunk={chunk or 30}s", fn)
        except Exception as e:
            print("failed:", beam, hint, lang, chunk, str(e)[:100])
