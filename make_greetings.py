#!/usr/bin/env python
"""Generate the spoken replies Tommy plays on wake, with the offline Kokoro voice of your choice.

    .venv-train/bin/python make_greetings.py                              # default: Microsoft neural voice en-GB-ThomasNeural
    .venv-train/bin/python make_greetings.py edge:en-GB-RyanNeural        # another Microsoft voice (en-IN-PrabhatNeural = Indian English)
    .venv-train/bin/python make_greetings.py bm_fable                     # offline Kokoro voice instead (bm_george, bm_lewis, am_michael ...)

Microsoft voices (edge-tts) need internet ONLY while generating, and only the short fixed phrases below are sent, nothing about you.
Kokoro voices run fully offline.
Writes models/tts/greetings/*.wav once; at run time the voice loop only plays those files (no speech synthesis cost).
"""
import subprocess, sys, wave
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
VOICE = sys.argv[1] if len(sys.argv) > 1 else "edge:en-GB-ThomasNeural"
# Several lines per situation, so Tommy does not repeat itself. File names: <situation>_<n>.wav; the voice loop picks one at random.
PHRASES = {
    "greet": ["Hello sir.", "Yes sir?", "At your service, sir.", "I'm listening, sir.", "Right here, sir.", "How can I help, sir?",
              "Ready when you are, sir.", "Go ahead, sir.", "Good to hear from you, sir.", "What can I do for you, sir?"],
    "done": ["Done, sir.", "Okay sir, done.", "Consider it done, sir.", "All set, sir.", "Taken care of, sir.", "As you wish, sir.",
             "That's sorted, sir.", "Certainly, sir. Done.", "Right away, sir. Done.", "Done and dusted, sir."],
    "nevermind": ["Never mind, sir.", "No problem, sir.", "Understood, sir. Forget it.", "Very well, sir."],
    "cancelled": ["Cancelled, sir.", "Alright, I won't, sir.", "Okay sir, scrapped.", "Understood, sir. Cancelled."],
    "sorry": ["Sorry sir, I didn't catch that.", "Pardon, sir? Could you say that again?", "I'm afraid I missed that, sir.", "Come again, sir?"],
    "cant": ["I couldn't do that, sir.", "I'm afraid that didn't work, sir.", "Sorry sir, that failed.", "Something went wrong there, sir."],
    "what_time": ["What time, sir?", "When should I do it, sir?", "And when would you like that, sir?", "By what time, sir?"],
    "confirm": ["Shall I go ahead, sir?", "Should I proceed, sir?", "Are you sure, sir?", "Shall I do it, sir?", "Do you confirm, sir?"],
}
ITEMS = [(f"{cat}_{i}", text) for cat, texts in PHRASES.items() for i, text in enumerate(texts)]

out = ROOT / "models/tts/greetings"
out.mkdir(parents=True, exist_ok=True)
for old in out.glob("*.wav"):
    old.unlink()                              # drop clips from an earlier voice or an earlier phrase list


def to_wav(src, dst):
    """Any audio file -> 24 kHz mono wav, with leading/trailing silence trimmed (the Microsoft clips start with a pause)."""
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ar", "24000", "-ac", "1", "-af",
                    "silenceremove=start_periods=1:start_threshold=-45dB:stop_periods=1:stop_threshold=-45dB:stop_duration=0.3",
                    str(dst)], check=True)


if VOICE.startswith("edge:"):
    import asyncio, edge_tts, tempfile
    voice = VOICE[5:]

    async def make_all():
        for name, text in ITEMS:
            with tempfile.TemporaryDirectory() as tmp:
                await edge_tts.Communicate(text, voice).save(f"{tmp}/x.mp3")
                to_wav(f"{tmp}/x.mp3", out / f"{name}.wav")
            print(f"{name}: {text!r}")
    asyncio.run(make_all())
else:
    from kokoro_onnx import Kokoro
    LANG = "en-gb" if VOICE.startswith("b") else "en-us"
    k = Kokoro(str(ROOT / "models/tts/kokoro/kokoro-v1.0.onnx"), str(ROOT / "models/tts/kokoro/voices-v1.0.bin"))
    for name, text in ITEMS:
        audio, sr = k.create(text, voice=VOICE, speed=0.95, lang=LANG)
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        with wave.open(str(out / f"{name}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
        print(f"{name}: {text!r} {len(pcm) / sr:.2f}s")
print("voice:", VOICE)
