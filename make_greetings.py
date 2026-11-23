#!/usr/bin/env python
"""Generate the spoken replies Tommy plays on wake, with the offline Kokoro voice of your choice.

    .venv-train/bin/python make_greetings.py                              # default: Microsoft neural voice en-GB-ThomasNeural
    .venv-train/bin/python make_greetings.py edge:en-GB-RyanNeural        # another Microsoft voice (en-IN-PrabhatNeural = Indian English)
    .venv-train/bin/python make_greetings.py bm_fable                     # offline Kokoro voice instead (bm_george, bm_lewis, am_michael ...)

Microsoft voices (edge-tts) need internet ONLY while generating, and only the short fixed phrases below are sent, nothing about you.
Kokoro voices run fully offline.
Writes models/tts/greetings/*.wav once; at run time the voice loop only plays those files (no speech synthesis cost).
"""
import shutil, subprocess, sys, wave
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
VOICE = sys.argv[1] if len(sys.argv) > 1 else "edge:en-GB-ThomasNeural"
# Several lines per situation, so Tommy does not repeat itself. File names: <situation>_<n>.wav; the voice loop picks one at random.
PHRASES = {
    "greet": ["Hello sir.", "Yes sir?", "At your service, sir.", "I'm listening, sir.", "Right here, sir.", "How can I help, sir?",
              "Ready when you are, sir.", "Go ahead, sir.", "What can I do for you, sir?"],
    # one set per kind of Reply (tommy/registry.py)
    "done": ["Done, sir.", "Okay sir, done.", "Consider it done, sir.", "All set, sir.", "Taken care of, sir.", "As you wish, sir.",
             "That's sorted, sir.", "Certainly, sir. Done.", "Right away, sir."],
    "declined": ["Never mind, sir.", "No problem, sir.", "Okay sir, I won't.", "Very well, sir."],
    "cancelled": ["Cancelled, sir.", "Understood, sir. Cancelled.", "Okay sir, it's off."],
    "not_understood": ["Sorry sir, I didn't get that.", "I didn't understand that one, sir.", "I'm afraid I can't do that one yet, sir.", "Pardon, sir?"],
    "no_app": ["There's no app by that name, sir.", "I couldn't find that app, sir.", "I don't see an app with that name, sir."],
    "not_open": ["That one isn't open, sir.", "It's not open, sir."],
    "no_time": ["I couldn't work out the time, sir.", "Sorry sir, I didn't get the time."],
    "failed": ["I couldn't do that, sir.", "I'm afraid that didn't work, sir.", "Something went wrong there, sir."],
    "locked": ["The screen is locked, sir."],
    # questions
    "what_time": ["What time, sir?", "When should I do it, sir?", "And when would you like that, sir?"],
    "remind_what": ["What should I remind you about, sir?", "Remind you about what, sir?"],
    "note_what": ["What should I note down, sir?", "What would you like me to write down, sir?"],
    "confirm": ["Shall I go ahead, sir?", "Should I proceed, sir?", "Are you sure, sir?"],
    "confirm_sleep": ["Shall I put the PC to sleep, sir?"], "confirm_poweroff": ["Shall I shut the PC down, sir?"],
    "confirm_lock": ["Shall I lock the screen, sir?"], "confirm_cancel_scheduled": ["Shall I cancel your timers, sir?"],
    "confirm_close_app": ["Shall I close it, sir?"], "confirm_close_all": ["There are several open, sir. Shall I close them all?"],
    "confirm_hotspot": ["Shall I change the hotspot, sir?"],
    "did_you_mean": ["Did you mean the one on the screen, sir?", "Is this the one you meant, sir?"],
}
ITEMS = [(f"{cat}_{i}", text) for cat, texts in PHRASES.items() for i, text in enumerate(texts)]

final = ROOT / "models/tts/greetings"
out = final.with_name("greetings.new")        # built apart, so a failure half way leaves the working voice alone
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True)


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
shutil.rmtree(final, ignore_errors=True)
out.rename(final)
print("voice:", VOICE)
