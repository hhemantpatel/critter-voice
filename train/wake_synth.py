"""Make synthetic audio for the Jarvis wake word: many voices saying "Jarvis", and many sentences that are NOT the wake word.

    .venv-train/bin/python train/wake_synth.py            # -> data/wake_synth/{pos,neg}.npz   (about 10-15 minutes)

Positives: Kokoro voices (offline, includes Hindi-voiced and British ones) and Microsoft neural voices (edge-tts, needs internet;
only the words "Jarvis", "Hey Jarvis" ... are sent). Negatives: the sentences Tommy is meant to understand, sentences from the docs,
and look-alike words (service, Marvel, harvest, Travis ...). Audio is held as 16 kHz int16 arrays; nothing personal is involved.
"""
import asyncio, json, random, re, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "wake_synth"
R = random.Random(11)
POS_TEXTS = ["Jarvis.", "Hey Jarvis.", "Hey, Jarvis!", "Hi Jarvis.", "Okay Jarvis.", "Jarvis?", "Hello Jarvis."]
LOOKALIKES = ["Service", "Marvel", "Harvest", "Travis", "Davis", "Garvis", "Marvin", "Jarring", "Carvis", "Nervous", "Purpose", "Hey Siri",
              "Okay Google", "Hey buddy", "Hey Alexa", "Jar", "Jargon", "Harvey", "Hey there", "Hey Tommy", "Charges", "Starvis"]
EDGE_VOICES = ["en-IN-PrabhatNeural", "en-IN-NeerjaNeural", "en-GB-RyanNeural", "en-GB-ThomasNeural", "en-GB-SoniaNeural",
               "en-US-GuyNeural", "en-US-JennyNeural", "en-AU-WilliamNeural", "en-US-AriaNeural", "en-US-ChristopherNeural",
               "en-IE-ConnorNeural", "en-ZA-LukeNeural", "en-NG-AbeoNeural", "en-KE-ChilembaNeural"]


def sentences():
    out = []
    for f in ("train.jsonl", "test.jsonl"):
        p = ROOT / "data" / "out" / f
        if p.exists():
            out += [json.loads(l)["text"] for l in p.read_text().splitlines()]
    for f in ("README.md", "docs/DESIGN.md"):
        txt = (ROOT / f).read_text()
        txt = re.sub(r"[`#*|>\-]|\(.*?\)|\[|\]", " ", txt)
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt) if 25 < len(s.strip()) < 140]
    out = [s for s in dict.fromkeys(out) if "jarvis" not in s.lower()]
    R.shuffle(out)
    return out


def trim(a, thr=0.02):
    """Cut leading and trailing silence from a float array in [-1, 1]."""
    idx = np.where(np.abs(a) > thr * max(1e-6, np.abs(a).max()))[0]
    return a[max(0, idx[0] - 800): idx[-1] + 1200] if len(idx) else a


def to_int16(a):
    return (np.clip(a, -1, 1) * 32767).astype(np.int16)


def kokoro_clips(texts_per_voice, speeds):
    from kokoro_onnx import Kokoro
    from scipy.signal import resample_poly
    k = Kokoro(str(ROOT / "models/tts/kokoro/kokoro-v1.0.onnx"), str(ROOT / "models/tts/kokoro/voices-v1.0.bin"))
    voices = [v for v in k.get_voices() if v[:2] in ("af", "am", "bf", "bm", "hf", "hm")]
    print(f"kokoro: {len(voices)} voices", flush=True)
    clips = []
    for v in voices:
        for text in texts_per_voice(v):
            sp = R.choice(speeds)
            try:
                a, sr = k.create(text, voice=v, speed=sp, lang="en-gb" if v[0] == "b" else "en-us")
            except Exception as e:
                print("skip", v, text, str(e)[:60], flush=True)
                continue
            clips.append(to_int16(trim(resample_poly(a, 16000, sr))))
    return clips


async def edge_clips(texts, voices, rates, pitches, n, concurrency=5):
    import edge_tts
    sem = asyncio.Semaphore(concurrency)
    jobs = [(R.choice(texts), R.choice(voices), R.choice(rates), R.choice(pitches)) for _ in range(n)]

    async def one(text, voice, rate, pitch):
        async with sem:
            try:
                async def fetch():
                    data = b""
                    async for chunk in edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).stream():
                        if chunk["type"] == "audio":
                            data += chunk["data"]
                    return data
                data = await asyncio.wait_for(fetch(), timeout=20)        # one stuck request must not stall the whole run
                pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "-", "-f", "s16le", "-ar", "16000", "-ac", "1", "-"],
                                     input=data, capture_output=True).stdout
                a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
                return to_int16(trim(a)) if len(a) else None
            except Exception as e:
                print("edge skip", voice, str(e)[:60], flush=True)
                return None
    res = await asyncio.gather(*(one(*j) for j in jobs))
    return [r for r in res if r is not None]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sents = sentences()
    print(f"{len(sents)} negative sentences", flush=True)
    # positives
    pos = kokoro_clips(lambda v: R.sample(POS_TEXTS, 5), speeds=[0.85, 1.0, 1.15, 1.3])
    print("kokoro positives:", len(pos), flush=True)
    pos += asyncio.run(edge_clips(POS_TEXTS, EDGE_VOICES, ["-20%", "-10%", "+10%", "+25%"], ["-8Hz", "+8Hz"], 280))
    print("positives total:", len(pos), flush=True)
    np.savez_compressed(OUT / "pos.npz", *pos)
    # negatives
    neg = kokoro_clips(lambda v: R.sample(sents, 12) + R.sample(LOOKALIKES, 4), speeds=[0.9, 1.0, 1.1, 1.25])
    print("kokoro negatives:", len(neg), flush=True)
    neg += asyncio.run(edge_clips(sents + LOOKALIKES * 4, EDGE_VOICES, ["-15%", "+15%"], ["-5Hz", "+5Hz"], 450))
    print("negatives total:", len(neg), "  seconds:", round(sum(len(x) for x in neg) / 16000), flush=True)
    np.savez_compressed(OUT / "neg.npz", *neg)


if __name__ == "__main__":
    main()
