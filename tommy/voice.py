"""Voice front end: mic -> wake word ("hey jarvis") -> voice-activity detection -> speech-to-text -> Tommy agent -> bubble.

    .venv-train/bin/python -m tommy.voice                 # always listening for "hey jarvis"
    .venv-train/bin/python -m tommy.voice --no-wake       # press Enter to talk instead (no wake word)
    .venv-train/bin/python -m tommy.voice --wav clip.wav  # run one recording through the whole chain

Audio is held in memory only, never written to disk. Needs the packages in .venv-train.
"""
import argparse, collections, difflib, importlib.machinery, re, subprocess, sys, time, types, wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RATE, FRAME = 16000, 1280            # 80 ms frames (what the wake word model expects)
GAIN = 4.0                           # the internal mic is quiet (peak ~4% of full scale); wake word and VAD like louder input
WAKE_THRESHOLD, COOLDOWN = 0.4, 2.0
SPEECH_P, QUIET_P, END_SILENCE_FRAMES, MIN_SPEECH_FRAMES = 0.4, 0.15, 21, 3   # speech starts above 0.4; only frames below 0.15 count as quiet; end = 1.7 s of quiet (21 x 80 ms)
PREROLL = 4
HINT = "Jarvis, set a timer, reminder, sleep, shutdown, lock, focus, stopwatch, notes, do not disturb, hotspot."   # nudges the speech model toward our commands
SEG_QUIET_FRAMES, SEG_MAX_FRAMES = 7, 40       # a wake phrase is a short burst: ends after 0.56 s of quiet, longer than 3.2 s = ordinary talk
YES = {"yes", "yeah", "yep", "haan", "han", "ha", "ok", "okay", "sure", "theek", "thik", "kar", "karo", "confirm"}
NO = {"no", "nope", "nahi", "nahin", "mat", "cancel", "stop", "ruko"}


def _stub_sklearn():
    # openWakeWord imports scikit-learn only for training a custom verifier; the detector never uses it.
    for name, attrs in {"sklearn": [], "sklearn.linear_model": ["LogisticRegression"], "sklearn.pipeline": ["make_pipeline"],
                        "sklearn.preprocessing": ["FunctionTransformer", "StandardScaler"]}.items():
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)      # torch calls find_spec() on every loaded module
        for attr in attrs:
            setattr(mod, attr, object)
        sys.modules.setdefault(name, mod)


def after_jarvis(text):
    """If the text contains something like "Jarvis" (speech models spell it many ways), return what follows it, else None."""
    for m in re.finditer(r"[A-Za-z']+", text):
        if difflib.SequenceMatcher(None, "jarvis", m.group(0).lower()).ratio() >= 0.62:
            return text[m.end():].strip(" ,.!?-")
    return None


def make_beep(path, hz=880, ms=110, volume=0.35):
    """A short synthesised tone (not recorded audio), with a fade in/out so it does not click."""
    t = np.arange(int(RATE * ms / 1000)) / RATE
    env = np.minimum(1, np.minimum(t, t[-1] - t) / 0.01)
    tone = (np.sin(2 * np.pi * hz * t) * env * volume * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(tone.tobytes())


INFO_TOOLS = {"list_scheduled", "focus_status", "network_status", "read_notes"}


def reply_clip(reply):
    """Which situation a text reply belongs to (Tommy's replies are short, fixed-pattern sentences)."""
    if reply.startswith("Never mind"):
        return "nevermind"
    if reply.startswith("Cancelled"):
        return "cancelled"
    if reply.startswith(("Sorry", "I didn't", "I wasn't sure")):
        return "sorry"
    if reply.startswith(("Couldn't", "The screen is locked")):
        return "cant"
    return "done"


_last = {}


def play(mic, situation, beep_path=None):
    """Say one of the recorded lines for this situation ("greet", "done", "confirm" ...), chosen at random but never the same
    line twice in a row; a beep if the voice files are missing. The mic audio recorded meanwhile is thrown away so Tommy does
    not hear its own voice."""
    import random
    clips = sorted((ROOT / "models/tts/greetings").glob(f"{situation}_*.wav"))
    if len(clips) > 1:
        clips = [c for c in clips if c != _last.get(situation)]
    path = random.choice(clips) if clips else beep_path
    if path is None:
        return
    _last[situation] = path
    with wave.open(str(path)) as w:
        seconds = w.getnframes() / w.getframerate()
    subprocess.run(["pw-play", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(int(seconds / 0.08) + 4):        # the clip, its room echo, and a little margin
        mic.read()


class Mic:
    # parec, not pw-record: pw-record writes a big-endian .snd stream to a pipe, which reads as loud noise.
    def __init__(self):
        self.proc = subprocess.Popen(["parec", "--rate", str(RATE), "--channels", "1", "--format", "s16le", "--latency-msec", "40"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self):
        raw = self.proc.stdout.read(FRAME * 2)
        if len(raw) < FRAME * 2:
            raise EOFError("microphone stream ended")
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) * GAIN
        return np.clip(x, -32768, 32767).astype(np.int16)

    def drain(self):
        """Throw away audio that piled up while we were busy (speech-to-text takes seconds), so the next
        wake-word check only sees what is said from now on. Buffered reads return instantly; a live one blocks ~80 ms."""
        while True:
            t = time.monotonic()
            self.read()
            if time.monotonic() - t > 0.04:
                return

    def close(self):
        self.proc.terminate()


class Ears:
    def __init__(self, gpu=False):
        import torch
        from transformers import pipeline
        _stub_sklearn()
        from openwakeword.vad import VAD
        from tommy.wake import JarvisWake
        self.wake = JarvisWake()                  # trained on this user's voice; hears sound features only, never words
        self.vad = VAD()
        self._seg, self._pre, self._over, self._speech, self._quiet = [], collections.deque(maxlen=PREROLL), False, 0, 0
        dev = 0 if (gpu and torch.cuda.is_available()) else -1
        self.asr = pipeline("automatic-speech-recognition", model=str(ROOT / "models/asr/Whisper-Hindi2Hinglish-Swift"),
                            device=dev, dtype=torch.float16 if dev == 0 else torch.float32)
        ids = self.asr.tokenizer.get_prompt_ids(HINT)
        self.prompt_ids = torch.tensor(ids) if not isinstance(ids, torch.Tensor) else ids

    def woke(self, frame):
        return self.wake.predict(frame) >= self.wake.threshold

    def capture(self, mic, wait_s=5.0, max_s=14.0):
        """Record from the first speech until 1.7 s of quiet. Returns float32 audio, or None if nobody spoke."""
        self.vad.reset_states()
        pre, frames, speech, quiet = collections.deque(maxlen=PREROLL), [], 0, 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_s:
            f = mic.read()
            voiced = self.vad.predict(f, frame_size=320) > SPEECH_P
            if not frames:
                pre.append(f)
                if voiced:
                    frames, speech = list(pre), 1
                elif time.monotonic() - t0 > wait_s:
                    return None
                continue
            frames.append(f)
            p = self.vad.predict(f, frame_size=320)
            if p > SPEECH_P:
                speech, quiet = speech + 1, 0
            elif p < QUIET_P:
                quiet += 1                      # dips between 0.15 and 0.4 (soft syllables, short gaps) neither count nor reset
            if quiet >= END_SILENCE_FRAMES and speech >= MIN_SPEECH_FRAMES:
                break
        return np.concatenate(frames).astype(np.float32) / 32768 if frames else None

    def transcribe(self, audio):
        kw = {"generate_kwargs": {"prompt_ids": self.prompt_ids}} if self.prompt_ids is not None else {}
        text = self.asr({"raw": audio, "sampling_rate": RATE}, **kw)["text"].strip()
        return text[len(HINT):].strip() if text.lower().startswith(HINT[:20].lower()) else text

    def push(self, frame):
        """Idle listening: collect short bursts of speech. Returns a finished burst (float32 audio) or None."""
        voiced = self.vad.predict(frame, frame_size=320) > SPEECH_P
        if not self._seg:
            self._pre.append(frame)
            if voiced:
                self._seg, self._speech, self._quiet = list(self._pre), 1, 0
            return None
        self._seg.append(frame)
        self._speech, self._quiet = (self._speech + 1, 0) if voiced else (self._speech, self._quiet + 1)
        if len(self._seg) > SEG_MAX_FRAMES:                       # too long to be a wake phrase: ordinary conversation
            self._seg, self._over = [], True
            return None
        if self._quiet >= SEG_QUIET_FRAMES:
            seg, self._seg = self._seg, []
            return np.concatenate(seg).astype(np.float32) / 32768 if self._speech >= MIN_SPEECH_FRAMES else None
        return None

    def reset_idle(self):
        self._seg, self._pre, self._over = [], collections.deque(maxlen=PREROLL), False
        self.vad.reset_states()
        self.wake.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wake", action="store_true")
    ap.add_argument("--wav")
    ap.add_argument("--speech-wake", action="store_true", help="also wake on the word Jarvis found by transcribing every short phrase heard while idle (uses more CPU, hears more)")
    ap.add_argument("--gpu", action="store_true", help="run speech-to-text on the GPU (default: CPU, as on battery)")
    a = ap.parse_args()
    from tommy.agent import Agent
    from tommy import system
    t0 = time.monotonic()
    ears, agent = Ears(a.gpu), Agent()
    agent.start()
    print(f"[ready in {time.monotonic() - t0:.1f}s]", flush=True)
    import os, tempfile
    beep_file = Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()) / "tommy-beep.wav"
    make_beep(beep_file)

    def handle(audio, mic=None):
        t = time.monotonic()
        text = ears.transcribe(audio)
        print(f'  heard: "{text}"   (speech-to-text {time.monotonic() - t:.2f}s)', flush=True)
        handle_text(text, mic)

    def handle_text(text, mic=None):

        def confirm(question):
            if mic is None:
                return input(f"  confirm {question} [y/N] ").strip().lower() == "y"
            system.say(f"{question} Say yes or no.")
            print(f"  asking: {question}", flush=True)
            play(mic, "confirm", beep_file)
            ans = ears.capture(mic)
            words = set(ears.transcribe(ans).lower().replace(",", " ").replace(".", " ").split()) if ans is not None else set()
            print(f"  answer: {sorted(words)}", flush=True)
            return bool(words & YES) and not words & NO

        def ask_back(kind):
            """The model could not find a time in what was said: ask for it out loud and listen (twice, 8 s each)."""
            for attempt in range(2):
                system.say("What time, sir?")
                print("  asking: what time?", flush=True)
                if attempt:
                    play(mic, "sorry", beep_file)
                play(mic, "what_time", beep_file)
                ans = ears.capture(mic, wait_s=8.0)
                said = ears.transcribe(ans) if ans is not None else None
                print(f'  answer: "{said}"', flush=True)
                if said:
                    return said
            return None

        t = time.monotonic()
        reply, call = agent.ask(text, confirm, ask_back if mic is not None else None)
        print(f"  -> {call}\n  {reply}   (tommy {time.monotonic() - t:.2f}s)", flush=True)
        system.say(reply)
        if mic is not None and not (call and call[0] in INFO_TOOLS):       # answers to questions are read, not "Done, sir"
            play(mic, reply_clip(reply), beep_file)

    try:
        if a.wav:
            w = wave.open(a.wav)
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768 * GAIN
            handle(np.clip(audio, -1, 1))
            return
        mic = Mic()
        last = 0.0
        print('say "hey jarvis" then your command' if not a.no_wake else "press Enter, then speak", flush=True)
        ears.reset_idle()
        while True:
            command = None
            if a.no_wake:
                input()
            else:
                frame = mic.read()
                fast = ears.woke(frame)
                seg = ears.push(frame) if a.speech_wake else None
                if fast and time.monotonic() - last > COOLDOWN:
                    pass                                              # the wake word model heard "hey jarvis"
                elif seg is not None and time.monotonic() - last > COOLDOWN:
                    t = time.monotonic()
                    text = ears.transcribe(seg)                       # a short burst of speech: is it "Jarvis ..."?
                    rest = after_jarvis(text)
                    # only a match is shown: speech that is not for Tommy is checked in memory and dropped, never printed or stored
                    if rest is not None:
                        print(f'  (woken by "{text}" in {time.monotonic() - t:.1f}s)', flush=True)
                    if rest is None:
                        mic.drain(); ears.reset_idle()
                        continue
                    command = rest if len(rest.split()) >= 2 else None  # "hey jarvis set a timer" in one breath
                else:
                    continue
            play(mic, "greet", beep_file)
            if command is not None:
                print(f'  heard: "{command}"', flush=True)
                handle_text(command, mic)
            else:
                print("* listening", flush=True)
                system.say("Listening...")
                audio = ears.capture(mic)
                if audio is None:
                    print("  (heard nothing)", flush=True)
                else:
                    handle(audio, mic)
            mic.drain()                                               # audio that piled up while we were busy
            ears.reset_idle()
            last = time.monotonic()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
