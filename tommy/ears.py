"""Hearing: the wake word, where a command starts and ends, and speech-to-text."""
import collections, difflib, re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .mic import FRAME_S, RATE
from .speak import show_listening
from .system import ROOT
from .wake import JarvisWake, stub_sklearn

STT = ROOT / "models/asr/swift-ct2-int8"          # Hindi2Hinglish-Swift in CTranslate2 int8: same actions as PyTorch, 5x faster (bench/)
HINT = ("Jarvis, set a timer, reminder, sleep, shutdown, lock, focus, stopwatch, notes, do not disturb, hotspot. "
        "Open YouTube, WhatsApp, Classroom, Instagram, Spotify, Telegram.")      # nudges it toward our commands
WAKE_AT = 0.9
VOICE_NEEDED = 0.15               # a wake counts only if a voice was heard in the last 1.5 s: "Jarvis" is speech, room noise is not
END_S, THINK_S, MAX_S = 1.5, 0.5, 8.0     # stop after this much quiet / start working on it after this much / longest command


class Endpoint:
    """Decides, frame by frame, whether a command has started and whether you are still speaking.
    A frame is speech if the voice detector is sure, or half sure and as loud as this command has been (soft last syllables)."""
    START, SURE, HALF, LOUD = 0.2, 0.4, 0.15, 0.15

    def __init__(self):
        self.started, self.quiet, self.peak = False, 0, 1.0

    def feed(self, p, level):
        if not self.started:
            self.started = p > self.START
            self.peak = max(self.peak, level) if self.started else 1.0
            return self.started
        if p > self.SURE:
            self.peak = max(self.peak, level)
        speaking = p > self.SURE or (p > self.HALF and level > self.peak * self.LOUD)
        self.quiet = 0 if speaking else self.quiet + 1
        return speaking


def strip_wake_word(text):
    """Drop a leading "Jarvis" (a one-breath command; speech models spell it many ways), and the "... sir" that ends our own greeting."""
    text = re.sub(r"^(?:(?:yes|hello|ahead|here|service|listening|for you|you are|help)[,.]?\s+)?sir\b[\s,.?!]*", "", text, flags=re.I)
    words = text.split()
    for i, w in enumerate(words[:3]):
        if difflib.SequenceMatcher(None, "jarvis", re.sub(r"[^a-z]", "", w.lower())).ratio() >= 0.62:
            return " ".join(words[i + 1:]).strip(" ,.!?-")
    return text


class Ears:
    def __init__(self):
        stub_sklearn()
        import ctranslate2
        from faster_whisper.feature_extractor import FeatureExtractor
        from openwakeword.vad import VAD
        from tokenizers import Tokenizer
        self.wake, self.vad, self._ct2 = JarvisWake(), VAD(), ctranslate2
        self._mel, self._tok = FeatureExtractor(), Tokenizer.from_file(str(STT / "tokenizer.json"))
        self._model = ctranslate2.models.Whisper(str(STT), device="cpu", compute_type="int8", intra_threads=4)
        ids = lambda t: self._tok.encode(t, add_special_tokens=False).ids
        self._prompt = ids("<|startofprev|> " + HINT) + [self._tok.token_to_id(t) for t in ("<|startoftranscript|>", "<|hi|>", "<|transcribe|>")]
        self._worker = ThreadPoolExecutor(1)
        self.score = self.voice = 0.0
        self.forget()
        self.transcribe(np.zeros(RATE, np.float32))      # warm up, so the first real command is not slower

    # ---- waiting for "Jarvis" ----
    def forget(self):
        """Start listening for the wake word from a clean state."""
        self._voices = collections.deque(maxlen=int(1.5 / FRAME_S))
        self.vad.reset_states()
        self.wake.reset()

    def woke(self, frame):
        """True if "Jarvis" was just said. Sets .score and .voice; a high score without a voice is room noise (see .noise)."""
        self._voices.append(float(self.vad.predict(frame, frame_size=320)))
        self.score, self.voice = self.wake.predict(frame), max(self._voices)
        self.noise = self.score >= WAKE_AT and self.voice < VOICE_NEEDED
        return self.score >= WAKE_AT and not self.noise

    # ---- a command ----
    def capture(self, mic, wait_s, skip=0, work=None):
        """Record one command: from the first speech to END_S of quiet. Returns (audio, job, last_word_time) or None if nobody spoke.

        `skip` frames at the start may hold our own voice: kept, but not counted as you starting. After THINK_S of quiet
        `work(audio)` starts in the background, so the answer is nearly ready when the pause ends; if you go on speaking that
        job is dropped and a new one is started at the next pause. `work` must have no side effects."""
        show_listening(True)
        try:
            self.vad.reset_states()
            end, frames, pre = Endpoint(), [], collections.deque(maxlen=max(4, skip + 2))
            t0 = last_word = time.monotonic()
            n, job = 0, None
            while True:
                f = mic.read()
                n += 1
                p, level = float(self.vad.predict(f, frame_size=320)), float(f.astype(np.float32).std())
                if not frames:
                    pre.append(f)
                    if n > skip and end.feed(p, level):
                        frames, last_word = list(pre), time.monotonic()
                    elif time.monotonic() - t0 > wait_s:
                        return None
                    continue
                frames.append(f)
                if end.feed(p, level):
                    last_word = time.monotonic()
                    if job:
                        job.cancel()
                        job = None
                if work and job is None and end.quiet * FRAME_S >= THINK_S:
                    job = self._worker.submit(work, self._join(frames))
                if end.quiet * FRAME_S >= END_S or len(frames) * FRAME_S >= MAX_S:
                    if work and job is None:
                        job = self._worker.submit(work, self._join(frames))
                    return self._join(frames), job, last_word
        finally:
            show_listening(False)

    @staticmethod
    def _join(frames):
        return np.concatenate(frames).astype(np.float32) / 32768

    def transcribe(self, audio):
        pad = np.zeros(30 * RATE, np.float32)
        pad[:min(len(audio), len(pad))] = audio[:len(pad)]
        mel = self._ct2.StorageView.from_array(np.ascontiguousarray(self._mel(pad)[None, :, :3000].astype(np.float32)))
        out = self._model.generate(mel, [self._prompt], beam_size=1, max_length=220)
        text = re.sub(r"<\|[^|]*\|>", "", self._tok.decode(out[0].sequences_ids[0], skip_special_tokens=True))     # timestamp marks
        return strip_wake_word(text.strip())
