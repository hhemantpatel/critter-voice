"""Spoken lines (recorded once by make_greetings.py, only played here) and the critter's listening animation."""
import random, subprocess, time, wave
from .mic import FRAME_S
from .system import ROOT

CLIPS = ROOT / "models/tts/greetings"
_last = {}


def pick_clip(situation):
    """A recorded line for this situation, random but never the same one twice in a row; None if there is none."""
    clips = sorted(CLIPS.glob(f"{situation}_[0-9]*.wav"))
    if len(clips) > 1:
        clips = [c for c in clips if c != _last.get(situation)]
    if not clips:
        return None
    _last[situation] = random.choice(clips)
    return _last[situation]


def clip_seconds(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def say_async(situation):
    """Start a line and return at once, with the seconds it will take (0 if there is nothing to say)."""
    path = pick_clip(situation)
    if path is None:
        return 0.0
    subprocess.Popen(["pw-play", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return clip_seconds(path)


def ask(mic, situation, lead=0.7):
    """Say a line and return `lead` seconds before it ends, so you can answer over its last words.
    Returns how many coming frames still hold our own voice: they are kept but never counted as you starting to speak."""
    seconds = say_async(situation)
    if not seconds:
        return 0
    time.sleep(max(0.0, seconds - lead))
    mic.drain()                                     # what was recorded so far is our own voice
    return int((lead + 0.25) / FRAME_S)


def show_listening(on):
    """The desktop critter's listening animation (Caelestia shell IPC; does nothing elsewhere)."""
    subprocess.Popen(["qs", "-c", "caelestia", "ipc", "call", "buddy", "hearing", "true" if on else "false"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
