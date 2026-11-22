#!/usr/bin/env python
"""Record yourself for the Jarvis wake word. About 5 minutes.

    .venv/bin/python record_wake.py

For each line: read it, press Enter, SAY IT the way you would naturally, press Enter again.
  - "JARVIS" lines are the wake word. Vary it: soft, loud, fast, slow, from your usual seat, and leaning back.
  - "NOT" lines are ordinary speech that must NOT wake it. Read them naturally.
Keys: s = skip, r = redo, q = quit (you can resume later).
Files stay on this PC in data/wake/ (ignored by git). Nothing is uploaded.
"""
import json, random, signal, subprocess, sys, time
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "wake"
LABELS = OUT / "labels.jsonl"

WAKE = [
    "Jarvis", "Hey Jarvis", "Jarvis", "Hey Jarvis", "Jarvis?", "Hey Jarvis!", "Jarvis (say it softly)", "Hey Jarvis (say it softly)",
    "Jarvis (say it louder)", "Hey Jarvis (say it louder)", "Jarvis (fast)", "Hey Jarvis (fast)", "Jarvis (slow, drawn out)",
    "Hey Jarvis (slow)", "Jarvis (casual, like calling a friend)", "Hey Jarvis (casual)", "Jarvis (lean back from the screen first)",
    "Hey Jarvis (lean back first)", "Jarvis (sit forward, close to the mic)", "Hey Jarvis (close to the mic)",
    "Jarvis, set a timer for ten minutes", "Hey Jarvis, put my PC to sleep", "Jarvis, 10 sec ka timer lagao",
    "Hey Jarvis, play some music", "Jarvis, open YouTube", "Hey Jarvis, remind me at nine", "Jarvis, pause the music",
    "Hey Jarvis, what is scheduled", "Jarvis (say it the way you said it in the test earlier)", "Hey Jarvis (the way you said it earlier)",
    "Jarvis (tired voice)", "Hey Jarvis (tired voice)", "Jarvis (with a short pause before it)", "Hey Jarvis (with a short pause before it)",
    "Jarvis (while typing)", "Hey Jarvis (while moving the mouse)", "Jarvis (turn your head a little away)", "Hey Jarvis (head turned away)",
    "Jarvis (one more time)", "Hey Jarvis (one more time)",
]
NOT = [
    "So what are you doing today", "I will check it later and tell you", "Can you send me the file", "Kal subah college jana hai",
    "That is a good idea, let us try it", "Mujhe nahi pata kya hua", "The service is down again", "Let me look at the report",
    "Chal, khana kha lete hain", "What time does the exam start", "I think the battery is low", "Harvest festival is next week",
    "Marvel movie dekhne chalein", "Who is calling me", "Okay wait a second", "Garbage collection runs automatically",
    "Yaar ye code kaam nahi kar raha", "Let us start the meeting", "Turn the music down a little", "Is the charger plugged in",
    "He said Travis and Davis were there", "Service pack, Jarvis-like names, nothing to do with me", "Hey, can you hear me",
    "Hello, is anyone there", "Hey Siri, what is the time", "Okay Google, play a song", "Hey buddy, come here",
    "Jar, wiss, jar wiss, jarring noise", "Just talking to myself while working", "Aaj ka kaam khatam kar leta hoon",
]


def done():
    return {json.loads(l)["i"] for l in LABELS.read_text().splitlines()} if LABELS.exists() else set()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = [("JARVIS", t) for t in WAKE] + [("NOT", t) for t in NOT]
    random.Random(3).shuffle(items)                       # mixed order, so the two kinds sound equally natural
    seen = done()
    todo = [(i, k, t) for i, (k, t) in enumerate(items) if i not in seen]
    print(f"{len(todo)} lines to record. Enter = start / stop.   s skip   r redo   q quit\n")
    for i, kind, text in todo:
        while True:
            tag = "\033[1;32mJARVIS\033[0m" if kind == "JARVIS" else "\033[1;33mNOT   \033[0m"
            print(f"[{i + 1}/{len(items)}] {tag}  {text}")
            c = input("   Enter to start (s skip, q quit) > ").strip().lower()
            if c == "q":
                return
            if c == "s":
                break
            wav = OUT / f"{i:03d}.wav"
            p = subprocess.Popen(["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16", str(wav)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            t0 = time.time()
            input("   \033[31m● recording\033[0m  say it, then press Enter > ")
            p.send_signal(signal.SIGINT)
            p.wait()
            print(f"   {time.time() - t0:.1f}s")
            if input("   Enter = keep, r = redo > ").strip().lower() in ("r", "redo"):
                continue
            with LABELS.open("a") as f:
                f.write(json.dumps({"i": i, "file": wav.name, "kind": kind, "text": text}, ensure_ascii=False) + "\n")
            break
    print("\nDone. Thank you.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
