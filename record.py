#!/usr/bin/env python
"""Record yourself saying test phrases, for the speech-to-text benchmark.

    .venv/bin/python record.py            # records the next 60 phrases you haven't done yet
    .venv/bin/python record.py --n 20     # just 20 this session

For each phrase: read it on screen, press Enter, SAY IT, press Enter again. Type 's' to skip, 'r' to redo, 'q' to quit.
Say each phrase the way you naturally would; if the printed wording feels wrong, say it your own way and type
what you actually said when asked (press Enter to keep the printed text). Sit where you normally sit, at your normal
volume. Files stay on this PC: data/recordings/NNN.wav plus labels.jsonl. Nothing is uploaded.
"""
import argparse, json, random, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "recordings"
LABELS = OUT / "labels.jsonl"

# Phrases that matter most for the ASR choice: numbers, units and Hinglish, plus out-of-scope and plain commands.
CORE = [
    "set a timer for 10 minutes", "25 minute ka timer lagao", "put my pc to sleep after 25 minutes", "25 min baad laptop sula do",
    "pachees minute baad sleep", "paune do ghante baad laptop band kar do", "dedh ghante baad pc sula do", "aadha ghanta baad lock kar do",
    "remind me to drink water in 20 minutes", "shaam paanch baje yaad dilana call mom", "10 minute baad yaad dilana pani peena hai",
    "note this down buy milk", "note kar lo kal professor se milna hai", "read my notes", "mere notes sunao",
    "start a focus session", "pachees minute ka focus shuru karo", "focus status", "pause focus", "stop focus",
    "start the stopwatch", "stopwatch kitna hua", "reset the stopwatch", "dnd on for 30 minutes", "notifications band karo",
    "turn on the hotspot", "hotspot band karo", "is my internet working", "net chal raha hai kya", "what is scheduled",
    "cancel all timers", "sab timers cancel karo", "cancel the sleep timer", "lock the screen in 5 minutes",
    "shut down the pc in an hour", "ek ghanta baad shutdown", "what is the weather", "kya haal hai", "play some music", "hello tommy",
    "turn on sleep mode after 25 minutes", "can you set a reminder for 7 30 am", "subah saat baje uthana", "twenty five minute timer",
    "tees minute baad yaad dilana", "sixty seconds timer", "ek minute ka timer", "sawa ghanta baad sleep", "teen ghante baad pc off",
]

def done_ids():
    return {json.loads(l)["id"] for l in LABELS.read_text().splitlines()} if LABELS.exists() else set()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); a = ap.parse_args()
    extra = HERE / "data" / "test" / "user.txt"
    phrases = CORE + ([l.strip() for l in extra.read_text().splitlines() if l.strip()] if extra.exists() else [])
    done = done_ids(); todo = [(i, p) for i, p in enumerate(phrases) if i not in done][: a.n]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} phrases to record. Enter = start/stop. s = skip, r = redo, q = quit.\n")
    for i, text in todo:
        while True:
            print(f"[{i + 1}/{len(phrases)}]  \033[1m{text}\033[0m")
            c = input("   Enter to start (s skip, q quit) > ").strip().lower()
            if c == "q": return
            if c == "s": break
            wav = OUT / f"{i:03d}.wav"
            p = subprocess.Popen(["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16", str(wav)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            t0 = time.time(); input("   \033[31m● recording\033[0m  say it, then Enter > ")
            p.send_signal(signal.SIGINT); p.wait()
            print(f"   {time.time() - t0:.1f}s")
            r = input("   what you actually said (Enter = as printed, r = redo) > ").strip()
            if r.lower() in ("r", "redo"): continue
            with LABELS.open("a") as f:
                f.write(json.dumps({"id": i, "file": wav.name, "said": r or text, "printed": text}, ensure_ascii=False) + "\n")
            break
    print("\nDone. Thank you.")

if __name__ == "__main__":
    try: main()
    except (KeyboardInterrupt, EOFError): print()
