#!/usr/bin/env python
"""Learn from real use: review what Tommy heard and did, then retrain the command model and the Jarvis wake word.

    .venv-train/bin/python learn.py review     # go through new events: was it right? did you really say Jarvis?
    .venv-train/bin/python learn.py retrain    # retrain both models on everything reviewed; keeps a new model only if it is not worse
    .venv-train/bin/python learn.py stats      # what has been collected so far
    .venv-train/bin/python learn.py wrong [N]  # the last (N) wake(s) were false: you did not say Jarvis  (also: `jarvis wrong`)
    .venv-train/bin/python learn.py right [N]  # the last (N) wake(s) were real

Everything lives in data/usage/ on this PC (git-ignored). Delete that folder to forget it all.
Nothing is learned without review: an unchecked log would teach the model its own mistakes.
"""
import json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
USAGE = ROOT / "data" / "usage"
EVENTS, REVIEWED = USAGE / "events.jsonl", USAGE / "reviewed.jsonl"
PY = str(ROOT / ".venv-train" / "bin" / "python")


def read(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def append(p, row):
    with p.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ask_call():
    """Ask for the right tool and its arguments by number and plain text."""
    from tommy.registry import load_all, check_args
    tools = list(load_all().values())
    for i, t in enumerate(tools):
        print(f"   {i:2d} {t.name:18} {t.description}")
    while True:
        c = input("   right tool number (Enter = skip this one) > ").strip()
        if not c:
            return None
        if c.isdigit() and int(c) < len(tools):
            break
    t = tools[int(c)]
    args = {}
    for name, spec in t.params.items():
        hint = f" one of {spec['enum']}" if "enum" in spec else ""
        v = input(f"   {name}{hint}{'' if name in t.required else ' (Enter = leave out)'} > ").strip()
        if v:
            args[name] = v
    try:
        _, clean = check_args(t.name, args)
        return t.name, clean
    except ValueError as e:
        print("   not valid:", e)
        return None


def review():
    done = {r["ts"] for r in read(REVIEWED)}
    todo = [e for e in read(EVENTS) if e["ts"] not in done]
    print(f"{len(todo)} new events. Keys: Enter = right, n = wrong, s = skip, q = quit\n")
    for e in todo:
        print(f"[{e['ts']}] wake score {e.get('wake_score')}")
        row = {"ts": e["ts"], "wake_audio": e.get("wake_audio")}
        if e.get("wake_audio"):
            if input("   play what it heard before waking? (y / Enter = no) > ").strip().lower() == "y":
                subprocess.run(["pw-play", str(USAGE / "wake" / e["wake_audio"])])
        said = input("   did you say Jarvis just before this? (Enter = yes, n = no, s = skip) > ").strip().lower()
        if said == "q":
            return
        row["wake"] = None if said == "s" else (said != "n")
        if e.get("heard"):
            print(f'   heard: "{e["heard"]}"\n   did:   {e.get("call")} -> {e.get("reply")}')
            c = input("   was that the right action? (Enter = yes, n = no, s = skip) > ").strip().lower()
            if c == "q":
                return
            if c == "":
                row["text"], (row["tool"], row["args"]) = e["heard"], e["call"]
            elif c == "n":
                fixed = ask_call()
                if fixed:
                    row["text"], (row["tool"], row["args"]) = e["heard"], fixed
        append(REVIEWED, row)
        print()
    print("review done")


def mark(real, n=1):
    """Label the most recent wake(s) right away, e.g. `jarvis wrong` just after a false wake."""
    done = {r["ts"] for r in read(REVIEWED)}
    events = [e for e in read(EVENTS) if e.get("wake_audio") and e["ts"] not in done][-n:]
    if not events:
        print("no unlabelled wake to mark")
        return
    for e in events:
        append(REVIEWED, {"ts": e["ts"], "wake_audio": e["wake_audio"], "wake": real})
        print(f"marked the wake at {e['ts']} as {'real' if real else 'FALSE'} (score {e.get('wake_score')}, heard: {e.get('heard') or 'nothing'})")


def stats():
    ev, rv = read(EVENTS), read(REVIEWED)
    print(f"events {len(ev)}  reviewed {len(rv)}")
    print(f"  wake: real {sum(r.get('wake') is True for r in rv)}, false {sum(r.get('wake') is False for r in rv)}")
    print(f"  commands with a known right answer: {sum('tool' in r for r in rv)}")


def score(run):
    out = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "bench/semantic.py", str(run)], capture_output=True, text=True, cwd=ROOT).stdout
    print("  ", out.strip())
    return float(out.split("by-meaning ")[1].split("%")[0]) if "by-meaning" in out else 0.0


def retrain():
    rv = read(REVIEWED)
    rows = [{"text": r["text"], "tool": r["tool"], "args": r["args"]} for r in rv if "tool" in r]
    wake_rows = [r for r in rv if r.get("wake") is not None and r.get("wake_audio")]
    print(f"{len(rows)} reviewed commands, {len(wake_rows)} reviewed wake clips")
    if rows:
        (ROOT / "data/out").mkdir(parents=True, exist_ok=True)
        (ROOT / "data/out/usage.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "data/gen.py"], cwd=ROOT, check=True)
        subprocess.run([PY, "train/sft.py", "--model", "Qwen3-0.6B", "--mode", "lora", "--epochs", "3", "--bs", "8", "--tag", "_usage",
                        "--extra", "data/out/usage.jsonl", "--extra-reps", "5"], cwd=ROOT, check=True)
        run = ROOT / "runs" / "Qwen3-0.6B-lora_usage"
        current = ROOT / "models/tommy/current-score.txt"
        old = float(current.read_text()) if current.exists() else 95.2
        new = score(run)
        if new >= old - 0.3:
            subprocess.run([PY, "vendor/llama.cpp/convert_hf_to_gguf.py", str(run), "--outfile", str(run / "model-f16.gguf"), "--outtype", "f16"],
                           cwd=ROOT, check=True, capture_output=True)
            subprocess.run([str(ROOT / "vendor/llama.cpp/build/bin/llama-quantize"), str(run / "model-f16.gguf"),
                            str(ROOT / "models/tommy/qwen3-0.6b-usage-q8.gguf"), "Q8_0"], check=True, capture_output=True)
            current.write_text(str(new))
            print(f"command model: {new:.1f}% (was {old:.1f}%) -> now using models/tommy/qwen3-0.6b-usage-q8.gguf")
        else:
            print(f"command model: {new:.1f}% is worse than {old:.1f}%, keeping the old one")
    if wake_rows:
        w = ROOT / "models/tommy/jarvis_wake.npz"
        if w.exists():
            shutil.copy(w, w.with_suffix(".prev.npz"))
        subprocess.run([PY, "train/wake_train.py", "--final", "--epochs", "25"], cwd=ROOT, check=True)
        print("wake word retrained (previous model kept as jarvis_wake.prev.npz)")


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 1
    {"review": review, "retrain": retrain, "stats": stats, "wrong": lambda: mark(False, n), "right": lambda: mark(True, n)
     }.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
