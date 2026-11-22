#!/usr/bin/env python
"""Always-on listening cost probe for the critter.

    mic -> voice gate -> wake word -> on_wake()   (everything after on_wake is a stub)

Audio stays in memory: frames are read, scored and dropped, never written anywhere.

Stages (each adds one layer so the cost of each layer can be measured on its own):
    capture   read the mic, throw frames away
    vad       + cheap energy voice gate
    gated     + wake-word model, fed only while the gate is open   <- the real design
    ungated   + wake-word model on every frame                     <- worst case

Wake word is openWakeWord's stock "hey_jarvis": same network as a custom "Tommy" model,
so the CPU cost is a faithful stand-in.
"""
import argparse, collections, json, os, signal, subprocess, sys, time, types

import numpy as np

RATE, FRAME = 16000, 1280          # 80 ms frames, what openWakeWord expects
PREROLL, HANG = 6, 19              # 0.5 s of audio kept before the gate opens, 1.5 s kept after speech
WORD, THRESHOLD, COOLDOWN = "hey_jarvis", 0.5, 2.0
STAGES = ("capture", "vad", "gated", "ungated")


def source_cmd(file):
    if file:
        return ["ffmpeg", "-loglevel", "error", "-re", "-i", file, "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-"]
    # parec gives raw little-endian samples; pw-record to a pipe writes a big-endian .snd stream (reads as loud noise)
    return ["parec", "--rate", str(RATE), "--channels", "1", "--format", "s16le", "--latency-msec", "40"]


def load_wake_model():
    # openWakeWord imports scikit-learn only for its verifier-training feature; the detector
    # never touches it. Stubbing keeps a 100 MB+ dependency out of an always-on process.
    for name, attrs in {"sklearn": [], "sklearn.linear_model": ["LogisticRegression"],
                        "sklearn.pipeline": ["make_pipeline"],
                        "sklearn.preprocessing": ["FunctionTransformer", "StandardScaler"]}.items():
        mod = types.ModuleType(name); mod.__path__ = []
        for attr in attrs:
            setattr(mod, attr, object)
        sys.modules.setdefault(name, mod)
    import openwakeword
    models = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
    return openwakeword.Model(wakeword_models=[os.path.join(models, f"{WORD}_v0.1.onnx")],
                              inference_framework="onnx")


class EnergyGate:
    """Opens on frames clearly louder than the running noise floor."""
    def __init__(self):
        self.floor = 200.0

    def speech(self, frame):
        level = float(frame.std())                     # std, not rms: ignores any DC offset
        loud = level > self.floor * 3 + 150
        # Quiet frames pull the floor in fast; loud frames only let it creep, so a room that
        # gets permanently louder can't pin the gate open forever (the old floor never rose).
        self.floor += (0.02 if level < self.floor else 0.0005) * (level - self.floor)
        return loud


def on_wake(score):
    # Stub for the rest of the workflow: load LLM -> streaming STT -> tool call -> "done" bubble.
    print(f"WAKE {time.strftime('%H:%M:%S')} score={score:.2f}", file=sys.stderr, flush=True)


def proc_cpu_s(pid):
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def run(stage, seconds, file):
    use_vad, use_wake = stage != "capture", stage in ("gated", "ungated")
    model = load_wake_model() if use_wake else None
    gate, preroll, hang, open_ = EnergyGate(), collections.deque(maxlen=PREROLL), 0, False
    src = subprocess.Popen(source_cmd(file), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    stop = False
    def halt(*_):
        nonlocal stop; stop = True
    signal.signal(signal.SIGTERM, halt); signal.signal(signal.SIGINT, halt)

    frames = fed = wakes = loud_frames = 0
    last_wake = 0.0
    t0, cpu0 = time.monotonic(), time.process_time()
    src_cpu = 0.0

    def feed(frame):
        nonlocal fed, wakes, last_wake
        fed += 1
        score = max(model.predict(frame).values())
        if score > THRESHOLD and time.monotonic() - last_wake > COOLDOWN:
            last_wake, wakes = time.monotonic(), wakes + 1
            on_wake(score)

    while not stop and (seconds is None or time.monotonic() - t0 < seconds):
        raw = src.stdout.read(FRAME * 2)
        if len(raw) < FRAME * 2:
            break
        frames += 1
        if not use_vad:
            continue
        frame = np.frombuffer(raw, dtype=np.int16)
        loud = gate.speech(frame)
        loud_frames += loud
        if stage == "vad":
            hang = HANG if loud else max(hang - 1, 0)
            open_ = hang > 0
        elif stage == "ungated":
            feed(frame)
            open_ = True
        else:  # gated
            if loud:
                hang = HANG
            if hang > 0:
                if not open_:
                    open_ = True
                    for old in preroll:
                        feed(old)
                    preroll.clear()
                feed(frame)
                if not loud:
                    hang -= 1
            else:
                if open_:
                    open_ = False
                    model.reset()
                preroll.append(frame)

    wall, cpu = time.monotonic() - t0, time.process_time() - cpu0
    try:
        src_cpu = proc_cpu_s(src.pid)
    except OSError:
        pass
    src.terminate(); src.wait()
    ru = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF)
    print(json.dumps({"stage": stage, "wall_s": round(wall, 1), "frames": frames,
                      "model_frames": fed, "loud_pct": round(100 * loud_frames / max(frames, 1), 1),
                      "wakes": wakes,
                      "py_cpu_pct": round(100 * cpu / wall, 2),
                      "recorder_cpu_pct": round(100 * src_cpu / wall, 2),
                      "ctx_switches_per_s": round((ru.ru_nvcsw + ru.ru_nivcsw) / wall, 1),
                      "rss_mb": round(ru.ru_maxrss / 1024, 1)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--seconds", type=float, help="stop after this long (default: until Ctrl-C)")
    ap.add_argument("--file", help="read this audio file instead of the mic (testing)")
    args = ap.parse_args()
    run(args.stage, args.seconds, args.file)
