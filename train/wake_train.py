"""Train the Jarvis wake word classifier and test it honestly.

    .venv-train/bin/python train/wake_train.py            # trains with some of YOUR clips held out, prints detection / false-alarm numbers
    .venv-train/bin/python train/wake_train.py --final    # trains on everything and writes models/tommy/jarvis_wake.npz

Training windows are 16 embedding frames (about 1.3 s) from openWakeWord's frozen audio features. Positive = a window that ends
just after "Jarvis" finishes. Negative = everything else: other speech, look-alike words, silence, room noise, and the moments before/after the word.
Held-out test (no --final): 10 of your plain wake clips, your 11 "wake + command" clips, 10 of your ordinary-speech clips, and 15% of the
synthetic audio, all run through the same streaming code used live (tommy/wake.py).
"""
import argparse, json, random, sys, wave
from pathlib import Path
import numpy as np
import torch   # first: the openWakeWord feature code installs a scikit-learn stub that torch's module scan cannot handle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tommy.wake import FRAMES, WEIGHTS, JarvisWake, _features, forward

R = np.random.default_rng(5)
RNG = random.Random(5)
SR, HOP, SPAN = 16000, 1280, 12400            # one embedding every 1280 samples; each looks at ~12,400 samples
CLIP = int(3.6 * SR)


def load_wav(p):
    w = wave.open(str(p))
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)


def speech_span(a, floor_pct=20):
    """(start, end) sample of the speech in a clip, from short-frame energy."""
    n = len(a) // 320 * 320
    f = a[:n].reshape(-1, 320).std(axis=1)
    thr = max(np.percentile(f, floor_pct) * 4, f.max() * 0.08, 3.0)
    idx = np.where(f > thr)[0]
    return (int(idx[0]) * 320, int((idx[-1] + 1) * 320)) if len(idx) else (0, len(a))


def noise(n, kind=None):
    kind = kind or RNG.choice(["white", "pink", "brown", "hum", "room"])
    x = R.standard_normal(n).astype(np.float32)
    if kind in ("pink", "brown"):
        f = np.fft.rfft(x)
        k = np.maximum(np.arange(len(f)), 1) ** (0.5 if kind == "pink" else 1.0)
        x = np.fft.irfft(f / k, n).astype(np.float32)
    elif kind == "hum":
        t = np.arange(n) / SR
        x = 0.6 * np.sin(2 * np.pi * RNG.choice([50, 60, 100]) * t) + 0.2 * x
    elif kind == "room" and ROOM:
        seg = ROOM[RNG.randrange(len(ROOM))]
        x = np.resize(seg, n).astype(np.float32)
        return x / (x.std() + 1e-6) * RNG.uniform(3, 40)
    return x / (x.std() + 1e-6) * RNG.uniform(3, 60)


ROOM = []


def collect_room(clips):
    for a in clips:
        n = len(a) // 8000 * 8000
        if n < 8000:
            continue
        segs = a[:n].reshape(-1, 8000)
        quiet = segs[np.argsort(segs.std(axis=1))[: max(1, len(segs) // 4)]]
        ROOM.extend(quiet)


def place(word, pre_s=None, tail_s=None):
    """Build a CLIP-long clip with `word` (int16/float array) inside background; returns (audio, word_start, word_end)."""
    word = word.astype(np.float32)
    pre = int(SR * (pre_s if pre_s is not None else RNG.uniform(1.0, 1.6)))
    tail = int(SR * (tail_s if tail_s is not None else RNG.uniform(0.05, 0.5)))
    word = word[: CLIP - pre - tail]
    bg = noise(CLIP)
    w = word / (np.abs(word).max() + 1e-6) * 10 ** RNG.uniform(np.log10(250), np.log10(14000))
    sig = np.zeros(CLIP, np.float32)
    sig[pre: pre + len(w)] = w
    snr = RNG.uniform(6, 30)
    sig = sig + bg * (np.abs(w).std() / (bg.std() + 1e-6)) / 10 ** (snr / 20) * RNG.choice([0.3, 1.0])
    sig = sig + noise(CLIP, "room") * 0.5
    return np.clip(sig, -32768, 32767).astype(np.int16), pre, pre + len(w)


def speed(a, lo=0.8, hi=1.25):
    """Play the clip faster or slower (changes pitch a little too): cheap way to get new 'voices' from one recording."""
    from scipy.signal import resample_poly
    r = RNG.uniform(lo, hi)
    return resample_poly(a.astype(np.float32), 1000, int(1000 * r)).astype(np.float32)


def crop(a, n=CLIP):
    a = a.astype(np.float32)
    if len(a) >= n:
        s = RNG.randrange(len(a) - n + 1)
        a = a[s: s + n]
    else:
        pad = RNG.randrange(n - len(a) + 1)
        a = np.concatenate([noise(pad), a, noise(n - len(a) - pad)])
    a = a / (np.abs(a).max() + 1e-6) * 10 ** RNG.uniform(np.log10(250), np.log10(14000))
    a = a + noise(n) / 10 ** (RNG.uniform(8, 35) / 20) * (np.abs(a).std() + 1)
    return np.clip(a, -32768, 32767).astype(np.int16)


def embed(clips, feats, batch=64):
    out = []
    for i in range(0, len(clips), batch):
        out.append(feats.embed_clips(np.stack(clips[i: i + batch]), batch_size=batch))
    return np.concatenate(out)


def window_end(sample):             # index of the last embedding frame whose look-back ends at or before `sample`
    return int((sample - SPAN) // HOP)


def windows(emb, w_start=None, w_end=None):
    """Yield (window, label) from one clip's embeddings. label 1 near the end of the word, 0 elsewhere, None = ambiguous."""
    F = emb.shape[0]
    for e in range(FRAMES - 1, F):
        win = emb[e - FRAMES + 1: e + 1]
        if w_start is None:
            yield win, 0
            continue
        lo, hi = window_end(w_end), window_end(w_end + int(0.25 * SR))
        if lo <= e <= hi:
            yield win, 1
        elif e < lo - 1 or e > window_end(w_end + int(0.6 * SR)):
            yield win, 0                      # before the word, half-said word, or long after it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()

    pos_synth = [x for x in np.load(ROOT / "data/wake_synth/pos.npz").values()]
    neg_synth = [x for x in np.load(ROOT / "data/wake_synth/neg.npz").values()]
    labels = [json.loads(l) for l in (ROOT / "data/wake/labels.jsonl").read_text().splitlines()]
    user_pos = [load_wav(ROOT / "data/wake" / r["file"]) for r in labels if r["kind"] == "JARVIS" and "," not in r["text"]]
    user_cmd = [load_wav(ROOT / "data/wake" / r["file"]) for r in labels if r["kind"] == "JARVIS" and "," in r["text"]]
    user_neg = [load_wav(ROOT / "data/wake" / r["file"]) for r in labels if r["kind"] == "NOT"]
    cmd_recs = [load_wav(p) for p in sorted((ROOT / "data/recordings").glob("*.wav"))]
    collect_room(user_pos + user_neg + cmd_recs)
    print(f"synthetic: {len(pos_synth)} positive, {len(neg_synth)} negative | yours: {len(user_pos)} wake, {len(user_cmd)} wake+command, "
          f"{len(user_neg)} ordinary, {len(cmd_recs)} command recordings | room-noise pieces: {len(ROOM)}", flush=True)

    # split: hold some of everything out (not with --final)
    def split(xs, frac):
        idx = list(range(len(xs)))
        RNG.shuffle(idx)
        k = 0 if a.final else max(1, int(len(xs) * frac))
        return [xs[i] for i in idx[k:]], [xs[i] for i in idx[:k]]
    tr_up, te_up = split(user_pos, 0.34)
    tr_un, te_un = split(user_neg, 0.34)
    tr_ps, te_ps = split(pos_synth, 0.15)
    tr_ns, te_ns = split(neg_synth, 0.15)
    te_uc = [] if a.final else user_cmd
    print(f"training on {len(tr_up)} of your wake clips, {len(tr_un)} ordinary; testing on {len(te_up)} / {len(te_un)} / {len(te_uc)} wake+command", flush=True)

    feats = _features()
    X, Y = [], []

    def add(clips, spans, reps):
        for _ in range(reps):
            built = [place(speed(c[s:e])) for c, (s, e) in zip(clips, spans)]
            audios = [b[0] for b in built]
            emb = embed(audios, feats)
            for em, (au, ws, we) in zip(emb, built):
                for win, lab in windows(em, ws, we):
                    X.append(win); Y.append(lab)

    # positives: your clips (heavily augmented) and synthetic ones
    add(tr_up, [speech_span(c) for c in tr_up], 60)
    add(tr_ps, [(0, len(c)) for c in tr_ps], 2)
    n_pos = sum(Y)
    # negatives: speech (yours and synthetic), cropped at random, plus pure noise
    for clips, reps in ((tr_un, 40), (cmd_recs, 6), (tr_ns, 1)):
        for _ in range(reps):
            sel = [crop(speed(c)) for c in clips]
            for em in embed(sel, feats):
                for win, lab in windows(em):
                    X.append(win); Y.append(0)
    noise_clips = [np.clip(noise(CLIP, k), -32768, 32767).astype(np.int16) for k in ["white", "pink", "brown", "hum", "room"] * 60]
    for em in embed(noise_clips, feats):
        for win, lab in windows(em):
            X.append(win); Y.append(0)
    X = np.stack(X).astype(np.float32); Y = np.array(Y, np.float32)
    print(f"windows: {len(Y)} ({int(Y.sum())} positive, {int((1 - Y).sum())} negative)", flush=True)

    torch.manual_seed(0)
    mean = X.reshape(len(X), -1).mean(0); std = X.reshape(len(X), -1).std(0) + 1e-3
    norm = lambda A: torch.tensor((A.reshape(len(A), -1) - mean) / std)
    net = torch.nn.Sequential(torch.nn.Linear(FRAMES * 96, 96), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(96, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)

    def fit(Xa, Ya, Wa, epochs):
        Xt, Yt, Wt = norm(Xa), torch.tensor(Ya), torch.tensor(Wa)
        for ep in range(epochs):
            net.train(); perm = torch.randperm(len(Xt)); tot = 0
            for i in range(0, len(Xt), 256):
                b = perm[i: i + 256]
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]).squeeze(1), Yt[b], reduction="none") * Wt[b]).mean()
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        print(f"  trained {epochs} epochs, loss {tot / len(Xt):.4f}", flush=True)

    def scores(A):
        net.eval()
        with torch.no_grad():
            return torch.sigmoid(net(norm(A)).squeeze(1)).numpy()

    W = np.where(Y == 1, 1.0, 1.0).astype(np.float32)
    fit(X, Y, W, a.epochs)
    # hard-negative mining: score fresh negative audio, add every window it wrongly fires on, retrain on top
    for rnd in range(3):
        hard = []
        for clips in (tr_un, cmd_recs, tr_ns, tr_un, cmd_recs):
            sel = [crop(speed(c)) for c in clips]
            for em in embed(sel, feats):
                wins = np.stack([win for win, _ in windows(em)])
                sc = scores(wins)
                hard.extend(wins[sc > 0.2])
        print(f"mining round {rnd + 1}: {len(hard)} windows scored above 0.2", flush=True)
        if not hard:
            break
        H = np.stack(hard)
        X = np.concatenate([X, H, H]); Y = np.concatenate([Y, np.zeros(2 * len(H), np.float32)])
        W = np.concatenate([W, np.full(2 * len(H), 3.0, np.float32)])
        fit(X, Y, W, 12)
    w = {"mean": mean.astype(np.float32), "std": std.astype(np.float32),
         "w1": net[0].weight.detach().numpy().T.astype(np.float32), "b1": net[0].bias.detach().numpy().astype(np.float32),
         "w2": net[3].weight.detach().numpy().T.astype(np.float32), "b2": net[3].bias.detach().numpy().astype(np.float32)}

    # ---- honest test through the real streaming path ----
    def stream_max(audio):
        det = JarvisWake.__new__(JarvisWake); det.w = w; det.feats = _features()
        pad = np.zeros(int(0.8 * SR), np.int16)
        x = np.concatenate([pad, audio.astype(np.int16), pad])
        best, prev = 0.0, 0.0
        for i in range(0, len(x) - HOP + 1, HOP):
            sc = det.predict(x[i: i + HOP])
            best, prev = max(best, min(sc, prev)), sc          # fires only if two frames in a row are high
        return best
    if not a.final:
        pos_scores = {"your wake clips": [stream_max(c * 1.0) for c in te_up], "your wake+command clips": [stream_max(c) for c in te_uc],
                      "synthetic wake clips": [stream_max(place(c, 1.0, 0.3)[0]) for c in te_ps[:60]]}
        neg_scores = {"your ordinary speech": [stream_max(c) for c in te_un], "synthetic speech": [stream_max(crop(c)) for c in te_ns[:150]],
                      "your command recordings": [stream_max(c) for c in cmd_recs[:48]],
                      "noise": [stream_max(c) for c in noise_clips[::20]]}
        neg_seconds = {"your ordinary speech": sum(len(c) for c in te_un) / SR, "synthetic speech": 150 * CLIP / SR,
                       "your command recordings": sum(len(c) for c in cmd_recs[:48]) / SR, "noise": len(noise_clips[::20]) * CLIP / SR}
        print("\nthreshold  " + "  ".join(f"{k[:22]:>22}" for k in pos_scores) + "   false alarms per hour of speech/noise")
        for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
            det = "  ".join(f"{100 * np.mean(np.array(v) >= t):21.0f}%" for v in pos_scores.values())
            fa = sum(int(np.sum(np.array(v) >= t)) for v in neg_scores.values())
            hrs = sum(neg_seconds.values()) / 3600
            per = {k: int(np.sum(np.array(v) >= t)) for k, v in neg_scores.items()}
            print(f"  {t:.2f}    {det}   {fa} false alarms in {hrs * 60:.1f} min (~{fa / hrs:.0f}/h)  {per}")
        for k, v in pos_scores.items():
            print(f"  scores {k}: {np.round(sorted(v), 2).tolist()}")
        return
    # threshold for live use: chosen from the held-out run (default 0.8), stored with the weights
    w["threshold"] = np.float32(0.8)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(WEIGHTS, **w)
    print("saved", WEIGHTS)


if __name__ == "__main__":
    main()
