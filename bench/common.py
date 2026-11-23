"""Scoring shared by the speech benchmarks: word errors, and whether a time phrase still means the same."""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tommy.durations import parse_span, parse_clock


def norm(s):
    s = re.sub(r"[^a-z0-9 :]", " ", s.lower().replace("-", " "))
    return re.sub(r"\s+", " ", s).strip().split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def meaning(t):
    s = parse_span(t)
    if s:
        return ("span", s)
    c = parse_clock(t)
    return ("clock", c.strftime("%H:%M")) if c else None
