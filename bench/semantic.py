"""Re-score a run's fails.txt by meaning: durations compare by parsed seconds, clarify ignores the question text.
usage: python bench/semantic.py runs/<name> [more runs...]"""
import ast, re, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tommy.durations import parse_span, parse_clock

TIME_KEYS = {"duration", "after", "when"}

def same(want, got):
    if set(want) != set(got): return False
    for k, w in want.items():
        g = got[k]
        if k in TIME_KEYS:
            a, b = parse_span(w), parse_span(g)
            if a is None and b is None: a, b = parse_clock(w), parse_clock(g)
            if a != b or a is None: return False
        elif k != "question" and str(w).strip().lower() != str(g).strip().lower():
            return False
    return True

for run in sys.argv[1:]:
    run = Path(run); sc = json.loads((run / "score.json").read_text()); n = sc["n"]
    real = 0; risky = 0
    for line in (run / "fails.txt").read_text().splitlines():
        m = re.match(r"'(.*)' want (\w+) (\{.*?\}) got (.*)", line)
        if not m: continue
        text, tool, wargs, got = m.groups()
        want = ast.literal_eval(wargs)
        g = ast.literal_eval(got) if got != "None" else None
        if g and g[0] == tool and same(want, g[1]): continue
        real += 1; risky += tool in ("schedule_power", "hotspot")
    print(f"{run.name:34} strict {100*sc['exact']:.1f}%  by-meaning {100*(1-real/n):.1f}%  (real misses {real}/{n}, of which risky {risky})")
