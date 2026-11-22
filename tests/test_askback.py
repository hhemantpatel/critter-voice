"""Ask-back logic with a stubbed model and a stubbed executor: nothing real runs."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tommy import agent as A

ran = []
A.run_call = lambda name, args, confirm=None: (ran.append((name, dict(args))), "ran")[1]
class Fake(A.Agent):
    def __init__(self, call): A.load_all(); self.call = call
    def propose(self, text): return self.call

def run(utter, call, answer=None, yes=True):
    ran.clear(); asked = []
    ab = (lambda kind: (asked.append(kind), answer)[1])
    reply, _ = Fake(call).ask(utter, lambda q: yes, ab)
    return reply, list(ran), asked

CASES = [   # (utterance, model call, spoken answer to "what time?", expected asked, expected args or None)
    ("put my pc to sleep", ("schedule_power", {"action": "sleep", "after": "0"}), "in ten minutes", ["time"], {"action": "sleep", "after": "in ten minutes"}),
    ("pc on sleep", ("schedule_power", {"action": "sleep"}), "25 minutes", ["time"], {"action": "sleep", "after": "25 minutes"}),
    ("put my pc to sleep", ("schedule_power", {"action": "sleep", "after": "x"}), None, ["time"], None),
    ("put my pc to sleep", ("schedule_power", {"action": "sleep", "after": "x"}), "banana", ["time"], None),
    ("sleep the laptop now", ("schedule_power", {"action": "sleep", "after": "now"}), None, [], {"action": "sleep", "after": "now"}),
    ("abhi laptop sula do", ("schedule_power", {"action": "sleep", "after": "0"}), None, [], {"action": "sleep", "after": "now"}),
    ("sleep after 25 minutes", ("schedule_power", {"action": "sleep", "after": "25 minutes"}), None, [], {"action": "sleep", "after": "25 minutes"}),
    ("10 second baad laptop ko sleep mein daal do", ("schedule_power", {"action": "sleep", "after": "second"}), None, [], {"action": "sleep", "after": "10 second baad laptop ko sleep mein daal do"}),
    ("start a focus session", ("focus_start", {}), None, [], {}),
    ("paune do ghante baad sula do", ("schedule_power", {"action": "sleep", "after": "do ghante"}), None, [], {"action": "sleep", "after": "paune do ghante baad sula do"}),
]
bad = 0
for utter, call, ans, want_asked, want_args in CASES:
    reply, did, asked = run(utter, call, ans)
    got_args = did[0][1] if did else None
    ok = asked == want_asked and got_args == want_args
    bad += not ok
    print(("ok  " if ok else "FAIL"), f"{utter!r:50} asked={asked} ran={got_args}  reply={reply!r}")
print(f"{len(CASES) - bad}/{len(CASES)} passed"); raise SystemExit(bad)
