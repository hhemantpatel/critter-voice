"""python -m tommy.cli --direct start_timer duration="25 minutes"   (tool harness, no model)"""
import argparse, sys
from .dispatch import run_call
from .registry import load_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", nargs="+", metavar="TOOL [k=v ...]", help="call a tool without the model")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for n, t in load_all().items():
            print(f"{t.risk:8}{n}  {t.description}")
    elif a.direct:
        name, kv = a.direct[0], dict(x.split("=", 1) for x in a.direct[1:])
        ask = lambda q, *_: input(f"{q} [y/N] ").strip().lower() == "y"
        print(run_call(name, kv, ask if sys.stdin.isatty() else None).text)      # no terminal to ask on: risky tools are refused
