"""python -m tommy "25 min baad laptop sula do"   -- type a request, Tommy answers and acts."""
import sys, time
from .agent import Agent

def main():
    texts = sys.argv[1:]
    a = Agent(); t0 = time.monotonic(); load = a.start()
    print(f"[model ready in {load:.2f}s]", file=sys.stderr)
    ask = lambda q, *_: input(f"  {q} [y/N] ").strip().lower() == "y"
    ask_back = lambda kind: input(f"  {'What time' if kind == 'time' else 'What should it say'}, sir? > ").strip() or None
    try:
        for text in texts or iter(lambda: input("tommy> "), "quit"):
            t = time.monotonic(); reply, call = a.ask(text, ask, ask_back)
            print(f"  -> {call}\n  {reply.text}   ({time.monotonic() - t:.2f}s)")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        a.stop()

main()
