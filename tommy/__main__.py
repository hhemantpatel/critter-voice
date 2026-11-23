"""python -m tommy "25 min baad laptop sula do"   -- type a request, Jarvis answers and acts (also: jarvis "...").
With no request: a prompt that also shows what the model chose."""
import sys, time
from .agent import Agent
from .dispatch import is_yes


def main():
    texts = sys.argv[1:]
    confirm = ask_back = None                        # no terminal to ask on: risky requests are refused
    if sys.stdin.isatty():
        confirm = lambda q, *_: is_yes(input(f"{q} [yes/no] "))
        ask_back = lambda kind: input(f"{'What time' if kind == 'time' else 'What should it say'}, sir? ").strip() or None
    a = Agent()
    try:
        a.start()
        for text in texts or iter(lambda: input("jarvis> "), "quit"):
            t = time.monotonic()
            reply, call = a.ask(text, confirm, ask_back)
            print(reply.text if texts else f"  -> {call}\n  {reply.text}   ({time.monotonic() - t:.2f}s)")
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        a.stop()


main()
