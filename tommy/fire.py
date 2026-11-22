"""Entry point that timers run when they go off: python -m tommy.fire reminder "text" """
import sys
from . import system

if __name__ == "__main__":
    text = " ".join(sys.argv[2:]) or "Time's up"
    system.say(text)
    system.notify("Tommy", text)
