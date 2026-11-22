# Tommy: design

Tommy is the voice of Sprout, the desktop critter. You say or type something short, in English or Hinglish
("25 min baad pc sula do"). A very small local language model picks ONE whitelisted Python function and fills in its
arguments. Python then does the work. The model never runs a command.

This file explains what each part is and why it was chosen. `LOG.md` is the dated journal of what was actually done
and measured. If the two disagree, the log is right and this file is out of date.

## 1. The one rule: the model chooses, Python acts
The model outputs a tool name plus arguments. `tommy/registry.py` looks the name up in a table and type-checks the
arguments. Anything unknown or malformed becomes "I didn't understand". No model text is ever put into a shell
string; every external effect in `tommy/system.py` is an argv list. So the worst a confused model can do is call
one of the tools we wrote, with arguments of the right type.

Risky tools carry `risk="confirm"` (sleep/poweroff/lock after a delay, hotspot). `tommy/dispatch.py` asks the user
first, and refuses everything risky while the screen is locked.

## 2. The model copies phrases; Python does the maths
Small models are bad at arithmetic ("pachees minute" -> 1500 s) and good at copying a span of text. So the tool takes
the time phrase as a string (`after="pachees minute"`) and `tommy/durations.py` converts it. This:
- needs far fewer training examples (the model learns "which tool, which words", not number systems),
- is testable (`tests/test_durations.py`, 57 cases: English and Hinglish numbers, "dedh ghanta", "sawa paanch baje"),
- fails loudly: if the phrase cannot be parsed the user is asked, instead of a silent wrong time.

Roman-script Hindi has no fixed spelling, so `durations.py` lists several spellings per number. A spelling it does
not know gives "could not read a duration", never a guess. Add spellings to `_HI_SPELL` as they turn up.

## 3. Timers are systemd, not a daemon
"Sleep in 25 minutes", reminders and countdowns use `systemd-run --user --on-active=...` (`system.schedule`).
Why: the timer survives a crash or restart of the Tommy daemon, `systemctl --user list-timers 'tommy-*'` shows them,
and `systemctl --user stop` cancels them. Nothing to write or maintain. Every unit is named `tommy-*`, and `cancel`
refuses any other name, so Tommy can only stop its own timers. The due time is stored in the unit's description
because systemd reports the next run relative to boot, which is awkward to read back.

## 4. Adding a tool
1. Write a function in `tommy/tools/<area>.py` and decorate it with `@tool(risk=..., arg="what it means")`.
   The schema the model sees is built from the type hints and docstring. Use `Literal[...]` for choices.
2. Return a short string; it becomes the bubble text.
3. Add 15-30 example phrasings to `data/seeds/<tool>.yaml` (English, Hinglish, broken English), then regenerate data
   and retrain. Until retrained the model will not know the tool.
4. Add the tool to the benchmark test set.
Nothing else needs editing: `load_all()` imports every module in `tools/`.

## 5. Replying and talking to Sprout
Replies use `qs -c caelestia ipc call buddy say "<text>"`, which already exists (`services/Buddy.qml`). Focus and
stopwatch tools wrap the existing `focus` and `stopwatch` IPC handlers instead of reimplementing them.
Notifications use `notify-send -a Sprout`, because Buddy.qml ignores notifications from that app name (otherwise
Sprout would startle at its own reminder).

## 6. Models and runtime (decided in phases 1-7; results go in LOG.md)
- Candidates: FunctionGemma 270M (made to be fine-tuned for function calling) and Qwen3 0.6B (Apache-2.0). A 1B model
  is the fallback only if both fail the decision rule.
- Runtime: llama.cpp built from source (`vendor/llama.cpp`, CPU, AVX2). One build gives the server, the benchmark
  tool, the quantiser and the HF -> GGUF converter we need after training. The server loads the GGUF with mmap, so
  after the first load the file is in the page cache and a restart takes a fraction of a second.
- Server lifecycle: started when you hover Sprout, stopped after about 90 s idle, so nothing sits in RAM.
- Decision rule on held-out phrases: at least 95% correct tool and arguments, at most 1% wrong on risky tools, and
  it must ask instead of guessing when the request is unclear. Smallest model that passes wins.

## 7. Listening (later phases)
- On battery the mic is off. Hover preloads the model; double-click opens the mic until you stop talking.
- On AC an always-on wake word ("hey tommy") may run. Battery measurement (see `results/`): keeping the mic open costs
  about +0.9 W and a wake-word model about +1.3 W in total, roughly an hour of battery, which is why it is AC-only.
- Audio is never stored, except recordings you choose to make for testing; those stay on this PC.

## 8. Layout
```
tommy/registry.py     @tool decorator, schema from type hints, argument validation
tommy/durations.py    time phrases -> seconds / clock times (English + Hinglish)
tommy/system.py       the only place that touches the machine (argv lists only)
tommy/dispatch.py     validate -> risk tier -> run -> reply text
tommy/tools/*.py      the tools
tommy/fire.py         what a timer runs when it goes off (bubble + notification)
tommy/cli.py          `python -m tommy.cli --direct <tool> k=v` test harness
tests/                parser tests
```
