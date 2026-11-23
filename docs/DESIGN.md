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

## 6. Models and runtime
- **Command model:** Qwen3-0.6B + LoRA, 8-bit, in llama.cpp's server on the CPU. It beat FunctionGemma 270M (95.2% vs 83% by meaning) at about 0.5 s per answer.
  The prompt holds no tool descriptions; the fine-tune taught them, which saves about 1,400 tokens per request.
- **Speech-to-text:** Hindi2Hinglish-Swift (best of four on the user's voice) in CTranslate2 int8, 0.26 s per command.
- **Wake word:** a small classifier on openWakeWord's audio features, trained on the user's voice plus synthetic voices. A wake also needs a voice in the last 1.5 s.
- **Rules that came from measurements** are in `LOG.md`; the important one: check training clips for an actual voice before using them.

## 7. One command, step by step
1. `mic.py`: a thread keeps the last 30 s of microphone frames, so nothing is lost while Jarvis is busy.
2. `ears.py` `woke()`: every 80 ms, wake score and voice level. Not printed, not transcribed.
3. `ears.py` `capture()`: `Endpoint` decides where the command starts and ends (1.5 s of quiet). After 0.5 s of quiet the next two steps start in the background.
4. `transcribe()` then `agent.propose()`: text, then one tool call. Neither has side effects, so a wrong guess about the pause costs nothing.
5. `agent.ask()`: checks the proposal against the words (times, app names), asks for what is missing, confirms risky actions, runs the tool.
6. `speak.py`: the bubble shows the text; the spoken line is chosen by the reply's *kind*.

## 8. Listening and power
- Jarvis runs on mains power only (`system/`): an open microphone costs about 0.9 W and the wake model 0.3 W more on this laptop.
- `jarvis pause` ignores the name for a while (dictation, calls). Wake clips are kept for `learn.py` on this PC only.

## 9. Layout
```
tommy/voice.py      the loop            tommy/ears.py   wake word, end of speech, speech-to-text
tommy/mic.py        microphone          tommy/speak.py  spoken lines, listening animation
tommy/agent.py      proposal + checks   tommy/dispatch.py  risk tier, run, Reply
tommy/registry.py   @tool, Reply        tommy/apps.py   installed-app matching
tommy/durations.py  time phrases        tommy/system.py the only place that touches the machine (argv lists only)
tommy/tools/*.py    the tools           tests/          run all of them before restarting jarvis.service
```
