# critter-voice

A tiny, fully local assistant for a desktop critter. You say or type something short, in English or Hinglish
(Hindi in Roman letters), and a small fine-tuned language model picks **one** whitelisted Python function.
Python then does the work. The model never runs a command.

```
"Jarvis, 25 min baad laptop sula do"
   -> wake word / speech detection -> speech-to-text -> 0.6B model picks a tool -> safety checks -> action -> spoken reply
```

Everything runs on your own machine. Audio is kept in memory and is never written to disk (except recordings you
choose to make for benchmarking).

> **Status: working prototype.** It is built for one laptop (Arch Linux, Hyprland, a Quickshell-based shell, PipeWire)
> and one speaker. Expect to adapt paths and commands. See [Limits](#limits).

## What it can do

19 tools, each one a small Python function in `tommy/tools/`:

| Area | Examples of what you can say |
|---|---|
| Timers and reminders | "set a timer for 10 minutes", "remind me to drink water in 20 minutes", "shaam 5 baje yaad dilana" |
| Power | "put the pc to sleep after 25 minutes", "shut down in an hour", "lock the screen in 5 minutes" (asks first) |
| Focus and stopwatch | "start a focus session", "focus status", "start the stopwatch" |
| Notes | "note this down: buy milk", "read my notes" |
| Do not disturb, hotspot, network | "dnd on for 30 minutes", "turn on the hotspot" (asks first), "is my internet working" |
| Apps, sites and music | "open classroom", "whatsapp kholo" (any installed app or web app, by fuzzy name), "play believer by imagine dragons" (opens a YouTube search), "pause the music", "agla gaana" |
| Close apps | "close whatsapp", "brave band karo", "close this window" (with several windows: the one on your current screen; "close all brave windows" asks first, and so do terminals) |
| Manage | "what is scheduled", "cancel all timers" |
| Anything else | it says it didn't understand, instead of guessing |

Missing details are asked for out loud: "put my PC to sleep" gets "What time, sir?".

## Design highlights

### 1. Listening without draining the battery
An always-open microphone is expensive on a laptop, so it was measured first (`listen.py`, `measure-listen.sh`, 5 minutes per
stage, battery power sensor, noise floor 0.12 W on a 6.2 W baseline):

| Stage | Extra power | Battery life lost per full charge |
|---|---|---|
| Keeping the microphone open | +0.93 W | about 0.9 h |
| Microphone + wake word model | +1.27 W | about 1.15 h |

The cost is the audio path waking the system about 47 times a second, not the Python code (under 1% CPU). So the policy is:

- **On mains power:** always listening. Say "Jarvis" or "hey Jarvis". The wake word is a tiny classifier on openWakeWord's audio
  features that never turns speech into text; the speech model runs only after it fires.
- **On battery:** the microphone stays off. Hovering the critter preloads the model; a double-click opens the mic for one command and
  closes it when you stop talking. About 50 commands a day then cost roughly 1 Wh, versus about 7 Wh for 8 hours of an open mic (estimates).

The mains/battery switch comes from UPower. *Status: the measurements and the always-listening path on mains are built; the
hover and double-click hooks in the critter's shell are designed but not built yet.*

### 2. Loading the model at the right moment
The model is not kept in memory. It is started when you hover the critter, so by the time you speak it is ready, and it is stopped after
about 90 seconds idle. Measured with llama.cpp and memory-mapped weights: the server is ready in 0.4-1.1 s depending on how warm the file cache is,
and most of that is vocabulary setup, not weights. Two more things keep the answer quick:

- **A compact prompt.** The tool descriptions are *not* sent with every request. Fine-tuning teaches the model the tool names, which cuts
  about 1,400 prompt tokens and takes prompt processing from over 1 s to about 75 ms.
- **A small model** (Qwen3-0.6B, 8-bit, about 0.6 GB), so the answer itself takes 0.4-0.8 s on the CPU.

### 3. Python functions plus a little retraining give it "agentic" ability
The model is not asked to be smart; it is asked to choose. Every ability is a plain Python function with a type-hinted signature and
a one-line docstring. A decorator turns that into the schema, and a short fine-tune teaches the model when to call it:

1. Write the function (for example `open_app(name)` in `tommy/tools/media.py`).
2. Add 15-30 example phrases in English, Hinglish and broken English to `data/gen.py`.
3. Run a LoRA fine-tune (about 5-7 minutes on a 6 GB laptop GPU) and re-run the held-out test.

That is how it went from 16 to 19 tools in one sitting, with the score rising from 92.7% to 93.6% on a harder test. The model only picks
the tool and copies words; deterministic code does the arithmetic, validates arguments, asks for confirmation on risky actions
and asks you back when something is missing ("What time, sir?").

## How it works

- **The model chooses, Python acts.** `tommy/registry.py` looks the model's answer up in a table and type-checks the
  arguments. Anything unknown becomes "I didn't understand". No model text ever becomes a shell command.
- **The model copies words; Python does the maths.** "pachees minute", "dedh ghanta" and "shaam paune paanch baje" are
  turned into numbers by `tommy/durations.py` (English and Hinglish, 61 tests), not by the model.
- **Safety checks before anything runs** (`tommy/agent.py`, `tommy/dispatch.py`): the time the model copied must match the
  time you said, risky actions (sleep, shutdown, lock, hotspot) ask for a yes/no first, and nothing risky runs while the
  screen is locked.
- **Timers are systemd user timers** named `tommy-*`, so they survive a crash and can be listed and cancelled with
  `systemctl --user`.
- **Voice:** `parec` mic -> a small "Jarvis" wake-word model trained on the user's own voice plus synthetic voices
  (`tommy/wake.py`, `train/wake_train.py`) -> Silero voice detection for the end of the command -> Hindi2Hinglish
  speech-to-text -> the model. The wake word hears sound features only, so nothing is transcribed while idle. Replies are pre-recorded phrases
  (`make_greetings.py`), several per situation so it does not repeat itself.

More detail, with the reasoning behind each choice, is in [docs/DESIGN.md](docs/DESIGN.md). Every measurement and
dead end is in [docs/LOG.md](docs/LOG.md).

### 4. It learns from your own use
Every wake and every request is kept locally in `data/usage/` (git-ignored; `--no-learn` turns it off): the 2.5 s the wake word heard,
what was transcribed, and what Tommy did. `learn.py review` walks through them (did you really say Jarvis? was that the right action,
and if not, which one?), and `learn.py retrain` retrains both the command model and the wake word on what you checked. A new command
model is only used if it scores at least as well as the old one on the held-out test. Nothing is learned without review, because an
unchecked log would teach the model its own mistakes.

### 5. Fast, because it works while you pause
It stops listening 1.5 s after your last word. After 0.5 s of quiet it already transcribes and works out the action in the background (nothing runs yet),
and drops that work if you go on speaking. Measured in a real-time test: 1.5 s from the last word to the action (median), i.e. the pause itself.

## Results so far

Held-out test (288 phrases for the 16-tool rows, 330 for the 19-tool row) whose sentence patterns and time wordings never appear in training:

| Model | Exact by meaning | Right tool | Training time (RTX 3050 6 GB) |
|---|---|---|---|
| Qwen3-0.6B + LoRA (19 tools, chat + ask-back phrases, current) | **95.2%** | 96.9% | about 5.5 min |
| Qwen3-0.6B + LoRA (19 tools + chat phrases) | 94.5% | 96.8% | about 5.5 min |
| Qwen3-0.6B + LoRA (19 tools) | 93.6% | 97.3% | about 5.5 min |
| Qwen3-0.6B + LoRA (16 tools) | 92.7% | 98.3% | about 6.5 min |
| FunctionGemma 270M + LoRA | 83.0% | 96.5% | about 4.5 min |
| FunctionGemma 270M, full fine-tune | 80.2% | 94.1% | about 5.5 min |

The target was 95%: the current model just reaches it on generated test phrases; real speech is harder. Speech-to-text on one speaker's 48 recordings: the best of four models
(Hindi2Hinglish-Swift) had a 23-27% word error rate and understood 21 of 24 time phrases. Both are small samples.

## Setup

You need Python 3.13, `uv`, `cmake` and a C++ compiler, PipeWire (`parec`, `pw-play`) and ffmpeg. A CUDA GPU is only
needed for training. Nothing below is downloaded for you automatically.

```bash
# 1. runtime environment (tools, tests, typed demo)
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python numpy pyyaml "huggingface_hub[cli]"

# 2. llama.cpp (CPU build)
git clone --depth 1 https://github.com/ggml-org/llama.cpp vendor/llama.cpp
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build vendor/llama.cpp/build -j --target llama-server llama-quantize

# 3. base models (Hugging Face)
.venv/bin/hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
.venv/bin/hf download Oriserve/Whisper-Hindi2Hinglish-Swift --local-dir models/asr/Whisper-Hindi2Hinglish-Swift

#    the speech model in CTranslate2 int8 (needs step 4): 5x faster on the CPU
.venv-train/bin/ct2-transformers-converter --model models/asr/Whisper-Hindi2Hinglish-Swift --output_dir models/asr/swift-ct2-int8 --quantization int8
cp models/asr/Whisper-Hindi2Hinglish-Swift/tokenizer.json models/asr/swift-ct2-int8/

# 4. training/voice environment (faster-whisper, openwakeword, onnxruntime, edge-tts; torch, transformers, peft for training)
uv venv --python 3.13 .venv-train   # then install torch for your CUDA version plus the packages above

# 5. data, training, export
.venv/bin/python data/gen.py
.venv-train/bin/python train/sft.py --model Qwen3-0.6B --mode lora --epochs 3 --bs 8 --tag _c
.venv-train/bin/python vendor/llama.cpp/convert_hf_to_gguf.py runs/Qwen3-0.6B-lora_c --outfile runs/q.f16.gguf --outtype f16
vendor/llama.cpp/build/bin/llama-quantize runs/q.f16.gguf models/tommy/qwen3-0.6b-lora_f-q8.gguf Q8_0

# 6. your own Jarvis wake word (about 8 minutes of recording, then about 25 minutes of generating and training)
.venv/bin/python record_wake.py
.venv-train/bin/python train/wake_synth.py
.venv-train/bin/python train/wake_train.py --final     # (without --final it holds some clips out and prints detection / false-alarm numbers)

# 7. spoken replies (needs internet once for the default Microsoft voice; or use an offline Kokoro voice)
.venv-train/bin/python make_greetings.py
```

## Run

On the author's machine Jarvis runs as a user service that starts when the charger is plugged in and stops on battery
(`system/`: `jarvis.service`, `jarvis-power.path`, `jarvis-power.sh`; `jarvis on | off | status | log`; `jarvis help` lists every command). While it listens, the desktop
critter wears headphones and shows a pulsing dot (needs the matching shell IPC, `buddy hearing true|false`).


```bash
.venv/bin/python -m tommy                       # type requests, no speech
.venv-train/bin/python -m tommy.voice           # say "Jarvis ..." (or "hey jarvis ...")
.venv-train/bin/python -m tommy.voice --no-wake # press Enter instead of using a wake word
.venv/bin/python -m tommy.cli --direct start_timer duration="25 minutes"   # call a tool without the model
for t in durations askback apps replies; do .venv/bin/python tests/test_$t.py; done
.venv-train/bin/python tests/test_endpoint.py && .venv-train/bin/python tests/test_voice_loop.py   # need your own recordings
```

Careful: sleep, shutdown and lock really do happen after you confirm.

## Add a tool

1. Write a function in `tommy/tools/<area>.py` and decorate it with `@tool(risk="safe" | "confirm", arg="what it means")`.
   The schema is built from the type hints and docstring.
2. Add 15-30 example phrases (English, Hinglish, broken English) to `data/gen.py`, regenerate the data and run a short
   LoRA top-up. The model learns tool names from training, so it needs this step; the function itself needs nothing else.

## Limits

- Tested on one machine and one voice. The reply bubble uses `qs -c caelestia ipc call buddy say`, which only exists in a
  customised Quickshell setup; elsewhere the bubble is a no-op.
- The model is below the 95% target and still misreads unusual Hinglish; the confirm step is the safety net.
- The wake word is trained on one voice (27 usable recordings plus synthetic voices). On held-out clips: 20 of 20 "Jarvis" caught, 0 of 10 silent-room
  triggers, 1 of 10 ordinary-speech clips wakes it. Small samples.
- Hover/click hooks for the critter itself are not built yet.
- Always-on microphone listening costs about 1-1.3 W on the author's laptop (measured in `listen.py` / `measure-listen.sh`).

## Models and licences

Model weights are not in this repository. Base models keep their own licences: Qwen3 (Apache-2.0), FunctionGemma (Gemma
terms), Hindi2Hinglish-Swift (Apache-2.0), Kokoro (Apache-2.0), Silero VAD and openWakeWord (see their projects). The default
spoken voice is generated with `edge-tts` (Microsoft online voices), once, from short fixed phrases. This repository has no
licence file yet, so by default all rights are reserved.
