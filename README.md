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

16 tools, each one a small Python function in `tommy/tools/`:

| Area | Examples of what you can say |
|---|---|
| Timers and reminders | "set a timer for 10 minutes", "remind me to drink water in 20 minutes", "shaam 5 baje yaad dilana" |
| Power | "put the pc to sleep after 25 minutes", "shut down in an hour", "lock the screen in 5 minutes" (asks first) |
| Focus and stopwatch | "start a focus session", "focus status", "start the stopwatch" |
| Notes | "note this down: buy milk", "read my notes" |
| Do not disturb, hotspot, network | "dnd on for 30 minutes", "turn on the hotspot" (asks first), "is my internet working" |
| Manage | "what is scheduled", "cancel all timers" |
| Anything else | it says it didn't understand, instead of guessing |

Missing details are asked for out loud: "put my PC to sleep" gets "What time, sir?".

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
- **Voice:** `parec` mic -> openWakeWord "hey jarvis" plus a speech-based wake path (Silero voice detection, then the
  speech model listens for "Jarvis") -> Hindi2Hinglish speech-to-text -> the model. Replies are pre-recorded phrases
  (`make_greetings.py`), several per situation so it does not repeat itself.

More detail, with the reasoning behind each choice, is in [docs/DESIGN.md](docs/DESIGN.md). Every measurement and
dead end is in [docs/LOG.md](docs/LOG.md).

## Results so far

Held-out test of 288 phrases whose sentence patterns and time wordings never appear in training:

| Model | Exact by meaning | Right tool | Training time (RTX 3050 6 GB) |
|---|---|---|---|
| Qwen3-0.6B + LoRA | **92.7%** | 98.3% | about 6.5 min |
| FunctionGemma 270M + LoRA | 83.0% | 96.5% | about 4.5 min |
| FunctionGemma 270M, full fine-tune | 80.2% | 94.1% | about 5.5 min |

The target was 95%, so this is not there yet. Speech-to-text on one speaker's 48 recordings: the best of four models
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

# 4. training/voice environment (torch, transformers, peft, openwakeword, kokoro/edge-tts)
uv venv --python 3.13 .venv-train   # then install torch for your CUDA version plus the packages above

# 5. data, training, export
.venv/bin/python data/gen.py
.venv-train/bin/python train/sft.py --model Qwen3-0.6B --mode lora --epochs 3 --bs 8 --tag _b
.venv-train/bin/python vendor/llama.cpp/convert_hf_to_gguf.py runs/Qwen3-0.6B-lora_b --outfile runs/q.f16.gguf --outtype f16
vendor/llama.cpp/build/bin/llama-quantize runs/q.f16.gguf models/tommy/qwen3-0.6b-lora_b-q8.gguf Q8_0

# 6. spoken replies (needs internet once for the default Microsoft voice; or use an offline Kokoro voice)
.venv-train/bin/python make_greetings.py
```

## Run

```bash
.venv/bin/python -m tommy                       # type requests, no speech
.venv-train/bin/python -m tommy.voice           # say "Jarvis ..." (or "hey jarvis ...")
.venv-train/bin/python -m tommy.voice --no-wake # press Enter instead of using a wake word
.venv/bin/python -m tommy.cli --direct start_timer duration="25 minutes"   # call a tool without the model
.venv/bin/python tests/test_durations.py && .venv/bin/python tests/test_askback.py
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
- The speech wake path runs the speech model on every short phrase it hears, so it is meant for mains power, not battery.
- Hover/click hooks for the critter itself are not built yet.
- Always-on microphone listening costs about 1-1.3 W on the author's laptop (measured in `listen.py` / `measure-listen.sh`).

## Models and licences

Model weights are not in this repository. Base models keep their own licences: Qwen3 (Apache-2.0), FunctionGemma (Gemma
terms), Hindi2Hinglish-Swift (Apache-2.0), Kokoro (Apache-2.0), Silero VAD and openWakeWord (see their projects). The default
spoken voice is generated with `edge-tts` (Microsoft online voices), once, from short fixed phrases. This repository has no
licence file yet, so by default all rights are reserved.
