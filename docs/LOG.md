# Tommy build log

## 2026-11-22: setup and tool layer
- Checked Hugging Face. Google's FunctionGemma repo needs a manual licence click; the `unsloth/functiongemma-270m-it`
  mirror is not gated. Qwen3-0.6B is Apache-2.0 and not gated. Used the unsloth mirror for FunctionGemma.
- Cloned llama.cpp into `vendor/llama.cpp` (shallow), CPU build, native flags, curl off.
- Downloaded: GGUF Q8_0 of both models (`models/gguf/`) plus safetensors of both (`models/<name>/`, for training).
- Wrote the registry, duration parser, system layer, dispatcher and 16 tools (`python -m tommy.cli --list`).
- Duration parser: 57/57 tests. Two first-run failures: "an hour and a half" (the "and a half" was only handled after a
  bare number, not after a unit) and one wrong expected value in my own test (1 h + 25 min is 5100 s, not 6300).
- Timers: first version read the next fire time from systemd, which reports it relative to boot (showed "0 s left").
  Fixed by writing the due time into the unit description. A 5 s test timer started and ran its service.
- Not yet verified: that `lock isLocked` is true while the screen is locked (the confirm-tier guard depends on it).

## 2026-11-22: runtime check and zero-shot baseline (nothing trained yet)
- llama.cpp build `e300919` OK (server, cli, bench, quantize). Both GGUF Q8_0 models run through `llama-server --jinja`.
- **Load time, measured: FunctionGemma 270M 0.75-0.93 s, Qwen3 0.6B 0.6-1.1 s, to /health OK with the file already in
  the page cache.** That misses the 0.5 s target. Server log shows ~0.65 s between "loading model" and the tokenizer
  warning, i.e. vocabulary setup (Gemma's vocabulary is 262k tokens), not weight loading. `--no-warmup` saves ~0.04 s.
  Options for later: (a) hover preload hides it (you hover before you speak); (b) keep the server resident, since an
  idle server uses no CPU and its model pages are evictable; (c) after fine-tuning, prune unused vocabulary rows,
  which also shrinks the 270M model a lot (most of its parameters are the embedding table).
- **Prompt size matters:** the 16 tool schemas are ~1,400 tokens. A first run with context 1024 failed with HTTP 400
  (context exceeded); context is now 4096. Fine-tuned models can learn the tool names, so the schemas can later be
  dropped from the prompt, cutting latency.
- **llama.cpp does not parse FunctionGemma's call format** (`call:NAME{k:<escape>v<escape>}`), and the model does not
  stop after the first call (it repeated calls until max_tokens). Fixed with stop string `<end_function_call>` and a
  small parser (`parse_gemma_call` in `tommy/model.py`). Qwen3's calls are parsed by llama.cpp.
- Zero-shot, 8 hand-picked phrases (not a real benchmark, only a sanity check):
  - Qwen3-0.6B: right tool for ~4/8. Wrong on "put my pc to sleep after 25 minutes" (picked `lock`, a wrong risky
    action), on a Hinglish "sula do" (picked focus_start), on a reminder (picked a timer), and it called
    `network_status` for "what is the weather" instead of `clarify`. Latency ~1-1.5 s per call.
  - FunctionGemma-270M: right tool for ~3/8, 0.2-0.4 s per call. Wrong on sleep (picked a timer), on Hinglish, on focus
    (picked pause), and returned nothing for the weather question.
  - Both are well below the 95% bar, as expected before fine-tuning. This is the number fine-tuning has to beat.

## 2026-11-22: data, training script, CPU pilot
- `data/gen.py`: 965 train / 156 test rows, 16 tools plus `clarify`. Test uses different templates and time phrases
  from train (asserted no shared utterance), so passing it cannot come from memorising.
- `train/sft.py`: one script for CPU and GPU, full or LoRA, compact prompt (no tool schemas), loss on the call only,
  greedy-decode score on the test set at the end. Training env `.venv-train` (CPU torch 2.15, transformers 5.24, trl,
  peft). The GPU run needs the CUDA torch wheel swapped in after the reboot.
- **CPU pilot** (FunctionGemma 270M full fine-tune, 96 examples, 12 steps, batch 8): about 5.5 s per step, loss 5.84 -> 0.62.
  Score on 12 test rows: 0% exact (expected after 12 steps). Full speed on CPU would be ~11 min per pass over 965 rows.
- **Pipeline bug found and fixed:** the script re-saved the tokenizer, and `convert_hf_to_gguf.py` rejected it
  (`assert max(vocab) < vocab_size`). It now copies the original tokenizer files instead (tokens are unchanged).
- **End to end proven:** pilot weights -> GGUF f16 -> Q8_0 -> llama-server `/completion` with the compact prompt returned
  `call:schedule_power{action:<escape>poweroff<escape>,after:<escape>25 min<escape>}`. The time span is copied right;
  the action is wrong, which is what a 12-step pilot should look like. **Prompt processing took 75 ms**, versus
  1+ s with the 1,400-token tool schemas. The compact prompt is a large latency win.
- Saved pilot weights are fp32 (1 GB); Q8_0 GGUF is ~0.3 GB.

## 2026-11-22: first real training runs (RTX 3050, CUDA torch 2.14 cu130)
Setup: GPU mode `nvidia` after reboot, `.venv-train` switched to the CUDA wheel. Compact prompt (no tool schemas), 3 epochs,
greedy decode on the held-out test set. `train/sft.py`. FunctionGemma full fine-tune at batch 16 ran out of memory (the
262k-token vocabulary makes the output layer large); batch 4 fits. Peak GPU use was under 6 GB for all runs.

Round a (965 train / 156 test): FunctionGemma full 73.7% exact (tool 88.5%, risky-wrong 11.9%, 208 s);
Qwen3-0.6B LoRA 89.1% exact (tool 94.9%, risky-wrong 0%, 244 s). Failures were mostly few sentence patterns per tool.
Round b: widened the data to 1337 train / 288 test (more patterns, more number words, more out-of-scope phrases).
Scores below are by `bench/semantic.py`, which compares durations by parsed seconds ("8 mins" = "8 minutes") and ignores
the clarify question text. Strict = exact string match.

| run | strict | by meaning | right tool | train time |
|---|---|---|---|---|
| FunctionGemma 270M full | 74.3% | 80.2% | 94.1% | 325 s |
| FunctionGemma 270M LoRA r16 | 79.9% | 83.0% | 96.5% | 279 s |
| Qwen3-0.6B LoRA r16 | 85.4% | 92.7% | 98.3% | 393 s |

- Qwen3-0.6B leads. Still below the 95% bar; 2 of its 21 misses are risky tools.
- Real errors seen: a dropped word in a copied time ("paune do ghante" -> "do ghante" = 2 h instead of 1 h 45), focus
  pause/stop mix-ups on unseen Hinglish, and out-of-scope phrases answered with a tool. Harmless misses: "8 mins" vs "8
  minutes", a stray "do" at the start of note text.
- Planned fixes: cross-check each model-copied time against the parser's reading of the original utterance and ask
  when they disagree; snap/validate tool names; more patterns for the confused tool pairs; GBNF grammar in the benchmark.

## 2026-11-22: speech-to-text benchmark on the user's own voice
- 48 recordings (2.4 min, 16 kHz mono, internal mic at 12% gain), checked: every clip has clear speech, SNR 26-36 dB,
  none clipped. Peak level is low (median ~4.5% of full scale) but clean. `record.py` / `bench/asr.py`.
- One recording was saved with the label "redo" (script only accepted `r`); removed, script fixed.
- **lipi-hn dropped:** custom `.lipi` 4-bit format, no runtime published (the PyPI package "lipi" is an unrelated font tool),
  and its licence is non-commercial. Could not be run.
- Results (GPU, fp16; 48 clips; "time understood" = the parser reads the transcript the same as what was said, 24 clips with a time):

| model | WER | time understood | s/clip |
|---|---|---|---|
| whisper-base (control, forced English) | 195% | 15/24 | 0.16 |
| whisper-tiny-hinglish (151 MB) | 40.6% | 15/24 | 0.06 |
| whisper-base-hinglish (290 MB) | 35.7% | 16/24 | 0.09 |
| **Hindi2Hinglish-Swift (283 MB)** | **26.8%** | **19/24** | 0.12 |

- Winner on this voice: Hindi2Hinglish-Swift. Most "errors" are spelling variants ("Pachchis" for pachees), which the parser
  already reads. Real mishearings: "paune" -> "Pahunch" (read as 2 h instead of 1 h 45), "sula" -> "suna", "sawa" -> "Savaag".
- Parser bugs found by this test and fixed (now 61/61): "a.m." was read as minutes ("7 thirty a.m." = 30 minutes), "saam"
  (spelling of shaam) was unknown, "7 30 am" and "7 thirty am" were not read as clock times.
- Limits: 24 time phrases is a small sample; one speaker, one mic. Not yet measured: CPU speed/energy of the speech model
  (GPU numbers above), and the full speech -> Tommy accuracy.
- The "cross-check the time against the utterance" safety net does not help once the input is a transcript: the speech model's
  mistake is already in the text. The defence is the confirm step (it shows the parsed time) and retraining Tommy on transcripts.

## 2026-11-22: voice front end (`tommy/voice.py`)
- Chain: mic (pw-record, 4x software gain because the internal mic is quiet) -> openWakeWord "hey jarvis" (stock model; the
  user's name for the critter is Jarvis) -> Silero VAD (via openWakeWord's `VAD`, speech > 0.4, end = 0.8 s quiet, 4-frame
  preroll) -> Hindi2Hinglish-Swift on CPU -> `Agent.ask` -> `buddy say` bubble. Risky actions are confirmed by voice
  ("say yes or no"; yes = yes/haan/ok/kar do..., no = no/nahi/mat/cancel...). Audio stays in memory.
- Silero on a real recording: speech 0.6-0.8, silence ~0.01. Mic noise alone never opens it.
- Run on three recordings with `--wav`: 2 of 3 right. Speech-to-text on CPU ~2.4 s per clip, Tommy 0.5-0.8 s, startup 5 s.
  One miss: "pachees minute baad sleep" heard as "Pachchis minute baad slip" -> set_reminder instead of sleep (the retrain on
  ASR output is meant to fix this kind of error). Test timers created by these runs were cancelled.
- Bug: the scikit-learn stub (needed by openWakeWord) broke torch's module inspection (`sklearn.__spec__ is None`); fixed by
  importing torch/transformers before installing the stub.
- Not yet tested live: the wake word on the user's real voice and mic (false wakes, misses). Run needs `.venv-train`
  (torch + transformers + openwakeword); the runtime venv has no speech-to-text yet.

## 2026-11-22: "hey jarvis does nothing" - root cause: the mic stream was being misread
- Symptom: wake word never fired; diagnostics showed std 12,000-18,000 and peaks at full scale even in silence, while a
  file recording of the same mic had std ~100.
- Cause: `pw-record ... -` writes a **`.snd` (AU) stream to a pipe: 24-byte header, big-endian samples**. `voice.py` and
  `listen.py` read it as raw little-endian int16, so every frame was noise. Fix: use `parec --format=s16le` (raw
  little-endian). Verified: parec through a pipe gives std 160, peak 1526, the same as the file recordings.
- **This corrects the 21 Nov conclusion** that the mic "was saturating at 27% gain" (std ~18.9k, 2.4% clipped). That
  reading came from the same misparse, so the gain was never clipping. The mic gain was lowered 27% -> 5% for nothing; it now
  sits at 12%. The battery measurements (watts) are unaffected, but the gate-stuck-open diagnosis was wrong. Speech at 12% is quiet
  (peak ~1,500), which is why `voice.py` applies 4x software gain; restoring ~27% in PipeWire would give a better signal.
- Not yet verified: that "hey jarvis" fires on the user's voice now (the diagnostic after the fix was run while nobody spoke).

## 2026-11-22: wake word does not recognise this voice; speech-based wake added
- With the mic stream fixed (see above) the user could trigger "hey jarvis" only by saying it twice. Measured with the correct
  audio: three normal-volume "hey jarvis" attempts scored 0.00 at gains x1, x4 and x8 (a 0.24 near-miss was seen once earlier).
  The stock openWakeWord model is trained on synthetic voices and does not fit this accent reliably; lowering the threshold
  alone would not help (scores are ~0, not ~0.4).
- Added a second wake path in `tommy/voice.py`: Silero VAD collects short bursts of speech (0.24 s .. 3.2 s, ends after
  0.56 s quiet); each burst is transcribed by the Hinglish speech model and matched against "jarvis" (difflib ratio >= 0.7).
  Whatever follows "jarvis" is run as the command, so "hey jarvis, 10 sec ka timer lagao" works in one breath; "hey jarvis"
  alone beeps and listens. The old wake model stays as a fast path (threshold lowered 0.5 -> 0.4). Cost: one speech-to-text
  run (~2.4 s CPU) per short phrase heard while idle, so this is for AC only. Audio stays in memory.
- Speech-to-text now gets a vocabulary hint ("Jarvis, set a timer, reminder, sleep, ..."): WER 29.5% -> 23.2% and time phrases
  understood 20/24 -> 21/24 on the 48 recordings (CPU, 2.4 s/clip either way).
- After each command the loop now throws away audio that piled up while it was busy and resets the wake model and the burst
  collector, so it needs a fresh "hey jarvis" for the next command.
- Observed: "10 sec ka timer lagao" was heard as "Das second ka time mein" and became a stopwatch start. The retrain on
  speech-model output is the planned fix.

## 2026-11-22: spoken greeting instead of a beep, longer pause
- Added Piper TTS (`piper-tts` in `.venv-train`, voice `en_GB-alan-medium`, 61 MB in `models/tts/`). Phrases are synthesised once to
  `models/tts/greetings/*.wav` ("Hello sir.", "Yes sir?", plus two longer ones unused), and played from file on wake: no TTS cost
  at run time. Mic audio recorded during the clip is discarded so Tommy does not hear itself. Falls back to the beep if files are missing.
- End-of-speech pause raised: speech starts above VAD 0.4, only frames below 0.15 count as quiet, ends after 1.7 s (21 frames);
  max command length 14 s. Cause of "cuts me off": 0.8 s end and in-sentence dips counted as silence.
- "Jarvis" match loosened to ratio >= 0.62 so "Hazaarvis" wakes it ("Service", "Marvel" still do not).
- Replaced Piper with **Kokoro** (kokoro-onnx 0.6.1, int8 model 89 MB + voices 27 MB in `models/tts/kokoro/`), a more natural voice.
  `make_greetings.py <voice>` regenerates the greeting files (default now `bm_fable` (the user picked the third sample); also bm_lewis, bm_fable, bm_daniel, am_*). Samples
  of the three British voices are in `models/tts/samples/`. `piper-tts` stays installed in `.venv-train` but is no longer used.

## 2026-11-22: ask-back for missing times, "now", plain "Jarvis"
- `Agent.ask(text, confirm, ask_back)` now checks every time argument against the words actually said:
  - no time in the sentence -> ask back ("What time, sir?" spoken and in the bubble), then use the answer; "now"/"abhi"/"turant" is accepted;
  - a time in the sentence but the model failed to copy it -> use the sentence's own words;
  - the model's copy disagrees with the sentence ("paune do ghante" -> "do ghante") -> confirm "Did you mean 1 h 45 min?".
  The earlier check wrongly accepted an invented "0" (read as midnight); the rule is now "no time said = ask".
- "now" means a 1 second delay in `schedule_power` (still behind the confirm step).
- Spoken phrases added: "What time, sir?", "Shall I go ahead, sir?" (before every yes/no), "Sorry sir, I didn't catch that."
  (generated by `make_greetings.py bm_fable`).
- Tests: `tests/test_askback.py` (10 cases, stubbed model and executor, nothing real runs) 10/10; durations 61/61.
- Plain "Jarvis" and "Hey Jarvis" both wake it through the speech path; one-breath "Jarvis, set a timer for ten minutes" returns the
  command. Checked with synthesized speech (Kokoro voices) through the real VAD, speech model and matcher; not yet with the user's voice.
- A test run that piped "y" into the sleep confirmation was blocked by the permission check (it could have suspended the laptop);
  the logic is now tested with stubs only.

## 2026-11-22: spoken confirmations
- After each command Tommy now says a short phrase matching the result (Kokoro `bm_fable`, pre-generated): "Done, sir." / "Okay sir, done."
  on success, "Cancelled, sir.", "Never mind, sir.", "Sorry sir, I didn't catch that.", "I couldn't do that, sir." (`reply_clip` in
  `tommy/voice.py`). Answers to questions (list_scheduled, focus_status, network_status, read_notes) stay text-only.
- The time question ("What time, sir?") now waits 8 s per try and asks twice (the first live test returned an empty answer after 5 s).

## 2026-11-22: voice choice and variety
- User compared five voices and picked **Microsoft en-GB-ThomasNeural** (via `edge-tts`; internet needed only while generating,
  only the short fixed phrases are sent; the saved wav files are what Tommy plays). Kokoro full-precision model (311 MB) is also
  installed as an offline option: `make_greetings.py bm_fable`.
- Spoken lines are now several per situation (greet 10, done 10, nevermind 4, cancelled 4, sorry 4, cant 4, what_time 4, confirm 5 = 45 clips),
  chosen at random, never the same line twice in a row (`play()` in `tommy/voice.py`; edit `PHRASES` in `make_greetings.py`).

## 2026-11-22: media tools, 19 tools, retrain `_c`
- Added `open_site` (youtube, whatsapp, gmail, github, chatgpt, maps, spotify; a fixed table, the model never supplies a URL or command),
  `play_music` (opens a YouTube search; `yt-dlp` is not installed, so no autoplay) and `media_control` (playerctl play/pause/next/previous).
- Data now 1574 train / 330 test, 19 tools. Phrases that are now valid ("open youtube", "play some music", "gaana chalao", "open spotify")
  were removed from the "refuse" list. Qwen3-0.6B LoRA `_c`: 88.8% strict, **93.6% by meaning**, 97.3% right tool, 0 risky misses (21/330), 323 s.
  All 12 spot-check phrases proposed the right tool (nothing executed).
- A test started a second model server on the same port as the user's running voice session and silently talked to the old model
  ("ready in 0.01 s" gave it away); re-tested on a separate port.

## 2026-11-22: custom Jarvis wake word (replaces the speech-based wake path)
- Why: the speech-based wake path transcribed every short phrase heard while idle (the user saw a long stream of their own talking
  scroll past). The user chose a wake word trained on their voice instead. Transcripts of non-wake speech are no longer printed.
- Data: 70 recordings by the user (`record_wake.py`: 29 plain wake, 11 wake+command, 30 ordinary speech) + 438 synthetic "Jarvis" clips
  (32 Kokoro voices incl. Hindi-voiced, 14 Microsoft voices incl. en-IN, speeds/pitches) + 959 synthetic non-wake clips (28 min: Tommy's own
  sentences, doc sentences, look-alike words: service, Marvel, harvest, Travis, Hey Siri ...). `train/wake_synth.py`.
- Model: 16 x 96 openWakeWord embeddings (about 1.3 s) -> 96 hidden units -> score, in `tommy/wake.py`; fires only if two frames in a row are high.
- First run: 100% detection but ~150-240 false alarms per hour (4 of 10 held-out ordinary clips fired). Causes: half-said words were never
  negatives, positives weighted far above negatives, too little ordinary speech. Fixes: partial-word windows as negatives, speed/pitch variation of
  the user's clips, equal weights, three rounds of hard-negative mining, two-in-a-row rule.
- Held-out result (9 of the user's plain wake clips, 11 wake+command, 10 ordinary clips, 15% of synthetic, 13 min of negative audio):
  8/9 plain, 11/11 with command, 100% synthetic; 2 false alarms in 12.9 min (~9/h, both on the user's ordinary clips with look-alike
  lines such as "Hey Siri"/"Hey buddy"). Threshold 0.8.
- Final model trained on everything: 95% of the user's wake clips fire (they were in training), 1 of 30 ordinary clips fires, room noise max 0.15.
- Bugs: the scikit-learn stub (needed by openWakeWord) must have a module spec or torch's lazy imports fail (`sklearn.__spec__ is None`);
  a hung Microsoft-voice request stalled generation for 10 minutes (added a 20 s timeout); a `pkill -f` matched its own shell (twice).
- Not yet tested live with the user's voice in a real room for hours.

## 2026-11-22: "hears nothing" after the wake word
- Symptom (live log): the Jarvis wake fired often, then `(heard nothing)` after 5 s of waiting; one command worked only after the user waited.
- Cause: the loop played the greeting (about 1-2 s) first and discarded the mic audio recorded during it, so a command spoken right after
  "Jarvis" was thrown away. (False wakes may also contribute; not measured yet.)
- Fix: listen immediately after the wake word (ignoring the first 3 frames, the tail of "Jarvis" itself, as a speech start); if speech begins
  within 1.3 s it is a one-breath command and no greeting is played; if not, greet and then wait up to 6 s. A leading "Jarvis" in the transcript is
  stripped; a transcript that is only the wake word counts as "no command yet". Detector logits are clipped (overflow warnings were harmless
  but noisy: very large negative logits on out-of-range audio).

## 2026-11-22: earlier listening during the greeting, wake score shown, chat is not a command
- Live: "I am just checking. Are you listening to me or not?" turned do-not-disturb ON. Added 40 chatty non-command phrases (English and
  Hinglish: "are you listening", "testing testing", "sun raha hai kya" ...) as `clarify` and retrained (`_d`, 1604 train / 346 test):
  88.2% strict, **94.5% by meaning** (19 real misses, 0 risky), up from 93.6%. Spot check: the chatty sentences now get "say that differently";
  normal commands unchanged. `tommy/agent.py` now loads `qwen3-0.6b-lora_d-q8.gguf`.
- User asked to start listening 0.5-0.8 s before the greeting ends. `play(..., lead=0.7)` starts the clip without waiting, drains the mic
  0.7 s before its end and returns how many frames still hold our own voice; `capture()` ignores speech *starts* in those frames but keeps
  them (preroll = skip + 2), so words spoken over the end of the greeting are kept.
- Each wake now prints its score (`* woke (score 0.97)`), to tell real wakes from false ones. Wake threshold raised 0.8 -> 0.9 at run time
  (held-out scores saturate near 0 or 1, so this costs almost no detection).
- Still unknown: how often it wakes falsely in this room. Next step if it does: keep the audio around false wakes (with the user's consent) as
  hard negatives and retrain the wake model.

## 2026-11-22: learning from use, mic thread, live fixes
- Live: "set a reminder" was misheard as "Shatter reminder" -> `cancel_scheduled(reminder)` -> 2 real reminders cancelled without asking.
  `cancel_scheduled` is now confirm-tier.
- Live: "What time, sir?" got no answer twice: the answer was spoken over the question and discarded. Questions (confirm, what time) now listen
  during their last 0.7 s, like the greeting.
- Live: "after this it starts ignoring me". Likely cause: the mic was read only when Tommy was free; during playback and transcription
  the pipe (about 2 s) filled and the recorder could stall. `Mic` now reads on a background thread into a 30 s queue and restarts parec if it stops.
  Tested: still streaming after 5 s and 8 s busy periods. Not proven to be the cause.
- Live: "10 second stopwatch" -> stopwatch with an invalid duration; "set an alarm" -> refused. Added stopwatch/alarm phrasings for timers and
  timers with no time (the agent then asks "What time, sir?"); removed "set a timer"/"set an alarm" from the refuse list. Retrain `_e`.
- Learning from use: each wake writes `data/usage/events.jsonl` (+ the 2.5 s of audio before the wake in `data/usage/wake/`). `learn.py review` labels
  wake clips (real / false) and actions (right / corrected); `learn.py retrain` retrains the command model with reviewed requests x5
  (`sft.py --extra`) and keeps it only if its held-out score is not worse, and retrains the wake word with reviewed clips added
  (`wake_train.py` reads `data/usage/reviewed.jsonl`), keeping the previous model as `jarvis_wake.prev.npz`.
- Retrain `_f` (1679 train / 355 test) adds "set a reminder" with nothing said (Tommy asks the time, then "What should I remind you about, sir?")
  and stopwatch/alarm phrasings: 93.0% strict, **95.2% by meaning** (17 real misses, 0 risky): first run above the 95% target. Live-log phrases
  now right: "10 second stopwatch" -> timer, "set an alarm" -> asks time, "Set a reminder." -> asks (was cancel). `tommy/agent.py` loads `_f`;
  `models/tommy/current-score.txt` = 95.2 is the bar `learn.py retrain` must not drop below. Agent now also asks for any missing non-time required
  argument (reminder text, note text); spoken questions added. `tests/test_askback.py`: 10/10 + reminder flow.

## 2026-11-22: always listening on the charger, listening animation on Sprout
- `system/jarvis.service` runs `python -m tommy.voice`; `system/jarvis-power.path` watches `/run/power-source` (written by the system's
  udev battery rule) and `jarvis-power.sh` starts the service on AC and stops it on battery. `jarvis off` (`system/jarvis`, linked to
  `~/.local/bin/jarvis`) holds until the next plug-in; `jarvis on|status|log`. Units are symlinked into `~/.config/systemd/user/`.
  Tested: start on AC, `jarvis off` stops it with no model server left behind, a repeat charger signal keeps it off, `jarvis on` restarts it.
  SIGTERM now exits like Ctrl-C (stops the model server, clears the animation).
- Caelestia fork: `buddy hearing true|false` IPC sets `Buddy.listening`; Sprout wears headphones, stays awake and shows a pulsing dot
  (desktop and bar) only while a capture is running. `qs ipc call buddy listen` failed: qs reserves `listen`, renamed to `hearing`.
  qmllint clean, check-sprout ok, shell restarted twice, no new log errors. The animation itself was not seen in a screenshot (a terminal
  covered that corner).
- End-of-speech pause raised 1.7 s -> 3 s at the user's request (each command now takes about 1.3 s longer to start processing).
- Marking false wakes: say "wrong" / "galti se" / "false alarm" / "not you" / "kuch nahi" right after a wake (logged as outcome "false" and
  labelled at once), or run `jarvis wrong [N]` / `jarvis right [N]` (`learn.py wrong|right`) for the latest unlabelled wake(s);
  `jarvis review` opens the full review. Labelled wakes go into the next `learn.py retrain`.

## 2026-11-23: floating timer clock
- Timers ran as invisible systemd timers. Now `system.write_state()` writes `~/.local/state/tommy/timers.json` ({kind, label, due}) on every
  schedule/cancel (and once when the agent starts), and the shell shows a small click-through pill top-right (`TimerPill.qml` in the
  Caelestia fork): countdown timers, scheduled sleep/shutdown/lock, and the shell's own stopwatch. Reminders stay as notifications. It exists
  only while something runs and ticks once a second.
- First try showed nothing: the file did not exist when the shell started, so its file watcher never attached. Fixed by creating the file at
  agent start, and writing it in place (an atomic rename can drop a file watch). Verified by screenshot: a 2-minute test timer and an existing
  26-minute timer appeared and counted down; removing the test timer made it vanish within 2 s.
- Live: "Cancel timer." -> confirm -> answer "cancel, haan, it, yeah" was read as NO because "cancel" was in the no-words, so the timer
  kept running while Tommy said "Cancelled." (meaning: not done). The floating clock was right to keep showing it. Fixes: no-words are only
  clear negatives (no, nahi, mat, don't, not); a refused action now says "Okay, I won't do that." / "Never mind, sir.", and "Cancelled, sir."
  is spoken only when a cancel really happened. The timer the user meant to cancel was cancelled by hand.
- End-of-speech pause set to 1.5 s (19 frames) at the user's request (was 3 s).

## 2026-11-23: random wakes and missed wakes
- Analysed all 32 logged wakes (`data/usage/`): 11 of 14 "nothing" wakes had NO speech before them and very low levels (frame std 28-60,
  peaks 116-472): the wake model misfires on quiet room noise (score 0.93-1.0). It does not misfire on synthetic noise or on a fresh reset;
  the recorded clips re-trigger it, so it is the sound itself (levels never seen in training). 14 of 17 "command" wakes had clear speech
  (max voice 0.49-0.95 in the last 1.5 s); the other 3 had none, and their captures were background talk.
- Fix 1: a wake counts only if the voice detector reached 0.3 in the last 1.5 s (`VOICE_NEEDED`). Replayed over the log it blocks 11/14
  silent wakes and the marked false wake, and keeps 14/17 command wakes (the 3 it drops had no speech). Blocked wakes are logged as
  "ignored" so they can be reviewed.
- Fix 2 (missed "Jarvis"): the spoken reply now plays without blocking (`say_async`), so the mic is not thrown away for the reply's whole
  duration; our own voice is ignored via `mute_until` (reply length + 0.3 s). Cooldown after a wake 2 s -> 1 s.
- My edits broke the service twice (a duplicated `if` line, then a removed `seg` line); compile checks do not catch a missing name,
  so every restart is now preceded by running the voice loop for ~15 s and checking its output.
- Not proven live yet: whether misses are really fixed (no log of misses exists) and whether 11 silent wakes was the whole problem.
- Added `tests/test_voice_loop.py`: the real `voice.main()` with a fake microphone playing the user's recorded "Jarvis, <command>" clips (agent, speakers,
  desktop stubbed). It exercises wake -> capture -> speech-to-text -> agent. Run it before any restart of the service.

## 2026-11-23 (evening): optimisation pass
Complaint: "start a timer of 25 minutes" took about 20 s, random wakes, missed wakes, generic answers.
- **Listening ran to its 14 s cap.** On 48 recordings + real room noise the old rule never ended on 14: it needed 3 loud voice frames (short commands
  have 1-2) and fed every frame to the recurrent voice detector twice. New `Endpoint`: one pass; speech = voice > 0.4, or > 0.15 and louder than 15% of the
  command's peak; start at 0.2; 1.5 s of quiet ends it; 8 s cap counted from your first word. Now 48/48 end, none cut early, median 1.48 s after the last word.
- **Speech-to-text 6.35 s** that time because a retrain had silently fallen back to the CPU (NVIDIA libraries upgraded to 620.73 without a reboot, module still
  620.70). `sft.py` now refuses a CPU run without `--cpu`.
- **Speech engine: CTranslate2 int8** instead of PyTorch, same Swift weights: 0.26 s vs 1.36 s per clip. The prompt must be exactly the one the model was trained
  with (hint, `<|hi|>`, timestamps allowed); the faster-whisper wrapper's own prompt lost 3 time phrases. Final, 48 recordings: times understood 21/24 both; WER 25.0%
  vs 22.8%; through the command model: same tool 45/48 both, same full action 40 vs 41. PyTorch is no longer loaded (start 1.2 s; 612 MB).
- **Work during the pause:** after 0.5 s of quiet, speech-to-text and the model's proposal run on a worker thread (no side effects; dropped if you go on).
  Real-time loop test (`tests/test_voice_loop.py`): last word -> action **median 1.51 s, max 1.73 s**, with a training run in the background.
- **Wake word fired on silence because I had trained silence as "Jarvis":** 013.wav is blank and 022.wav is a whisper; both were in the positives, 60 copies each, amplified.
  Positives now need an audible voice. 31 no-voice wakes from real use and quiet-room sound at its true level were added as negatives. Held out:
  Jarvis 20/20 (was 18/20), no-voice wakes 0/10 (was 10/10), ordinary speech 1/10 (was 0/10, but the old model had trained on those clips).
- **Missed wakes:** every ignored wake reset the detector (0.1 s rebuild + 1.3 s refill + cooldown), up to once a minute. Ignored wakes now reset nothing. Voice gate 0.3 -> 0.15
  (ignored wakes measured 0.00-0.05).
- **Typed answers:** tools, dispatcher and agent return `Reply(text, kind)`; each kind has its own lines (no such app, didn't get that, no time, locked, failed ...).
  Questions are plain words. The hotspot, focus stop and media keys used to claim success without checking; now they check. A generic "confirm" could have picked
  a "shut the PC down?" line (glob matched confirm_*): fixed and tested.
- **open_app** replaces the fixed site list: fuzzy match against installed .desktop entries (incl. Brave web apps), asks when unsure, "There's no app called X" when nothing fits.
  The command model is not retrained for it yet (no GPU until a reboot); "open X" / "X kholo" is recognised in Python meanwhile.
- `jarvis pause [min]` / `jarvis resume`. "7.30 a.m." was read as 7:18: fixed.
- Code: voice.py 472 lines -> voice, ears, mic, speak (415); the speech-based wake path, beep, zero-shot code and --wav/--gpu modes removed.

## 2026-11-23 (night): closing apps
- Live: "close WhatsApp" was guessed as opening it, and once as "cancel everything", which (after a yes to "Shall I cancel it, sir?") cancelled a real 30-minute timer.
  The model knows no close tool. No retrain: "close/quit/exit X" at the start of a sentence, and "X band karo" when X is an app, are read by rule before the model.
  Timers, power, focus, hotspot, notifications, music keep going to the model.
- Windows are matched to apps by class (web apps have their own, so "close WhatsApp" never touches Brave and the reverse). Several windows, terminals, unsure names
  and "this window" are asked first. Closing is a normal close by address (`hl.dsp.window.close({window = "address:..."})`, verified with two throwaway windows: the
  address is respected, an unknown one does nothing); nothing is killed, and if a window stays open Jarvis says so.
- Our own greeting's last word was transcribed ("Sir, close WhatsApp", "Yes sir, ikattis minute ..."): a leading "... sir" is dropped. "ikattis" (31) added.
  The cancel question is now "Shall I cancel your timers, sir?".
- With several windows "close X" now means the one on the screen in use (visible workspace of the focused monitor; the focused window if several are there). "all" / "sab"
  means every window and asks. None on this screen: asks about all of them. First version failed silently (an unused special workspace has no `id` in `hyprctl monitors -j`,
  and the fake test data had one); found by a dry run on the real desktop, then verified with three throwaway windows on three workspaces: closed 1, left 2.
