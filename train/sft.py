"""Fine-tune a small model to turn an utterance into one tool call. Same script for CPU pilot and GPU runs.

    python train/sft.py --model functiongemma-270m-it --mode full --epochs 3
    python train/sft.py --model Qwen3-0.6B --mode lora --epochs 3

Prompt is compact on purpose (no tool schemas): the model learns the 16 tool names and argument shapes, which
cuts ~1,400 prompt tokens per request. Loss is on the call only. Ends with a greedy-decode score on the test set.
"""
import argparse, json, random, re, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SYSTEM = "You control a desktop assistant. Reply with exactly one function call."


def call_text(kind, name, args):
    if kind == "gemma":
        body = ",".join(f"{k}:<escape>{v}<escape>" for k, v in args.items())
        return f"<start_function_call>call:{name}{{{body}}}<end_function_call>"
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"


def prompt_text(tok, kind, text):
    role = "developer" if kind == "gemma" else "system"
    msgs = [{"role": role, "content": SYSTEM}, {"role": "user", "content": text}]
    kw = {"enable_thinking": False} if kind == "qwen" else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def parse(kind, text):
    if kind == "gemma":
        m = re.search(r"call:(\w+)\{(.*?)\}?(?:<end_function_call>|$)", text, re.S)
        return (m.group(1), dict(re.findall(r"(\w+):<escape>(.*?)<escape>", m.group(2), re.S))) if m else None
    m = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", text, re.S)
    try:
        d = json.loads(m.group(1)); return d["name"], d["arguments"]
    except (AttributeError, ValueError, KeyError):
        return None


def encode(tok, kind, row):
    p = tok(prompt_text(tok, kind, row["text"]), add_special_tokens=False)["input_ids"]
    end = "<end_of_turn>" if kind == "gemma" else "<|im_end|>"
    c = tok(call_text(kind, row["tool"], row["args"]) + end, add_special_tokens=False)["input_ids"]
    return p + c, [-100] * len(p) + c


def evaluate(model, tok, kind, rows, device, limit=None):
    model.eval(); ok = tool_ok = risky_wrong = risky = 0; fails = []
    rows = rows[:limit] if limit else rows
    for r in rows:
        ids = tok(prompt_text(tok, kind, r["text"]), return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=80, do_sample=False,
                                 eos_token_id=tok.convert_tokens_to_ids("<end_function_call>" if kind == "gemma" else "</tool_call>"))
        got = parse(kind, tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False))
        good = got is not None and got[0] == r["tool"] and {k: str(v) for k, v in got[1].items()} == {k: str(v) for k, v in r["args"].items()}
        ok += good; tool_ok += got is not None and got[0] == r["tool"]
        if r["tool"] in ("schedule_power", "hotspot"):
            risky += 1; risky_wrong += not good
        if not good: fails.append((r["text"], r["tool"], r["args"], got))
    n = len(rows)
    return {"n": n, "exact": ok / n, "tool_only": tool_ok / n, "risky_wrong": risky_wrong / max(risky, 1)}, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--mode", choices=["full", "lora"], default="full")
    ap.add_argument("--epochs", type=float, default=3); ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None); ap.add_argument("--tag", default="")
    ap.add_argument("--extra", help="extra rows (jsonl), e.g. your reviewed real use"); ap.add_argument("--extra-reps", type=int, default=5)
    ap.add_argument("--cpu", action="store_true", help="allow a (very slow) CPU run")
    a = ap.parse_args()
    kind = "gemma" if "gemma" in a.model.lower() else "qwen"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not a.cpu:
        raise SystemExit("No CUDA GPU (driver updated without a reboot, or integrated mode). A CPU run takes over an hour and slows "
                         "everything else: pass --cpu to do it anyway.")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(ROOT / "models" / a.model)
    model = AutoModelForCausalLM.from_pretrained(ROOT / "models" / a.model, torch_dtype=dtype).to(device)
    if a.mode == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    lr = a.lr or (2e-4 if a.mode == "lora" else 2e-5)
    load = lambda f: [json.loads(l) for l in (ROOT / "data/out" / f).read_text().splitlines()]
    train, test = load("train.jsonl"), load("test.jsonl")
    if a.max_rows: train = random.Random(0).sample(train, a.max_rows)
    if a.extra:
        extra = [json.loads(l) for l in (ROOT / a.extra).read_text().splitlines() if l.strip()]
        train += extra * a.extra_reps          # real requests count more than generated ones
        print(f"+ {len(extra)} real requests x{a.extra_reps}", flush=True)
    data = [encode(tok, kind, r) for r in train]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = int(len(data) / a.bs * a.epochs); step = 0; rng = random.Random(1); t0 = time.time()
    print(f"{a.model} {a.mode} on {device}: {len(data)} examples, {steps} steps, lr {lr}, trainable {sum(p.numel() for p in params)/1e6:.1f}M", flush=True)
    model.train()
    while step < steps:
        rng.shuffle(data)
        for i in range(0, len(data) - a.bs + 1, a.bs):
            b = data[i:i + a.bs]; L = max(len(x[0]) for x in b)
            pad = tok.pad_token_id
            ids = torch.tensor([x[0] + [pad] * (L - len(x[0])) for x in b], device=device)
            lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in b], device=device)
            att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in b], device=device)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
            for g in opt.param_groups: g["lr"] = lr * min(1, (step + 1) / 10) * max(0.05, 1 - step / steps)
            opt.step(); opt.zero_grad(); step += 1
            if step % 10 == 0 or step == 1:
                print(f"step {step}/{steps} loss {loss.item():.4f}  {time.time() - t0:.0f}s", flush=True)
            if step >= steps: break
    if a.mode == "lora": model = model.merge_and_unload()
    score, fails = evaluate(model, tok, kind, test, device, a.eval_limit)
    name = f"{a.model}-{a.mode}{a.tag}"
    out = ROOT / "runs" / name; out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    import shutil     # copy the original tokenizer files: tokens are unchanged and the GGUF converter needs the originals
    for f in (ROOT / "models" / a.model).iterdir():
        if f.name.startswith(("tokenizer", "special_tokens", "added_tokens", "chat_template", "vocab", "merges")):
            shutil.copy(f, out / f.name)
    (out / "score.json").write_text(json.dumps({**score, "train_seconds": round(time.time() - t0), "device": device, "epochs": a.epochs}, indent=1))
    (out / "fails.txt").write_text("\n".join(f"{t!r} want {w} {x} got {g}" for t, w, x, g in fails))
    print("SCORE", json.dumps(score), f"{time.time() - t0:.0f}s total -> {out}")


if __name__ == "__main__":
    main()
