"""llama-server lifecycle and one-call inference. Both models go through llama.cpp's OpenAI-style
/v1/chat/completions with --jinja, so each model's own chat template renders the same registry
schemas into its native tool-call format (FunctionGemma tokens, Qwen <tool_call> JSON)."""
import json, re, subprocess, time, urllib.request
from .registry import TOOLS, load_all
from .system import ROOT

SERVER = ROOT / "vendor/llama.cpp/build/bin/llama-server"
SYSTEM = ("You control a desktop assistant. Pick exactly one function for the user's request and copy time "
          "phrases and note text exactly as spoken. If the request is unclear or fits no function, call clarify.")


_GEMMA_CALL = re.compile(r"call:(\w+)\{(.*?)\}?(?:<end_function_call>|$)", re.S)
_GEMMA_ARG = re.compile(r"(\w+):<escape>(.*?)<escape>", re.S)


def parse_gemma_call(text):
    """FunctionGemma's own format: call:NAME{key:<escape>value<escape>,...}. llama.cpp does not parse it."""
    m = _GEMMA_CALL.search(text)
    return (m.group(1), dict(_GEMMA_ARG.findall(m.group(2)))) if m else None


class Server:
    def __init__(self, gguf, port=8199, threads=4, ctx=4096):
        self.port, self.proc = port, None
        self.cmd = [str(SERVER), "-m", str(gguf), "--port", str(port), "-c", str(ctx), "-t", str(threads),
                    "--no-webui", "--jinja", "--no-warmup", "-np", "1"]

    def start(self):
        t0 = time.monotonic()
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while time.monotonic() - t0 < 30:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=0.2).read()
                return time.monotonic() - t0
            except OSError:
                time.sleep(0.02)
        raise RuntimeError("llama-server did not start")

    def stop(self):
        if self.proc:
            self.proc.terminate(); self.proc.wait()

    def call(self, utterance, extra=None):
        """Returns ((tool_name, args_dict) or None, seconds)."""
        load_all()
        body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": utterance}],
                "tools": [{"type": "function", "function": t.schema()} for t in TOOLS.values()],
                "temperature": 0, "max_tokens": 96, "stop": ["<end_function_call>"], **(extra or {})}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.monotonic()
        msg = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]
        dt = time.monotonic() - t0
        calls = msg.get("tool_calls") or []
        if not calls:
            return parse_gemma_call(msg.get("content") or ""), dt
        fn = calls[0]["function"]
        args = fn["arguments"]
        return (fn["name"], json.loads(args) if isinstance(args, str) else args), dt
