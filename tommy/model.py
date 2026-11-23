"""llama-server lifecycle (llama.cpp)."""
import subprocess, time, urllib.request
from .system import ROOT

SERVER = ROOT / "vendor/llama.cpp/build/bin/llama-server"


class Server:
    def __init__(self, gguf, port=8199, threads=4, ctx=512):
        self.port, self.proc = port, None
        self.cmd = [str(SERVER), "-m", str(gguf), "--port", str(port), "-c", str(ctx), "-t", str(threads),
                    "--no-webui", "--no-warmup", "-np", "1"]

    def _up(self):
        try:
            return bool(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=0.2).read())
        except OSError:
            return False

    def start(self):
        """Use the server already running (the voice service's), which is then not ours to stop; else start one."""
        t0, self.proc = time.monotonic(), None
        if self._up():
            return 0.0
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while time.monotonic() - t0 < 30:
            if self._up():
                return time.monotonic() - t0
            time.sleep(0.02)
        raise RuntimeError("llama-server did not start")

    def stop(self):
        if self.proc:
            self.proc.terminate(); self.proc.wait()
