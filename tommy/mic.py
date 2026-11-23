"""The microphone, read on a background thread."""
import queue, subprocess, threading, time
import numpy as np

RATE, FRAME = 16000, 1280            # 80 ms frames (what the wake word model expects)
FRAME_S = FRAME / RATE
GAIN = 4.0                           # the internal mic is quiet; wake word and voice detector like louder input


class Mic:
    """Frames go into a queue that keeps the last 30 s. Reading only when Tommy was free let the pipe fill (about 2 s) and the
    recorder stall. The recorder is restarted if it stops. parec, not pw-record: pw-record writes a big-endian .snd stream to a pipe."""
    CMD = ["parec", "--rate", str(RATE), "--channels", "1", "--format", "s16le", "--latency-msec", "40"]

    def __init__(self):
        self.q = queue.Queue(maxsize=int(30 / FRAME_S))
        self._stopped = False
        self._open()
        threading.Thread(target=self._pump, daemon=True).start()

    def _open(self):
        self.proc = subprocess.Popen(self.CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _pump(self):
        while not self._stopped:
            raw = self.proc.stdout.read(FRAME * 2)
            if len(raw) < FRAME * 2:
                if self._stopped:
                    return
                print("  (microphone stream stopped; restarting it)", flush=True)
                time.sleep(0.5)
                self._open()
                continue
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) * GAIN
            if self.q.full():
                self.q.get_nowait()
            self.q.put(np.clip(x, -32768, 32767).astype(np.int16))

    def read(self):
        return self.q.get()                             # waits about 80 ms for the next frame

    def drain(self):
        """Throw away audio recorded while Tommy was busy, so what comes next is said from now on."""
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def close(self):
        self._stopped = True
        self.proc.terminate()
