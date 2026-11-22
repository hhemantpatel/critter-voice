#!/usr/bin/env python3
"""Summarise a measure-listen.sh run: watts per phase, cost over baseline, and the noise floor.

Usage: report.py results/<run>
Power is the median of battery power_now samples after a 60 s settle at the start of each phase.
Runtime lost is computed against this battery's full capacity (energy_full).
"""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path

SETTLE_S = 60
run = Path(sys.argv[1])
rows = [l.rstrip("\n").split("\t") for l in (run / "samples.tsv").read_text().splitlines()[1:]]
by_phase, order, start = defaultdict(list), [], {}
for epoch, phase, status, _cap, _e, p in rows:
    epoch = int(epoch)
    if phase not in start:
        start[phase] = epoch; order.append(phase)
    if status == "Discharging" and epoch - start[phase] >= SETTLE_S:
        by_phase[phase].append(int(p) / 1e6)

watts = {p: statistics.median(v) for p, v in by_phase.items() if v}
if "baseline" not in watts or "baseline2" not in watts:
    sys.exit("need both baselines with discharging samples (was the charger plugged in?)")
base = (watts["baseline"] + watts["baseline2"]) / 2
noise = abs(watts["baseline"] - watts["baseline2"])

print(f"\n{'phase':<10} {'watts':>7} {'vs base':>8} {'py cpu%':>8} {'rec cpu%':>9} {'wakes/s':>8}")
for p in order:
    if p not in watts:
        continue
    j = {}
    f = run / f"listen-{p}.json"
    if f.exists() and f.read_text().strip():
        j = json.loads(f.read_text().strip().splitlines()[-1])
    print(f"{p:<10} {watts[p]:7.2f} {watts[p] - base:+8.2f} {j.get('py_cpu_pct', ''):>8} "
          f"{j.get('recorder_cpu_pct', ''):>9} {j.get('ctx_switches_per_s', ''):>8}")
print(f"\nnoise floor (baseline vs baseline2): {noise:.2f} W  -> differences below this are not real")
batt_wh = int(Path("/sys/class/power_supply/BAT1/energy_full").read_text()) / 1e6
for p in ("capture", "vad", "gated", "ungated"):
    if p in watts:
        d = watts[p] - base
        print(f"{p:<9} +{max(d, 0):.2f} W  ~ {100 * max(d, 0) / base:.1f}% more drain "
              f"-> about {batt_wh / base - batt_wh / (base + max(d, 0)):.2f} h of runtime lost per full charge")
