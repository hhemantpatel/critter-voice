#!/bin/bash
# Battery cost of always-on listening, layer by layer.
# Run ON BATTERY, screen on, nothing else open, and do not touch the laptop.
# Usage: ./measure-listen.sh [minutes_per_phase]      (default 5, so 6 phases = 30 min)
#
# Phases: baseline -> capture -> vad -> gated -> ungated -> baseline
# The two baselines bracket the run: their difference is the noise floor of the whole test,
# so any stage that differs from baseline by less than that is "no measurable cost".
set -u
MIN="${1:-5}"
D="$(cd "$(dirname "$0")" && pwd)"; B=/sys/class/power_supply/BAT1
OUT="$D/results/$(date +%Y%m%d-%H%M)"; mkdir -p "$OUT"
PHASES=(baseline capture vad gated ungated baseline2)

[ "$(cat /sys/class/power_supply/ACAD/online)" = 0 ] || { echo "ERROR: unplug the charger first."; exit 1; }
CAP=$(cat $B/capacity)
[ "$CAP" -lt 20 ] && { echo "ERROR: battery at ${CAP}% - too low for a clean run."; exit 1; }

# pin the conditions that silently ruin comparability (same ones measure-rice.sh pins)
BL_DEV=nvidia_wmi_ec_backlight; OLD_BL=$(cat /sys/class/backlight/$BL_DEV/brightness 2>/dev/null)
command -v brightnessctl >/dev/null && brightnessctl -d $BL_DEV set 400 >/dev/null 2>&1
RESTART_IDLE=0; pgrep -x hypridle >/dev/null && { pkill -x hypridle; RESTART_IDLE=1; }
hyprctl dispatch 'hl.dsp.dpms({action = "on"})' >/dev/null 2>&1
restore() {
  [ -n "${OLD_BL:-}" ] && brightnessctl -d $BL_DEV set "$OLD_BL" >/dev/null 2>&1
  [ "$RESTART_IDLE" = 1 ] && setsid -f hypridle >/dev/null 2>&1
}
trap restore EXIT

{ echo "start: $(date)  battery: ${CAP}%  minutes/phase: $MIN"
  echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  echo "epp: $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"
  echo "platform_profile: $(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"
  echo "default mic: $(wpctl status 2>/dev/null | grep -m1 -E '^ │  \*' )"
} > "$OUT/env.txt"

echo -e "epoch\tphase\tstatus\tcap\tenergy_uWh\tpower_uW" > "$OUT/samples.tsv"
for P in "${PHASES[@]}"; do
  echo "[$(date +%H:%M:%S)] phase '$P' for ${MIN} min  (battery $(cat $B/capacity)%)"
  case "$P" in
    baseline*) sleep $((MIN*60)) & PID=$! ;;
    *) "$D/.venv/bin/python" "$D/listen.py" "$P" --seconds $((MIN*60)) > "$OUT/listen-$P.json" 2>"$OUT/listen-$P.err" & PID=$! ;;
  esac
  while kill -0 $PID 2>/dev/null; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$P" "$(cat $B/status)" "$(cat $B/capacity)" \
      "$(cat $B/energy_now)" "$(cat $B/power_now)" >> "$OUT/samples.tsv"
    sleep 10
  done
done
echo "done -> $OUT"
python3 "$D/report.py" "$OUT"
