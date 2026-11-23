#!/bin/sh
# Jarvis listens while the charger is in. Plugging in starts it; unplugging stops it (an open mic costs about 1 W on battery).
# "jarvis off" while plugged in is respected until the next plug-in. /run/power-source ("ac" or "bat") is written by the
# system's udev battery rule on every change.
state=$(cat /run/power-source 2>/dev/null)
dir=${XDG_RUNTIME_DIR:-/tmp}
last=$(cat "$dir/jarvis-last-power" 2>/dev/null)
echo "$state" > "$dir/jarvis-last-power"
if [ "$state" = ac ]; then
    [ "$last" != ac ] && rm -f "$dir/jarvis-off"          # a fresh plug-in re-arms it
    [ -e "$dir/jarvis-off" ] || systemctl --user start jarvis.service
else
    systemctl --user stop jarvis.service
fi
