#!/bin/bash
# Runs scripts/daily.py every morning at 9:07 on your Mac (launchd).
#   bash scripts/install_daily_job.sh            install
#   bash scripts/install_daily_job.sh remove     uninstall
#
# macOS note: background jobs can't read ~/Desktop until you allow it.
# If data/daily.log shows "Operation not permitted", either move this project
# out of Desktop (e.g. ~/Projects) and re-run this script, or give /bin/bash
# Full Disk Access in System Settings > Privacy & Security.
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$DIR/.venv/bin/python"
LABEL="com.$(whoami).nbaprops.daily"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ "$1" = "remove" ]; then
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed daily job."
  exit 0
fi

[ -x "$PY" ] || { echo "Create the virtual environment first (see README)."; exit 1; }
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$PY</string><string>$DIR/scripts/daily.py</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>StartCalendarInterval</key><dict>
    <key>Hour</key><integer>9</integer><key>Minute</key><integer>7</integer>
  </dict>
  <key>StandardOutPath</key><string>$DIR/data/daily.log</string>
  <key>StandardErrorPath</key><string>$DIR/data/daily.log</string>
</dict></plist>
EOF
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "Daily job installed: runs at 9:07 every morning (while your Mac is awake)."
echo "Log: $DIR/data/daily.log   Run now: launchctl start $LABEL"
