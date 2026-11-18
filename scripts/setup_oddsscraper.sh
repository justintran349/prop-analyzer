#!/bin/bash
# One-time setup: install your oddsscraper project into THIS project's virtualenv,
# then run a first scrape so the app has real lines.
#   source .venv/bin/activate && bash scripts/setup_oddsscraper.sh
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRAPER="${ODDSSCRAPER_DIR:-$HOME/Desktop/oddsscraper}"
[ -d "$SCRAPER" ] || { echo "oddsscraper not found at $SCRAPER (set ODDSSCRAPER_DIR=/path)"; exit 1; }
PY="${VIRTUAL_ENV:-$DIR/.venv}/bin/python"
[ -x "$PY" ] || { echo "Activate this project's .venv first: source .venv/bin/activate"; exit 1; }
# macOS Python 3.9 ships pip 21.2, which can't do editable installs from pyproject.toml
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install -e "$SCRAPER"
if [ ! -f "$SCRAPER/.env" ]; then
  cp "$DIR/scripts/oddsscraper.env" "$SCRAPER/.env"
  echo "Created $SCRAPER/.env (NBA, NFL, MLB; DraftKings, FanDuel, Bovada, PrizePicks, Underdog)"
fi
echo
"$PY" "$DIR/scripts/scrape_lines.py"
echo
echo "Done. Keep lines fresh with either:"
echo "  - the 'Pull fresh lines' button on Top Picks, or"
echo "  - a second terminal:  oddsscraper run   (scrapes every hour), or"
echo "  - the daily job (scripts/daily.py scrapes before scoring)."
