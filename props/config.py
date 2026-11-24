"""Project-wide settings. Edit these instead of hunting through the code."""
from pathlib import Path
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"        # raw HTML so we never re-download the same page
MODELS_DIR = ROOT / "models"
for _d in (DATA_DIR, CACHE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Basketball-Reference names a season by the year it ENDS: 2026-27 season -> 2027
SEASON = int(os.getenv("NBA_SEASON", "2027"))
# Extra past seasons to pull for model training (more data = better model)
# Seasons of game logs to download and train on (default: last 5, i.e. 2022-23 to now).
HISTORY_SEASONS = int(os.getenv("NBA_HISTORY_SEASONS", "5"))
TRAINING_SEASONS = list(range(SEASON - HISTORY_SEASONS + 1, SEASON + 1))

BREF_BASE = "https://www.basketball-reference.com"
# Basketball-Reference bans IPs that make more than ~20 requests/minute.
# 3.5 seconds between requests keeps us safely under that.
REQUEST_DELAY_SECONDS = 3.5
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)

# Stats we analyze. key = column name in our data, value = label in the UI
STATS = {
    "pts": "Points",
    "trb": "Rebounds",
    "ast": "Assists",
    "pra": "Pts+Reb+Ast",
    "pr": "Pts+Reb",
    "pa": "Pts+Ast",
    "ra": "Reb+Ast",
    "fg2m": "2-Pointers Made",
    "fg2a": "2-Pointers Attempted",
    "fg3": "3-Pointers Made",
    "fg3a": "3-Pointers Attempted",
}

# combo stats = sums of base stats
COMBOS = {"pra": ["pts", "trb", "ast"], "pr": ["pts", "trb"], "pa": ["pts", "ast"], "ra": ["trb", "ast"]}
# 2-pointers = all field goals minus 3-pointers
DIFFS = {"fg2m": ("fgm", "fg3"), "fg2a": ("fga", "fg3a")}


def model_ready(stat: str) -> bool:
    """A stat can be predicted once its model has been trained."""
    return (MODELS_DIR / f"over_{stat}.joblib").exists()


def ready_stats() -> list:
    return [s for s in STATS if model_ready(s)]


# Ignore games where a player barely played (garbage time / injury exits)
MIN_MINUTES = 10

# The Odds API (https://the-odds-api.com) - put your key in a .env file
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_MARKETS = {
    "player_points": "pts",
    "player_rebounds": "trb",
    "player_assists": "ast",
    "player_points_rebounds_assists": "pra",
    "player_points_rebounds": "pr",
    "player_points_assists": "pa",
    "player_rebounds_assists": "ra",
    "player_threes": "fg3",
    "player_threes_attempts": "fg3a",   # oddsscraper names (few books post these)
    "player_twos_made": "fg2m",
    "player_twos_attempts": "fg2a",
}
ODDS_BOOKMAKERS = os.getenv("ODDS_BOOKMAKERS", "draftkings,fanduel")

# File locations
PLAYERS_CSV = DATA_DIR / "players.csv"
GAMELOGS_CSV = DATA_DIR / "gamelogs.csv"
SCHEDULE_CSV = DATA_DIR / "schedule.csv"
MANUAL_LINES_CSV = DATA_DIR / "lines.csv"
LINE_HISTORY_CSV = DATA_DIR / "line_history.csv"
MATCHUPS_CSV = DATA_DIR / "matchups.csv"
LINE_SNAPSHOTS_CSV = DATA_DIR / "line_snapshots.csv"   # every pull, for line-movement alerts
GAME_LINES_CSV = DATA_DIR / "game_lines.csv"           # Vegas totals + spreads
INJURIES_CSV = DATA_DIR / "injuries.csv"               # latest ESPN injury report
MANUAL_INJURIES_CSV = DATA_DIR / "injuries_manual.csv" # your own overrides
PICKS_CSV = DATA_DIR / "picks.csv"                     # pick tracker
SLATES_DIR = DATA_DIR / "slates"                       # daily slate snapshots
SLATES_DIR.mkdir(parents=True, exist_ok=True)

# Alerts: free push notifications via the ntfy app (https://ntfy.sh)
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")
ALERT_MIN_EDGE = float(os.getenv("ALERT_MIN_EDGE", "0.06"))     # 6% edge
ALERT_LINE_MOVE = float(os.getenv("ALERT_LINE_MOVE", "1.0"))    # line moved 1+ points
