"""Lines from your oddsscraper project (~/Desktop/oddsscraper).

oddsscraper scrapes DraftKings, FanDuel, Bovada, PrizePicks and Underdog into a local
SQLite file. This module reads it through `OddsClient` (same format as The Odds API)
and turns it into the lines this app scores:

    player props      client.player_props(sport, market, odds_format="american")
    game lines        client.odds(sport, markets=["spreads", "totals"])
    alternate spreads client.alternate_spreads(sport, team=...)

Setup (once, inside this project's .venv):  pip install -e ~/Desktop/oddsscraper
Then keep the scraper running in another terminal:  cd ~/Desktop/oddsscraper && oddsscraper run
If the package isn't installed, the database is read directly with SQL instead.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, nfl_data as nd
from .teams import to_code

SPORT_KEYS = {"NBA": "basketball_nba", "NFL": "americanfootball_nfl"}
MARKETS = {"NBA": config.ODDS_MARKETS, "NFL": nd.ODDS_MARKETS}
DFS_BOOKS = {"prizepicks", "underdog"}

NFL_TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def team_code(sport: str, name: str) -> str:
    if sport == "NFL":
        return NFL_TEAMS.get(name, name)
    return to_code(name)


def db_path() -> Path:
    p = os.environ.get("ODDSSCRAPER_DB") or "~/Desktop/oddsscraper/oddsscraper.db"
    return Path(p).expanduser()


def available() -> bool:
    return db_path().exists()


def installed() -> bool:
    """Is the oddsscraper package installed in this Python environment?"""
    try:
        import oddsscraper  # noqa: F401
        return True
    except ImportError:
        return False


def project_dir() -> Path:
    try:
        import oddsscraper
        return Path(oddsscraper.__file__).resolve().parent.parent
    except ImportError:
        return Path("~/Desktop/oddsscraper").expanduser()


def _bundled_settings() -> dict:
    """oddsscraper settings tuned for this app (scripts/oddsscraper.env)."""
    out = {}
    f = Path(__file__).resolve().parent.parent / "scripts" / "oddsscraper.env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def run_scrape(timeout: int = 1200) -> tuple[bool, str]:
    """Run one oddsscraper pass (books + DFS + scores) and wait for it.
    Returns (ok, short message). Uses oddsscraper's own .env (sports, books, delays)."""
    import subprocess
    import sys
    if not installed():
        return False, ("oddsscraper isn't installed in this environment. Run "
                       "`bash scripts/setup_oddsscraper.sh` in the VS Code terminal.")
    import os
    env = dict(os.environ)
    if not (project_dir() / ".env").exists():
        # No oddsscraper .env yet: use the settings this project ships with.
        env.update(_bundled_settings())
    try:
        r = subprocess.run([sys.executable, "-m", "oddsscraper", "once"], cwd=str(project_dir()),
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return False, f"Scrape took longer than {timeout // 60} minutes and was stopped."
    if r.returncode != 0:
        return False, (r.stderr or r.stdout).strip().splitlines()[-1][:300] if (r.stderr or r.stdout) else "failed"
    summary = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
    return True, summary


def scrape_health() -> list[dict]:
    """Latest result per scraper/sport (ok or error) from oddsscraper's own status view."""
    if not available():
        return []
    try:
        con = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(
            "SELECT source, kind, sport_key, ok, n_events, error, ts FROM v_scraper_status "
            "WHERE sport_key IN ('basketball_nba','americanfootball_nfl','baseball_mlb') OR sport_key IS NULL")]
        con.close()
        return rows
    except Exception:
        return []


def status() -> str:
    """One line for the dashboard: where lines come from and how fresh they are."""
    if not available():
        return f"oddsscraper database not found at {db_path()}"
    try:
        con = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
        ts = con.execute("SELECT MAX(ts) FROM snapshots").fetchone()[0]
        con.close()
        return f"oddsscraper: last scrape {pd.to_datetime(ts, utc=True).tz_convert('America/Los_Angeles'):%a %I:%M %p}"
    except Exception as e:
        return f"oddsscraper database unreadable: {e}"


def _client():
    try:
        from oddsscraper import OddsClient
    except ImportError:
        return None
    return OddsClient(str(db_path()))


def _rows_from_sql(sport_key: str, group: str) -> list[dict]:
    """Fallback when the oddsscraper package isn't installed in this venv."""
    con = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT event_id, commence_time, home_team, away_team, bookmaker, market, outcome AS side, "
        "player, point, price_american AS price FROM v_odds WHERE sport_key = ? AND market_group = ?",
        (sport_key, group))]
    con.close()
    return rows


# ------------------------------------------------------------ player props
def prop_lines(sport: str, main_only: bool = True, all_books: bool = False) -> pd.DataFrame:
    """Every upcoming prop as one row per player/stat/line/book with over & under odds."""
    cols = ["date", "player", "stat", "line", "over_odds", "under_odds", "book",
            "home_team", "away_team", "event_id", "commence_time"]
    if not available():
        return pd.DataFrame(columns=cols)
    key, mapping = SPORT_KEYS[sport], MARKETS[sport]
    client = _client()
    if client is not None:
        raw = client.player_props(key, list(mapping), odds_format="american")
    else:
        raw = [r for r in _rows_from_sql(key, "player_props") if r["market"] in mapping]
    if not raw:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(raw)
    df["stat"] = df["market"].map(mapping)
    df = df.dropna(subset=["stat", "player"])
    sides = df["side"].astype(str).str.lower()
    df = df[~sides.isin(["yes", "no"]) | ((sides == "yes") & (df["stat"] == "anytime_td"))].copy()
    yes = df["side"].astype(str).str.lower() == "yes"   # anytime TD only; milestone "25+" bets aren't lines
    df.loc[yes, "point"] = 0.5                       # anytime TD: Yes = over 0.5
    df["side"] = np.where(yes, "over", df["side"].astype(str).str.lower())
    df = df.dropna(subset=["point"])
    wide = (df.pivot_table(index=["event_id", "commence_time", "home_team", "away_team", "bookmaker",
                                  "player", "stat", "point"], columns="side", values="price", aggfunc="first")
            .reset_index())
    for c in ("over", "under"):
        if c not in wide:
            wide[c] = np.nan
    t = pd.to_datetime(wide["commence_time"], utc=True).dt.tz_convert("America/Los_Angeles")
    out = pd.DataFrame({
        "date": t.dt.normalize().dt.tz_localize(None), "player": wide["player"], "stat": wide["stat"],
        "line": wide["point"].astype(float), "over_odds": wide["over"], "under_odds": wide["under"],
        "book": wide["bookmaker"],
        "home_team": wide["home_team"].map(lambda n: team_code(sport, n)),
        "away_team": wide["away_team"].map(lambda n: team_code(sport, n)),
        "event_id": wide["event_id"], "commence_time": t.dt.tz_localize(None),
    })
    # PrizePicks/Underdog have no per-pick odds: oddsscraper stores a nominal pick'em price
    # (DFS_NOMINAL_PRICE, default -122), which is used as-is.
    out = out[out["date"] >= pd.Timestamp.today().normalize()]
    out = out[~out["home_team"].astype(str).str.contains(r"\(") & ~out["away_team"].astype(str).str.contains(r"\(")]
    allowed = allowed_books()
    if allowed and not all_books:
        out = out[out["book"].isin(allowed)]
    if main_only:
        out = main_lines(out, _recent_averages(sport))
    return out.reset_index(drop=True)


# ------------------------------------------------------------ main line per book
def _implied(odds) -> np.ndarray:
    o = pd.to_numeric(odds, errors="coerce").astype(float)
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


def main_lines(df: pd.DataFrame, ref: dict | None = None) -> pd.DataFrame:
    """One line per game/player/stat/book: the book's real (main) line.

    Books like FanDuel list alternate, milestone (25+) and sometimes period lines next to
    the main one. The main line has prices on BOTH sides; among those, keep the one closest
    to the player's recent average (or the most evenly priced one when that's unknown)."""
    if df.empty:
        return df
    d = df.copy()
    d["_two"] = d["over_odds"].notna() & d["under_odds"].notna()
    d["_bal"] = np.abs(_implied(d["over_odds"].fillna(-110)) - _implied(d["under_odds"].fillna(-110)))
    if ref:
        avg = [ref.get((_norm(p), s)) for p, s in zip(d["player"], d["stat"])]
        d["_dist"] = [abs(l - a) if a is not None else np.nan for l, a in zip(d["line"], avg)]
    else:
        d["_dist"] = np.nan
    d["_dist"] = d["_dist"].fillna(d["_bal"] * 100)
    d = d.sort_values(["_two", "_dist", "_bal"], ascending=[False, True, True])
    d = d.drop_duplicates(["event_id", "player", "stat", "book"])
    return d.drop(columns=["_two", "_bal", "_dist"]).sort_index()


def _norm(name: str) -> str:
    from .nba_com import norm_name
    return norm_name(str(name))


def _recent_averages(sport: str) -> dict:
    """{(normalized player name, stat): average over his last 10 games} for NBA."""
    if sport != "NBA" or not config.GAMELOGS_CSV.exists() or not config.PLAYERS_CSV.exists():
        return {}
    try:
        logs = pd.read_csv(config.GAMELOGS_CSV, parse_dates=["date"])
        names = pd.read_csv(config.PLAYERS_CSV).drop_duplicates("player_id").set_index("player_id")["player"]
        stats = [c for c in set(config.ODDS_MARKETS.values()) if c in logs]
        last = logs.sort_values("date").groupby("player_id").tail(10)
        avg = last.groupby("player_id")[stats].mean()
        return {(_norm(names.get(pid, "")), s): float(v)
                for pid, row in avg.iterrows() for s, v in row.items() if pd.notna(v)}
    except Exception:
        return {}


# ------------------------------------------------------------ which books to use
def _settings_file() -> Path:
    return config.DATA_DIR / "line_settings.json"


def allowed_books() -> list[str]:
    """Sportsbooks you chose to use (empty = all of them)."""
    import json
    try:
        return list(json.loads(_settings_file().read_text()).get("books", []))
    except Exception:
        return []


def set_allowed_books(books: list[str]) -> None:
    import json
    _settings_file().parent.mkdir(parents=True, exist_ok=True)
    _settings_file().write_text(json.dumps({"books": sorted(books)}))


def books_available() -> list[str]:
    """Every book that has posted props in oddsscraper."""
    if not available():
        return []
    try:
        con = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
        rows = con.execute("SELECT DISTINCT bookmaker FROM v_odds WHERE market_group = 'player_props'").fetchall()
        return sorted(r[0] for r in rows)
    except Exception:
        return []


def player_lines(sport: str, player: str, stat: str) -> pd.DataFrame:
    """Every book's line for one player's stat (for side-by-side comparison)."""
    df = prop_lines(sport)
    if df.empty:
        return df
    from .nba_com import norm_name
    k = norm_name(player)
    last = k.split()[-1] if k else ""
    m = df["player"].map(norm_name)
    hit = df[(m == k) | (m.str.endswith(" " + last) & (m.str[:1] == k[:1]))]
    return hit[hit["stat"] == stat].sort_values(["line", "book"])


# -------------------------------------------------------------- game lines
def game_lines(sport: str) -> pd.DataFrame:
    """Median spread and total per upcoming game across books (home_spread < 0 = home favored)."""
    cols = ["date", "home_team", "away_team", "total", "home_spread", "books"]
    client = _client() if available() else None
    if client is None:
        return pd.DataFrame(columns=cols)
    rows = []
    for ev in client.odds(SPORT_KEYS[sport], markets=["spreads", "totals"], odds_format="american"):
        tot, spr, books = [], [], set()
        for b in ev.get("bookmakers", []):
            books.add(b.get("key"))
            for mk in b.get("markets", []):
                for o in mk.get("outcomes", []):
                    if mk["key"] == "totals" and o.get("name") == "Over" and o.get("point") is not None:
                        tot.append(o["point"])
                    if mk["key"] == "spreads" and o.get("name") == ev["home_team"] and o.get("point") is not None:
                        spr.append(o["point"])
        t = pd.to_datetime(ev["commence_time"], utc=True).tz_convert("America/Los_Angeles")
        rows.append({"date": t.normalize().tz_localize(None),
                     "home_team": team_code(sport, ev["home_team"]), "away_team": team_code(sport, ev["away_team"]),
                     "total": float(np.median(tot)) if tot else np.nan,
                     "home_spread": float(np.median(spr)) if spr else np.nan, "books": len(books)})
    return pd.DataFrame(rows, columns=cols)


def alternate_spreads(sport: str, team_name: str | None = None) -> pd.DataFrame:
    client = _client() if available() else None
    if client is None:
        return pd.DataFrame()
    df = pd.DataFrame(client.alternate_spreads(SPORT_KEYS[sport], team=team_name, odds_format="american"))
    if df.empty:
        return df
    df["team"] = df["team"].map(lambda n: team_code(sport, n))
    return df[["team", "point", "price", "bookmaker", "home_team", "away_team", "commence_time"]]
