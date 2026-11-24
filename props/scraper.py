"""Scrapes Basketball-Reference.

Pages used:
  * /leagues/NBA_{season}_per_game.html     -> every player, their team & position
  * /players/{x}/{id}/gamelog/{season}       -> one row per game for a player
  * /leagues/NBA_{season}_games-{month}.html -> schedule (who plays who, home/away)

Every page is cached in data/cache so re-running only downloads what's new.
Please keep the delay in config.py - BRef blocks fast scrapers for an hour.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from . import config
from .teams import to_code

_session = requests.Session()
_session.headers.update({"User-Agent": config.USER_AGENT})
_last_request = 0.0


# ----------------------------------------------------------------- fetching
def _cache_path(url: str) -> Path:
    name = re.sub(r"[^A-Za-z0-9]+", "_", url.replace(config.BREF_BASE, "")).strip("_")
    return config.CACHE_DIR / f"{name}.html"


def fetch(url: str, max_age_hours: float | None = 12) -> str:
    """Download a page (politely) or return the cached copy.

    max_age_hours=None means "cached copy never expires" (good for old seasons).
    """
    global _last_request
    path = _cache_path(url)
    if path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if max_age_hours is None or age_h < max_age_hours:
            return path.read_text(encoding="utf-8")

    wait = config.REQUEST_DELAY_SECONDS - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    resp = _session.get(url, timeout=30)
    _last_request = time.time()
    if resp.status_code == 429:
        raise RuntimeError(
            "Basketball-Reference says 'Too Many Requests'. You're rate-limited "
            "for about an hour - wait, then re-run (cached pages are kept)."
        )
    if resp.status_code == 404:
        return ""
    resp.raise_for_status()
    path.write_text(resp.text, encoding="utf-8")
    return resp.text


# ------------------------------------------------------------------ parsing
def _find_table(html: str, table_ids: list[str]):
    """BRef hides some tables inside HTML comments, so look there too."""
    soup = BeautifulSoup(html, "lxml")
    for tid in table_ids:
        t = soup.find("table", id=tid)
        if t is not None:
            return t
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        if "<table" in c:
            inner = BeautifulSoup(c, "lxml")
            for tid in table_ids:
                t = inner.find("table", id=tid)
                if t is not None:
                    return t
    return None


def _rows(table) -> list[dict]:
    """Turn a BRef table into a list of {data-stat: text} dicts.
    Using the data-stat attribute instead of the header text keeps this working
    even when BRef renames visible column headers."""
    out = []
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        if "thead" in (tr.get("class") or []):  # repeated header rows
            continue
        row = {}
        for cell in tr.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if not stat:
                continue
            row[stat] = cell.get_text(strip=True)
            if cell.get("data-append-csv"):
                row[stat + "__id"] = cell["data-append-csv"]
            a = cell.find("a")
            if a and a.get("href"):
                row[stat + "__href"] = a["href"]
        if row:
            out.append(row)
    return out


def _pick(row: dict, *keys, default=""):
    """BRef has used different data-stat names over the years; try each."""
    for k in keys:
        if k in row and row[k] != "":
            return row[k]
    return default


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _minutes(mp: str) -> float:
    if not mp or ":" not in mp:
        return _num(mp)
    m, s = mp.split(":")[:2]
    return int(m) + int(s) / 60


# -------------------------------------------------------------------- pages
def get_players(season: int = config.SEASON) -> pd.DataFrame:
    """All players this season with id, name, team, position, minutes/game."""
    url = f"{config.BREF_BASE}/leagues/NBA_{season}_per_game.html"
    html = fetch(url, max_age_hours=24 if season == config.SEASON else None)
    table = _find_table(html, ["per_game_stats"])
    if table is None:
        raise RuntimeError(f"Couldn't find the per-game table at {url}")

    players = {}
    for r in _rows(table):
        pid = _pick(r, "name_display__id", "player__id")
        if not pid:
            continue
        team = _pick(r, "team_name_abbr", "team_id")
        rec = {
            "player_id": pid,
            "player": _pick(r, "name_display", "player").rstrip("*"),
            "team": team,
            "pos": _pick(r, "pos"),
            "games": _num(_pick(r, "games", "g")),
            "mpg": _num(_pick(r, "mp_per_g")),
            "season": season,
        }
        # Traded players have a "2TM"/"TOT" summary row + one row per team.
        # Keep the summary row's stats but their latest team.
        if pid in players:
            if team not in ("TOT", "2TM", "3TM", "4TM"):
                players[pid]["team"] = team
            continue
        players[pid] = rec
    df = pd.DataFrame(players.values())
    df["primary_pos"] = df["pos"].str.split("-").str[0]
    return df


def get_gamelog(player_id: str, season: int = config.SEASON) -> pd.DataFrame:
    """One row per game played for a player in a season."""
    url = f"{config.BREF_BASE}/players/{player_id[0]}/{player_id}/gamelog/{season}"
    html = fetch(url, max_age_hours=12 if season == config.SEASON else None)
    if not html:
        return pd.DataFrame()
    table = _find_table(html, ["player_game_log_reg", "pgl_basic"])
    if table is None:
        return pd.DataFrame()

    games = []
    for r in _rows(table):
        date = _pick(r, "date", "date_game")
        opp = _pick(r, "opp_name_abbr", "opp_id")
        mp = _pick(r, "mp")
        # rows for DNP / Inactive have no minutes
        if not date or not opp or not mp or not mp[0].isdigit():
            continue
        games.append({
            "player_id": player_id,
            "season": season,
            "date": date,
            "team": _pick(r, "team_name_abbr", "team_id"),
            "home": 0 if _pick(r, "game_location") == "@" else 1,
            "opp": to_code(opp),
            "result": _pick(r, "game_result"),
            "started": 1 if _pick(r, "is_starter", "gs") in ("*", "1") else 0,
            "min": _minutes(mp),
            "pts": _num(_pick(r, "pts")),
            "trb": _num(_pick(r, "trb")),
            "ast": _num(_pick(r, "ast")),
            "fgm": _num(_pick(r, "fg")),
            "fga": _num(_pick(r, "fga")),
            "fg3": _num(_pick(r, "fg3")),
            "fg3a": _num(_pick(r, "fg3a")),
            "stl": _num(_pick(r, "stl")),
            "blk": _num(_pick(r, "blk")),
            "tov": _num(_pick(r, "tov")),
            "fta": _num(_pick(r, "fta")),
            "oreb": _num(_pick(r, "orb")),
        })
    df = pd.DataFrame(games)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        from .nba_com import add_combos
        add_combos(df)
    return df


MONTHS = ["october", "november", "december", "january", "february",
          "march", "april"]


def get_schedule(season: int = config.SEASON) -> pd.DataFrame:
    """Full season schedule: date, away team, home team (+scores once played)."""
    games = []
    for month in MONTHS:
        url = f"{config.BREF_BASE}/leagues/NBA_{season}_games-{month}.html"
        html = fetch(url, max_age_hours=24)
        if not html:
            continue
        table = _find_table(html, ["schedule"])
        if table is None:
            continue
        for r in _rows(table):
            date = _pick(r, "date_game")
            if not date:
                continue
            games.append({
                "date": pd.to_datetime(date, errors="coerce"),
                "away": to_code(_pick(r, "visitor_team_name")),
                "home": to_code(_pick(r, "home_team_name")),
                "away_pts": _num(_pick(r, "visitor_pts")),
                "home_pts": _num(_pick(r, "home_pts")),
            })
    return pd.DataFrame(games)


def next_game(team: str, schedule: pd.DataFrame, today=None):
    """Return (opponent, is_home, date) for a team's next unplayed game."""
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    s = schedule[(schedule["date"] >= today)
                 & ((schedule["home"] == team) | (schedule["away"] == team))]
    if s.empty:
        return None
    g = s.sort_values("date").iloc[0]
    is_home = g["home"] == team
    return (g["away"] if is_home else g["home"]), bool(is_home), g["date"]
