"""NFL data from nflverse (free, public, updated during the season).

    schedules   games.csv: every game with Vegas spread/total, roof, surface,
                temperature, wind, rest days, division game, stadium
    stats       weekly player box scores (passing / rushing / receiving)
    snaps       snap counts (who was on the field, even with 0 stats)
    injuries    weekly injury reports (Out / Doubtful / Questionable)
    players     id map between nflverse (gsis) and Pro-Football-Reference ids

Everything is saved under data/nfl_props/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

NFL_DIR = config.DATA_DIR / "nfl_props"
RAW = NFL_DIR / "raw"
RAW.mkdir(parents=True, exist_ok=True)
PLAYER_GAMES_CSV = NFL_DIR / "player_games.csv"
GAMES_CSV = NFL_DIR / "games.csv"
INJURIES_CSV = NFL_DIR / "injuries.csv"
WEATHER_CSV = NFL_DIR / "weather.csv"
LINES_CSV = NFL_DIR / "lines.csv"              # your manual NFL lines
LINE_HISTORY_CSV = NFL_DIR / "line_history.csv"
LINE_SNAPSHOTS_CSV = NFL_DIR / "line_snapshots.csv"
MODELS_DIR = config.MODELS_DIR / "nfl"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
URLS = {
    "games": f"{BASE}/schedules/games.csv",
    "stats": BASE + "/stats_player/stats_player_week_{season}.csv",
    "snaps": BASE + "/snap_counts/snap_counts_{season}.csv",
    "injuries": BASE + "/injuries/injuries_{season}.csv",
    "players": f"{BASE}/players/players.csv",
}

# The NFL season is named by the year it STARTS (2026 season = fall 2026)
NFL_SEASON = int(pd.Timestamp.today().year if pd.Timestamp.today().month >= 3
                 else pd.Timestamp.today().year - 1)
TRAINING_SEASONS = [NFL_SEASON - 3, NFL_SEASON - 2, NFL_SEASON - 1, NFL_SEASON]

POSITIONS = ["QB", "RB", "WR", "TE"]

# stat key -> (label, positions it applies to)
STATS = {
    "passing_yards": ("Passing Yards", ["QB"]),
    "passing_tds": ("Passing TDs", ["QB"]),
    "completions": ("Completions", ["QB"]),
    "attempts": ("Pass Attempts", ["QB"]),
    "passing_interceptions": ("Interceptions", ["QB"]),
    "rushing_yards": ("Rushing Yards", ["QB", "RB"]),
    "carries": ("Rush Attempts", ["QB", "RB"]),
    "receiving_yards": ("Receiving Yards", ["RB", "WR", "TE"]),
    "receptions": ("Receptions", ["RB", "WR", "TE"]),
    "rush_rec_yds": ("Rush+Rec Yards", ["RB", "WR", "TE"]),
    "anytime_td": ("Anytime TD (0.5)", ["RB", "WR", "TE", "QB"]),
}
LABEL = {k: v[0] for k, v in STATS.items()}

# The Odds API market -> our stat
ODDS_MARKETS = {
    "player_pass_yds": "passing_yards", "player_pass_tds": "passing_tds",
    "player_pass_completions": "completions", "player_pass_attempts": "attempts",
    "player_pass_interceptions": "passing_interceptions",
    "player_rush_yds": "rushing_yards", "player_rush_attempts": "carries",
    "player_reception_yds": "receiving_yards", "player_receptions": "receptions",
    "player_rush_reception_yds": "rush_rec_yds", "player_anytime_td": "anytime_td",
}

TEAM_FIX = {"LA": "LA", "LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV", "WSH": "WAS", "JAC": "JAX"}


def _download(url: str, dest: Path, max_age_hours: float | None = 6) -> Path:
    if dest.exists() and max_age_hours is not None:
        age = (pd.Timestamp.now().timestamp() - dest.stat().st_mtime) / 3600
        if age < max_age_hours:
            return dest
    if dest.exists() and max_age_hours is None:
        return dest
    with requests.get(url, timeout=120, stream=True) as r:
        if r.status_code == 404:
            return dest
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        tmp.replace(dest)
    return dest


def download(seasons=None, log=print):
    """Fetch raw nflverse files (old seasons are only downloaded once)."""
    seasons = seasons or TRAINING_SEASONS
    log("schedules ...")
    _download(URLS["games"], RAW / "games.csv")
    _download(URLS["players"], RAW / "players.csv", max_age_hours=24 * 7)
    for s in seasons:
        age = 6 if s == NFL_SEASON else None
        for kind in ("stats", "snaps", "injuries"):
            log(f"{kind} {s} ...")
            _download(URLS[kind].format(season=s), RAW / f"{kind}_{s}.csv", age)


# ------------------------------------------------------------------ build
def _read(kind, season):
    p = RAW / f"{kind}_{season}.csv"
    return pd.read_csv(p, low_memory=False) if p.exists() and p.stat().st_size > 0 else pd.DataFrame()


def build_games() -> pd.DataFrame:
    g = pd.read_csv(RAW / "games.csv", low_memory=False)
    g = g[(g["season"] >= min(TRAINING_SEASONS)) & (g["game_type"] == "REG")].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_FIX)
    g["date"] = pd.to_datetime(g["gameday"])
    g["kick_hour"] = pd.to_numeric(g["gametime"].astype(str).str.split(":").str[0], errors="coerce")
    # roof missing for some upcoming games -> use the stadium's usual roof
    usual = g.dropna(subset=["roof"]).groupby("stadium_id")["roof"].agg(lambda s: s.mode().iat[0])
    g["roof"] = g["roof"].fillna(g["stadium_id"].map(usual)).fillna("outdoors")
    usual_s = g.dropna(subset=["surface"]).groupby("stadium_id")["surface"].agg(lambda s: s.mode().iat[0])
    g["surface"] = g["surface"].fillna(g["stadium_id"].map(usual_s)).fillna("grass")
    keep = ["game_id", "season", "week", "date", "kick_hour", "weekday", "away_team", "home_team",
            "away_score", "home_score", "spread_line", "total_line", "away_rest", "home_rest",
            "div_game", "roof", "surface", "temp", "wind", "stadium_id", "stadium",
            "away_qb_name", "home_qb_name"]
    g = g[keep]
    # weather from Open-Meteo (archive for past games, forecast for upcoming)
    if WEATHER_CSV.exists():
        w = pd.read_csv(WEATHER_CSV).drop_duplicates("game_id", keep="last")
        g = g.merge(w[["game_id", "wx_temp", "wx_wind", "wx_gust", "wx_precip", "wx_snow"]],
                    on="game_id", how="left")
    else:
        for c in ("wx_temp", "wx_wind", "wx_gust", "wx_precip", "wx_snow"):
            g[c] = np.nan
    indoor = g["roof"].isin(["dome", "closed"])
    # official game-day readings win; otherwise the Open-Meteo numbers
    g["game_temp"] = g["temp"].fillna(g["wx_temp"])
    g["game_wind"] = g["wind"].fillna(g["wx_wind"])
    g.loc[indoor, ["game_temp", "game_wind", "wx_gust", "wx_precip", "wx_snow"]] = [70, 0, 0, 0, 0]
    g["indoor"] = indoor.astype(int)
    g["turf"] = (~g["surface"].str.contains("grass", na=False)).astype(int)
    return g


def _pfr_to_gsis() -> dict:
    p = pd.read_csv(RAW / "players.csv", low_memory=False)
    a = p.get("pfr_id", p.get("pfr_player_id"))
    b = p.get("gsis_id", p.get("player_id"))
    m = pd.DataFrame({"pfr": a, "gsis": b}).dropna()
    return dict(zip(m["pfr"], m["gsis"]))


def build_player_games(seasons=None) -> pd.DataFrame:
    seasons = seasons or TRAINING_SEASONS
    pfr = _pfr_to_gsis()
    frames = []
    for s in seasons:
        st = _read("stats", s)
        if st.empty:
            continue
        st = st[(st["season_type"] == "REG") & st["position"].isin(POSITIONS)].copy()
        sn = _read("snaps", s)
        if not sn.empty:
            sn = sn[(sn["game_type"] == "REG")].copy()
            sn["player_id"] = sn["pfr_player_id"].map(pfr)
            sn = sn.dropna(subset=["player_id"])
            sn["team"] = sn["team"].replace(TEAM_FIX)
            st = st.merge(sn[["player_id", "week", "offense_snaps", "offense_pct"]],
                          on=["player_id", "week"], how="left")
            # players on the field who recorded no stats -> real zero games
            extra = sn[sn["position"].isin(POSITIONS) & (sn["offense_snaps"] > 0)]
            extra = extra.merge(st[["player_id", "week"]], on=["player_id", "week"], how="left",
                                indicator=True)
            extra = extra[extra["_merge"] == "left_only"]
            if not extra.empty:
                z = pd.DataFrame({"player_id": extra["player_id"], "player_display_name": extra["player"],
                                  "position": extra["position"], "season": s, "week": extra["week"],
                                  "season_type": "REG", "game_id": extra["game_id"], "team": extra["team"],
                                  "opponent_team": extra["opponent"].replace(TEAM_FIX),
                                  "offense_snaps": extra["offense_snaps"], "offense_pct": extra["offense_pct"]})
                st = pd.concat([st, z], ignore_index=True)
        else:
            st["offense_snaps"] = np.nan
            st["offense_pct"] = np.nan
        frames.append(st)
    pg = pd.concat(frames, ignore_index=True)
    num = ["completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
           "sacks_suffered", "carries", "rushing_yards", "rushing_tds", "receptions", "targets",
           "receiving_yards", "receiving_tds", "target_share", "air_yards_share", "passing_air_yards",
           "receiving_air_yards"]
    for c in num:
        if c not in pg:
            pg[c] = 0.0
        pg[c] = pd.to_numeric(pg[c], errors="coerce").fillna(0.0)
    pg["rush_rec_yds"] = pg["rushing_yards"] + pg["receiving_yards"]
    pg["anytime_td"] = pg["rushing_tds"] + pg["receiving_tds"]
    pg["team"] = pg["team"].replace(TEAM_FIX)
    pg["opp"] = pg["opponent_team"].replace(TEAM_FIX)
    pg["player"] = pg["player_display_name"]
    keep = ["player_id", "player", "position", "season", "week", "game_id", "team", "opp",
            "offense_snaps", "offense_pct"] + num + ["rush_rec_yds", "anytime_td"]
    pg = pg[keep]
    return pg.drop_duplicates(["player_id", "game_id"])


def build_injuries(season=None) -> pd.DataFrame:
    season = season or NFL_SEASON
    inj = _read("injuries", season)
    if inj.empty:
        return pd.DataFrame(columns=["season", "week", "team", "player_id", "player", "position", "status"])
    inj = inj.rename(columns={"gsis_id": "player_id", "full_name": "player", "report_status": "status"})
    inj["team"] = inj["team"].replace(TEAM_FIX)
    return inj[["season", "week", "team", "player_id", "player", "position", "status"]]


def build_all(log=print):
    games = build_games()
    pg = build_player_games()
    pg = pg.merge(games[["game_id", "date"]], on="game_id", how="left").dropna(subset=["date"])
    games.to_csv(GAMES_CSV, index=False)
    pg.to_csv(PLAYER_GAMES_CSV, index=False)
    build_injuries().to_csv(INJURIES_CSV, index=False)
    log(f"{len(pg):,} player-games, {len(games):,} games "
        f"(latest week played: {int(pg[pg['season'] == pg['season'].max()]['week'].max())})")
    return games, pg


def load():
    games = pd.read_csv(GAMES_CSV, parse_dates=["date"])
    pg = pd.read_csv(PLAYER_GAMES_CSV, parse_dates=["date"])
    inj = pd.read_csv(INJURIES_CSV) if INJURIES_CSV.exists() else pd.DataFrame()
    return games, pg.sort_values(["date", "player_id"]).reset_index(drop=True), inj


def upcoming_games(games: pd.DataFrame, today=None) -> pd.DataFrame:
    """Games in the next unplayed week."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    fut = games[(games["date"] >= today) & games["home_score"].isna()]
    if fut.empty:
        return fut
    s, w = fut.sort_values("date").iloc[0][["season", "week"]]
    return games[(games["season"] == s) & (games["week"] == w)].sort_values(["date", "kick_hour"])
