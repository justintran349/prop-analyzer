"""Game context the model uses on top of player form:

* Missing teammates  - who normally plays but sat out (injuries, rest).
                       For past games this is worked out from the box scores;
                       for upcoming games it comes from the ESPN injury report.
* Pace / totals      - each team's points scored & allowed to date -> projected
                       game total and margin (blowout risk).
* Usage              - share of the team's possessions a player uses.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import requests

from . import config
from .teams import to_code

ROTATION_MIN_MPG = 15     # "normally plays" = averages 15+ minutes
ROTATION_MIN_GAMES = 3
RECENT_TEAM_GAMES = 10    # ...and played for this team in its last 10 games
HOME_EDGE = 2.5           # points


# --------------------------------------------------------------- team games
def team_game_table(logs: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with box-score totals."""
    played = logs[logs["min"].notna() & (logs["min"] > 0)]
    agg = {"pts": "sum", "min": "sum", "fga": "sum", "tov": "sum",
           "opp": "first", "home": "first"}
    if "fta" in played:
        agg["fta"] = "sum"
    tg = played.groupby(["season", "team", "date"]).agg(agg).reset_index()
    if "fta" not in tg:
        tg["fta"] = 0.0
    tg["poss"] = tg["fga"] + 0.44 * tg["fta"].fillna(0) + tg["tov"]
    opp = tg[["season", "team", "date", "pts"]].rename(columns={"team": "opp", "pts": "pts_allowed"})
    return tg.merge(opp, on=["season", "opp", "date"], how="left").drop_duplicates(
        ["season", "team", "date"])


def team_scoring_to_date(tg: pd.DataFrame) -> pd.DataFrame:
    """Cumulative points for/against THROUGH each date (use with an as-of
    merge that excludes the same date to get pre-game values)."""
    t = tg.sort_values("date").copy()
    g = t.groupby(["season", "team"])
    t["gp"] = g.cumcount() + 1
    t["ppg_thru"] = g["pts"].cumsum() / t["gp"]
    t["papg_thru"] = g["pts_allowed"].cumsum() / t["gp"]
    return t[["season", "team", "date", "ppg_thru", "papg_thru"]]


# -------------------------------------------------------- missing teammates
def missing_teammates(logs: pd.DataFrame) -> pd.DataFrame:
    """For every past team-game: season averages of rotation players who did NOT play."""
    import bisect
    played = logs[logs["min"].notna() & (logs["min"] > 0)].sort_values("date")
    # each player's games in date order, to know which team he was on most recently
    history = {pid: (list(g["date"]), list(g["team"]))
               for pid, g in played.groupby("player_id")}

    def team_before(pid, d):
        dates, teams = history[pid]
        i = bisect.bisect_left(dates, d) - 1
        return teams[i] if i >= 0 else None

    rows = []
    for (season, team), g in played.groupby(["season", "team"], sort=False):
        dates = sorted(g["date"].unique())
        by_date = {d: gg for d, gg in g.groupby("date")}
        state = defaultdict(lambda: {"n": 0, "min": 0.0, "pts": 0.0, "ast": 0.0,
                                     "trb": 0.0, "last": -99})
        for i, d in enumerate(dates):
            gg = by_date[d]
            present = set(gg["player_id"])
            miss = {"min": 0.0, "pts": 0.0, "ast": 0.0, "trb": 0.0, "n": 0}
            for pid, s in state.items():
                if pid in present or s["n"] < ROTATION_MIN_GAMES:
                    continue
                if s["min"] / s["n"] < ROTATION_MIN_MPG or i - s["last"] > RECENT_TEAM_GAMES:
                    continue
                if team_before(pid, d) != team:   # traded away / waived
                    continue
                for k in ("min", "pts", "ast", "trb"):
                    miss[k] += s[k] / s["n"]
                miss["n"] += 1
            rows.append({"season": season, "team": team, "date": d,
                         "miss_min": miss["min"], "miss_pts": miss["pts"],
                         "miss_ast": miss["ast"], "miss_trb": miss["trb"],
                         "miss_n": miss["n"]})
            for r in gg.itertuples():
                s = state[r.player_id]
                s["n"] += 1
                s["min"] += r.min
                s["pts"] += r.pts
                s["ast"] += r.ast
                s["trb"] += r.trb
                s["last"] = i
    return pd.DataFrame(rows)


def season_averages(logs: pd.DataFrame, season: int) -> pd.DataFrame:
    cur = logs[(logs["season"] == season) & (logs["min"] > 0)]
    if cur.empty:
        cur = logs[logs["season"] == logs["season"].max()]
    last = cur.sort_values("date").groupby("player_id").tail(1)[["player_id", "team"]]
    avg = cur.groupby("player_id")[["min", "pts", "ast", "trb"]].mean()
    avg["games"] = cur.groupby("player_id").size()
    return avg.join(last.set_index("player_id"))


def missing_from_report(logs: pd.DataFrame, players: pd.DataFrame, team: str,
                        injuries: pd.DataFrame, season: int = config.SEASON) -> dict:
    """Same 'missing' numbers for an UPCOMING game, from the injury report."""
    out = {"miss_min": 0.0, "miss_pts": 0.0, "miss_ast": 0.0, "miss_trb": 0.0,
           "miss_n": 0, "names": []}
    if injuries is None or injuries.empty:
        return out
    inj = injuries[(injuries["team"] == team) & injuries["status"].str.lower().isin(
        ["out", "doubtful"])]
    if inj.empty:
        return out
    from .nba_com import norm_name
    avg = season_averages(logs, season)
    names = (players.sort_values("season").drop_duplicates("player_id", keep="last")
             .set_index("player_id")["player"])
    avg["key"] = avg.index.map(lambda p: norm_name(names.get(p, "")))
    keys = set(inj["player"].map(norm_name))
    hit = avg[(avg["key"].isin(keys)) & (avg["team"] == team)
              & (avg["min"] >= ROTATION_MIN_MPG) & (avg["games"] >= ROTATION_MIN_GAMES)]
    for k in ("min", "pts", "ast", "trb"):
        out[f"miss_{k}"] = float(hit[k].sum())
    out["miss_n"] = int(len(hit))
    out["names"] = [names.get(p, p) for p in hit.index]
    return out


# ------------------------------------------------------------------ injuries
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


def fetch_injuries() -> pd.DataFrame:
    """Current NBA injury report from ESPN -> data/injuries.csv."""
    r = requests.get(ESPN_INJURIES, timeout=30, headers={"User-Agent": config.USER_AGENT})
    r.raise_for_status()
    rows = []
    for team in r.json().get("injuries", []):
        code = to_code(team.get("displayName", ""))
        for inj in team.get("injuries", []):
            ath = inj.get("athlete", {}) or {}
            rows.append({
                "team": code,
                "player": ath.get("displayName", ""),
                "status": inj.get("status", ""),
                "detail": (inj.get("shortComment") or inj.get("longComment") or "")[:200],
                "updated": inj.get("date", ""),
            })
    df = pd.DataFrame(rows, columns=["team", "player", "status", "detail", "updated"])
    df.to_csv(config.INJURIES_CSV, index=False)
    return df


def load_injuries() -> pd.DataFrame:
    """ESPN report + your manual overrides (data/injuries_manual.csv: team,player,status)."""
    frames = []
    for p in (config.INJURIES_CSV, config.MANUAL_INJURIES_CSV):
        if p.exists():
            try:
                frames.append(pd.read_csv(p))
            except pd.errors.EmptyDataError:
                pass
    if not frames:
        return pd.DataFrame(columns=["team", "player", "status", "detail"])
    df = pd.concat(frames, ignore_index=True)
    df["team"] = df["team"].map(to_code)
    return df.drop_duplicates(["team", "player"], keep="last")


# --------------------------------------------------------------------- usage
def add_usage(logs: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Usage rate for each player-game (share of team possessions while on court)."""
    df = logs.copy()
    fta = df["fta"].fillna(0) if "fta" in df else 0
    df["poss_used"] = df["fga"] + 0.44 * fta + df["tov"]
    df = df.merge(tg[["season", "team", "date", "poss", "min"]].rename(
        columns={"poss": "tm_poss", "min": "tm_min"}),
        on=["season", "team", "date"], how="left")
    df["usg"] = (df["poss_used"] * (df["tm_min"] / 5)) / (df["min"] * df["tm_poss"])
    df.loc[~np.isfinite(df["usg"]), "usg"] = np.nan
    return df.drop(columns=["tm_poss", "tm_min", "poss_used"]).drop_duplicates(["player_id", "date"])
