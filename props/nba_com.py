"""NBA.com game logs: every player's games for a whole season in ONE request.

NBA.com only lists positions as G / F / C, so positions (PG/SG/SF/PF/C) still
come from Basketball-Reference's per-game page, matched up by player name.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from . import config
from .teams import to_code


def add_combos(df: pd.DataFrame) -> pd.DataFrame:
    """Add combo stats (pra, pr, pa, ra) and 2-pointers (fg2m, fg2a) in place."""
    for name, parts in config.COMBOS.items():
        if all(p in df for p in parts):
            df[name] = df[parts].sum(axis=1, min_count=len(parts))
    for name, (total, threes) in config.DIFFS.items():
        if total in df and threes in df:
            df[name] = df[total] - df[threes]
    return df


def season_str(season: int) -> str:
    """2027 -> '2026-27' (NBA.com's format)."""
    return f"{season - 1}-{str(season)[-2:]}"


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.']", "", s).replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return NICKNAMES.get(s, s)


# NBA.com name -> Basketball-Reference name, where they differ
NICKNAMES = {"ronald holland": "ron holland"}


def collapse_traded(bp: pd.DataFrame) -> pd.DataFrame:
    """BRef lists traded players as a 2TM/TOT summary row + one row per team.
    Keep one row per player-season: summary stats, latest team."""
    multi = bp["team"].astype(str).str.match(r"^\dTM$|^TOT$")
    rows = []
    for _, g in bp.groupby(["player_id", "season"], sort=False):
        summ = g[multi.loc[g.index]]
        base = (summ.iloc[0] if len(summ) else g.iloc[0]).copy()
        teams = g[~multi.loc[g.index]]["team"]
        if len(teams):
            base["team"] = teams.iloc[-1]
        rows.append(base)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["primary_pos"] = out["pos"].astype(str).str.split("-").str[0]
    return out


def fetch_league_gamelog(season: int) -> pd.DataFrame:
    """Raw NBA.com league game log (needs the nba_api package + internet)."""
    from nba_api.stats.endpoints import leaguegamelog
    df = leaguegamelog.LeagueGameLog(
        season=season_str(season), player_or_team_abbreviation="P",
        season_type_all_star="Regular Season", timeout=60,
    ).get_data_frames()[0]
    df["SEASON"] = season_str(season)
    return df


def convert(raw: pd.DataFrame, bref_players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NBA.com rows -> (gamelogs, players) in this project's format.

    Players are matched to Basketball-Reference ids by name so positions line up.
    Anyone who can't be matched keeps an 'nba<id>' id and no position."""
    raw = raw.copy()
    raw["season"] = raw["SEASON"].astype(str).str[:4].astype(int) + 1

    bp = bref_players.copy()
    bp["key"] = bp["player"].map(norm_name)
    # prefer the most recent season's entry for each name
    lookup = bp.sort_values("season").drop_duplicates("key", keep="last").set_index("key")

    raw["key"] = raw["PLAYER_NAME"].map(norm_name)
    raw["player_id"] = raw["key"].map(lookup["player_id"])
    unmatched = raw["player_id"].isna()
    raw.loc[unmatched, "player_id"] = "nba" + raw.loc[unmatched, "PLAYER_ID"].astype(str)

    m = raw["MATCHUP"].astype(str)
    logs = pd.DataFrame({
        "player_id": raw["player_id"],
        "player": raw["PLAYER_NAME"],
        "season": raw["season"],
        "date": pd.to_datetime(raw["GAME_DATE"]),
        "team": raw["TEAM_ABBREVIATION"].map(to_code),
        "home": m.str.contains("vs.").astype(int),
        "opp": m.str.split(r" vs\. | @ ", regex=True).str[-1].map(to_code),
        "result": raw["WL"],
        "started": np.nan,           # not in this NBA.com endpoint
        "min": pd.to_numeric(raw["MIN"], errors="coerce"),
        "pts": pd.to_numeric(raw["PTS"], errors="coerce"),
        "trb": pd.to_numeric(raw["REB"], errors="coerce"),
        "ast": pd.to_numeric(raw["AST"], errors="coerce"),
        "fgm": pd.to_numeric(raw["FGM"], errors="coerce"),
        "fga": pd.to_numeric(raw["FGA"], errors="coerce"),
        "fg3": pd.to_numeric(raw["FG3M"], errors="coerce"),
        "fg3a": pd.to_numeric(raw["FG3A"], errors="coerce"),
        "stl": pd.to_numeric(raw["STL"], errors="coerce"),
        "blk": pd.to_numeric(raw["BLK"], errors="coerce"),
        "tov": pd.to_numeric(raw["TOV"], errors="coerce"),
        "fta": pd.to_numeric(raw["FTA"], errors="coerce"),
        "oreb": pd.to_numeric(raw["OREB"], errors="coerce"),
    })
    logs = logs[logs["min"] > 0].copy()
    add_combos(logs)

    # players table: BRef rows + any unmatched NBA.com players
    players = bref_players.copy()
    if "primary_pos" not in players:
        players["primary_pos"] = players["pos"].astype(str).str.split("-").str[0]
    extra = (logs[logs["player_id"].str.startswith("nba")]
             .sort_values("date").groupby(["player_id", "season"])
             .agg(player=("player", "last"), team=("team", "last"),
                  games=("min", "size"), mpg=("min", "mean")).reset_index())
    extra["pos"] = ""
    extra["primary_pos"] = np.nan
    players = pd.concat([players, extra], ignore_index=True)
    return logs.drop(columns="player"), players
