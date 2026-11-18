"""Stats analysis: home/away splits, hit rates vs a line, and
defense-vs-position (how much each team gives up to PGs, SGs, SFs, PFs, Cs)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

POSITIONS = ["PG", "SG", "SF", "PF", "C"]


def load_data():
    """Read the CSVs written by scripts/update_data.py."""
    if not config.GAMELOGS_CSV.exists():
        return pd.DataFrame(), pd.DataFrame()
    logs = pd.read_csv(config.GAMELOGS_CSV, parse_dates=["date"])
    logs = logs.sort_values(["date", "player_id"]).reset_index(drop=True)
    players = pd.read_csv(config.PLAYERS_CSV)
    return logs, players


def attach_positions(logs: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Give every game row the player's name and primary position."""
    pos = (players.sort_values("season")
           .drop_duplicates("player_id", keep="last")[["player_id", "player", "primary_pos"]])
    out = logs.drop(columns=[c for c in ("player", "primary_pos") if c in logs.columns])
    return out.merge(pos, on="player_id", how="left")


# ------------------------------------------------------ defense vs position
def defense_vs_position(logs: pd.DataFrame, season: int | None = None,
                        last_n_games: int | None = None) -> pd.DataFrame:
    """For every (opponent team, position, stat): what that team allows per 36 min,
    compared with the league average for that position.

    last_n_games=15 uses only each defense's 15 most recent games (recent form:
    picks up trades, injuries and lineup changes).

    factor 1.10 = this defense allows 10% MORE than average to that position (good for overs)
    rank 1      = allows the most in the league
    """
    df = logs[logs["min"] >= config.MIN_MINUTES]
    if season is not None:
        df = df[df["season"] == season]
    if last_n_games:
        recent = (df[["opp", "date"]].drop_duplicates().sort_values("date")
                  .groupby("opp").tail(last_n_games))
        df = df.merge(recent, on=["opp", "date"])
    df = df.dropna(subset=["primary_pos"])
    stats = [s for s in config.STATS if s in df]

    team = df.groupby(["opp", "primary_pos"])[stats + ["min"]].sum()
    league = df.groupby("primary_pos")[stats + ["min"]].sum()

    rows = []
    for (opp, pos), r in team.iterrows():
        lg = league.loc[pos]
        for s in stats:
            allowed = r[s] / r["min"] * 36
            lg_avg = lg[s] / lg["min"] * 36
            rows.append({"opp": opp, "pos": pos, "stat": s,
                         "allowed_per36": allowed,
                         "league_per36": lg_avg,
                         "factor": allowed / lg_avg if lg_avg else 1.0})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank"] = (out.groupby(["pos", "stat"])["factor"]
                   .rank(ascending=False, method="min").astype(int))
    return out


def dvp_lookup(dvp: pd.DataFrame, opp: str, pos: str, stat: str) -> dict:
    hit = dvp[(dvp["opp"] == opp) & (dvp["pos"] == pos) & (dvp["stat"] == stat)]
    if hit.empty:
        return {"factor": 1.0, "rank": None, "allowed_per36": None}
    return hit.iloc[0].to_dict()


# ---------------------------------------------------------- player analysis
def hit_rate(values: pd.Series, line: float) -> dict:
    values = values.dropna()
    n = len(values)
    if n == 0:
        return {"games": 0, "overs": 0, "hit_rate": None, "avg": None}
    overs = int((values > line).sum())
    return {"games": n, "overs": overs, "hit_rate": overs / n,
            "avg": float(values.mean())}


def player_report(logs: pd.DataFrame, dvp: pd.DataFrame, player_id: str,
                  stat: str, line: float, opp: str, is_home: bool,
                  season: int = config.SEASON) -> dict:
    """Everything the dashboard shows for one player / stat / line."""
    p = logs[(logs["player_id"] == player_id) & (logs["min"] >= config.MIN_MINUTES)]
    p = p.sort_values("date")
    cur = p[p["season"] == season]
    if cur.empty:          # early season: fall back to last season
        cur = p
    pos = p["primary_pos"].dropna().iloc[-1] if p["primary_pos"].notna().any() else None

    venue = cur[cur["home"] == int(is_home)]
    splits = {
        "This season": hit_rate(cur[stat], line),
        "Home" if is_home else "Away": hit_rate(venue[stat], line),
        "Last 10": hit_rate(cur[stat].tail(10), line),
        "Last 5": hit_rate(cur[stat].tail(5), line),
        f"vs {opp} (all seasons)": hit_rate(p[p["opp"] == opp][stat], line),
    }

    d = dvp_lookup(dvp, opp, pos, stat) if pos else {"factor": 1.0, "rank": None}

    # Simple projection: blend season + venue + recent form, then adjust for defense
    parts, weights = [], []
    for key, w in (("This season", 0.4), ("Home" if is_home else "Away", 0.35),
                   ("Last 10", 0.25)):
        if splits[key]["avg"] is not None:
            parts.append(splits[key]["avg"])
            weights.append(w)
    base = float(np.average(parts, weights=weights)) if parts else None
    # shrink the defense factor toward 1 - one team's sample is noisy
    factor = 1 + 0.6 * (d["factor"] - 1)
    projection = base * factor if base is not None else None

    std = float(cur[stat].std()) if len(cur) > 2 else None
    prob_over = None
    if projection is not None and std:
        from scipy.stats import norm
        prob_over = float(1 - norm.cdf(line + 0.5 if line == int(line) else line,
                                       loc=projection, scale=std))

    return {
        "player_id": player_id, "pos": pos, "stat": stat, "line": line,
        "opp": opp, "is_home": is_home, "splits": splits,
        "dvp_factor": d["factor"], "dvp_rank": d.get("rank"),
        "dvp_allowed": d.get("allowed_per36"),
        "projection": projection, "std": std, "prob_over_simple": prob_over,
        "games": cur,
    }
