"""Pick tracker: log picks, grade them from box scores, measure win rate / ROI.

data/picks.csv - one row per pick. Stake is 1 unit unless you change it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, odds

COLUMNS = ["created", "date", "player", "player_id", "stat", "line", "pick", "odds",
           "model_prob", "book", "stake", "source", "result", "actual", "profit"]


def load() -> pd.DataFrame:
    p = config.PICKS_CSV
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(p, parse_dates=["date", "created"])
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[COLUMNS]


def save(df: pd.DataFrame):
    df[COLUMNS].to_csv(config.PICKS_CSV, index=False)


def add(date, player, player_id, stat, line, pick, odds_=None, model_prob=None,
        book=None, stake=1.0, source="manual") -> bool:
    """Log a pick. Returns False if the same pick is already logged."""
    df = load()
    d = pd.Timestamp(date).normalize()
    dup = ((df["date"] == d) & (df["player_id"] == player_id) & (df["stat"] == stat)
           & (df["pick"] == pick) & (df["line"] == float(line)))
    if dup.any():
        return False
    row = {"created": pd.Timestamp.now().floor("s"), "date": d, "player": player,
           "player_id": player_id, "stat": stat, "line": float(line), "pick": pick,
           "odds": odds_ if odds_ is not None and not pd.isna(odds_) else -110,
           "model_prob": model_prob, "book": book, "stake": stake, "source": source,
           "result": "pending", "actual": np.nan, "profit": np.nan}
    save(pd.concat([df, pd.DataFrame([row])], ignore_index=True))
    return True


def grade(logs: pd.DataFrame) -> int:
    """Fill in results for pending picks whose games are in the data. Returns # graded."""
    df = load()
    if df.empty:
        return 0
    last_date = logs["date"].max()
    is_nfl = df["source"].astype(str).str.startswith("nfl")
    pending = (df["result"] == "pending") & (df["date"] <= last_date) & ~is_nfl
    n = 0
    for i in df[pending].index:
        r = df.loc[i]
        g = logs[(logs["player_id"] == r["player_id"]) & (logs["date"] == r["date"])]
        if g.empty or pd.isna(g.iloc[0].get(r["stat"])):
            df.loc[i, ["result", "profit"]] = ["void (DNP)", 0.0]
        else:
            actual = float(g.iloc[0][r["stat"]])
            if actual == r["line"]:
                res, profit = "push", 0.0
            else:
                won = actual > r["line"] if r["pick"] == "OVER" else actual < r["line"]
                res = "win" if won else "loss"
                profit = r["stake"] * odds.payout(r["odds"]) if won else -r["stake"]
            df.loc[i, ["result", "actual", "profit"]] = [res, actual, profit]
        n += 1
    save(df)
    return n


def grade_nfl(pg: pd.DataFrame) -> int:
    """Grade NFL picks (source 'nfl...') from NFL weekly stats."""
    df = load()
    if df.empty:
        return 0
    is_nfl = df["source"].astype(str).str.startswith("nfl")
    pending = (df["result"] == "pending") & is_nfl & (df["date"] <= pg["date"].max())
    n = 0
    for i in df[pending].index:
        r = df.loc[i]
        g = pg[(pg["player_id"] == r["player_id"]) & (pg["date"] == r["date"])]
        if g.empty:
            df.loc[i, ["result", "profit"]] = ["void (DNP)", 0.0]
        else:
            actual = float(g.iloc[0][r["stat"]])
            won = actual > r["line"] if r["pick"] == "OVER" else actual < r["line"]
            res = "push" if actual == r["line"] else ("win" if won else "loss")
            profit = 0.0 if res == "push" else (r["stake"] * odds.payout(r["odds"]) if won else -r["stake"])
            df.loc[i, ["result", "actual", "profit"]] = [res, actual, profit]
        n += 1
    save(df)
    return n


def summary(df: pd.DataFrame | None = None) -> dict:
    df = load() if df is None else df
    done = df[df["result"].isin(["win", "loss", "push"])]
    decided = done[done["result"] != "push"]
    staked = done["stake"].sum()
    return {
        "picks": int(len(df)), "graded": int(len(done)),
        "pending": int((df["result"] == "pending").sum()),
        "wins": int((done["result"] == "win").sum()), "losses": int((done["result"] == "loss").sum()),
        "win_rate": float((decided["result"] == "win").mean()) if len(decided) else None,
        "profit": float(done["profit"].sum()) if len(done) else 0.0,
        "roi": float(done["profit"].sum() / staked) if staked else None,
        # how well calibrated: model's average confidence vs actual win rate
        "avg_model_prob": float(decided["model_prob"].mean()) if decided["model_prob"].notna().any() else None,
    }


def by_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    done = df[df["result"].isin(["win", "loss", "push"])]
    if done.empty:
        return pd.DataFrame()
    g = done.groupby(col)
    out = pd.DataFrame({
        "picks": g.size(),
        "win_rate": g.apply(lambda x: (x["result"] == "win").sum() / max(1, (x["result"] != "push").sum())),
        "profit": g["profit"].sum(),
    })
    out["roi"] = out["profit"] / g["stake"].sum()
    return out.reset_index()
