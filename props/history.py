"""Pick history: every Top Picks run is logged and graded, plus an honest backtest.

Sources
  live      data/pick_history.csv   - logged automatically each time Top Picks is built
                                       (the last version before the game is kept)
  backtest  data/backtest_picks.csv - this season's NBA games re-played out-of-sample:
                                       models trained ONLY on games before opening night,
                                       every pick scored with only what was known before
                                       that game, against the estimated line Top Picks uses
  tracked   data/picks.csv          - bets you added with the tracker

All three are graded from the box scores (NBA game logs / NFL weekly stats).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, odds

HISTORY_CSV = config.DATA_DIR / "pick_history.csv"
BACKTEST_CSV = config.DATA_DIR / "backtest_picks.csv"
BREAK_EVEN = 110 / 210          # win rate needed at -110

COLS = ["date", "sport", "player", "player_id", "team", "opp", "game", "stat", "stat_label",
        "line", "line_type", "book", "pick", "odds", "prob_over", "confidence",
        "generated_at", "source", "actual", "result", "profit"]


def _blank(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLS:
        if c not in df:
            df[c] = np.nan
    return df[COLS]


def _read(path) -> pd.DataFrame:
    if not path.exists():
        return _blank(pd.DataFrame())
    df = pd.read_csv(path, parse_dates=["date", "generated_at"], low_memory=False)
    return _blank(df)


# ------------------------------------------------------------- live logging
def log_picks(picks: pd.DataFrame) -> int:
    """Save a Top Picks run. For the same game/player/stat the newest pre-game version wins."""
    if picks is None or picks.empty:
        return 0
    p = picks.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p["odds"] = np.where(p["pick"] == "OVER", p.get("over_odds"), p.get("under_odds"))
    p["source"] = "live"
    p = _blank(p)
    old = _read(HISTORY_CSV)
    key = ["date", "sport", "player_id", "stat"]
    graded = old[old["result"].isin(["win", "loss", "push", "void"])]
    keep_old = old[~old.set_index(key).index.isin(p.set_index(key).index) | old.index.isin(graded.index)]
    p = p[~p.set_index(key).index.isin(graded.set_index(key).index)]   # never overwrite a graded pick
    out = pd.concat([keep_old, p], ignore_index=True).sort_values(["date", "sport", "confidence"],
                                                                  ascending=[True, True, False])
    out.to_csv(HISTORY_CSV, index=False)
    return len(p)


# ------------------------------------------------------------------ grading
def _outcome(row, actual):
    if pd.isna(actual):
        return "void", np.nan, 0.0
    if actual == row["line"]:
        return "push", actual, 0.0
    won = actual > row["line"] if row["pick"] == "OVER" else actual < row["line"]
    o = row["odds"] if pd.notna(row["odds"]) else -110
    return ("win", actual, odds.payout(o)) if won else ("loss", actual, -1.0)


def grade(df: pd.DataFrame) -> pd.DataFrame:
    """Fill actual / result / profit (1-unit stakes) for games that are in the data."""
    df = df.copy()
    todo = ~df["result"].isin(["win", "loss", "push", "void"])
    if not todo.any():
        return df
    from .analysis import load_data
    logs, _ = load_data()
    nba_last = logs["date"].max() if not logs.empty else pd.Timestamp.min
    try:
        from . import nfl_data as nd
        _, pg, _ = nd.load()
        pg = pg.assign(date=pd.to_datetime(pg["date"]).dt.normalize())
        nfl_last = pg["date"].max()
    except Exception:
        pg, nfl_last = pd.DataFrame(), pd.Timestamp.min
    for sport, box, last in (("NBA", logs, nba_last), ("NFL", pg, nfl_last)):
        m = todo & (df["sport"] == sport) & (df["date"] <= last)
        if not m.any() or box.empty:
            continue
        sub = df[m]
        for stat, rows in sub.groupby("stat"):
            if stat not in box:
                continue
            look = box[["player_id", "date", stat]].drop_duplicates(["player_id", "date"])
            got = rows[["player_id", "date"]].merge(look, on=["player_id", "date"], how="left")[stat].to_numpy()
            for idx, actual in zip(rows.index, got):
                res, act, prof = _outcome(df.loc[idx], actual)
                df.loc[idx, ["result", "actual", "profit"]] = [res, act, prof]
    df.loc[~df["result"].isin(["win", "loss", "push", "void"]), "result"] = "pending"
    return df


def grade_history() -> int:
    """Grade the live log in place. Returns how many picks were newly graded."""
    df = _read(HISTORY_CSV)
    if df.empty:
        return 0
    before = df["result"].isin(["win", "loss", "push", "void"]).sum()
    df = grade(df)
    df.to_csv(HISTORY_CSV, index=False)
    return int(df["result"].isin(["win", "loss", "push", "void"]).sum() - before)


def tracked() -> pd.DataFrame:
    """Your tracker bets in the same shape."""
    from . import tracker
    t = tracker.load()
    if t.empty:
        return _blank(pd.DataFrame())
    is_nfl = t["source"].astype(str).str.startswith("nfl")
    res = t["result"].replace({"void (DNP)": "void"})
    out = pd.DataFrame({
        "date": pd.to_datetime(t["date"]), "sport": np.where(is_nfl, "NFL", "NBA"),
        "player": t["player"], "player_id": t["player_id"], "stat": t["stat"],
        "stat_label": t["stat"].map(lambda s: config.STATS.get(s, s)),
        "line": t["line"], "line_type": "sportsbook", "book": t["book"], "pick": t["pick"],
        "odds": t["odds"],
        "confidence": np.where(t["pick"] == "OVER", t["model_prob"], 1 - t["model_prob"]),
        "source": "tracked", "actual": t["actual"], "result": res,
        "profit": t["profit"] / t["stake"].replace(0, 1),
    })
    return _blank(out)


def load_all(sources=("live", "backtest", "tracked")) -> pd.DataFrame:
    frames = []
    if "live" in sources:
        frames.append(_read(HISTORY_CSV))
    if "backtest" in sources:
        frames.append(_read(BACKTEST_CSV))
    if "tracked" in sources:
        frames.append(tracked())
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _blank(pd.DataFrame())
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[(df["sport"] != "NBA") | df["stat"].isin(list(config.STATS))]   # only stats still offered


# ----------------------------------------------------------------- backtest
def backtest_nba(start: str | None = None, log=print) -> pd.DataFrame:
    """Re-play this NBA season out-of-sample.

    1. minutes + over/under models are trained only on games before `start`
       (default: this season's opening night), calibrated on the last 15% of those
    2. every rotation player's game since then is scored at the line Top Picks would
       have used (his last-10 average rounded to x.5), with features from earlier games
    3. the same 'realistic line' filter as Top Picks is applied, then graded
    Injury inputs come from who actually sat (box scores), the live app uses the
    injury report, so this is slightly kinder than real life.
    """
    from . import model
    from .analysis import attach_positions, load_data
    from .context import ROTATION_MIN_GAMES, ROTATION_MIN_MPG
    from .top_picks import realistic

    logs, players = load_data()
    logs = attach_positions(logs, players)
    cut = pd.Timestamp(start) if start else logs.loc[logs["season"] == config.SEASON, "date"].min()
    ctx = model.Context(logs)
    names = players.drop_duplicates("player_id", keep="last").set_index("player_id")["player"]

    log(f"Backtest: training on games before {cut:%b %d, %Y}, testing on games since ...")
    base = model.build_features(logs, "pts", None, ctx)
    mins = base.dropna(subset=["min_last10"])
    pre = mins[(mins["date"] < cut) & mins["min"].notna()]
    reg = model._reg().fit(pre[model.MIN_FEATURES], pre["min"])
    pred_min = mins[["player_id", "date"]].assign(pred_min=reg.predict(mins[model.MIN_FEATURES]))

    out = []
    for stat in config.ready_stats():
        if stat not in logs:
            continue
        log(f"  {config.STATS[stat]} ...")
        f = model.build_features(logs, stat, None, ctx)
        f = f.drop(columns="pred_min").merge(pred_min, on=["player_id", "date"], how="left")
        f = model.set_line_features(f, stat)
        f = f.dropna(subset=["avg_last10", "avg_season"])
        f = f[f.groupby("player_id").cumcount() >= 3]
        tr = f[(f["date"] < cut)].dropna(subset=["target_over"]).sort_values("date")
        te = f[(f["date"] >= cut) & f[stat].notna()].copy()
        te = te[(te["min_season"] >= ROTATION_MIN_MPG) & (te["games_played"] >= ROTATION_MIN_GAMES)]
        if len(tr) < 500 or te.empty:
            continue
        # calibrator on the last 15% of the pre-season data, then a final fit on all of it
        c2 = tr["date"].quantile(0.85)
        fit, hold = tr[tr["date"] < c2], tr[tr["date"] >= c2]
        aug = model.augment_lines(fit, stat)
        cal, _ = model.fit_calibrator(model._clf().fit(aug[model.FEATURES], aug["target_over"])
                                      .predict_proba(hold[model.FEATURES])[:, 1],
                                      hold["target_over"].to_numpy())
        aug = model.augment_lines(tr, stat)
        clf = model._clf().fit(aug[model.FEATURES], aug["target_over"])
        # the line Top Picks would have used
        te["line"] = np.maximum(0.5, np.floor(te["avg_last10"]) + 0.5)
        te = model.set_line_features(te, stat)
        p = cal.predict(clf.predict_proba(te[model.FEATURES])[:, 1])
        out.append(pd.DataFrame({
            "date": te["date"].values, "sport": "NBA", "player_id": te["player_id"].values,
            "team": te["team"].values, "opp": te["opp"].values,
            "home": te["home"].values, "stat": stat, "stat_label": config.STATS[stat],
            "line": te["line"].values, "prob_over": p, "actual": te[stat].values,
            "avg_season": te["avg_season"].values}))
    if not out:
        return _blank(pd.DataFrame())
    bt = pd.concat(out, ignore_index=True)
    bt["player"] = bt["player_id"].map(names).fillna(bt["player_id"])
    bt["game"] = np.where(bt["home"].astype(bool), "vs " + bt["opp"], "@ " + bt["opp"])
    bt["line_type"] = "estimated"
    bt = bt[realistic(bt.assign(sport="NBA"))]
    bt["pick"] = np.where(bt["prob_over"] >= 0.5, "OVER", "UNDER")
    bt["confidence"] = np.where(bt["pick"] == "OVER", bt["prob_over"], 1 - bt["prob_over"])
    bt["odds"], bt["source"], bt["book"] = -110, "backtest", "estimated line"
    bt["generated_at"] = pd.Timestamp.now().floor("min")
    res = [_outcome(r, a) for r, a in zip(bt.to_dict("records"), bt["actual"])]
    bt["result"] = [r[0] for r in res]
    bt["profit"] = [r[2] for r in res]
    bt = _blank(bt).sort_values(["date", "confidence"], ascending=[True, False])
    bt.to_csv(BACKTEST_CSV, index=False)
    log(f"Backtest: {len(bt):,} picks from {bt['date'].min():%b %d} to {bt['date'].max():%b %d} "
        f"-> {BACKTEST_CSV.name}")
    return bt


# ---------------------------------------------------------------- summaries
def top_per_day(df: pd.DataFrame, n: int | None, per_player: int = 2) -> pd.DataFrame:
    """What Top Picks would have shown: best n picks a day, at most 2 per player."""
    if not n or df.empty:
        return df
    d = df.sort_values("confidence", ascending=False)
    d = d[d.groupby(["date", "source", "player"]).cumcount() < per_player]
    return d[d.groupby(["date", "source"]).cumcount() < n]


def record(df: pd.DataFrame) -> dict:
    g = df[df["result"].isin(["win", "loss", "push"])]
    w, l, p = (g["result"] == "win").sum(), (g["result"] == "loss").sum(), (g["result"] == "push").sum()
    return {"picks": int(len(g)), "wins": int(w), "losses": int(l), "pushes": int(p),
            "hit_rate": w / (w + l) if w + l else np.nan,
            "units": float(g["profit"].sum()), "roi": float(g["profit"].sum() / len(g)) if len(g) else np.nan,
            "pending": int((df["result"] == "pending").sum())}


def by_period(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """freq: 'D' day, 'W' week (Mon-Sun), 'M' month."""
    g = df[df["result"].isin(["win", "loss", "push"])].copy()
    if g.empty:
        return pd.DataFrame()
    g["period"] = g["date"].dt.to_period({"D": "D", "W": "W-SUN", "M": "M"}[freq]).dt.start_time
    rows = [{"period": k, **record(v)} for k, v in g.groupby("period")]
    out = pd.DataFrame(rows).sort_values("period")
    out["cum_units"] = out["units"].cumsum()
    return out


def wilson_low(w, n, z=1.96):
    """Lower edge of a 95% confidence range for the true hit rate. Rewards a long
    record over a lucky short one (5-0 ranks below 40-12)."""
    if n == 0:
        return np.nan
    p = w / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / den


def players(df: pd.DataFrame, min_picks: int = 5) -> pd.DataFrame:
    g = df[df["result"].isin(["win", "loss"])].sort_values("date")
    rows = []
    for (sport, player), v in g.groupby(["sport", "player"]):
        n = len(v)
        if n < min_picks:
            continue
        hits = (v["result"] == "win").to_numpy()
        w = int(hits.sum())
        # streaks
        longest = cur = 0
        for h in hits:
            cur = cur + 1 if h else 0
            longest = max(longest, cur)
        last = hits[::-1]
        now_kind = "W" if last[0] else "L"
        diff = np.flatnonzero(last != last[0])
        now = int(diff[0]) if len(diff) else len(last)
        top_stat = v.groupby("stat_label")["result"].apply(lambda r: (r == "win").mean()).idxmax()
        rows.append({"sport": sport, "player": player, "picks": n, "wins": w, "losses": n - w,
                     "hit_rate": w / n, "consistency": wilson_low(w, n),
                     "units": float(v["profit"].sum()), "longest_win_streak": longest,
                     "current_streak": f"{now_kind}{now}", "best_stat": top_stat,
                     "last_pick": v["date"].max()})
    return pd.DataFrame(rows).sort_values("consistency", ascending=False) if rows else pd.DataFrame()


def by_confidence(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["result"].isin(["win", "loss"])].copy()
    if g.empty:
        return pd.DataFrame()
    g["bucket"] = pd.cut(g["confidence"], [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1.0],
                         labels=["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"], include_lowest=True)
    out = g.groupby("bucket", observed=True).agg(picks=("result", "size"),
                                                 hit_rate=("result", lambda r: (r == "win").mean()),
                                                 model_said=("confidence", "mean"),
                                                 units=("profit", "sum")).reset_index()
    return out
