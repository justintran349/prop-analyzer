"""Most confident picks across NBA + NFL, built automatically.

For every regular player in the next NBA slate and the upcoming NFL week, the
models score each stat. Real sportsbook lines are used where you have them
(data/lines.csv, Odds API pulls); otherwise the player's estimated line
(recent average rounded to x.5).

confidence     = the model's chance for the side it picks (max of over/under)
tested_hit     = how often picks at that confidence actually won in testing
                 (from the calibration data saved at training time)
Results are cached in data/top_picks.csv so the dashboard opens instantly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, model, nfl_data as nd, nfl_model as nm

CACHE = config.DATA_DIR / "top_picks.csv"
MAX_AGE_HOURS = 3


# ---------------------------------------------------------- model quality
def _metrics(sport, stat):
    return model.load_metrics(stat) if sport == "NBA" else nm.load_metrics(stat)


def model_ok(sport, stat) -> bool:
    """A model is 'trusted' when it beat the coin-flip baseline in testing."""
    m = _metrics(sport, stat)
    if not m:
        return False
    if sport == "NFL" and stat == "anytime_td":
        return False          # mostly predicts "no TD" - accuracy is just the base rate
    return m["brier"] < m["baseline_brier"] and (m.get("auc") or 0) > 0.55


def tested_hit_rate(sport, stat, prob_over, pick) -> float | None:
    """Historical win rate for picks the model made at this confidence."""
    m = _metrics(sport, stat)
    cal = pd.DataFrame((m or {}).get("calibration", []))
    if cal.empty or pd.isna(prob_over):
        return None
    row = cal.iloc[(cal["predicted"] - prob_over).abs().argmin()]
    if abs(row["predicted"] - prob_over) > 0.1:
        return None
    return float(row["actual"]) if pick == "OVER" else float(1 - row["actual"])


# ------------------------------------------------------------------- NBA
def nba_slate_date(lg, now=None):
    """Next date with games that aren't in the data yet (after 8pm look at tomorrow)."""
    now = pd.Timestamp(now or pd.Timestamp.now())
    start = now.normalize() + pd.Timedelta(days=1 if now.hour >= 20 else 0)
    start = max(start, lg.logs["date"].max() + pd.Timedelta(days=1))
    if lg.sched.empty:
        return None
    fut = lg.sched[lg.sched["date"] >= start]
    return None if fut.empty else fut["date"].min()


def build_nba(stats=None, log=print) -> pd.DataFrame:
    from . import odds, slate
    lg = slate.League()
    day = nba_slate_date(lg)
    if day is None:
        return pd.DataFrame()
    games = slate.todays_games(lg, day)
    stats = stats or list(config.STATS)
    log(f"NBA: {len(games)} games on {day:%a %b %d}, {len(stats)} stats")
    proj = slate.projections_for_games(lg, games, tuple(stats))
    if proj.empty:
        return pd.DataFrame()
    proj["line_type"] = "estimated"
    proj["book"] = None
    proj["edge"] = np.nan
    # real lines win where available
    lines = pd.concat([odds.load_manual_lines(), odds.load_saved_api_lines()], ignore_index=True)
    if not lines.empty:
        lines = lines[pd.to_datetime(lines["date"]).dt.normalize() == day]
    if not lines.empty:
        board = slate.score_lines(lg, lines)
        if not board.empty:
            best = board.sort_values("edge", ascending=False).drop_duplicates(["player", "stat"])
            canon = best["player_id"].map(lg.current["player"]).fillna(best["player"])
            real = pd.DataFrame({"player": canon, "team": best["team"], "game": best["game"],
                                 "stat": best["stat"], "line": best["line"], "pred_min": best["pred_min"],
                                 "proj": best["proj"], "prob_over": best["prob_over"],
                                 "line_type": "sportsbook", "book": best["book"], "edge": best["edge"],
                                 "avg_season": np.nan, "over_odds": best["over_odds"],
                                 "under_odds": best["under_odds"], "onoff": best["onoff"],
                                 "prob_over_model": best["prob_over_model"]})
            key = set(zip(real["player"], real["stat"]))
            proj = pd.concat([proj[[k not in key for k in zip(proj["player"], proj["stat"])]], real],
                             ignore_index=True)
    out = pd.DataFrame({
        "sport": "NBA", "date": day, "player": proj["player"], "team": proj["team"],
        "game": proj["game"], "stat": proj["stat"], "stat_label": proj["stat"].map(config.STATS),
        "line": proj["line"], "line_type": proj["line_type"], "book": proj["book"],
        "prob_over": proj["prob_over"], "proj": proj["proj"], "edge": proj["edge"],
        "notes": proj["pred_min"].map(lambda m: f"proj. {m:.0f} min" if pd.notna(m) else ""),
        "avg_season": proj.get("avg_season"),
    })
    for c in ("avg_last10", "avg_last5", "hit_rate_last10_at_line", "pred_min", "miss_n", "miss_pts",
              "b2b", "dvp_factor", "proj_total", "over_odds", "under_odds"):
        out[c] = proj[c].values if c in proj else np.nan
    from . import onoff
    oo = proj["onoff"].values if "onoff" in proj else [None] * len(out)
    pm = proj["prob_over_model"].values if "prob_over_model" in proj else [np.nan] * len(out)
    out["prob_over_model"] = pm
    out["injury_note"] = [onoff.explain(i, config.STATS.get(st_, st_), pl, pb, pa)
                          if isinstance(i, dict) else ""
                          for i, st_, pl, pb, pa in zip(oo, out["stat"], out["player"], pm, out["prob_over"])]
    out["injury_short"] = [f"Without {' & '.join(n.split()[-1] for n in i['names'])}: "
                           f"{i['without']['min']:.0f} min vs {i['with']['min']:.0f} "
                           f"({i['without']['n']} games)" if isinstance(i, dict) else ""
                           for i in oo]
    for t in set(out["team"]):
        names = lg.missing(t)["names"]
        if names:
            m = out["team"] == t
            out.loc[m, "notes"] = out.loc[m, "notes"] + f" · {t} out: {', '.join(names[:3])}"
    return out


# ------------------------------------------------------------------- NFL
def build_nfl(stats=None, log=print) -> pd.DataFrame:
    from . import nfl_slate
    lg = nfl_slate.NFLLeague()
    up = lg.week[lg.week["home_score"].isna() & (lg.week["date"] >= pd.Timestamp.today().normalize())]
    if up.empty:
        return pd.DataFrame()
    stats = stats or [s for s in nd.STATS if s != "anytime_td"]
    log(f"NFL: week {int(up['week'].iloc[0])}, {len(up)} games, {len(stats)} stats")
    proj = lg.projections(tuple(stats))
    if proj.empty:
        return pd.DataFrame()
    proj = proj.assign(line_type="estimated", book=None, edge=np.nan)
    lines = nfl_slate.load_lines()
    if not lines.empty:
        board = lg.score_lines(lines)
        if not board.empty:
            best = board.sort_values("edge", ascending=False).drop_duplicates(["player", "stat"])
            key = set(zip(best["player"], best["stat"]))
            proj = proj[[k not in key for k in zip(proj["player"], proj["stat"])]]
            g = lg.games.set_index("game_id")
            canon = best["player_id"].map(lg.current["player"]).fillna(best["player"])
            best = best.assign(player=canon)
            key = set(zip(best["player"], best["stat"]))
            proj = proj[[k not in key for k in zip(proj["player"], proj["stat"])]]
            real = pd.DataFrame({"player": best["player"], "position": best["pos"], "team": best["team"],
                                 "matchup": best["game"], "stat": best["stat"], "line": best["line"],
                                 "avg_last5": best["avg_last5"], "proj": np.nan,
                                 "prob_over": best["prob_over"], "weather": best["weather"],
                                 "line_type": "sportsbook", "book": best["book"], "edge": best["edge"],
                                 "over_odds": best["over_odds"], "under_odds": best["under_odds"]})
            proj = pd.concat([proj, real], ignore_index=True)
    gdate = lg.week.assign(m=lg.week["away_team"] + " @ " + lg.week["home_team"]).set_index("m")["date"]
    return pd.DataFrame({
        "sport": "NFL", "date": proj["matchup"].map(gdate).fillna(up["date"].min()),
        "player": proj["player"], "team": proj["team"], "game": proj["matchup"],
        "stat": proj["stat"], "stat_label": proj["stat"].map(nd.LABEL), "line": proj["line"],
        "line_type": proj["line_type"], "book": proj["book"], "prob_over": proj["prob_over"],
        "proj": proj["proj"], "edge": proj["edge"],
        "notes": proj["weather"].fillna("") + " · " + proj["position"].fillna(""),
        "avg_season": proj.get("avg_season"),
    }).assign(**{c: (proj[c].values if c in proj else np.nan) for c in (
        "avg_last5", "avg_last3", "hit_rate_last5_at_line", "dvp_factor", "game_wind", "game_temp",
        "indoor", "qb_out", "implied_pts", "team_spread", "over_odds", "under_odds")})


# ---------------------------------------------------------------- combine
# Smallest estimated line a sportsbook would realistically post. Below these,
# "confident" picks are things like a center going UNDER 0.5 threes - true but useless.
MIN_LINE = {
    ("NBA", "pts"): 9.5, ("NBA", "trb"): 3.5, ("NBA", "ast"): 2.5, ("NBA", "fg3"): 1.5,
    ("NBA", "pra"): 14.5, ("NBA", "pr"): 11.5, ("NBA", "pa"): 11.5, ("NBA", "ra"): 5.5,
    ("NBA", "stl"): 0.5, ("NBA", "blk"): 0.5, ("NBA", "stocks"): 1.5, ("NBA", "tov"): 1.5,
    ("NFL", "passing_yards"): 160.5, ("NFL", "completions"): 14.5, ("NFL", "attempts"): 22.5,
    ("NFL", "passing_tds"): 0.5, ("NFL", "passing_interceptions"): 0.5,
    ("NFL", "rushing_yards"): 14.5, ("NFL", "carries"): 5.5, ("NFL", "receiving_yards"): 14.5,
    ("NFL", "receptions"): 1.5, ("NFL", "rush_rec_yds"): 19.5,
}
MAX_LINE = {("NFL", "passing_interceptions"): 0.5, ("NFL", "passing_tds"): 2.5}
MIN_AVG_FOR_HALF = {"stl": 0.9, "blk": 0.9, "passing_tds": 1.0, "passing_interceptions": 0.5}


def realistic(df: pd.DataFrame) -> pd.Series:
    """Keep sportsbook lines always; keep estimated lines only when a book would post
    something like them and the player's role is stable (line near his season average)."""
    est = df["line_type"] == "estimated"
    min_line = [MIN_LINE.get((s, st), 0.5) for s, st in zip(df["sport"], df["stat"])]
    max_line = [MAX_LINE.get((s, st), 1e9) for s, st in zip(df["sport"], df["stat"])]
    ok = (df["line"] >= np.array(min_line)) & (df["line"] <= np.array(max_line))
    half_need = df["stat"].map(MIN_AVG_FOR_HALF)
    ok &= ~((df["line"] == 0.5) & half_need.notna() & (df["avg_season"].fillna(0) < half_need.fillna(0)))
    avg = df["avg_season"]
    stable = avg.isna() | ((df["line"] - avg).abs() <= 0.3 * avg.abs().clip(lower=1) + 1)
    return ~est | (ok & stable)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["prob_over"]).copy()
    if df.empty:
        return df
    df = df[realistic(df)]
    df["pick"] = np.where(df["prob_over"] >= 0.5, "OVER", "UNDER")
    df["confidence"] = np.where(df["pick"] == "OVER", df["prob_over"], 1 - df["prob_over"])
    df["tested_hit"] = [tested_hit_rate(s, st, p, pk) for s, st, p, pk in
                        zip(df["sport"], df["stat"], df["prob_over"], df["pick"])]
    df["trusted_model"] = [model_ok(s, st) for s, st in zip(df["sport"], df["stat"])]
    from .explain import short_reason
    df["reason"] = [short_reason(sp, rec) for sp, rec in zip(df["sport"], df.to_dict("records"))]
    df = df.sort_values("confidence", ascending=False)
    df["player_rank"] = df.groupby(["sport", "player"]).cumcount() + 1   # for "max 2 per player"
    return df.reset_index(drop=True)


LAST_ERRORS: dict = {}


def build(log=print) -> pd.DataFrame:
    """Score everything. Errors are kept in LAST_ERRORS (sport -> message) so the
    dashboard can show what went wrong instead of an empty page."""
    import traceback
    frames = []
    LAST_ERRORS.clear()
    for name, fn in (("NBA", build_nba), ("NFL", build_nfl)):
        try:
            f = fn(log=log)
            if not f.empty:
                frames.append(f)
        except Exception as e:
            LAST_ERRORS[name] = f"{type(e).__name__}: {e}"
            log(f"{name} skipped: {e}")
            traceback.print_exc()
    df = finalize(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
    if not df.empty:
        df["generated_at"] = pd.Timestamp.now().floor("min")
        df.to_csv(CACHE, index=False)
    return df


def data_mtime() -> float:
    files = [config.GAMELOGS_CSV, config.INJURIES_CSV, nd.PLAYER_GAMES_CSV, nd.WEATHER_CSV,
             config.LINE_SNAPSHOTS_CSV, config.MANUAL_LINES_CSV, nd.LINES_CSV, nd.LINE_SNAPSHOTS_CSV,
             config.MODELS_DIR / "over_pts.joblib", nd.MODELS_DIR / "over_passing_yards.joblib"]
    try:
        from .scraped_lines import db_path
        files += [db_path(), db_path().with_name(db_path().name + "-wal")]
    except Exception:
        pass
    return max((f.stat().st_mtime for f in files if f.exists()), default=0)


def load_cached() -> tuple[pd.DataFrame, bool]:
    """(picks, is_fresh). Fresh = built within MAX_AGE_HOURS, after the latest data/model change."""
    if not CACHE.exists():
        return pd.DataFrame(), False
    df = pd.read_csv(CACHE, parse_dates=["date", "generated_at"])
    if df.empty:
        return df, False
    built = df["generated_at"].max()
    age_h = (pd.Timestamp.now() - built).total_seconds() / 3600
    changed = pd.Timestamp.fromtimestamp(data_mtime())   # local time, like generated_at
    fresh = age_h < MAX_AGE_HOURS and built >= changed - pd.Timedelta(minutes=2)
    fresh = fresh and (df["date"].max() >= pd.Timestamp.today().normalize())
    return df, fresh


def get(force=False, log=print) -> pd.DataFrame:
    df, fresh = load_cached()
    if fresh and not force:
        return df
    return build(log=log)


# ------------------------------------------------------------- hit rates
def _norm(n):
    from .nba_com import norm_name
    return norm_name(n)


def player_games(sport: str, player: str) -> pd.DataFrame:
    """A player's game log (oldest first) with date, season, opp, home and every stat."""
    if sport == "NBA":
        from .analysis import load_data
        logs, players = load_data()
        ids = players.assign(k=players["player"].map(_norm))
        pid = ids.loc[ids["k"] == _norm(player), "player_id"]
        if pid.empty:
            return pd.DataFrame()
        g = logs[(logs["player_id"] == pid.iloc[-1]) & (logs["min"] >= config.MIN_MINUTES)].copy()
        g["label"] = g["date"].dt.strftime("%m/%d") + " " + g["home"].map({1: "vs", 0: "@"}) + " " + g["opp"]
    else:
        games, pg, _ = nd.load()
        pg = pg.assign(k=pg["player"].map(_norm))
        g = pg[pg["k"] == _norm(player)].copy()
        if g.empty:
            return g
        g = g.merge(games[["game_id", "home_team"]], on="game_id", how="left")
        g["home"] = (g["team"] == g["home_team"]).astype(int)
        g["label"] = (g["season"].astype(str).str[-2:] + " W" + g["week"].astype(str) + " "
                      + g["home"].map({1: "vs", 0: "@"}) + " " + g["opp"])
    return g.sort_values("date").reset_index(drop=True)


def opponent_of(sport: str, team: str, game: str) -> str:
    """'vs LAL' / '@ PHO' (NBA) or 'MIN @ GB' (NFL) -> the opponent's code."""
    parts = [p for p in str(game).replace("vs", " ").replace("@", " ").split() if p]
    others = [p for p in parts if p != team]
    return others[-1] if others else ""


def hit_rates(g: pd.DataFrame, stat: str, line: float, pick: str, opp: str) -> pd.DataFrame:
    """Hit rate of the pick (OVER or UNDER at this line) over different windows."""
    if g.empty or stat not in g:
        return pd.DataFrame()
    season = g["season"].max()
    windows = [
        (f"Head to head vs {opp}", g[g["opp"] == opp]),
        ("Last 5", g.tail(5)), ("Last 10", g.tail(10)), ("Last 15", g.tail(15)),
        ("This season", g[g["season"] == season]),
    ]
    rows = []
    for name, w in windows:
        v = w[stat].dropna()
        hits = (v > line) if pick == "OVER" else (v < line)
        rows.append({"Window": name, "Games": len(v), "Hits": int(hits.sum()),
                     "Hit rate": hits.mean() if len(v) else None,
                     "Average": v.mean() if len(v) else None,
                     "Pushes": int((v == line).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------- compare book lines
def model_chances(sport: str, league, player: str, stat: str, lines: list[float]) -> dict:
    """Model's P(over) at each line for one player's next game. league = slate.League
    (NBA) or nfl_slate.NFLLeague (NFL), passed in so it's only loaded once."""
    if not lines:
        return {}
    pid = league.player_id(player)
    if pid is None:
        return {}
    if sport == "NBA":
        from . import slate
        reqs = slate.build_requests(league, pd.DataFrame(
            [{"player": player, "stat": stat, "line": ln, "date": pd.Timestamp.today().normalize()}
             for ln in lines]))
        if reqs.empty:
            return {}
        sc = league.predict(reqs)
    else:
        rows = [league.request_row(pid, stat, ln) for ln in lines]
        rows = [r for r in rows if r]
        if not rows:
            return {}
        sc = nm.predict_batch(league.pg, league.ctx, pd.DataFrame(rows))
    return dict(zip(sc["line"], sc["prob_over"]))


def compare_books(sport: str, league, player: str, stat: str) -> pd.DataFrame:
    """Every book's line for this prop + the model's chance and edge at that line."""
    from . import odds, scraped_lines
    bl = scraped_lines.player_lines(sport, player, stat)
    if bl.empty:
        return bl
    ch = model_chances(sport, league, player, stat, sorted(bl["line"].unique()))
    bl = bl.assign(model_over=bl["line"].map(ch))
    io = bl["over_odds"].map(lambda x: odds.implied_prob(x) or 0.524)
    iu = bl["under_odds"].map(lambda x: odds.implied_prob(x) or 0.524)
    bl["edge_over"] = bl["model_over"] - io
    bl["edge_under"] = (1 - bl["model_over"]) - iu
    bl["best_side"] = np.where(bl["edge_over"] >= bl["edge_under"], "OVER", "UNDER")
    bl["best_edge"] = bl[["edge_over", "edge_under"]].max(axis=1)
    return bl[["book", "line", "over_odds", "under_odds", "model_over", "best_side", "best_edge"]]


# --------------------------------------------------- try a different line
def rescore_pick(sport: str, league, r: dict, line: float, over_odds=None, under_odds=None):
    """Re-run the model (and injury adjustment) for this pick at a line you type in.
    Returns (updated pick dict, model row) or (None, None) if it can't be scored."""
    from . import explain, odds
    row = explain.model_row(sport, league, {**r, "line": float(line)})
    if row is None or pd.isna(row.get("prob_over")):
        return None, None
    p = float(row["prob_over"])
    new = dict(r)
    custom = float(line) != float(r["line"]) or over_odds is not None or under_odds is not None
    new.update({
        "line": float(line), "prob_over": p,
        "prob_over_raw": row.get("prob_over_raw"), "prob_over_model": row.get("prob_over_model"),
        "proj": row.get("proj"), "pred_min": row.get("pred_min"),
        "hit_rate_last10_at_line": row.get("hit_rate_last10_at_line"),
        "hit_rate_last5_at_line": row.get("hit_rate_last5_at_line"),
        "line_type": "custom" if custom else r.get("line_type"),
        "over_odds": over_odds if over_odds is not None else (r.get("over_odds") if not custom else None),
        "under_odds": under_odds if under_odds is not None else (r.get("under_odds") if not custom else None),
        "injury_note": None,
    })
    io = odds.implied_prob(new["over_odds"]) or 0.524
    iu = odds.implied_prob(new["under_odds"]) or 0.524
    new["edge_over"], new["edge_under"] = p - io, (1 - p) - iu
    new["pick"] = "OVER" if p >= 0.5 else "UNDER"
    new["confidence"] = p if p >= 0.5 else 1 - p
    new["tested_hit"] = tested_hit_rate(sport, r["stat"], p, new["pick"])
    return new, row
