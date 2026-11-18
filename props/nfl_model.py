"""NFL over/under model, one per stat (passing / rushing / receiving yards, TDs, ...).

Features (all known before kickoff):
  weather      temperature, wind, gusts, rain, snow, indoors, turf, cold/windy/wet flags
  Vegas        game total, team spread, team implied points (game script: trailing
               teams pass more, leading teams run more)
  game         home/away, rest days, short week, division game, prime time
  player       season / last-3 / last-5 averages, consistency, snap share, targets,
               target share, air-yards share, carries, pass attempts, yards per opportunity
  team         pass attempts and rush attempts per game, points per game
  defense      what the opponent allows to the player's position (season + last 4 games)
  teammates    targets / carries of injured teammates, starting QB out
  the line     and how often the player cleared it in his last 5 games

Real sportsbook lines are used when data/nfl_props/line_history.csv has them;
otherwise estimated lines (plus copies above/below so the model learns line distance).
"""
from __future__ import annotations

import bisect
import json
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from . import nfl_data as nd

POS_CODE = {p: i for i, p in enumerate(nd.POSITIONS)}
WEATHER = ["game_temp", "game_wind", "wx_gust", "wx_precip", "wx_snow", "indoor", "turf",
           "cold", "windy", "wet"]
FEATURES = WEATHER + [
    "total_line", "team_spread", "implied_pts", "opp_implied_pts",
    "home", "rest", "short_week", "div_game", "primetime",
    "avg_season", "avg_last3", "avg_last5", "std_last5", "games_played",
    "snap_last3", "targets_last3", "tshare_last3", "airshare_last3", "carries_last3",
    "attempts_last3", "per_opp_last5",
    "team_pass_pg", "team_rush_pg", "team_ppg", "opp_papg",
    "dvp_factor", "dvp_recent",
    "miss_targets", "miss_carries", "qb_out", "opp_miss_n",
    "line", "line_minus_avg5", "line_minus_season", "hit_rate_last5_at_line",
    "pos_code",
]
MISS_COLS = ["miss_targets", "miss_carries", "qb_out", "opp_miss_n"]
LAGS = [f"_lag{k}" for k in range(1, 6)]
LINE_OFFSETS = (-1.5, -0.75, 0.75, 1.5)
# which "opportunity" drives each stat (for yards-per-opportunity)
OPPORTUNITY = {"passing_yards": "attempts", "passing_tds": "attempts", "completions": "attempts",
               "attempts": "attempts", "passing_interceptions": "attempts",
               "rushing_yards": "carries", "carries": "carries", "receiving_yards": "targets",
               "receptions": "targets", "rush_rec_yds": "touches", "anytime_td": "touches"}


def _asof(left, right, by, cols):
    left = left.sort_values("date")
    right = right.sort_values("date")[by + ["date"] + cols]
    return pd.merge_asof(left, right, on="date", by=by, direction="backward",
                         allow_exact_matches=False)


# ------------------------------------------------------------- context
class Context:
    """League-wide tables built once: game info, team rates, missing teammates."""

    def __init__(self, pg: pd.DataFrame, games: pd.DataFrame):
        self.games = games
        played = pg[pg["offense_snaps"].fillna(1) > 0]
        tg = played.groupby(["season", "team", "game_id", "date"]).agg(
            pass_att=("attempts", "sum"), rush_att=("carries", "sum")).reset_index()
        sc = games[["game_id", "home_team", "away_team", "home_score", "away_score"]]
        tg = tg.merge(sc, on="game_id", how="left")
        home = tg["team"] == tg["home_team"]
        tg["pts"] = np.where(home, tg["home_score"], tg["away_score"])
        tg["pts_allowed"] = np.where(home, tg["away_score"], tg["home_score"])
        tg = tg.sort_values("date")
        g = tg.groupby(["season", "team"])
        n = g.cumcount() + 1
        tg["team_pass_pg"] = g["pass_att"].cumsum() / n
        tg["team_rush_pg"] = g["rush_att"].cumsum() / n
        tg["team_ppg"] = g["pts"].cumsum() / n
        tg["papg"] = g["pts_allowed"].cumsum() / n
        self.team_rates = tg[["season", "team", "date", "team_pass_pg", "team_rush_pg", "team_ppg", "papg"]]
        self.missing = missing_teammates(played)


def missing_teammates(played: pd.DataFrame) -> pd.DataFrame:
    """For each past team-game: targets & carries per game of regulars who didn't play,
    and whether the team's starting QB was out."""
    played = played.sort_values("date")
    hist = {pid: (list(g["date"]), list(g["team"])) for pid, g in played.groupby("player_id")}

    def team_before(pid, d):
        dates, teams = hist[pid]
        i = bisect.bisect_left(dates, d) - 1
        return teams[i] if i >= 0 else None

    rows = []
    for (season, team), g in played.groupby(["season", "team"], sort=False):
        dates = sorted(g["date"].unique())
        by_date = {d: x for d, x in g.groupby("date")}
        st = defaultdict(lambda: {"n": 0, "tgt": 0.0, "car": 0.0, "att": 0.0, "last": -99, "pos": ""})
        for i, d in enumerate(dates):
            present = set(by_date[d]["player_id"])
            mt = mc = 0.0
            nmiss = 0
            # starting QB = most pass attempts per game so far
            qbs = [(s["att"] / s["n"], pid) for pid, s in st.items() if s["pos"] == "QB" and s["n"] >= 1]
            qb_out = 0
            if qbs:
                avg, qb = max(qbs)
                if avg >= 15 and qb not in present and i - st[qb]["last"] <= 4 and team_before(qb, d) == team:
                    qb_out = 1
            for pid, s in st.items():
                if pid in present or s["n"] < 2 or i - s["last"] > 4:
                    continue
                at, ac = s["tgt"] / s["n"], s["car"] / s["n"]
                if (at >= 4 or ac >= 7) and team_before(pid, d) == team:
                    mt += at
                    mc += ac
                    nmiss += 1
            rows.append({"team": team, "date": d, "miss_targets": mt, "miss_carries": mc,
                         "qb_out": qb_out, "miss_n": nmiss})
            for r in by_date[d].itertuples():
                s = st[r.player_id]
                s["n"] += 1
                s["tgt"] += r.targets
                s["car"] += r.carries
                s["att"] += r.attempts
                s["last"] = i
                s["pos"] = r.position
    return pd.DataFrame(rows).drop_duplicates(["team", "date"])


def missing_from_injuries(pg: pd.DataFrame, injuries: pd.DataFrame, team: str, season: int,
                          week: int) -> dict:
    """Same numbers for an upcoming game, from that week's injury report (Out/Doubtful/IR)."""
    out = {"miss_targets": 0.0, "miss_carries": 0.0, "qb_out": 0, "miss_n": 0, "names": []}
    if injuries is None or injuries.empty:
        return out
    inj = injuries[(injuries["team"] == team) & (injuries["season"] == season)]
    if inj.empty:
        return out
    wk = inj["week"].max() if week is None else min(week, inj["week"].max())
    inj = inj[(inj["week"] == wk) & inj["status"].isin(["Out", "Doubtful"])]
    cur = pg[(pg["season"] == season) & (pg["team"] == team)]
    if cur.empty:
        return out
    avg = cur.groupby("player_id").agg(tgt=("targets", "mean"), car=("carries", "mean"),
                                       att=("attempts", "mean"), n=("week", "size"),
                                       name=("player", "last"), pos=("position", "last"))
    hit = avg[avg.index.isin(inj["player_id"])]
    reg = hit[(hit["n"] >= 2) & ((hit["tgt"] >= 4) | (hit["car"] >= 7))]
    out["miss_targets"] = float(reg["tgt"].sum())
    out["miss_carries"] = float(reg["car"].sum())
    out["miss_n"] = int(len(reg))
    qbs = avg[avg["pos"] == "QB"]
    if not qbs.empty and qbs["att"].max() >= 15 and qbs["att"].idxmax() in set(inj["player_id"]):
        out["qb_out"] = 1
    out["names"] = list(hit.sort_values("tgt", ascending=False)["name"])
    return out


# ------------------------------------------------------------- features
def build_features(pg: pd.DataFrame, stat: str, ctx: Context, lines: pd.DataFrame | None = None,
                   overrides: pd.DataFrame | None = None, league: pd.DataFrame | None = None) -> pd.DataFrame:
    games = ctx.games
    df = pg.copy()
    df = df.merge(games.drop(columns=["season", "week", "date"]), on="game_id", how="left")
    df["home"] = (df["team"] == df["home_team"]).astype(int)
    df["team_spread"] = np.where(df["home"] == 1, df["spread_line"], -df["spread_line"])
    df["implied_pts"] = df["total_line"] / 2 + df["team_spread"] / 2
    df["opp_implied_pts"] = df["total_line"] - df["implied_pts"]
    df["rest"] = np.where(df["home"] == 1, df["home_rest"], df["away_rest"])
    df["short_week"] = (df["rest"] < 6).astype(int)
    df["primetime"] = (df["kick_hour"] >= 19).astype(int)
    df["cold"] = (df["game_temp"] < 35).astype(int)
    df["windy"] = (df["game_wind"] >= 15).astype(int)
    df["wet"] = ((df["wx_precip"].fillna(0) > 0.05) | (df["wx_snow"].fillna(0) > 0.05)).astype(int)

    df["touches"] = df["carries"] + df["receptions"]
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    gp = df.groupby("player_id")
    gs = df.groupby(["player_id", "season"])

    def roll(col, n, minp=1, fn="mean"):
        return gp[col].transform(lambda s: getattr(s.shift(1).rolling(n, minp), fn)())

    df["avg_season"] = gs[stat].transform(lambda s: s.shift(1).expanding().mean())
    df["avg_season"] = df["avg_season"].fillna(roll(stat, 8))
    df["avg_last3"] = roll(stat, 3)
    df["avg_last5"] = roll(stat, 5)
    df["std_last5"] = roll(stat, 5, 2, "std")
    df["games_played"] = gs.cumcount()
    df["snap_last3"] = roll("offense_pct", 3)
    df["targets_last3"] = roll("targets", 3)
    df["tshare_last3"] = roll("target_share", 3)
    df["airshare_last3"] = roll("air_yards_share", 3)
    df["carries_last3"] = roll("carries", 3)
    df["attempts_last3"] = roll("attempts", 3)
    opp_col = OPPORTUNITY[stat]
    df["_per_opp"] = df[stat] / df[opp_col].replace(0, np.nan)
    df["per_opp_last5"] = roll("_per_opp", 5)

    # team rates to date
    tr = ctx.team_rates
    df = _asof(df, tr[["season", "team", "date", "team_pass_pg", "team_rush_pg", "team_ppg"]],
               ["season", "team"], ["team_pass_pg", "team_rush_pg", "team_ppg"])
    df = _asof(df, tr[["season", "team", "date", "papg"]].rename(columns={"team": "opp", "papg": "opp_papg"}),
               ["season", "opp"], ["opp_papg"])

    # defense vs position, per game (season to date + last 4 games)
    lsrc = league if league is not None else df
    base = lsrc[lsrc[stat].notna()]
    tot = base.groupby(["season", "opp", "position", "date"])[stat].sum().reset_index().sort_values("date")
    g = tot.groupby(["season", "opp", "position"])
    tot["o_cum"], tot["o_n"] = g[stat].cumsum(), g.cumcount() + 1
    tot["o_rec"] = g[stat].transform(lambda s: s.rolling(4, 1).mean())
    lg = tot.groupby(["season", "position", "date"])[stat].agg(["sum", "size"]).reset_index().sort_values("date")
    gl = lg.groupby(["season", "position"])
    lg["l_cum"], lg["l_n"] = gl["sum"].cumsum(), gl["size"].cumsum()
    df = _asof(df, tot, ["season", "opp", "position"], ["o_cum", "o_n", "o_rec"])
    df = _asof(df, lg, ["season", "position"], ["l_cum", "l_n"])
    lg_pg = df["l_cum"] / df["l_n"]
    w = (df["o_n"].fillna(0) / 5).clip(upper=1)
    df["dvp_factor"] = (1 + w * ((df["o_cum"] / df["o_n"]) / lg_pg - 1)).fillna(1.0)
    df["dvp_recent"] = (1 + w * (df["o_rec"] / lg_pg - 1)).fillna(1.0)

    # missing teammates
    miss = ctx.missing
    df = df.merge(miss.drop(columns="miss_n"), on=["team", "date"], how="left")
    df = df.merge(miss[["team", "date", "miss_n"]].rename(columns={"team": "opp", "miss_n": "opp_miss_n"}),
                  on=["opp", "date"], how="left")
    if overrides is not None and not overrides.empty:
        ov = overrides[["player_id", "date"] + MISS_COLS].rename(columns={c: c + "_o" for c in MISS_COLS})
        df = df.merge(ov, on=["player_id", "date"], how="left")
        for c in MISS_COLS:
            df[c] = df[c].fillna(df[c + "_o"])
        df = df.drop(columns=[c + "_o" for c in MISS_COLS])
    up = df[stat].isna()
    df.loc[up, MISS_COLS] = df.loc[up, MISS_COLS].fillna(0)

    # the line
    ref = df["avg_last5"].fillna(df["avg_season"])
    df["line"] = 0.5 if stat == "anytime_td" else (np.floor(ref.fillna(0)) + 0.5).clip(lower=0.5)
    df["line_source"] = "estimated"
    if lines is not None and not lines.empty:
        real = (lines[lines["stat"] == stat].assign(date=lambda x: pd.to_datetime(x["date"]))
                .groupby(["player_id", "date"])["line"].median().reset_index().rename(columns={"line": "real"}))
        df = df.merge(real, on=["player_id", "date"], how="left")
        has = df["real"].notna()
        df.loc[has, "line"] = df.loc[has, "real"]
        df.loc[has, "line_source"] = "real"
        df = df.drop(columns="real")
    df["pos_code"] = df["position"].map(POS_CODE)
    df = set_line_features(df, stat)
    df["target_over"] = np.where(df[stat].notna(), (df[stat] > df["line"]).astype(float), np.nan)
    return df


def set_line_features(df, stat):
    if LAGS[0] not in df:
        df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
        g = df.groupby("player_id")[stat]
        for k, c in enumerate(LAGS, 1):
            df[c] = g.shift(k)
    df["line_minus_avg5"] = df["line"] - df["avg_last5"]
    df["line_minus_season"] = df["line"] - df["avg_season"]
    lags = df[LAGS].to_numpy(dtype=float)
    valid = ~np.isnan(lags)
    n = valid.sum(axis=1)
    over = ((lags > df["line"].to_numpy()[:, None]) & valid).sum(axis=1)
    df["hit_rate_last5_at_line"] = np.where(n > 0, over / np.maximum(n, 1), np.nan)
    return df


def eligible(df: pd.DataFrame, stat: str) -> pd.Series:
    """Rows where a sportsbook would actually post this prop (real role in the offense)."""
    pos_ok = df["position"].isin(nd.STATS[stat][1])
    if stat in ("passing_yards", "passing_tds", "completions", "attempts", "passing_interceptions"):
        role = df["attempts_last3"].fillna(0) >= 15
    elif stat in ("rushing_yards", "carries"):
        role = df["carries_last3"].fillna(0) >= (3 if df is not None else 3)
    elif stat in ("receiving_yards", "receptions"):
        role = df["targets_last3"].fillna(0) >= 2.5
    else:
        role = (df["targets_last3"].fillna(0) + df["carries_last3"].fillna(0)) >= 4
    return pos_ok & role & (df["games_played"] >= 2)


# -------------------------------------------------------------- training
def _clf():
    cat = [f == "pos_code" for f in FEATURES]
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=24,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          categorical_features=cat, random_state=42)


def augment(data, stat):
    if stat == "anytime_td":
        return data
    est = data[data["line_source"] == "estimated"]
    spread = est["std_last5"].fillna(est["avg_last5"].abs() ** 0.5).clip(lower=0.5)
    copies = [data]
    for k in LINE_OFFSETS:
        c = est.copy()
        c["line"] = (np.floor(c["line"] + k * spread) + 0.5).clip(lower=0.5)
        copies.append(c)
    out = set_line_features(pd.concat(copies, ignore_index=True), stat)
    out["target_over"] = (out[stat] > out["line"]).astype(float)
    return out


# ------------------------------------------------------------ calibration
def fit_calibrator(p, y, dates=None):
    """Platt scaling: a logistic fit on the raw model's log-odds. It smoothly pulls
    overconfident predictions toward what actually happened, and extrapolates
    sensibly at the extremes (where few test games exist).
    Returns (calibrator fit on all test rows, honestly cross-fitted calibrated probs)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    h = len(p) // 2                      # test rows are in date order: first half / second half
    pc = np.empty_like(p)
    pc[h:] = PlattCalibrator().fit(p[:h], y[:h]).predict(p[h:])
    pc[:h] = PlattCalibrator().fit(p[h:], y[h:]).predict(p[:h])
    return PlattCalibrator().fit(p, y), pc


class PlattCalibrator:
    """Stores only two plain numbers (slope, intercept), so saved models work with any
    scikit-learn version: P(over) = 1 / (1 + exp(-(a * logit(raw) + b)))."""

    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6).fit(self._logit(p), y)
        self.a, self.b = float(lr.coef_[0][0]), float(lr.intercept_[0])
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict(self, p):
        z = self.a * self._logit(p)[:, 0] + self.b
        return np.clip(1 / (1 + np.exp(-z)), 0.02, 0.98)

    @property
    def slope(self):
        return self.a


def _calibration(y, p, bins=10):
    out = []
    for lo in np.linspace(0, 0.9, bins):
        m = (p >= lo) & (p < lo + 0.1)
        if m.sum() >= 15:
            out.append({"predicted": float(p[m].mean()), "actual": float(y[m].mean()), "n": int(m.sum())})
    return out


def train(pg, games, stat, ctx=None, lines=None, test_fraction=0.15) -> dict:
    ctx = ctx or Context(pg, games)
    f = build_features(pg, stat, ctx, lines)
    data = f[eligible(f, stat) & f["target_over"].notna() & f["avg_last5"].notna()].sort_values("date")
    if len(data) < 300:
        raise RuntimeError(f"only {len(data)} rows for {stat}")
    cutoff = data["date"].quantile(1 - test_fraction)
    tr, te = data[data["date"] < cutoff], data[data["date"] >= cutoff]
    tr_a = augment(tr, stat)
    clf = _clf().fit(tr_a[FEATURES], tr_a["target_over"])
    raw = clf.predict_proba(te[FEATURES])[:, 1]
    y = te["target_over"].to_numpy()
    cal, p = fit_calibrator(raw, y)
    joblib.dump(cal, nd.MODELS_DIR / f"over_{stat}_cal.joblib")
    conf = (p >= 0.6) | (p <= 0.4)
    m = {"stat": stat, "train_games": int(len(tr)), "test_games": int(len(te)),
         "test_from": str(cutoff.date()), "accuracy": float(accuracy_score(y, p > 0.5)),
         "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
         "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
         "brier": float(brier_score_loss(y, p)),
         "baseline_brier": float(brier_score_loss(y, np.full(len(y), tr["target_over"].mean()))),
         "confident_picks": int(conf.sum()),
         "confident_accuracy": float(accuracy_score(y[conf], p[conf] > 0.5)) if conf.any() else None,
         "real_line_rows": int((data["line_source"] == "real").sum()), "calibration": _calibration(y, p),
         "raw_brier": float(brier_score_loss(y, raw)), "raw_calibration": _calibration(y, raw)}
    # weather check: accuracy in bad-weather test games
    bad = ((te["windy"] == 1) | (te["wet"] == 1) | (te["cold"] == 1)).to_numpy()
    m["bad_weather_test_games"] = int(bad.sum())
    m["bad_weather_accuracy"] = float(accuracy_score(y[bad], p[bad] > 0.5)) if bad.sum() >= 20 else None
    final = _clf().fit(augment(data, stat)[FEATURES], augment(data, stat)["target_over"])
    joblib.dump(final, nd.MODELS_DIR / f"over_{stat}.joblib")
    (nd.MODELS_DIR / f"over_{stat}_metrics.json").write_text(json.dumps(m, indent=2))
    return m


def train_all(pg, games, stats=None, lines=None, log=print) -> dict:
    ctx = Context(pg, games)
    out = {}
    for s in stats or list(nd.STATS):
        log(f"Training {nd.LABEL[s]} ...")
        out[s] = train(pg, games, s, ctx, lines)
    return out


def load_metrics(stat):
    p = nd.MODELS_DIR / f"over_{stat}_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def model_exists(stat):
    return (nd.MODELS_DIR / f"over_{stat}.joblib").exists()


_cache = {}


def calibrate(stat, raw):
    p = nd.MODELS_DIR / f"over_{stat}_cal.joblib"
    if not p.exists():
        return raw
    key = ("cal", stat, p.stat().st_mtime)
    if key not in _cache:
        _cache[key] = joblib.load(p)
    return _cache[key].predict(np.asarray(raw, float))


def _load(stat):
    p = nd.MODELS_DIR / f"over_{stat}.joblib"
    key = (stat, p.stat().st_mtime)
    if key not in _cache:
        _cache[key] = joblib.load(p)
    return _cache[key]


# ------------------------------------------------------------- prediction
def predict_batch(pg, ctx: Context, req: pd.DataFrame) -> pd.DataFrame:
    """req columns: player_id, stat, line, game_id, team  (+ optional MISS_COLS)."""
    req = req.copy().reset_index(drop=True)
    req["_rid"] = np.arange(len(req))
    games = ctx.games.set_index("game_id")
    last = pg.sort_values("date").groupby("player_id").tail(1).set_index("player_id")
    results = []
    for stat, grp in req.groupby("stat"):
        fut = grp.drop_duplicates(["player_id", "game_id"])
        fut = fut[fut["player_id"].isin(last.index) & fut["game_id"].isin(games.index)]
        if fut.empty:
            continue
        gi = games.loc[fut["game_id"]]
        lr = last.loc[fut["player_id"]]
        team = fut["team"].fillna(pd.Series(lr["team"].values, index=fut.index)).values
        future = pd.DataFrame({
            "player_id": fut["player_id"].values, "player": lr["player"].values,
            "position": lr["position"].values, "season": gi["season"].values, "week": gi["week"].values,
            "game_id": fut["game_id"].values, "team": team,
            "opp": np.where(gi["home_team"].values == team, gi["away_team"].values, gi["home_team"].values),
            "date": gi["date"].values})
        mine = pg[pg["player_id"].isin(future["player_id"])]
        ov = fut[["player_id"] + MISS_COLS].assign(date=gi["date"].values) if all(c in fut for c in MISS_COLS) else None
        feats = build_features(pd.concat([mine, future], ignore_index=True), stat, ctx,
                               overrides=ov, league=pg)
        fr = feats[feats[stat].isna()].merge(future[["player_id", "game_id"]], on=["player_id", "game_id"])
        keep = [c for c in fr.columns if c not in ("line", "line_source", "team", "player")]
        m = grp[["_rid", "player_id", "game_id", "line"]].merge(fr[keep], on=["player_id", "game_id"], how="left")
        recent = {p: s.dropna().tail(5).to_numpy() for p, s in
                  pg[pg["player_id"].isin(m["player_id"])].sort_values("date").groupby("player_id")[stat]}
        m["line_minus_avg5"] = m["line"] - m["avg_last5"]
        m["line_minus_season"] = m["line"] - m["avg_season"]
        m["hit_rate_last5_at_line"] = [float((recent[p] > ln).mean()) if p in recent and len(recent[p]) else np.nan
                                       for p, ln in zip(m["player_id"], m["line"])]
        m["proj"] = m["avg_last5"] * 0.5 + m["avg_season"] * 0.5
        m["proj"] = m["proj"] * (1 + 0.5 * (m["dvp_factor"] - 1))
        m["prob_over"] = np.nan
        ok = m["avg_last5"].notna()
        if model_exists(stat) and ok.any():
            raw = _load(stat).predict_proba(m.loc[ok, FEATURES])[:, 1]
            m.loc[ok, "prob_over_raw"] = raw
            m.loc[ok, "prob_over"] = calibrate(stat, raw)
        results.append(m.drop(columns=["player_id", "game_id", "line"]))
    if not results:
        return req.drop(columns="_rid").assign(prob_over=np.nan)
    sc = pd.concat(results, ignore_index=True)
    out = req.drop(columns=[c for c in MISS_COLS if c in req]).merge(sc, on="_rid", how="left")
    return out.sort_values("_rid").drop(columns="_rid").reset_index(drop=True)


# ------------------------------------------------------- weather impact
def weather_impact(pg: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """League-wide: how each stat changes by weather bucket (per regular, vs normal)."""
    df = pg.merge(games[["game_id", "game_temp", "game_wind", "wx_precip", "wx_snow", "indoor"]],
                  on="game_id", how="left")
    df = df[(df["offense_pct"].fillna(0) >= 0.5)]
    cond = pd.Series("Normal outdoors", index=df.index)
    cond[df["indoor"] == 1] = "Indoors"
    cond[(df["indoor"] == 0) & (df["game_temp"] < 32)] = "Freezing (<32°F)"
    cond[(df["indoor"] == 0) & (df["game_wind"] >= 15)] = "Windy (15+ mph)"
    cond[(df["indoor"] == 0) & ((df["wx_precip"] > 0.05) | (df["wx_snow"] > 0.05))] = "Rain / snow"
    df["condition"] = cond
    rows = []
    for stat, (label, poss) in nd.STATS.items():
        sub = df[df["position"].isin(poss)]
        base = sub.loc[sub["condition"] == "Normal outdoors", stat].mean()
        for c, g in sub.groupby("condition"):
            rows.append({"stat": stat, "label": label, "condition": c, "avg": g[stat].mean(),
                         "vs_normal": g[stat].mean() / base - 1 if base else np.nan, "games": len(g)})
    return pd.DataFrame(rows)
