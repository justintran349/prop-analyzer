"""Machine-learning model: P(player goes OVER the line) for each stat.

How it works
------------
1. Every past player-game becomes one training row.
2. Features only use information known BEFORE tip-off (no peeking):
     player form     recent / season averages, consistency, minutes, usage, shots
     minutes model   a separate model's projected minutes for tonight
     game context    home/away, rest, back-to-back, projected game total & margin
     teammates out   minutes/points/assists/rebounds of rotation players missing
     defense         opponent vs the player's position (season + last 15 games),
                     opponent overall, opponent's own missing players
     history         the player's results vs this specific opponent
     the line        and how often the player cleared it in his last 10
3. Target: did the stat go over the line? Real sportsbook lines are used when
   data/line_history.csv has them, otherwise an estimated book line.
4. Gradient-boosted trees (scikit-learn), one per stat, tested on the most
   recent games it never saw.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             mean_absolute_error, roc_auc_score)
from sklearn.model_selection import KFold

from . import config, context
from .analysis import POSITIONS

TEAM_CODES = sorted([
    "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU",
    "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL",
    "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"])

MIN_FEATURES = [
    "min_season", "min_last5", "min_last10", "started_rate", "games_played",
    "rest_days", "b2b", "home", "miss_min", "miss_n", "abs_proj_margin", "proj_total",
]
FEATURES = [
    "home", "rest_days", "b2b", "started_rate",
    "avg_season", "avg_last5", "avg_last10", "std_last10",
    "min_season", "min_last5", "pred_min", "per_min_last10",
    "usg_last10", "fga_last10",
    "line", "line_minus_avg10", "line_minus_season", "line_minus_proj",
    "hit_rate_last10_at_line",
    "dvp_factor", "dvp_recent", "opp_overall_factor",
    "proj_total", "proj_margin", "abs_proj_margin",
    "miss_min", "miss_pts", "miss_ast", "miss_trb", "miss_n", "opp_miss_min",
    "vs_opp_games", "vs_opp_diff",
    "pos_code", "opp_code", "team_code",
]
CATEGORICAL = ["pos_code", "opp_code", "team_code"]
MISS_COLS = ["miss_min", "miss_pts", "miss_ast", "miss_trb", "miss_n", "opp_miss_min"]
RECENT_DVP_DATES = 15
HOME_EDGE = 2.5


def _code(values: pd.Series, vocab: list[str]) -> pd.Series:
    m = {v: i for i, v in enumerate(vocab)}
    return values.map(m).astype(float)


def _asof(left, right, by, cols):
    """Attach the latest `right` row strictly BEFORE each left date."""
    left = left.sort_values("date")
    right = right.sort_values("date")[by + ["date"] + cols]
    return pd.merge_asof(left, right, on="date", by=by, direction="backward",
                         allow_exact_matches=False)


# ---------------------------------------------------------------- features
class Context:
    """League-wide tables computed once and reused for every stat."""

    def __init__(self, logs: pd.DataFrame):
        played = logs[logs["min"].notna() & (logs["min"] > 0)]
        self.tg = context.team_game_table(played)
        self.scoring = context.team_scoring_to_date(self.tg)
        self.missing = context.missing_teammates(played)
        self.usage = context.add_usage(played, self.tg)[["player_id", "date", "usg"]]
        self.missing = self.missing.drop_duplicates(["team", "date"])


def build_features(logs: pd.DataFrame, stat: str, lines: pd.DataFrame | None = None,
                   ctx: Context | None = None, overrides: pd.DataFrame | None = None,
                   league: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per player-game with pre-game features + target.

    Rows with a NaN stat are upcoming games: they get features from history.
    `overrides` (player_id, date, miss_*, opp_miss_min) supplies injury-report
    values for upcoming games. `league` = all players' games (for defense
    tables) when `logs` holds only a few players - makes scoring much faster."""
    ctx = ctx or Context(logs if league is None else league)
    df = logs.copy()
    df = df[(df["min"] >= config.MIN_MINUTES) | df[stat].isna()]
    df = df.merge(ctx.usage, on=["player_id", "date"], how="left")
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)

    gp = df.groupby("player_id")
    gs = df.groupby(["player_id", "season"])

    def roll(col, n, minp=1, fn="mean"):
        return gp[col].transform(lambda s: getattr(s.shift(1).rolling(n, minp), fn)())

    df["avg_season"] = gs[stat].transform(lambda s: s.shift(1).expanding().mean())
    df["avg_season"] = df["avg_season"].fillna(
        gp[stat].transform(lambda s: s.shift(1).expanding().mean()))
    df["avg_last5"] = roll(stat, 5)
    df["avg_last10"] = roll(stat, 10)
    df["std_last10"] = roll(stat, 10, 3, "std")
    df["min_season"] = gs["min"].transform(lambda s: s.shift(1).expanding().mean())
    df["min_last5"] = roll("min", 5)
    df["min_last10"] = roll("min", 10)
    df["min_season"] = df["min_season"].fillna(df["min_last10"])
    df["games_played"] = gs.cumcount()
    df["per_min"] = df[stat] / df["min"]
    df["per_min_last10"] = roll("per_min", 10)
    df["usg_last10"] = roll("usg", 10)
    df["fga_last10"] = roll("fga", 10)

    # NBA.com data has no "started" flag - estimate it from minutes (24+ = starter)
    started = df["started"] if "started" in df else pd.Series(np.nan, index=df.index)
    df["started"] = started.where(started.notna(), (df["min"] >= 24).astype(float))
    df["started_rate"] = roll("started", 10)

    df["rest_days"] = gp["date"].diff().dt.days.clip(upper=7)
    df["b2b"] = (df["rest_days"] == 1).astype(int)

    go = df.groupby(["player_id", "opp"])[stat]
    df["vs_opp_games"] = go.cumcount()
    df["vs_opp_diff"] = go.transform(lambda s: s.shift(1).expanding().mean()) - df["avg_season"]

    # ---- defense vs position: season-to-date and last 15 game dates
    lsrc = league if league is not None else df
    base = lsrc[(lsrc["min"] >= config.MIN_MINUTES) & lsrc[stat].notna()]
    daily = (base.groupby(["season", "opp", "primary_pos", "date"])[[stat, "min"]]
             .sum().reset_index().sort_values("date"))
    g = daily.groupby(["season", "opp", "primary_pos"])
    daily["o_cum"], daily["o_min"] = g[stat].cumsum(), g["min"].cumsum()
    daily["o_rec"] = g[stat].transform(lambda s: s.rolling(RECENT_DVP_DATES, 1).sum())
    daily["o_rec_min"] = g["min"].transform(lambda s: s.rolling(RECENT_DVP_DATES, 1).sum())
    lg = (base.groupby(["season", "primary_pos", "date"])[[stat, "min"]]
          .sum().reset_index().sort_values("date"))
    gl = lg.groupby(["season", "primary_pos"])
    lg["l_cum"], lg["l_min"] = gl[stat].cumsum(), gl["min"].cumsum()
    df = _asof(df, daily, ["season", "opp", "primary_pos"], ["o_cum", "o_min", "o_rec", "o_rec_min"])
    df = _asof(df, lg, ["season", "primary_pos"], ["l_cum", "l_min"])
    lg_rate = df["l_cum"] / df["l_min"]
    w = (df["o_min"].fillna(0) / 400).clip(upper=1)    # trust grows with sample size
    df["dvp_factor"] = 1 + w * ((df["o_cum"] / df["o_min"]) / lg_rate - 1).fillna(0)
    w2 = (df["o_rec_min"].fillna(0) / 250).clip(upper=1)
    df["dvp_recent"] = 1 + w2 * ((df["o_rec"] / df["o_rec_min"]) / lg_rate - 1).fillna(0)
    df[["dvp_factor", "dvp_recent"]] = df[["dvp_factor", "dvp_recent"]].fillna(1.0)

    tot = base.groupby(["season", "opp", "date"])[[stat, "min"]].sum().reset_index().sort_values("date")
    gt = tot.groupby(["season", "opp"])
    tot["t_cum"], tot["t_min"] = gt[stat].cumsum(), gt["min"].cumsum()
    lt = base.groupby(["season", "date"])[[stat, "min"]].sum().reset_index().sort_values("date")
    lt["lt_cum"] = lt.groupby("season")[stat].cumsum()
    lt["lt_min"] = lt.groupby("season")["min"].cumsum()
    df = _asof(df, tot, ["season", "opp"], ["t_cum", "t_min"])
    df = _asof(df, lt, ["season"], ["lt_cum", "lt_min"])
    df["opp_overall_factor"] = ((df["t_cum"] / df["t_min"]) / (df["lt_cum"] / df["lt_min"])).fillna(1.0)

    # ---- pace: projected game total & margin from points for/against to date
    sc = ctx.scoring
    df = _asof(df, sc.rename(columns={"ppg_thru": "t_ppg", "papg_thru": "t_papg"}),
               ["season", "team"], ["t_ppg", "t_papg"])
    df = _asof(df, sc.rename(columns={"team": "opp", "ppg_thru": "o_ppg", "papg_thru": "o_papg"}),
               ["season", "opp"], ["o_ppg", "o_papg"])
    for c in ("t_ppg", "t_papg", "o_ppg", "o_papg"):
        df[c] = df[c].fillna(115.0)
    team_exp = (df["t_ppg"] + df["o_papg"]) / 2
    opp_exp = (df["o_ppg"] + df["t_papg"]) / 2
    df["proj_total"] = team_exp + opp_exp
    df["proj_margin"] = team_exp - opp_exp + np.where(df["home"] == 1, HOME_EDGE, -HOME_EDGE)
    df["abs_proj_margin"] = df["proj_margin"].abs()

    # ---- missing teammates (past: box scores; upcoming: injury-report overrides)
    miss = ctx.missing
    df = df.merge(miss.drop(columns="season"), on=["team", "date"], how="left")
    opp_miss = miss[["team", "date", "miss_min"]].rename(columns={"team": "opp", "miss_min": "opp_miss_min"})
    df = df.merge(opp_miss, on=["opp", "date"], how="left")
    if overrides is not None and not overrides.empty:
        ov = overrides[["player_id", "date"] + MISS_COLS].rename(columns={c: c + "_ovr" for c in MISS_COLS})
        df = df.merge(ov, on=["player_id", "date"], how="left")
        for c in MISS_COLS:
            df[c] = df[c].fillna(df[c + "_ovr"])
        df = df.drop(columns=[c + "_ovr" for c in MISS_COLS])
    upcoming = df[stat].isna()
    df.loc[upcoming, MISS_COLS] = df.loc[upcoming, MISS_COLS].fillna(0)

    # ---- the line: real one when we have it, else the estimated book line
    df["line"] = (np.round(df["avg_last10"].fillna(df["avg_season"]) * 2) / 2 - 0.5).clip(lower=0.5)
    df["line_source"] = "estimated"
    if lines is not None and not lines.empty and "player" in df:
        real = (lines[lines["stat"] == stat].assign(date=lambda x: pd.to_datetime(x["date"]))
                .groupby(["player", "date"])["line"].median().reset_index()
                .rename(columns={"line": "real_line"}))
        df = df.merge(real, on=["player", "date"], how="left")
        has = df["real_line"].notna()
        df.loc[has, "line"] = df.loc[has, "real_line"]
        df.loc[has, "line_source"] = "real"
        df = df.drop(columns="real_line")

    df["pos_code"] = _code(df["primary_pos"], POSITIONS)
    df["opp_code"] = _code(df["opp"], TEAM_CODES)
    df["team_code"] = _code(df["team"], TEAM_CODES)
    df["pred_min"] = np.nan
    df = set_line_features(df, stat)
    df["target_over"] = np.where(df[stat].notna(), (df[stat] > df["line"]).astype(float), np.nan)
    return df


LAGS = [f"_lag{k}" for k in range(1, 11)]


def set_line_features(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Features that depend on the line (recomputed when the line changes).
    The player's previous 10 results are stored once as _lag columns, so copies
    of a row at other lines reuse the same history (no leakage)."""
    if LAGS[0] not in df:
        df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
        g = df.groupby("player_id")[stat]
        for k, c in enumerate(LAGS, start=1):
            df[c] = g.shift(k)
    df["line_minus_avg10"] = df["line"] - df["avg_last10"]
    df["line_minus_season"] = df["line"] - df["avg_season"]
    df["line_minus_proj"] = df["line"] - df["per_min_last10"] * df["pred_min"].fillna(df["min_last10"])
    lags = df[LAGS].to_numpy(dtype=float)
    valid = ~np.isnan(lags)
    over = (lags > df["line"].to_numpy()[:, None]) & valid
    n = valid.sum(axis=1)
    df["hit_rate_last10_at_line"] = np.where(n > 0, over.sum(axis=1) / np.maximum(n, 1), np.nan)
    return df


# ------------------------------------------------------------ minutes model
def _reg():
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05,
                                         min_samples_leaf=40, random_state=42)


def train_minutes(feats: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Fit the minutes model; return out-of-fold predictions for training rows."""
    data = feats.dropna(subset=["min", "min_last10"])
    oof = pd.Series(np.nan, index=data.index)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(data):
        m = _reg().fit(data.iloc[tr][MIN_FEATURES], data.iloc[tr]["min"])
        oof.iloc[te] = m.predict(data.iloc[te][MIN_FEATURES])
    final = _reg().fit(data[MIN_FEATURES], data["min"])
    joblib.dump(final, config.MODELS_DIR / "minutes.joblib")
    metrics = {"mae": float(mean_absolute_error(data["min"], oof)),
               "baseline_mae_last10_avg": float(mean_absolute_error(data["min"], data["min_last10"])),
               "rows": int(len(data))}
    (config.MODELS_DIR / "minutes_metrics.json").write_text(json.dumps(metrics, indent=2))
    return oof, metrics


# ---------------------------------------------------------------- training
def _clf():
    cat_mask = [f in CATEGORICAL for f in FEATURES]
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features=cat_mask, random_state=42)


LINE_OFFSETS = (-1.5, -0.75, 0.75, 1.5)   # in units of the player's std dev


def augment_lines(data: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Teach the model what happens when the line is far from a player's average.

    Rows with estimated lines get copies at lower/higher lines (real sportsbook
    lines are kept as-is). Without this, every training line sits right next to
    the average and the model can't tell a 28.5 line from a 33.5 line."""
    est = data[data["line_source"] == "estimated"]
    copies = [data]
    spread = est["std_last10"].fillna(est["avg_last10"].abs() ** 0.5).clip(lower=0.5)
    for k in LINE_OFFSETS:
        c = est.copy()
        c["line"] = (np.round((c["line"] + k * spread) * 2) / 2)
        c["line"] = np.floor(c["line"]) + 0.5          # books use x.5 lines
        c["line"] = c["line"].clip(lower=0.5)
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


def _calibration(y, p, bins=10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 20:
            out.append({"bin": f"{lo:.0%}-{hi:.0%}", "predicted": float(p[m].mean()),
                        "actual": float(np.asarray(y)[m].mean()), "n": int(m.sum())})
    return out


def train(logs: pd.DataFrame, stat: str, lines: pd.DataFrame | None = None,
          ctx: Context | None = None, pred_min: pd.DataFrame | None = None,
          test_fraction: float = 0.15) -> dict:
    """Train + evaluate on the most recent games (time split, like real life)."""
    ctx = ctx or Context(logs)
    feats = build_features(logs, stat, lines, ctx)
    if pred_min is not None:
        feats = feats.drop(columns="pred_min").merge(pred_min, on=["player_id", "date"], how="left")
        feats = set_line_features(feats, stat)
    data = feats.dropna(subset=["target_over", "avg_last10", "avg_season"])
    data = data[data.groupby("player_id").cumcount() >= 3].sort_values("date")
    if len(data) < 500:
        raise RuntimeError(f"Only {len(data)} usable games - get more data first.")

    cutoff = data["date"].quantile(1 - test_fraction)
    tr, te = data[data["date"] < cutoff], data[data["date"] >= cutoff]
    tr_aug = augment_lines(tr, stat)
    clf = _clf().fit(tr_aug[FEATURES], tr_aug["target_over"])
    raw = clf.predict_proba(te[FEATURES])[:, 1]
    y = te["target_over"].to_numpy()
    cal, p = fit_calibrator(raw, y)          # p = calibrated, measured honestly
    joblib.dump(cal, config.MODELS_DIR / f"over_{stat}_cal.joblib")
    conf = (p >= 0.6) | (p <= 0.4)
    metrics = {
        "stat": stat, "train_games": int(len(tr)), "test_games": int(len(te)),
        "test_from": str(cutoff.date()),
        "accuracy": float(accuracy_score(y, p > 0.5)),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "raw_brier": float(brier_score_loss(y, raw)),
        "raw_calibration": _calibration(y, raw),
        "baseline_brier": float(brier_score_loss(y, np.full(len(y), tr["target_over"].mean()))),
        "real_line_rows": int((data["line_source"] == "real").sum()),
        "confident_picks": int(conf.sum()),
        "confident_accuracy": float(accuracy_score(y[conf], p[conf] > 0.5)) if conf.any() else None,
        "calibration": _calibration(y, p),
    }
    all_aug = augment_lines(data, stat)
    final = _clf().fit(all_aug[FEATURES], all_aug["target_over"])
    joblib.dump(final, config.MODELS_DIR / f"over_{stat}.joblib")
    (config.MODELS_DIR / f"over_{stat}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def train_all(logs: pd.DataFrame, stats=None, lines=None, log=print) -> dict:
    """Minutes model first, then one over/under model per stat."""
    stats = stats or list(config.STATS)
    ctx = Context(logs)
    log("Training minutes model ...")
    base = build_features(logs, "pts", lines, ctx)
    oof, mm = train_minutes(base)
    log(f"  minutes error {mm['mae']:.2f} min/game "
        f"(simple last-10 average: {mm['baseline_mae_last10_avg']:.2f})")
    pred_min = base.loc[oof.index, ["player_id", "date"]].assign(pred_min=oof.values)
    results = {"minutes": mm}
    for stat in stats:
        log(f"Training {config.STATS[stat]} ...")
        results[stat] = train(logs, stat, lines, ctx, pred_min)
    return results


def load_metrics(name: str) -> dict | None:
    p = (config.MODELS_DIR / "minutes_metrics.json" if name == "minutes"
         else config.MODELS_DIR / f"over_{name}_metrics.json")
    return json.loads(p.read_text()) if p.exists() else None


def model_exists(stat: str) -> bool:
    return (config.MODELS_DIR / f"over_{stat}.joblib").exists()


# -------------------------------------------------------------- prediction
_cache: dict = {}


def _load(name):
    path = config.MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise RuntimeError(f"No model '{name}' yet - run: python scripts/train_model.py")
    key = (name, path.stat().st_mtime)
    if key not in _cache:
        _cache[key] = joblib.load(path)
    return _cache[key]


def calibrate(stat: str, raw):
    """Apply the saved calibration (if this model has one)."""
    path = config.MODELS_DIR / f"over_{stat}_cal.joblib"
    if not path.exists():
        return raw
    return _load(f"over_{stat}_cal").predict(np.asarray(raw, float))


def predict_batch(logs: pd.DataFrame, requests_df: pd.DataFrame,
                  ctx: Context | None = None) -> pd.DataFrame:
    """Score many upcoming props at once (features built once per stat).

    requests_df columns: player_id, stat, line, opp, is_home, date, team
      optional: miss_min, miss_pts, miss_ast, miss_trb, miss_n, opp_miss_min
    Returns one row per request with prob_over, pred_min, proj + features."""
    ctx = ctx or Context(logs)
    req = requests_df.copy().reset_index(drop=True)
    req["_rid"] = np.arange(len(req))
    if "team" not in req:
        req["team"] = None
    req["date"] = pd.to_datetime(req["date"]).dt.normalize()
    minutes = _load("minutes") if (config.MODELS_DIR / "minutes.joblib").exists() else None
    last = logs.sort_values("date").groupby("player_id").tail(1).set_index("player_id")
    results = []
    for stat, grp in req.groupby("stat"):
        fut = grp.drop_duplicates(["player_id", "date"])
        fut = fut[fut["player_id"].isin(last.index)]
        if fut.empty:
            continue
        lr = last.loc[fut["player_id"]]
        future = pd.DataFrame({
            "player_id": fut["player_id"].values, "player": lr["player"].values if "player" in lr else None,
            "primary_pos": lr["primary_pos"].values, "season": config.SEASON,
            "date": fut["date"].values,
            "team": fut["team"].fillna(pd.Series(lr["team"].values, index=fut.index)).values,
            "opp": fut["opp"].values, "home": fut["is_home"].astype(int).values})
        overrides = fut[["player_id", "date"] + MISS_COLS] if all(c in fut for c in MISS_COLS) else None
        mine = logs[logs["player_id"].isin(future["player_id"])]
        feats = build_features(pd.concat([mine, future], ignore_index=True), stat,
                               ctx=ctx, overrides=overrides, league=logs)
        fr = feats[feats[stat].isna()].merge(future[["player_id", "date"]], on=["player_id", "date"])
        if minutes is not None and not fr.empty:
            fr["pred_min"] = minutes.predict(fr[MIN_FEATURES])
        keep = [c for c in fr.columns if c not in ("line", "line_source", "stat", "team", "opp", "player",
                                                   "over_odds", "under_odds", "book")]
        m = grp[["_rid", "player_id", "date", "line"]].merge(fr[keep], on=["player_id", "date"], how="left")
        # line-dependent features against the player's real history
        hist = logs[logs["player_id"].isin(m["player_id"])].sort_values("date")
        recent = {pid: s.dropna().tail(10).to_numpy() for pid, s in hist.groupby("player_id")[stat]}
        m["line_minus_avg10"] = m["line"] - m["avg_last10"]
        m["line_minus_season"] = m["line"] - m["avg_season"]
        m["proj"] = m["per_min_last10"] * m["pred_min"].fillna(m["min_last10"])
        m["line_minus_proj"] = m["line"] - m["proj"]
        m["hit_rate_last10_at_line"] = [
            float((recent[p] > ln).mean()) if p in recent and len(recent[p]) else np.nan
            for p, ln in zip(m["player_id"], m["line"])]
        m["prob_over"] = np.nan
        ok = m["avg_season"].notna()
        if model_exists(stat) and ok.any():
            raw = _load(f"over_{stat}").predict_proba(m.loc[ok, FEATURES])[:, 1]
            m.loc[ok, "prob_over_raw"] = raw
            m.loc[ok, "prob_over"] = calibrate(stat, raw)
        results.append(m.drop(columns=["player_id", "date", "line"]))
    if not results:
        return req.drop(columns="_rid").assign(prob_over=np.nan, pred_min=np.nan, proj=np.nan)
    scored = pd.concat(results, ignore_index=True)
    out = req.drop(columns=[c for c in MISS_COLS if c in req]).merge(scored, on="_rid", how="left")
    return out.sort_values("_rid").drop(columns="_rid").reset_index(drop=True)


def predict(logs, player_id, stat, line, opp, is_home, date=None, team=None,
            missing: dict | None = None, ctx: Context | None = None) -> dict:
    """Probability one player goes over `line` in an upcoming game."""
    req = {"player_id": player_id, "stat": stat, "line": float(line), "opp": opp,
           "is_home": bool(is_home), "team": team,
           "date": pd.Timestamp(date or pd.Timestamp.today().normalize())}
    if missing:
        req.update({k: missing.get(k, 0) for k in MISS_COLS})
    r = predict_batch(logs, pd.DataFrame([req]), ctx).iloc[0]
    if pd.isna(r.get("prob_over")):
        raise RuntimeError("Not enough history to score this player.")
    return {"prob_over": float(r["prob_over"]), "pred_min": float(r.get("pred_min", np.nan)),
            "proj": float(r.get("proj", np.nan)), "features": r[FEATURES].to_dict()}
