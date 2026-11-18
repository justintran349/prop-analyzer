"""With / without splits: how a player's minutes, usage and production change this
season when a teammate ahead of him in the pecking order doesn't play.

Example: Austin Reaves is out tonight. In the games Reaves missed this season, how
many minutes did Quentin Grimes play, what share of possessions did he use, and how
much did he produce per minute? That's compared with the games Reaves played, and
the difference adjusts tonight's projection and probability.

The math (shown in plain English on every pick):
  1. minutes and per-minute production, WITH vs WITHOUT the teammate, this season
  2. small samples are shrunk toward "no change":  weight = n / (n + PRIOR_GAMES)
  3. start from his last 10 games (minutes, per-minute rate) and move w of the way to
     his "without" numbers:  target = (min10 + w*(min_wo - min10)) * (rate10 + w*(rate_wo - rate10))
     (comparing to recent form, not the "with" games, avoids double counting when the
      star has already been out for most of the last 10)
  4. boost = target / (his recent form); final projection = model projection x boost,
     so the model's read on rest, matchup and blowout risk is kept
  5. shift = final projection - model projection
  6. chance moves by shift / (his game-to-game spread), on the normal curve
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .context import ROTATION_MIN_GAMES, ROTATION_MIN_MPG

PRIOR_GAMES = 4          # a 4-game split counts half; 12 games counts 75%
MIN_GAMES_WITHOUT = 1    # one game counts only 20% (1 / (1 + 4)), so it barely moves anything
MAX_Z_SHIFT = 0.75       # never move the probability more than ~0.75 standard deviations


class OnOff:
    """Absences for the current season, computed once per League."""

    def __init__(self, logs: pd.DataFrame, usage: pd.DataFrame, season: int):
        cur = logs[(logs["season"] == season) & (logs["min"] > 0)].copy()
        cur = cur.merge(usage, on=["player_id", "date"], how="left")
        self.cur = cur
        self.team_dates = {t: set(g["date"]) for t, g in cur.groupby("team")}
        self.played = {(p, t): set(g["date"]) for (p, t), g in cur.groupby(["player_id", "team"])}
        last_team = cur.sort_values("date").groupby("player_id")["team"].last()
        span = cur.groupby(["player_id", "team"])["date"].agg(["min", "max", "size"])
        self.span = span
        self.last_team = last_team
        tot = cur.groupby(["player_id", "team"]).agg(min=("min", "mean"), usg=("usg", "mean"))
        self.avg = tot

    def season_avg(self, pid, team) -> dict:
        if (pid, team) not in self.avg.index:
            return {}
        r = self.avg.loc[(pid, team)]
        return {"min": float(r["min"]), "usg": float(r["usg"]) if pd.notna(r["usg"]) else np.nan}

    def absent_dates(self, pid, team) -> set:
        """Team games this season he missed while on the roster."""
        if (pid, team) not in self.span.index:
            return set()
        first, last, n = self.span.loc[(pid, team)]
        if n < ROTATION_MIN_GAMES or self.avg.loc[(pid, team), "min"] < ROTATION_MIN_MPG:
            return set()
        still_here = self.last_team.get(pid) == team
        dates = {d for d in self.team_dates.get(team, ()) if d >= first and (still_here or d <= last)}
        return dates - self.played[(pid, team)]

    def ahead_of(self, out_pid, pid, team) -> bool:
        """Is the missing teammate ahead of this player (more minutes or more usage)?"""
        a, b = self.season_avg(out_pid, team), self.season_avg(pid, team)
        if not a or not b:
            return False
        return a["min"] > b["min"] or (pd.notna(a["usg"]) and pd.notna(b["usg"]) and a["usg"] > b["usg"] + 0.01)

    def split(self, pid, team, out_pids: list, stat: str) -> dict | None:
        """His numbers with vs without the given teammates this season."""
        g = self.cur[(self.cur["player_id"] == pid) & (self.cur["team"] == team)]
        if g.empty or stat not in g:
            return None
        absent = [self.absent_dates(o, team) for o in out_pids]
        together = set.intersection(*absent) if absent else set()
        used = list(out_pids)
        if len(together & set(g["date"])) < MIN_GAMES_WITHOUT and len(out_pids) > 1:
            # not enough games with ALL of them out: use the biggest one alone
            biggest = max(out_pids, key=lambda o: self.season_avg(o, team).get("min", 0))
            together, used = self.absent_dates(biggest, team), [biggest]
        wo = g[g["date"].isin(together)]
        wi = g[~g["date"].isin(together)]
        if len(wo) < MIN_GAMES_WITHOUT or len(wi) < 1:
            return None

        def side(x):
            mins = float(x["min"].mean())
            return {"n": int(len(x)), "min": mins, "usg": float(x["usg"].mean()),
                    "rate": float(x[stat].sum() / x["min"].sum()) if x["min"].sum() else np.nan,
                    "per_game": float(x[stat].mean())}

        a, b = side(wi), side(wo)
        w = b["n"] / (b["n"] + PRIOR_GAMES)
        sd = float(g[stat].std()) if len(g) >= 5 else np.nan
        return {"with": a, "without": b, "weight": w, "used": used,
                "sd": sd if pd.notna(sd) and sd > 0 else max(1.0, np.sqrt(max(g[stat].mean(), 0.5)))}


def apply(lg, scored: pd.DataFrame) -> pd.DataFrame:
    """Adjust model projections/probabilities for teammates out tonight (NBA)."""
    if scored.empty or "prob_over" not in scored:
        return scored
    oo = getattr(lg, "onoff", None)
    if oo is None:
        return scored
    out = scored.copy()
    for c in ("onoff_shift", "prob_over_model"):
        out[c] = np.nan
    out["onoff"] = None
    for i, r in out.iterrows():
        team, pid, stat = r.get("team"), r.get("player_id"), r.get("stat")
        names = lg.missing(team)["names"] if team else []
        out_pids = [p for p in (lg.player_id(n) for n in names) if p and p != pid]
        out_pids = [p for p in out_pids if oo.ahead_of(p, pid, team)]
        if not out_pids or pd.isna(r.get("prob_over")):
            continue
        s = oo.split(pid, team, out_pids, stat)
        if s is None:
            continue
        recent_rate, recent_min = r.get("per_min_last10"), r.get("min_last10")
        model_proj = r.get("proj")
        if pd.isna(recent_rate) or pd.isna(recent_min) or pd.isna(model_proj):
            continue
        # move his recent minutes and per-minute rate toward his "without" numbers, by weight w
        w, b = s["weight"], s["without"]
        min_t = recent_min + w * (b["min"] - recent_min)
        rate_t = recent_rate + w * (b["rate"] - recent_rate) if pd.notna(b["rate"]) else recent_rate
        boost = (min_t * rate_t) / (recent_rate * recent_min) if recent_rate * recent_min > 0 else 1.0
        target = model_proj * boost      # keep the model's read on rest, matchup, blowout risk
        s = {**s, "min_t": min_t, "rate_t": rate_t, "boost": boost}
        shift = target - model_proj
        z = float(np.clip(shift / s["sd"], -MAX_Z_SHIFT, MAX_Z_SHIFT))
        p0 = float(np.clip(r["prob_over"], 0.02, 0.98))
        p1 = float(np.clip(norm.cdf(norm.ppf(p0) + z), 0.02, 0.98))
        out.at[i, "prob_over_model"] = p0
        out.at[i, "prob_over"] = p1
        out.at[i, "onoff_shift"] = shift
        out.at[i, "onoff"] = {**s, "names": [lg.current.loc[p, "player"] if p in lg.current.index else p
                                             for p in s["used"]],
                              "target": target, "model_proj": model_proj, "z": z,
                              "recent_rate": recent_rate, "recent_min": recent_min,
                              "pred_min": r.get("pred_min")}
    return out


def explain(info: dict, stat_label: str, player: str, p_before, p_after) -> str:
    """Plain-English version of the with/without math."""
    a, b, w = info["with"], info["without"], info["weight"]
    who = " and ".join(info["names"])
    last = player.split()[-1]
    lab = stat_label.lower()
    usg = (f", used {b['usg']:.0%} of the team's possessions while on the floor (vs {a['usg']:.0%})"
           if pd.notna(a["usg"]) and pd.notna(b["usg"]) else "")
    pm = info.get("pred_min")
    pb = f"{p_before:.1%}" if p_before is not None and pd.notna(p_before) else "the model's number"
    pa = f"{p_after:.1%}" if p_after is not None and pd.notna(p_after) else "?"
    already = (f" The minutes model (which already reacts to rotation players being out) has him at "
               f"{pm:.1f} minutes, for a projection of {info['model_proj']:.1f}." if pd.notna(pm) else "")
    return (f"**Injuries: with vs without {who} this season.** In the {b['n']} game(s) {last} played without "
            f"{who}, he averaged {b['min']:.1f} minutes (vs {a['min']:.1f} in {a['n']} with){usg}, and "
            f"{b['rate']:.2f} {lab} per minute (vs {a['rate']:.2f}), so {b['per_game']:.1f} {lab} a game without "
            f"vs {a['per_game']:.1f} with. {b['n']} game(s) is a small sample, so it counts "
            f"{w:.0%} ({b['n']} / ({b['n']} + {PRIOR_GAMES})). Starting from his last 10 games "
            f"({info['recent_min']:.1f} min, {info['recent_rate']:.2f}/min) and moving {w:.0%} of the way to "
            f"his without-{info['names'][0].split()[-1]} numbers: {info['min_t']:.1f} min × "
            f"{info['rate_t']:.2f}/min = {info['min_t'] * info['rate_t']:.1f} {lab}, which is **×{info['boost']:.2f}** "
            f"his recent form.{already} Applying ×{info['boost']:.2f} to that: {info['model_proj']:.1f} × "
            f"{info['boost']:.2f} = **{info['target']:.1f}**. The difference, "
            f"{info['target'] - info['model_proj']:+.1f}, against his usual "
            f"game-to-game swing of ±{info['sd']:.1f}, moves the chance of the over from {pb} to **{pa}**.")


# ---------------------------------------------------- who was out, game by game
def _absences(logs: pd.DataFrame, team: str, season, weight: str, min_games: int, min_weight: float,
              names: dict, fmt) -> dict:
    """{date: ["Austin Reaves (34 mpg)", ...]} for key players on `team` who missed that game."""
    t = logs[(logs["team"] == team) & (logs["season"] == season)]
    if t.empty:
        return {}
    dates = sorted(t["date"].unique())
    last_team = (logs[logs["season"] == season].sort_values("date")
                 .groupby("player_id")["team"].last())
    prev = logs[logs["season"] == season - 1]
    prev_max = prev.groupby("player_id").size().max() if not prev.empty else 0
    prev_stats = prev.groupby("player_id").agg(n=(weight, "size"), w=(weight, "mean"))
    out: dict = {d: [] for d in dates}
    for pid, g in t.groupby("player_id"):
        avg = g[weight].mean()
        if not (avg >= min_weight):
            continue
        # key = a regular this season, or a regular last season (hurt for most of this one).
        # Skips backups who only played while the starter was out.
        regular_now = len(g) >= max(min_games, 0.5 * len(dates))
        ps = prev_stats.loc[pid] if pid in prev_stats.index else None
        regular_before = (ps is not None and bool(prev_max) and ps["n"] >= 0.35 * prev_max
                          and ps["w"] >= min_weight)
        if not (regular_now or regular_before):
            continue
        first, last = g["date"].min(), g["date"].max()
        still = last_team.get(pid) == team
        played = set(g["date"])
        for d in dates:
            if d >= first and (still or d <= last) and d not in played:
                out[d].append((avg, fmt(names.get(pid, str(pid)), avg, g)))
    return {d: [s for _, s in sorted(v, key=lambda x: -x[0])] for d, v in out.items()}


def teammates_out(sport: str, games: pd.DataFrame, top: int = 5) -> pd.Series:
    """For each row of a player's game log: the key teammates who didn't play, as hover text."""
    if games.empty:
        return pd.Series(dtype=str)
    if sport == "NBA":
        from .analysis import load_data
        logs, players = load_data()
        names = players.drop_duplicates("player_id", keep="last").set_index("player_id")["player"].to_dict()
        kw = dict(weight="min", min_games=ROTATION_MIN_GAMES, min_weight=20.0,
                  fmt=lambda n, a, g: f"{n} ({a:.0f} mpg)")
    else:
        from . import nfl_data as nd
        _, logs, _ = nd.load()
        logs = logs.copy()
        logs["date"] = pd.to_datetime(logs["date"])
        names = logs.drop_duplicates("player_id", keep="last").set_index("player_id")["player"].to_dict()
        kw = dict(weight="offense_pct", min_games=3, min_weight=0.55,
                  fmt=lambda n, a, g: f"{n} ({g['position'].iloc[-1]}, {a:.0%} of snaps)")
    me = games["player_id"].iloc[-1] if "player_id" in games else None
    cache: dict = {}
    texts = []
    for _, r in games.iterrows():
        key = (r["team"], r["season"])
        if key not in cache:
            cache[key] = _absences(logs, r["team"], r["season"], names=names, **kw)
        miss = [s for s in cache[key].get(pd.Timestamp(r["date"]), []) if names.get(me, "") not in s]
        texts.append("<br>".join(miss[:top]) + (f"<br>+{len(miss) - top} more" if len(miss) > top else "")
                     if miss else "none")
    return pd.Series(texts, index=games.index)
