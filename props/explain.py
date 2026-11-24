"""Plain-English reasoning behind each pick.

short_reason(...)  one or two sentences for the pick card (built from the numbers the
                   model already computed - no extra work)
breakdown(...)     the full step-by-step math for the pop-up, plus a "what moved the
                   needle" table: each group of factors is reset to a neutral value and
                   the model is re-run, so the % change shows how much that group
                   pushed the prediction. Effects overlap, so they don't add up exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, nfl_data as nd


def _f(x, d=1):
    return "?" if x is None or pd.isna(x) else f"{x:.{d}f}"


def _pct(x):
    return "?" if x is None or pd.isna(x) else f"{x:+.0%}"


def _label(sport, stat):
    return (config.STATS.get(stat) if sport == "NBA" else nd.LABEL.get(stat, stat)) or stat


# ------------------------------------------------------------- card text
def short_reason(sport: str, r: dict) -> str:
    """1-2 sentences: the strongest reasons, using numbers already on the row."""
    line, pick = r.get("line"), r.get("pick")
    over = pick == "OVER"
    bits = []
    avg_s, avg_r = r.get("avg_season"), r.get("avg_last10") if sport == "NBA" else r.get("avg_last5")
    window = "last 10" if sport == "NBA" else "last 5"
    if pd.notna(avg_s) and pd.notna(avg_r):
        side = "above" if line > max(avg_s, avg_r) else "below" if line < min(avg_s, avg_r) else "between"
        bits.append(f"line {line:g} is {side} his season avg ({_f(avg_s)}) and {window} avg ({_f(avg_r)})")
    hr = r.get("hit_rate_last10_at_line") if sport == "NBA" else r.get("hit_rate_last5_at_line")
    if pd.notna(hr):
        n = 10 if sport == "NBA" else 5
        hits = round(hr * n) if over else n - round(hr * n)
        bits.append(f"{'over' if over else 'under'} hit in {hits} of his last {n}")
    proj = r.get("proj")
    if pd.notna(proj):
        bits.append(f"model projects {_f(proj)}" + (f" in {_f(r.get('pred_min'), 0)} min" if pd.notna(r.get("pred_min")) else ""))
    dvp = r.get("dvp_factor")
    if pd.notna(dvp) and abs(dvp - 1) >= 0.05:
        bits.append(f"opponent allows {abs(dvp - 1):.0%} {'more' if dvp > 1 else 'fewer'} than average to his position")
    if sport == "NFL":
        if r.get("indoor") == 0 and pd.notna(r.get("game_wind")) and r["game_wind"] >= 15:
            bits.append(f"{r['game_wind']:.0f} mph wind")
        if r.get("qb_out") == 1:
            bits.append("starting QB out")
    else:
        if r.get("injury_short"):
            bits.insert(0, r["injury_short"])
        elif pd.notna(r.get("miss_n")) and r["miss_n"] > 0:
            bits.append(f"{int(r['miss_n'])} teammate(s) out ({_f(r.get('miss_pts'), 0)} pts/g missing)")
        if r.get("b2b") == 1:
            bits.append("second night of a back-to-back")
    text = "; ".join(bits[:4])
    return (text[0].upper() + text[1:] + ".") if text else ""


# ---------------------------------------------------------- full breakdown
def _line_at_average(x, avg_col, short_col, lag_n):
    """Put the line right at his season average (so it no longer sits above/below his history)."""
    x = x.copy()
    x["line"] = x["avg_season"]
    for c in [k for k in x.columns if k.startswith("line_minus")]:
        x[c] = 0.0
    for c in [k for k in x.columns if k.startswith("hit_rate_last")]:
        x[c] = 0.5
    return x


GROUPS_NBA = {
    "Where the line sits vs his history": lambda x: _line_at_average(x, "avg_season", "avg_last10", 10),
    "Recent form (last 5/10 vs season)": lambda x: x.assign(avg_last5=x["avg_season"], avg_last10=x["avg_season"],
                                                           line_minus_avg10=x["line"] - x["avg_season"]),
    "Projected minutes / role": lambda x: x.assign(pred_min=x["min_season"], line_minus_proj=x["line"] - x["per_min_last10"] * x["min_season"]),
    "Opponent defense vs his position": lambda x: x.assign(dvp_factor=1.0, dvp_recent=1.0, opp_overall_factor=1.0),
    "History vs this opponent": lambda x: x.assign(vs_opp_diff=0.0, vs_opp_games=0),
    "Teammates out": lambda x: x.assign(miss_min=0.0, miss_pts=0.0, miss_ast=0.0, miss_trb=0.0, miss_n=0, opp_miss_min=0.0),
    "Rest / back-to-back": lambda x: x.assign(rest_days=2, b2b=0),
    "Game pace & spread": lambda x: x.assign(proj_total=230.0, proj_margin=0.0, abs_proj_margin=0.0),
}
GROUPS_NFL = {
    "Where the line sits vs his history": lambda x: _line_at_average(x, "avg_season", "avg_last5", 5),
    "Weather": lambda x: x.assign(game_temp=70, game_wind=0, wx_gust=0, wx_precip=0, wx_snow=0, indoor=1,
                                  cold=0, windy=0, wet=0),
    "Vegas (implied points & spread)": lambda x: x.assign(total_line=44.0, team_spread=0.0, implied_pts=22.0,
                                                         opp_implied_pts=22.0),
    "Recent form (last 3/5 vs season)": lambda x: x.assign(avg_last3=x["avg_season"], avg_last5=x["avg_season"],
                                                          line_minus_avg5=x["line"] - x["avg_season"]),
    "Opponent defense vs his position": lambda x: x.assign(dvp_factor=1.0, dvp_recent=1.0),
    "Injured teammates / QB": lambda x: x.assign(miss_targets=0.0, miss_carries=0.0, qb_out=0, opp_miss_n=0),
    "Rest / schedule": lambda x: x.assign(rest=7, short_week=0, primetime=0, div_game=0),
}


def factor_effects(sport: str, row: pd.Series, stat: str) -> pd.DataFrame:
    """Re-run the model with each group of factors set to neutral -> effect on P(over)."""
    if sport == "NBA":
        from . import model as m
        feats, clf, groups = m.FEATURES, m._load(f"over_{stat}"), GROUPS_NBA
    else:
        from . import nfl_model as m
        feats, clf, groups = m.FEATURES, m._load(stat), GROUPS_NFL

    def prob(X):
        return np.asarray(m.calibrate(stat, clf.predict_proba(X[feats])[:, 1]), float)
    base = pd.DataFrame([row[feats].astype(float)])
    variants = []
    for name, fn in groups.items():
        try:
            variants.append((name, fn(base.copy())[feats]))
        except KeyError:
            continue
    # home/away: average of both, so the effect is "being home (or away)"
    home_col = "home"
    h1, h0 = base.assign(**{home_col: 1}), base.assign(**{home_col: 0})
    p_full = float(prob(base)[0])
    X = pd.concat([v for _, v in variants] + [h1[feats], h0[feats]], ignore_index=True)
    ps = prob(X)
    rows = [{"factor": name, "effect_on_over": p_full - ps[i]} for i, (name, _) in enumerate(variants)]
    neutral_home = (ps[-1] + ps[-2]) / 2
    rows.append({"factor": "Home / away", "effect_on_over": p_full - neutral_home})
    out = pd.DataFrame(rows)
    out["abs"] = out["effect_on_over"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def odds_math(prob_pick: float, american) -> str:
    try:
        a = float(american)
    except (TypeError, ValueError):
        a = np.nan
    if pd.isna(a):
        return (f"No sportsbook price, so a standard -110 is assumed. At -110 you risk $110 to win $100, "
                f"so you need to win 110 / (110 + 100) = 52.4% to break even. The model says "
                f"{prob_pick:.1%}, a cushion of {prob_pick - 0.524:+.1%}.")
    be = -a / (-a + 100) if a < 0 else 100 / (a + 100)
    risk = f"risk ${-a:.0f} to win $100" if a < 0 else f"risk $100 to win ${a:.0f}"
    formula = f"{-a:.0f} / ({-a:.0f} + 100)" if a < 0 else f"100 / ({a:.0f} + 100)"
    ev = prob_pick * (100 / -a if a < 0 else a / 100) - (1 - prob_pick)
    return (f"At {a:+.0f} you {risk}, so the break-even win rate is {formula} = {be:.1%}. "
            f"The model gives this side {prob_pick:.1%}, so the edge is {prob_pick:.1%} - {be:.1%} = "
            f"{prob_pick - be:+.1%}. Expected value: {ev:+.2f} units per 1 unit bet.")


def breakdown(sport: str, r: dict, row: pd.Series | None) -> tuple[list[str], pd.DataFrame]:
    """Numbered plain-English steps + factor-effect table for one pick."""
    stat, line, pick = r["stat"], float(r["line"]), r["pick"]
    lab = _label(sport, stat)
    over = pick == "OVER"
    steps = []
    get = (lambda k: row.get(k) if row is not None else r.get(k))

    # 1. averages vs line
    if sport == "NBA":
        s, l10, l5 = get("avg_season"), get("avg_last10"), get("avg_last5")
        steps.append(f"**Averages vs the line.** The line is {line:g} {lab.lower()}. He averages "
                     f"{_f(s)} this season, {_f(l10)} over his last 10 and {_f(l5)} over his last 5. "
                     + ("All three are below the line, which points to the under."
                        if max(x for x in (s, l10, l5) if pd.notna(x)) < line else
                        "All three are above the line, which points to the over."
                        if min(x for x in (s, l10, l5) if pd.notna(x)) > line else
                        "The averages straddle the line, so other factors decide it."))
        hr = get("hit_rate_last10_at_line")
        if pd.notna(hr):
            n_over = round(hr * 10)
            steps.append(f"**Recent hit rate.** He's gone over {line:g} in {n_over} of his last 10 games "
                         f"({n_over * 10}%), so the {'over' if over else 'under'} hit "
                         f"{n_over if over else 10 - n_over} of 10.")
        pm, mins, proj = get("per_min_last10"), get("pred_min"), get("proj")
        if pd.notna(pm) and pd.notna(mins):
            steps.append(f"**Projection = rate × minutes.** Over his last 10 he's produced {pm:.2f} {lab.lower()} "
                         f"per minute. The minutes model (role, rest, injuries, blowout risk) projects "
                         f"{mins:.1f} minutes tonight (his recent average: {_f(get('min_last10'))}). "
                         f"{pm:.2f} × {mins:.1f} = **{pm * mins:.1f}**, which is {abs(pm * mins - line):.1f} "
                         f"{'above' if pm * mins > line else 'below'} the line.")
        dv, dr = get("dvp_factor"), get("dvp_recent")
        if pd.notna(dv):
            steps.append(f"**Matchup.** Per 36 minutes, this opponent allows {abs(dv - 1):.0%} "
                         f"{'more' if dv >= 1 else 'fewer'} {lab.lower()} than the league average to his position "
                         f"this season, and {abs((dr or 1) - 1):.0%} {'more' if (dr or 1) >= 1 else 'fewer'} "
                         f"over their last 15 games.")
        ctx = []
        if pd.notna(get("miss_n")) and get("miss_n") > 0:
            ctx.append(f"{int(get('miss_n'))} rotation teammate(s) are out ({_f(get('miss_pts'), 0)} points and "
                       f"{_f(get('miss_min'), 0)} minutes a game to replace)")
        if get("b2b") == 1:
            ctx.append("it's the second night of a back-to-back")
        if pd.notna(get("proj_total")):
            ctx.append(f"the game projects to about {get('proj_total'):.0f} total points")
        if ctx:
            steps.append("**Game context.** " + "; ".join(ctx).capitalize() + ".")
        oo = get("onoff")
        if isinstance(oo, dict):
            from .onoff import explain as onoff_explain
            steps.append(onoff_explain(oo, lab, r["player"], get("prob_over_model"), r["prob_over"]))
        elif r.get("injury_note"):
            steps.append(r["injury_note"])
        elif pd.notna(get("miss_n")) and get("miss_n") > 0:
            steps.append("**Injuries: with vs without.** Teammates are out, but none of them is ahead of him "
                         "in minutes or usage, or there aren't enough games this season without them to "
                         "measure, so only the minutes model's adjustment applies.")
    else:
        s, l5, l3 = get("avg_season"), get("avg_last5"), get("avg_last3")
        steps.append(f"**Averages vs the line.** The line is {line:g} {lab.lower()}. He averages {_f(s)} this "
                     f"season, {_f(l5)} over his last 5 and {_f(l3)} over his last 3.")
        hr = get("hit_rate_last5_at_line")
        if pd.notna(hr):
            n_over = round(hr * 5)
            steps.append(f"**Recent hit rate.** Over {line:g} in {n_over} of his last 5 games, so the "
                         f"{'over' if over else 'under'} hit {n_over if over else 5 - n_over} of 5.")
        role = []
        if pd.notna(get("snap_last3")):
            role.append(f"he's played {get('snap_last3'):.0%} of snaps over the last 3 games")
        if pd.notna(get("tshare_last3")) and get("tshare_last3") > 0:
            role.append(f"he's drawn {get('tshare_last3'):.0%} of the team's targets")
        if role:
            steps.append("**Role.** " + " and ".join(role).capitalize() + ".")
        if pd.notna(get("implied_pts")):
            sp = get("team_spread")
            steps.append(f"**Vegas.** The team is projected for {get('implied_pts'):.1f} points "
                         f"(total / 2 + spread / 2) and is a {abs(sp):.1f}-point "
                         f"{'favorite' if sp > 0 else 'underdog'}. Favorites tend to run more late in games, "
                         "and underdogs tend to throw more.")
        dv = get("dvp_factor")
        if pd.notna(dv):
            steps.append(f"**Matchup.** This defense allows {abs(dv - 1):.0%} {'more' if dv >= 1 else 'fewer'} "
                         f"{lab.lower()} per game than average to his position.")
        if get("indoor") == 1:
            steps.append("**Weather.** Indoors, so the weather has no effect.")
        elif pd.notna(get("game_wind")):
            wind, temp, rain = get("game_wind"), get("game_temp"), get("wx_precip") or 0
            cond = f"{temp:.0f}°F with {wind:.0f} mph wind" + (f" and {rain:.2f} in of rain" if rain > 0.02 else "")
            if wind >= 15 or rain > 0.05 or temp < 32:
                note = (" These are tough conditions: across every game since 2023, 15+ mph wind, rain/snow "
                        "and freezing temperatures have lowered passing and kicking numbers (see the Weather "
                        "impact tab).")
            else:
                note = " That's fairly normal, so weather is only a small factor here."
            steps.append(f"**Weather.** {cond}.{note}")
        if get("qb_out") == 1:
            steps.append("**Injuries.** The team's starting QB is out, which usually lowers passing and "
                         "receiving numbers.")
        elif pd.notna(get("miss_targets")) and get("miss_targets") > 0:
            steps.append(f"**Injuries.** Injured teammates leave about {get('miss_targets'):.1f} targets a game "
                         "to redistribute.")

    # model + calibration + odds
    conf = r["confidence"]
    raw = get("prob_over_raw")
    steps.append(f"**The model's answer.** Combining all of this (gradient-boosted trees trained on every game "
                 f"since 20{'25' if sport == 'NBA' else '23'}), "
                 + (f"the raw model says {raw:.1%} for the over. Raw models get overconfident at the "
                    f"extremes, so it's adjusted (calibrated) to how often predictions like this actually "
                    f"came true in testing: **{r['prob_over']:.1%}**. "
                    if raw is not None and pd.notna(raw) and abs(raw - r["prob_over"]) >= 0.005 else
                    f"the chance of the over is **{r['prob_over']:.1%}**. ")
                 + (f"That includes the injury adjustment above (it was {get('prob_over_model'):.1%} "
                    f"before). " if pd.notna(get("prob_over_model") if get("prob_over_model") is not None
                                             else np.nan) else "")
                 + f"So it picks the **{pick}** with {conf:.1%} confidence.")
    th = r.get("tested_hit")
    if th is not None and pd.notna(th):
        steps.append(f"**Reality check.** When the model gave about this confidence on {lab.lower()} in testing "
                     f"(games it had never seen), that side won {th:.0%} of the time.")
    odds_col = "over_odds" if over else "under_odds"
    american = r.get(odds_col) if r.get("line_type") in ("sportsbook", "custom", "mine") else None
    steps.append("**Is it worth betting?** " + odds_math(conf, american))
    if r.get("line_type") in ("custom", "mine"):
        steps.append(f"*You set this line to {line:g} by hand. Everything above was recalculated for it.*")
    elif r.get("line_type") != "sportsbook":
        steps.append("*This uses an estimated line (his recent average rounded to x.5). If the real line is "
                     "different, type it into **Try a different line** at the top to see the new math.*")

    effects = pd.DataFrame()
    if row is not None:
        try:
            effects = factor_effects(sport, row, stat)
        except Exception:
            effects = pd.DataFrame()
    return steps, effects


def model_row(sport: str, league, r: dict) -> pd.Series | None:
    """Re-score one pick to get every feature the model used."""
    pid = league.player_id(r["player"])
    if pid is None:
        return None
    if sport == "NBA":
        from . import model, slate
        req = slate.build_requests(league, pd.DataFrame([{
            "player": r["player"], "stat": r["stat"], "line": float(r["line"]),
            "date": pd.Timestamp(r.get("date") or pd.Timestamp.today()).normalize()}]))
        if req.empty:
            return None
        return league.predict(req).iloc[0]
    from . import nfl_model
    req = league.request_row(pid, r["stat"], float(r["line"]))
    if req is None:
        return None
    return nfl_model.predict_batch(league.pg, league.ctx, pd.DataFrame([req])).iloc[0]
