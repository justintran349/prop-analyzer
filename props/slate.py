"""Scoring props for upcoming games: shared by the dashboard and scripts/daily.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import analysis, config, context, model, odds, scraper
from .nba_com import norm_name

MISS_KEYS = ["miss_min", "miss_pts", "miss_ast", "miss_trb", "miss_n"]


def match_abbreviated(name: str, norm_to_id: dict):
    """'J. Tatum' / 'J Tatum' -> the one player whose name is J... Tatum (None if ambiguous)."""
    k = norm_name(name)
    parts = k.split()
    if len(parts) < 2 or len(parts[0]) > 1:
        return None
    hits = [pid for n, pid in norm_to_id.items()
            if n.split() and n.split()[-1] == parts[-1] and n[:1] == parts[0]]
    return hits[0] if len(set(hits)) == 1 else None


class League:
    """Everything needed to score games, loaded once."""

    def __init__(self):
        logs, players = analysis.load_data()
        if logs.empty:
            raise RuntimeError("No data yet - run: python scripts/update_data.py")
        self.logs = analysis.attach_positions(logs, players)
        self.players = players
        season = config.SEASON if (self.logs["season"] == config.SEASON).any() else self.logs["season"].max()
        self.season = season
        self.dvp = analysis.defense_vs_position(self.logs, season)
        self.dvp_recent = analysis.defense_vs_position(self.logs, season, last_n_games=15)
        self.sched = (pd.read_csv(config.SCHEDULE_CSV, parse_dates=["date"])
                      if config.SCHEDULE_CSV.exists() else pd.DataFrame())
        self.injuries = context.load_injuries()
        self.game_lines = odds.load_game_lines()
        self.ctx = model.Context(self.logs)
        cur = (players.sort_values("season").drop_duplicates("player_id", keep="last")
               .query("player_id in @self.logs.player_id.unique()"))
        # team = team of most recent game
        last_team = self.logs.sort_values("date").groupby("player_id")["team"].last()
        cur = cur.assign(team=cur["player_id"].map(last_team).fillna(cur["team"]))
        self.current = cur.set_index("player_id")
        self.name_to_id = dict(zip(cur["player"], cur["player_id"]))
        self.norm_to_id = {norm_name(n): p for n, p in self.name_to_id.items()}
        self._miss_cache: dict = {}
        from . import onoff
        self.onoff = onoff.OnOff(self.logs, self.ctx.usage, season)

    def predict(self, req: pd.DataFrame) -> pd.DataFrame:
        """Model chances, then the with/without-teammate adjustment for injuries tonight."""
        from . import onoff
        return onoff.apply(self, model.predict_batch(self.logs, req, self.ctx))

    def player_id(self, name: str):
        return (self.name_to_id.get(name) or self.norm_to_id.get(norm_name(name))
                or match_abbreviated(name, self.norm_to_id))

    def missing(self, team: str) -> dict:
        if team not in self._miss_cache:
            self._miss_cache[team] = context.missing_from_report(
                self.logs, self.players, team, self.injuries, self.season)
        return self._miss_cache[team]

    def game_for(self, team: str, date=None):
        """(opp, is_home, date) of the team's next game on/after date."""
        if self.sched.empty:
            return None
        return scraper.next_game(team, self.sched, date)

    def vegas(self, team: str, date) -> dict:
        gl = self.game_lines
        if gl.empty:
            return {}
        d = pd.Timestamp(date).normalize()
        g = gl[(gl["date"] == d) & ((gl["home_team"] == team) | (gl["away_team"] == team))]
        if g.empty:
            return {}
        r = g.iloc[0]
        spread = r["home_spread"] if r["home_team"] == team else -r["home_spread"]
        return {"total": r["total"], "spread": spread}


def build_requests(lg: League, lines: pd.DataFrame) -> pd.DataFrame:
    """Turn raw lines (player name, stat, line, odds, book, date) into model requests."""
    rows = []
    for r in lines.itertuples(index=False):
        rd = r._asdict()
        pid = lg.player_id(rd["player"])
        if pid is None or rd["stat"] not in config.STATS:
            continue
        team = lg.current.loc[pid, "team"]
        if pd.notna(rd.get("home_team")) and rd.get("home_team"):
            if team not in (rd["home_team"], rd.get("away_team")):
                continue        # player's team isn't in this game -> wrong player / stale team
            is_home = team == rd["home_team"]
            opp = rd["away_team"] if is_home else rd["home_team"]
            date = pd.Timestamp(rd["date"])
        else:
            g = lg.game_for(team, rd["date"])
            if not g:
                continue
            opp, is_home, date = g
        mine, theirs = lg.missing(team), lg.missing(opp)
        rows.append({**{k: rd.get(k) for k in ("player", "line", "over_odds", "under_odds", "book")},
                     "player_id": pid, "stat": rd["stat"], "team": team, "opp": opp,
                     "is_home": bool(is_home), "date": pd.Timestamp(date).normalize(),
                     **{k: mine[k] for k in MISS_KEYS}, "opp_miss_min": theirs["miss_min"]})
    return pd.DataFrame(rows)


def score_lines(lg: League, lines: pd.DataFrame) -> pd.DataFrame:
    """Model probability, pick and edge for every line."""
    req = build_requests(lg, lines)
    if req.empty:
        return pd.DataFrame()
    scored = lg.predict(req)
    out = []
    for r in scored.itertuples(index=False):
        d = r._asdict()
        p = d.get("prob_over")
        imp_o = odds.implied_prob(d.get("over_odds")) or 0.524
        imp_u = odds.implied_prob(d.get("under_odds")) or 0.524
        if p is None or pd.isna(p):
            pick, edge, prob_pick, odds_pick = "—", np.nan, np.nan, np.nan
        elif p - imp_o >= (1 - p) - imp_u:
            pick, edge, prob_pick, odds_pick = "OVER", p - imp_o, p, d.get("over_odds")
        else:
            pick, edge, prob_pick, odds_pick = "UNDER", (1 - p) - imp_u, 1 - p, d.get("under_odds")
        pos = lg.current.loc[d["player_id"], "primary_pos"] if d["player_id"] in lg.current.index else None
        dv = analysis.dvp_lookup(lg.dvp, d["opp"], pos, d["stat"]) if pos else {"rank": None}
        v = lg.vegas(d["team"], d["date"])
        out.append({
            "date": d["date"], "player": d["player"], "player_id": d["player_id"],
            "team": d["team"], "stat": d["stat"], "line": d["line"],
            "game": f"{'vs' if d['is_home'] else '@'} {d['opp']}", "opp": d["opp"],
            "is_home": d["is_home"], "book": d.get("book"),
            "over_odds": d.get("over_odds"), "under_odds": d.get("under_odds"),
            "proj": d.get("proj"), "pred_min": d.get("pred_min"),
            "l10_hit": d.get("hit_rate_last10_at_line"), "dvp_rank": dv.get("rank"),
            "teammates_out": int(d.get("miss_n") or 0) if "miss_n" in d else 0,
            "vegas_total": v.get("total"), "spread": v.get("spread"),
            "prob_over": p, "pick": pick, "prob_pick": prob_pick,
            "pick_odds": odds_pick, "edge": edge,
            "prob_over_model": d.get("prob_over_model"), "onoff": d.get("onoff"),
        })
    return pd.DataFrame(out).sort_values("edge", ascending=False, na_position="last")


def todays_games(lg: League, date=None) -> pd.DataFrame:
    d = pd.Timestamp(date or pd.Timestamp.today()).normalize()
    if lg.sched.empty:
        return pd.DataFrame()
    return lg.sched[lg.sched["date"] == d][["date", "away", "home"]].reset_index(drop=True)


def projections_for_games(lg: League, games: pd.DataFrame, stats=("pts", "trb", "ast")) -> pd.DataFrame:
    """Model projections (minutes + stat) for every rotation player in these games,
    even without sportsbook lines. The 'line' used is the player's estimated line."""
    if games.empty:
        return pd.DataFrame()
    avg = context.season_averages(lg.logs, lg.season)
    rot = avg[(avg["min"] >= 18) & (avg["games"] >= 3)]
    out_names = set()
    for t in pd.concat([games["home"], games["away"]]):
        out_names |= set(lg.missing(t)["names"])
    rows = []
    for g in games.itertuples():
        for team, opp, home in ((g.home, g.away, True), (g.away, g.home, False)):
            for pid in rot[rot["team"] == team].index:
                name = lg.current.loc[pid, "player"] if pid in lg.current.index else pid
                if name in out_names:
                    continue
                hist = lg.logs[lg.logs["player_id"] == pid].sort_values("date")
                mine, theirs = lg.missing(team), lg.missing(opp)
                for s in stats:
                    recent = hist[s].dropna().tail(10)
                    line = max(0.5, np.floor(recent.mean()) + 0.5) if len(recent) else 0.5
                    rows.append({"player": name, "player_id": pid, "stat": s, "line": line,
                                 "team": team, "opp": opp, "is_home": home, "date": g.date,
                                 **{k: mine[k] for k in MISS_KEYS}, "opp_miss_min": theirs["miss_min"]})
    req = pd.DataFrame(rows)
    if req.empty:
        return req
    scored = lg.predict(req)
    scored["game"] = np.where(scored["is_home"], "vs ", "@ ") + scored["opp"]
    keep = ["player", "team", "game", "stat", "line", "pred_min", "proj", "prob_over",
            "dvp_factor", "dvp_recent", "proj_total", "avg_season", "avg_last10", "avg_last5",
            "hit_rate_last10_at_line", "per_min_last10", "min_last10", "miss_n", "miss_pts",
            "is_home", "b2b", "rest_days", "usg_last10", "prob_over_model", "onoff", "player_id"]
    return scored[[c for c in keep if c in scored]]


def projected_game(lg: "League", home: str, away: str) -> dict:
    """Model's projected total and home margin from each team's scoring to date."""
    sc = lg.ctx.scoring.sort_values("date")
    sc = sc[sc["season"] == sc["season"].max()].groupby("team").last()
    if home not in sc.index or away not in sc.index:
        return {}
    h, a = sc.loc[home], sc.loc[away]
    home_pts = (h["ppg_thru"] + a["papg_thru"]) / 2 + 1.25
    away_pts = (a["ppg_thru"] + h["papg_thru"]) / 2 - 1.25
    return {"total": home_pts + away_pts, "home_margin": home_pts - away_pts}
