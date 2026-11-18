"""Scoring NFL props for the upcoming week (shared by the dashboard and scripts)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import nfl_data as nd, nfl_model as nm, nfl_weather as nw, odds
from .nba_com import norm_name


class NFLLeague:
    def __init__(self):
        if not nd.PLAYER_GAMES_CSV.exists():
            raise RuntimeError("No NFL data yet - run: python scripts/update_nfl_data.py")
        self.games, self.pg, self.injuries = nd.load()
        self.ctx = nm.Context(self.pg, self.games)
        self.week = nd.upcoming_games(self.games)
        last = self.pg.sort_values("date").groupby("player_id").tail(1)
        self.current = last.set_index("player_id")
        self.name_to_id = dict(zip(last["player"], last["player_id"]))
        self.norm_to_id = {norm_name(n): p for n, p in self.name_to_id.items()}
        self._miss = {}

    def player_id(self, name):
        from .slate import match_abbreviated
        return (self.name_to_id.get(name) or self.norm_to_id.get(norm_name(name))
                or match_abbreviated(name, self.norm_to_id))

    def game_for_team(self, team):
        g = self.week[(self.week["home_team"] == team) | (self.week["away_team"] == team)]
        g = g[g["home_score"].isna()]
        return None if g.empty else g.iloc[0]

    def missing(self, team):
        if team not in self._miss:
            s, w = (int(self.week["season"].iloc[0]), int(self.week["week"].iloc[0])) if not self.week.empty else (nd.NFL_SEASON, None)
            self._miss[team] = nm.missing_from_injuries(self.pg, self.injuries, team, s, w)
        return self._miss[team]

    def request_row(self, pid, stat, line, extra=None):
        team = self.current.loc[pid, "team"]
        g = self.game_for_team(team)
        if g is None:
            return None
        opp = g["away_team"] if g["home_team"] == team else g["home_team"]
        mine, theirs = self.missing(team), self.missing(opp)
        r = {"player_id": pid, "stat": stat, "line": float(line), "game_id": g["game_id"], "team": team,
             "miss_targets": mine["miss_targets"], "miss_carries": mine["miss_carries"],
             "qb_out": mine["qb_out"], "opp_miss_n": theirs["miss_n"]}
        r.update(extra or {})
        return r

    def predict(self, pid, stat, line):
        r = self.request_row(pid, stat, line)
        if r is None:
            raise RuntimeError("No upcoming game found for this player's team.")
        out = nm.predict_batch(self.pg, self.ctx, pd.DataFrame([r])).iloc[0]
        return out

    # ------------------------------------------------------------ lines
    def score_lines(self, lines: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for l in lines.itertuples(index=False):
            d = l._asdict()
            pid = self.player_id(d["player"])
            if pid is None or d["stat"] not in nd.STATS:
                continue
            r = self.request_row(pid, d["stat"], d["line"],
                                 {k: d.get(k) for k in ("player", "over_odds", "under_odds", "book")})
            if r:
                rows.append(r)
        if not rows:
            return pd.DataFrame()
        sc = nm.predict_batch(self.pg, self.ctx, pd.DataFrame(rows))
        g = self.games.set_index("game_id")
        out = []
        for r in sc.itertuples(index=False):
            d = r._asdict()
            p = d.get("prob_over")
            io = odds.implied_prob(d.get("over_odds")) or 0.524
            iu = odds.implied_prob(d.get("under_odds")) or 0.524
            if p is None or pd.isna(p):
                pick, edge, pp, po = "—", np.nan, np.nan, np.nan
            elif p - io >= (1 - p) - iu:
                pick, edge, pp, po = "OVER", p - io, p, d.get("over_odds")
            else:
                pick, edge, pp, po = "UNDER", (1 - p) - iu, 1 - p, d.get("under_odds")
            gm = g.loc[d["game_id"]]
            home = gm["home_team"] == d["team"]
            out.append({"player": d["player"], "player_id": d["player_id"], "pos": d.get("position"),
                        "team": d["team"], "game": f"{'vs' if home else '@'} {gm['away_team'] if home else gm['home_team']}",
                        "stat": d["stat"], "line": d["line"], "book": d.get("book"),
                        "over_odds": d.get("over_odds"), "under_odds": d.get("under_odds"),
                        "weather": nw.describe(gm), "implied_pts": d.get("implied_pts"),
                        "avg_last5": d.get("avg_last5"), "l5_hit": d.get("hit_rate_last5_at_line"),
                        "dvp_factor": d.get("dvp_factor"), "prob_over": p, "pick": pick,
                        "prob_pick": pp, "pick_odds": po, "edge": edge, "date": gm["date"]})
        return pd.DataFrame(out).sort_values("edge", ascending=False, na_position="last")

    def projections(self, stats=("passing_yards", "rushing_yards", "receiving_yards", "receptions")) -> pd.DataFrame:
        """Model view of every regular in this week's games, at estimated lines."""
        rows = []
        upcoming = self.week[self.week["home_score"].isna()]
        teams = set(upcoming["home_team"]) | set(upcoming["away_team"])
        season = int(self.pg["season"].max())
        cur = self.pg[(self.pg["season"] == season) & self.pg["team"].isin(teams)]
        cur = cur[cur["player_id"].map(self.current["team"]) == cur["team"]]
        for stat in stats:
            poss = nd.STATS[stat][1]
            opp_col = nm.OPPORTUNITY[stat] if nm.OPPORTUNITY[stat] != "touches" else "carries"
            avg = cur[cur["position"].isin(poss)].groupby("player_id").agg(
                n=(stat, "size"), opp=(opp_col, "mean"))
            last5 = (self.pg[self.pg["player_id"].isin(avg.index)].sort_values("date")
                     .groupby("player_id")[stat].apply(lambda s: s.tail(5).mean()))
            avg["v"] = last5
            thresh = {"attempts": 15, "carries": 6, "targets": 3}.get(nm.OPPORTUNITY[stat], 3)
            avg = avg[(avg["n"] >= 2) & (avg["opp"] >= thresh)]
            out_names = set()
            for t in teams:
                out_names |= set(self.missing(t)["names"])
            for pid, a in avg.iterrows():
                if self.current.loc[pid, "player"] in out_names:
                    continue
                line = 0.5 if stat == "anytime_td" else float(np.floor(a["v"]) + 0.5)
                r = self.request_row(pid, stat, line)
                if r:
                    rows.append(r)
        if not rows:
            return pd.DataFrame()
        sc = nm.predict_batch(self.pg, self.ctx, pd.DataFrame(rows))
        g = self.games.set_index("game_id")
        sc["player"] = sc["player_id"].map(self.current["player"])
        sc["weather"] = [nw.describe(g.loc[x]) for x in sc["game_id"]]
        sc["matchup"] = [f"{g.loc[x, 'away_team']} @ {g.loc[x, 'home_team']}" for x in sc["game_id"]]
        keep = ["player", "position", "team", "matchup", "stat", "line", "avg_last5", "proj",
                "prob_over", "implied_pts", "dvp_factor", "weather", "avg_season", "avg_last3",
                "hit_rate_last5_at_line", "dvp_recent", "game_wind", "game_temp", "wx_precip", "wx_snow",
                "indoor", "team_spread", "snap_last3", "tshare_last3", "qb_out", "miss_targets", "home"]
        return sc[[c for c in keep if c in sc]]


# ---------------------------------------------------------------- odds
def fetch_odds_api_lines():
    """NFL player props from The Odds API -> saved like the NBA lines."""
    if not odds.odds_api_available():
        raise RuntimeError("Add ODDS_API_KEY to your .env file first.")
    events, remaining = odds._get("/sports/americanfootball_nfl/events")
    rows = []
    for ev in events:
        data, remaining = odds._get(f"/sports/americanfootball_nfl/events/{ev['id']}/odds",
                                    regions="us", markets=",".join(nd.ODDS_MARKETS), oddsFormat="american",
                                    bookmakers=odds.config.ODDS_BOOKMAKERS)
        for book in data.get("bookmakers", []):
            for mk in book.get("markets", []):
                stat = nd.ODDS_MARKETS.get(mk["key"])
                if not stat:
                    continue
                by = {}
                for o in mk.get("outcomes", []):
                    point = o.get("point")
                    if stat == "anytime_td":       # Yes/No market with no number
                        point = 0.5
                        name = "over" if o["name"].lower() in ("yes", "over") else "under"
                    else:
                        name = o["name"].lower()
                    by.setdefault((o.get("description"), point), {})[name] = o.get("price")
                for (player, point), pr in by.items():
                    if player and point is not None:
                        rows.append({"date": odds._local_date(ev["commence_time"]), "player": player,
                                     "stat": stat, "line": float(point), "over_odds": pr.get("over"),
                                     "under_odds": pr.get("under"), "book": book["key"]})
    df = pd.DataFrame(rows, columns=["date", "player", "stat", "line", "over_odds", "under_odds", "book"])
    if not df.empty:
        hist = nd.LINE_HISTORY_CSV
        all_ = pd.concat([pd.read_csv(hist, parse_dates=["date"]), df]) if hist.exists() else df
        all_.drop_duplicates(["date", "player", "stat", "book"], keep="last").to_csv(hist, index=False)
        snap = df.assign(pulled_at=pd.Timestamp.now().floor("min"))
        snap.to_csv(nd.LINE_SNAPSHOTS_CSV, mode="a", header=not nd.LINE_SNAPSHOTS_CSV.exists(), index=False)
    return df, remaining


def load_lines() -> pd.DataFrame:
    frames = []
    if nd.LINES_CSV.exists():
        m = pd.read_csv(nd.LINES_CSV)
        if not m.empty:
            m["book"] = m.get("book", pd.Series("manual", index=m.index)).fillna("manual")
            frames.append(m)
    try:
        from . import scraped_lines
        sc = scraped_lines.prop_lines("NFL")
        if not sc.empty:
            frames.append(sc[["date", "player", "stat", "line", "over_odds", "under_odds", "book"]])
    except Exception:
        pass
    if nd.LINE_SNAPSHOTS_CSV.exists():
        s = pd.read_csv(nd.LINE_SNAPSHOTS_CSV, parse_dates=["pulled_at"])
        frames.append(s[s["pulled_at"] == s["pulled_at"].max()].drop(columns="pulled_at"))
    if not frames:
        return pd.DataFrame(columns=["date", "player", "stat", "line", "over_odds", "under_odds", "book"])
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("over_odds", "under_odds", "book"):
        if c not in df:
            df[c] = np.nan
    return df[df["date"] >= pd.Timestamp.today().normalize() - pd.Timedelta(days=1)]
