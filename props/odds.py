"""Betting lines: typed in by you (data/lines.csv) and/or pulled from The Odds API.

* Every line seen is saved to data/line_history.csv (one row per player/stat/book/day)
  so the model can train on REAL lines once those games are played.
* Every pull is also appended to data/line_snapshots.csv for line-movement alerts.
* Game totals and spreads go to data/game_lines.csv.
"""
from __future__ import annotations

import pandas as pd
import requests

from . import config
from .teams import to_code

LINE_COLUMNS = ["date", "player", "stat", "line", "over_odds", "under_odds",
                "book", "home_team", "away_team"]


# ------------------------------------------------------------ manual lines
def load_manual_lines() -> pd.DataFrame:
    """data/lines.csv columns: date,player,stat,line[,over_odds,under_odds]
    stat is one of the keys in config.STATS (pts, trb, ast, fg3, pra, ...)."""
    p = config.MANUAL_LINES_CSV
    if not p.exists():
        return pd.DataFrame(columns=LINE_COLUMNS)
    df = pd.read_csv(p)
    for c in LINE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df["book"] = df["book"].fillna("manual")
    df["date"] = pd.to_datetime(df["date"])
    return df[LINE_COLUMNS]


def add_manual_line(date, player, stat, line, over_odds=None, under_odds=None):
    df = load_manual_lines()
    new = pd.DataFrame([{"date": pd.to_datetime(date), "player": player, "stat": stat,
                         "line": float(line), "over_odds": over_odds,
                         "under_odds": under_odds, "book": "manual"}])
    pd.concat([df, new], ignore_index=True).to_csv(config.MANUAL_LINES_CSV, index=False)
    record_history(new)


def load_saved_api_lines() -> pd.DataFrame:
    """Upcoming lines from your oddsscraper database plus the most recent Odds API pull."""
    frames = []
    try:
        from . import scraped_lines
        sc = scraped_lines.prop_lines("NBA")
        if not sc.empty:
            frames.append(sc[[c for c in LINE_COLUMNS if c in sc]])
    except Exception:
        pass
    api = _load_odds_api_snapshot()
    if not api.empty:
        frames.append(api)
    if not frames:
        return pd.DataFrame(columns=LINE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _load_odds_api_snapshot() -> pd.DataFrame:
    p = config.LINE_SNAPSHOTS_CSV
    if not p.exists():
        return pd.DataFrame(columns=LINE_COLUMNS)
    df = pd.read_csv(p, parse_dates=["date", "pulled_at"])
    df = df[df["date"] >= pd.Timestamp.today().normalize()]
    if df.empty:
        return pd.DataFrame(columns=LINE_COLUMNS)
    latest = df["pulled_at"].max()
    return df[df["pulled_at"] == latest][[c for c in LINE_COLUMNS if c in df]]


# ----------------------------------------------------------- the odds api
def odds_api_available() -> bool:
    return bool(config.ODDS_API_KEY)


def _get(path: str, **params):
    params["apiKey"] = config.ODDS_API_KEY
    r = requests.get(f"{config.ODDS_API_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def _local_date(ts) -> pd.Timestamp:
    return pd.to_datetime(ts).tz_convert("America/Los_Angeles").normalize().tz_localize(None)


def fetch_game_lines() -> tuple[pd.DataFrame, str | None]:
    """Vegas totals + spreads for every upcoming game (one cheap request)."""
    data, remaining = _get("/sports/basketball_nba/odds", regions="us",
                           markets="totals,spreads", oddsFormat="american",
                           bookmakers=config.ODDS_BOOKMAKERS)
    rows = []
    for ev in data:
        home, away = to_code(ev["home_team"]), to_code(ev["away_team"])
        total, spread = [], []
        for book in ev.get("bookmakers", []):
            for m in book.get("markets", []):
                for o in m.get("outcomes", []):
                    if m["key"] == "totals" and o.get("name") == "Over":
                        total.append(o.get("point"))
                    if m["key"] == "spreads" and to_code(o.get("name", "")) == home:
                        spread.append(o.get("point"))
        rows.append({"date": _local_date(ev["commence_time"]), "home_team": home,
                     "away_team": away, "total": pd.Series(total).median(),
                     "home_spread": pd.Series(spread).median()})
    df = pd.DataFrame(rows)
    df.to_csv(config.GAME_LINES_CSV, index=False)
    return df, remaining


def load_game_lines() -> pd.DataFrame:
    """Spreads/totals: oddsscraper first (always current), then the last Odds API pull."""
    frames = []
    try:
        from . import scraped_lines
        sc = scraped_lines.game_lines("NBA")
        if not sc.empty:
            frames.append(sc)
    except Exception:
        pass
    if config.GAME_LINES_CSV.exists():
        frames.append(pd.read_csv(config.GAME_LINES_CSV, parse_dates=["date"]))
    if not frames:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "total", "home_spread"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "home_team", "away_team"])


def fetch_odds_api_lines() -> tuple[pd.DataFrame, str | None]:
    """Player prop lines for every upcoming NBA game, from every book you set.
    Costs about 1 credit per market per game, so it only runs when asked."""
    if not odds_api_available():
        raise RuntimeError("Add ODDS_API_KEY to your .env file first.")
    events, remaining = _get("/sports/basketball_nba/events")
    rows = []
    for ev in events:
        home, away = to_code(ev["home_team"]), to_code(ev["away_team"])
        data, remaining = _get(
            f"/sports/basketball_nba/events/{ev['id']}/odds", regions="us",
            markets=",".join(config.ODDS_MARKETS), oddsFormat="american",
            bookmakers=config.ODDS_BOOKMAKERS)
        for book in data.get("bookmakers", []):
            for market in book.get("markets", []):
                stat = config.ODDS_MARKETS.get(market["key"])
                if not stat:
                    continue
                by_player: dict[tuple, dict] = {}
                for o in market.get("outcomes", []):
                    d = by_player.setdefault((o.get("description"), o.get("point")), {})
                    d[o["name"].lower()] = o.get("price")
                for (player, point), prices in by_player.items():
                    if player is None or point is None:
                        continue
                    rows.append({"date": _local_date(ev["commence_time"]), "player": player,
                                 "stat": stat, "line": float(point),
                                 "over_odds": prices.get("over"), "under_odds": prices.get("under"),
                                 "book": book["key"], "home_team": home, "away_team": away})
    df = pd.DataFrame(rows, columns=LINE_COLUMNS)
    if not df.empty:
        record_history(df)
        snap = df.assign(pulled_at=pd.Timestamp.now().floor("min"))
        p = config.LINE_SNAPSHOTS_CSV
        snap.to_csv(p, mode="a", header=not p.exists(), index=False)
    return df, remaining


# ---------------------------------------------------------------- history
def record_history(lines: pd.DataFrame):
    """Keep every line ever seen so the model can learn from real lines later."""
    p = config.LINE_HISTORY_CSV
    keep = [c for c in ["date", "player", "stat", "line", "over_odds", "under_odds", "book"]
            if c in lines.columns]
    new = lines[keep].copy()
    if p.exists():
        new = pd.concat([pd.read_csv(p, parse_dates=["date"]), new], ignore_index=True)
    new.drop_duplicates(subset=["date", "player", "stat", "book"], keep="last").to_csv(p, index=False)


def implied_prob(american) -> float | None:
    """-110 -> 0.524. Lets us compare the model to the sportsbook."""
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if a != a:  # NaN
        return None
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def payout(american) -> float:
    """Profit per 1 unit staked if the bet wins (-110 -> 0.909)."""
    try:
        a = float(american)
    except (TypeError, ValueError):
        a = -110
    if a != a:
        a = -110
    return a / 100 if a > 0 else 100 / -a


def to_american(prob: float) -> int:
    if prob <= 0 or prob >= 1:
        return 0
    return int(round(-100 * prob / (1 - prob))) if prob >= 0.5 else int(round(100 * (1 - prob) / prob))


# ---------------------------------------------------------- line shopping
def best_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """For each player/stat/day: the best book to bet the over (lowest line,
    then best price) and the under (highest line, then best price)."""
    if lines.empty:
        return pd.DataFrame()
    from .nba_com import norm_name
    lines = lines.assign(_key=lines["player"].map(norm_name))
    rows = []
    for (date, _, stat), g in lines.groupby(["date", "_key", "stat"]):
        player = g["player"].iloc[0]
        g = g.assign(o_pay=g["over_odds"].map(payout), u_pay=g["under_odds"].map(payout))
        bo = g.sort_values(["line", "o_pay"], ascending=[True, False]).iloc[0]
        bu = g.sort_values(["line", "u_pay"], ascending=[False, False]).iloc[0]
        rows.append({"date": date, "player": player, "stat": stat, "books": g["book"].nunique(),
                     "best_over_line": bo["line"], "best_over_odds": bo["over_odds"],
                     "best_over_book": bo["book"],
                     "best_under_line": bu["line"], "best_under_odds": bu["under_odds"],
                     "best_under_book": bu["book"],
                     "line_range": g["line"].max() - g["line"].min()})
    return pd.DataFrame(rows)


def line_moves(min_move: float = config.ALERT_LINE_MOVE) -> pd.DataFrame:
    """Props whose line moved at least `min_move` points since the first pull today."""
    p = config.LINE_SNAPSHOTS_CSV
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date", "pulled_at"])
    df = df[df["date"] >= pd.Timestamp.today().normalize()]
    if df["pulled_at"].nunique() < 2:
        return pd.DataFrame()
    from .nba_com import norm_name
    df["player"] = df["player"].map(lambda n: n)  # keep display name
    df["_key"] = df["player"].map(norm_name)
    names = df.groupby("_key")["player"].first()
    med = df.groupby(["pulled_at", "date", "_key", "stat"])["line"].median().reset_index()
    med["player"] = med["_key"].map(names)
    first = med.sort_values("pulled_at").groupby(["date", "player", "stat"]).first()
    last = med.sort_values("pulled_at").groupby(["date", "player", "stat"]).last()
    out = first[["line"]].rename(columns={"line": "open"}).join(last[["line"]].rename(columns={"line": "now"}))
    out["move"] = out["now"] - out["open"]
    return out[out["move"].abs() >= min_move].reset_index()
