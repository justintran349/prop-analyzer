"""Run with:  pytest      (uses fake data - no internet needed)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from props import analysis, config, model, scraper  # noqa: E402
from props.model import TEAM_CODES  # noqa: E402

# ------------------------------------------------------------ scraper parsing
NEW_LAYOUT = """
<table id="player_game_log_reg"><tbody>
<tr><th data-stat="ranker">1</th><td data-stat="date">2026-10-22</td>
<td data-stat="team_name_abbr">LAL</td><td data-stat="game_location"></td>
<td data-stat="opp_name_abbr">GSW</td><td data-stat="game_result">W, 110-104</td>
<td data-stat="is_starter">*</td><td data-stat="mp">35:30</td><td data-stat="fg3">2</td>
<td data-stat="trb">8</td><td data-stat="ast">9</td><td data-stat="pts">27</td></tr>
<tr class="thead"><th>Rk</th></tr>
<tr><th data-stat="ranker">2</th><td data-stat="date">2026-10-24</td>
<td data-stat="team_name_abbr">LAL</td><td data-stat="game_location">@</td>
<td data-stat="opp_name_abbr">PHO</td><td data-stat="reason">Inactive</td></tr>
</tbody></table>"""

OLD_LAYOUT_IN_COMMENT = """<div><!--
<table id="pgl_basic"><tbody>
<tr><td data-stat="date_game">2025-01-05</td><td data-stat="team_id">LAL</td>
<td data-stat="game_location">@</td><td data-stat="opp_id">BOS</td>
<td data-stat="gs">1</td><td data-stat="mp">30:00</td><td data-stat="fg3">1</td>
<td data-stat="trb">5</td><td data-stat="ast">4</td><td data-stat="pts">18</td></tr>
</tbody></table>--></div>"""


def test_gamelog_new_layout(monkeypatch):
    monkeypatch.setattr(scraper, "fetch", lambda url, max_age_hours=None: NEW_LAYOUT)
    df = scraper.get_gamelog("jamesle01", 2027)
    assert len(df) == 1                      # inactive game dropped
    r = df.iloc[0]
    assert r["home"] == 1 and r["opp"] == "GSW" and r["started"] == 1
    assert r["min"] == pytest.approx(35.5) and r["pra"] == 44


def test_gamelog_old_layout_inside_comment(monkeypatch):
    monkeypatch.setattr(scraper, "fetch", lambda url, max_age_hours=None: OLD_LAYOUT_IN_COMMENT)
    df = scraper.get_gamelog("jamesle01", 2025)
    assert len(df) == 1 and df.iloc[0]["home"] == 0 and df.iloc[0]["opp"] == "BOS"


def test_players_page(monkeypatch):
    html = """<table id="per_game_stats"><tbody>
    <tr><td data-stat="name_display" data-append-csv="smithjo01"><a href="/x">John Smith</a></td>
    <td data-stat="team_name_abbr">2TM</td><td data-stat="pos">SG-SF</td>
    <td data-stat="games">20</td><td data-stat="mp_per_g">31.0</td></tr>
    <tr><td data-stat="name_display" data-append-csv="smithjo01">John Smith</td>
    <td data-stat="team_name_abbr">MIA</td><td data-stat="pos">SG</td>
    <td data-stat="games">8</td><td data-stat="mp_per_g">30.0</td></tr>
    </tbody></table>"""
    monkeypatch.setattr(scraper, "fetch", lambda url, max_age_hours=None: html)
    df = scraper.get_players(2027)
    assert len(df) == 1
    assert df.iloc[0]["team"] == "MIA" and df.iloc[0]["primary_pos"] == "SG"
    assert df.iloc[0]["games"] == 20


# ------------------------------------------------------------- fake league
@pytest.fixture(scope="module")
def fake_logs():
    """2 seasons, 150 players. Built-in truth: players score more at home and
    centers get extra rebounds against 'weak' defenses - the model should find it."""
    rng = np.random.default_rng(0)
    positions = ["PG", "SG", "SF", "PF", "C"]
    weak_vs_c = {"SAC", "WAS", "UTA"}
    rows, players = [], []
    for i in range(150):
        pid, pos = f"p{i:03d}", positions[i % 5]
        team = TEAM_CODES[i % 30]
        base_pts, base_reb = rng.uniform(8, 28), rng.uniform(3, 11) + (3 if pos == "C" else 0)
        players.append({"player_id": pid, "player": f"Player {i}", "team": team,
                        "pos": pos, "primary_pos": pos, "season": 2027, "games": 60, "mpg": 30})
        for season in (2026, 2027):
            dates = pd.date_range(f"{season - 1}-10-25", periods=60, freq="3D")
            for d in dates:
                opp = rng.choice([t for t in TEAM_CODES if t != team])
                home = int(rng.random() < 0.5)
                bump = 3 if (pos == "C" and opp in weak_vs_c) else 0
                pts = max(0, rng.normal(base_pts + 2.5 * home, 5))
                reb = max(0, rng.normal(base_reb + bump, 2.5))
                ast = max(0, rng.normal(4, 2))
                rows.append({"player_id": pid, "season": season, "date": d, "team": team,
                             "home": home, "opp": opp, "started": 1,
                             "min": rng.uniform(22, 38), "pts": round(pts), "trb": round(reb),
                             "ast": round(ast), "fg3": rng.integers(0, 5),
                             "fga": rng.integers(5, 20), "fta": rng.integers(0, 8),
                             "stl": rng.integers(0, 3), "blk": rng.integers(0, 3),
                             "tov": rng.integers(0, 5)})
    logs = pd.DataFrame(rows)
    from props.nba_com import add_combos
    add_combos(logs)
    return analysis.attach_positions(logs, pd.DataFrame(players))


def test_defense_vs_position_finds_weak_defenses(fake_logs):
    dvp = analysis.defense_vs_position(fake_logs, 2027)
    c_reb = dvp[(dvp.pos == "C") & (dvp.stat == "trb")].sort_values("rank")
    assert set(c_reb.head(3)["opp"]) == {"SAC", "WAS", "UTA"}


def test_player_report(fake_logs):
    dvp = analysis.defense_vs_position(fake_logs, 2027)
    rep = analysis.player_report(fake_logs, dvp, "p004", "trb", 9.5, "SAC", True)
    assert rep["pos"] == "C" and rep["dvp_rank"] <= 3
    assert 0 <= rep["prob_over_simple"] <= 1
    assert rep["splits"]["This season"]["games"] == 60


def test_model_trains_and_predicts(fake_logs, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    res = model.train_all(fake_logs, ["pts"], log=lambda *_: None)
    m = res["pts"]
    assert m["test_games"] > 500 and m["auc"] > 0.55      # learned the home effect
    assert (tmp_path / "minutes.joblib").exists()
    home = model.predict(fake_logs, "p000", "pts", 15.5, "BOS", True, date="2027-01-01")
    away = model.predict(fake_logs, "p000", "pts", 15.5, "BOS", False, date="2027-01-01")
    assert 0 < away["prob_over"] < 1
    assert home["prob_over"] > away["prob_over"]


def test_lower_line_means_higher_over_chance(fake_logs, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    model.train_all(fake_logs, ["pts"], log=lambda *_: None)
    req = pd.DataFrame([{"player_id": "p000", "stat": "pts", "line": ln, "opp": "BOS",
                         "is_home": True, "date": "2027-01-01", "team": None}
                        for ln in (5.5, 15.5, 30.5)])
    p = model.predict_batch(fake_logs, req)["prob_over"].tolist()
    assert p[0] > p[1] > p[2]


def test_missing_teammates_detected():
    from props import context
    rows = []
    for i, d in enumerate(pd.date_range("2026-10-20", periods=8, freq="2D")):
        for pid in ["a", "b", "c"]:
            if pid == "c" and i == 7:      # star sits the last game
                continue
            rows.append({"player_id": pid, "season": 2027, "team": "BOS", "date": d,
                         "opp": "NYK", "home": 1, "min": 30.0,
                         "pts": 25.0 if pid == "c" else 10.0, "ast": 5.0, "trb": 5.0,
                         "fga": 10, "tov": 2})
    m = context.missing_teammates(pd.DataFrame(rows))
    last = m.sort_values("date").iloc[-1]
    assert last["miss_n"] == 1 and last["miss_pts"] == 25.0
    assert m.iloc[:-1]["miss_n"].sum() == 0


def test_no_future_leakage(fake_logs):
    """A game's features must not change if we delete that game's result."""
    f1 = model.build_features(fake_logs, "pts")
    hidden = fake_logs.copy()
    target = (hidden["player_id"] == "p010") & (hidden["date"] == hidden["date"].max())
    hidden.loc[target, ["pts", "trb", "ast", "pra", "fg3"]] = np.nan
    f2 = model.build_features(hidden, "pts")
    key = lambda f: f[(f.player_id == "p010") & (f.date == fake_logs["date"].max())]
    cols = ["avg_season", "avg_last10", "dvp_factor", "dvp_recent", "vs_opp_diff", "line",
            "proj_total", "usg_last10", "hit_rate_last10_at_line"]
    pd.testing.assert_frame_equal(key(f1)[cols].reset_index(drop=True),
                                  key(f2)[cols].reset_index(drop=True))


# ------------------------------------------------------------------ NFL
def test_nfl_weather_helpers():
    from props import nfl_weather as nw
    assert nw.describe({"indoor": 1}).startswith("🏟️")
    windy = {"indoor": 0, "game_temp": 40, "game_wind": 18, "wx_precip": 0, "wx_snow": 0, "wx_gust": 30}
    assert nw.bad_weather(windy) and "wind 18" in nw.describe(windy)
    assert not nw.bad_weather({"indoor": 0, "game_temp": 65, "game_wind": 5, "wx_precip": 0, "wx_snow": 0})
    hourly = {"time": ["2026-11-22T12:00", "2026-11-22T13:00", "2026-11-22T14:00", "2026-11-22T15:00"],
              "temperature_2m": [40, 41, 42, 43], "wind_speed_10m": [10, 12, 14, 16],
              "wind_gusts_10m": [20, 22, 30, 25], "precipitation": [0, .1, .1, 0], "snowfall": [0, 0, 0, 0]}
    w = nw._window(hourly, "2026-11-22", 13, "g1", "forecast")
    assert w["wx_temp"] == 42 and w["wx_wind"] == 14.0 and w["wx_gust"] == 30 and w["wx_precip"] == 0.2


@pytest.mark.skipif(not (Path(__file__).resolve().parent.parent / "data/nfl_props/player_games.csv").exists(),
                    reason="NFL data not downloaded")
def test_nfl_predictions_respond_to_line_and_have_no_leak():
    from props import nfl_data as nd, nfl_model as nm
    games, pg, _ = nd.load()
    ctx = nm.Context(pg, games)
    # leakage: hiding a game's result must not change that game's features
    f1 = nm.build_features(pg, "receiving_yards", ctx)
    row = f1[f1["targets_last3"] > 5].iloc[-1]
    hidden = pg.copy()
    m = (hidden["player_id"] == row["player_id"]) & (hidden["game_id"] == row["game_id"])
    hidden.loc[m, ["receiving_yards", "receptions", "targets"]] = np.nan
    f2 = nm.build_features(hidden, "receiving_yards", ctx)
    pick = lambda f: f[(f.player_id == row["player_id"]) & (f.game_id == row["game_id"])].iloc[0]
    for c in ["avg_last5", "avg_season", "targets_last3", "dvp_factor", "game_wind", "implied_pts"]:
        assert pick(f1)[c] == pytest.approx(pick(f2)[c], nan_ok=True), c


def test_top_picks_filters_unrealistic_lines():
    from props import top_picks as tp
    df = pd.DataFrame([
        # center UNDER 0.5 threes on an estimated line -> dropped
        {"sport": "NBA", "stat": "fg3", "line": 0.5, "line_type": "estimated", "avg_season": 0.1},
        # normal points line near his average -> kept
        {"sport": "NBA", "stat": "pts", "line": 24.5, "line_type": "estimated", "avg_season": 25.0},
        # backup QB whose recent average is far from his role -> dropped
        {"sport": "NFL", "stat": "passing_yards", "line": 119.5, "line_type": "estimated", "avg_season": 60.0},
        # real sportsbook line -> always kept
        {"sport": "NBA", "stat": "fg3", "line": 0.5, "line_type": "sportsbook", "avg_season": None},
    ])
    assert tp.realistic(df).tolist() == [False, True, False, True]


def test_hit_rate_windows():
    from props import top_picks as tp
    g = pd.DataFrame({"season": [2026] * 4 + [2027] * 16, "opp": ["BOS", "NYK"] * 10,
                      "pts": list(range(20))})
    hr = tp.hit_rates(g, "pts", 14.5, "OVER", "BOS").set_index("Window")
    assert hr.loc["Last 5", "Hits"] == 5 and hr.loc["Last 10", "Hits"] == 5
    assert hr.loc["Last 15", "Games"] == 15 and hr.loc["This season", "Games"] == 16
    assert hr.loc["Head to head vs BOS", "Games"] == 10
    under = tp.hit_rates(g, "pts", 14.5, "UNDER", "BOS").set_index("Window")
    assert under.loc["Last 5", "Hits"] == 0
    assert tp.opponent_of("NBA", "LAL", "@ PHO") == "PHO"
    assert tp.opponent_of("NFL", "GB", "MIN @ GB") == "MIN"


# ------------------------------------------------------------ oddsscraper
class FakeOddsClient:
    """Mimics oddsscraper.OddsClient output (The Odds API v4 shapes)."""
    def __init__(self, *_):
        pass

    def player_props(self, sport, market=None, player=None, odds_format="decimal", **kw):
        if sport != "basketball_nba":
            return [{"event_id": "e2", "commence_time": "2026-11-23T18:00:00Z", "home_team": "Houston Texans",
                     "away_team": "Indianapolis Colts", "bookmaker": "fanduel", "market": "player_anytime_td",
                     "player": "Jonathan Taylor", "side": "Yes", "point": None, "price": -150}]
        base = {"event_id": "e1", "commence_time": "2099-11-16T03:00:00Z", "home_team": "Phoenix Suns",
                "away_team": "Los Angeles Lakers", "market": "player_points", "player": "Luka Doncic"}
        return [dict(base, bookmaker="draftkings", side="Over", point=33.5, price=-115),
                dict(base, bookmaker="draftkings", side="Under", point=33.5, price=-105),
                dict(base, bookmaker="fanduel", side="Over", point=34.5, price=-110),
                dict(base, bookmaker="fanduel", side="Under", point=34.5, price=-110),
                dict(base, bookmaker="prizepicks", side="Over", point=34.0, price=-122)]

    def odds(self, sport, markets=None, odds_format="decimal", **kw):
        return [{"commence_time": "2099-11-16T03:00:00Z", "home_team": "Phoenix Suns",
                 "away_team": "Los Angeles Lakers", "bookmakers": [{"key": "draftkings", "markets": [
                     {"key": "spreads", "outcomes": [{"name": "Phoenix Suns", "point": 4.5},
                                                     {"name": "Los Angeles Lakers", "point": -4.5}]},
                     {"key": "totals", "outcomes": [{"name": "Over", "point": 231.5}]}]}]}]

    def alternate_spreads(self, sport, team=None, **kw):
        return [{"team": "Phoenix Suns", "point": 8.5, "price": -250, "bookmaker": "fanduel",
                 "home_team": "Phoenix Suns", "away_team": "Los Angeles Lakers",
                 "commence_time": "2099-11-16T03:00:00Z"}]


def test_scraped_lines_conversion(tmp_path, monkeypatch):
    from props import scraped_lines as sl
    db = tmp_path / "oddsscraper.db"
    db.write_text("")
    monkeypatch.setenv("ODDSSCRAPER_DB", str(db))
    monkeypatch.setattr(sl, "_client", lambda: FakeOddsClient())
    nba = sl.prop_lines("NBA").set_index("book")
    assert nba.loc["draftkings", "line"] == 33.5 and nba.loc["draftkings", "over_odds"] == -115
    assert nba.loc["draftkings", "home_team"] == "PHO" and nba.loc["draftkings", "stat"] == "pts"
    assert nba.loc["prizepicks", "over_odds"] == -122          # DFS: nominal pick'em price
    gl = sl.game_lines("NBA").iloc[0]
    assert gl["home_team"] == "PHO" and gl["home_spread"] == 4.5 and gl["total"] == 231.5
    assert len(sl.player_lines("NBA", "Luka Dončić", "pts")) == 3
    assert sl.alternate_spreads("NBA").iloc[0]["team"] == "PHO"


def test_abbreviated_names():
    from props.slate import match_abbreviated
    ids = {"jayson tatum": "tatumja01", "jalen brunson": "brunsja01", "jalen green": "greenja05",
           "josh green": "greenjo02"}
    assert match_abbreviated("J. Tatum", ids) == "tatumja01"
    assert match_abbreviated("J. Green", ids) is None          # ambiguous -> no guess
    assert match_abbreviated("Jayson Tatum", ids) is None      # full names handled elsewhere


def test_research_pages_read_oddsscraper(tmp_path, monkeypatch):
    """sports_stats_db's imported-props section also sees oddsscraper lines."""
    import sqlite3
    from scrapers.csv_import import latest_player_props
    db = tmp_path / "oddsscraper.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE v_odds (sport_key, market_group, commence_time, home_team, away_team, "
                "bookmaker, market, outcome, player, point, price_american, last_update)")
    rows = [("americanfootball_nfl", "player_props", "2099-11-22T18:00:00Z", "Green Bay Packers",
             "Minnesota Vikings", "draftkings", "player_pass_yds", side, "Jordan Love", 245.5, price,
             "2099-11-20T12:00:00Z") for side, price in (("Over", -115), ("Under", -105))]
    con.executemany("INSERT INTO v_odds VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    monkeypatch.setenv("ODDSSCRAPER_DB", str(db))
    csv = tmp_path / "odds.csv"
    csv.write_text("sport,league,player_name,team,opponent,game_start_time,stat_type,line,over_odds,"
                   "under_odds,source,captured_at\n")
    p = latest_player_props("nfl", "Jordan Love", csv)
    assert len(p) == 1
    r = p.iloc[0]
    assert r["stat_type"] == "passing_yards" and r["line"] == 245.5 and r["over_odds"] == -115
    assert r["source"] == "draftkings"


def test_calibration_pulls_in_overconfident_predictions():
    """A model that says 5% when the truth is 25% gets corrected toward 25%."""
    rng = np.random.default_rng(1)
    raw = np.r_[np.full(2000, 0.05), np.full(2000, 0.95)]
    truth = np.r_[rng.random(2000) < 0.25, rng.random(2000) < 0.75].astype(float)
    cal, honest = model.fit_calibrator(raw, truth)
    assert abs(cal.predict([0.05])[0] - 0.25) < 0.05
    assert abs(cal.predict([0.95])[0] - 0.75) < 0.05
    assert abs(honest[:2000].mean() - 0.25) < 0.05


def test_calibrator_is_version_proof():
    """Saved calibrators hold plain floats only (no scikit-learn objects), so models
    trained on one scikit-learn version still load on another."""
    import pickle
    cal = model.PlattCalibrator().fit(np.r_[np.full(500, .1), np.full(500, .9)],
                                      np.r_[np.zeros(400), np.ones(100), np.zeros(100), np.ones(400)])
    assert set(vars(cal)) == {"a", "b"} and all(isinstance(v, float) for v in vars(cal).values())
    assert b"sklearn" not in pickle.dumps(cal)
    assert 0.15 < cal.predict([0.1])[0] < 0.25


def test_main_line_per_book():
    from props import scraped_lines as sl
    df = pd.DataFrame({
        "event_id": ["e"] * 3, "player": ["Devin Booker"] * 3, "stat": ["pts"] * 3, "book": ["fanduel"] * 3,
        "line": [7.5, 24.5, 29.5], "over_odds": [-108, -115, 150], "under_odds": [-122, -111, np.nan]})
    assert sl.main_lines(df, {("devin booker", "pts"): 25.1})["line"].tolist() == [24.5]
    assert sl.main_lines(df)["line"].tolist() == [24.5]          # no average: most evenly priced two-sided line


def test_book_filter(tmp_path, monkeypatch):
    from props import scraped_lines as sl
    monkeypatch.setattr(sl, "_settings_file", lambda: tmp_path / "s.json")
    assert sl.allowed_books() == []
    sl.set_allowed_books(["fanduel", "bovada"])
    assert sl.allowed_books() == ["bovada", "fanduel"]


def test_with_without_split():
    """Bench guard plays more and shoots more in the games the star missed."""
    from props import onoff
    rows = []
    dates = pd.date_range("2026-10-21", periods=12, freq="2D")
    for i, d in enumerate(dates):
        star_out = i in (3, 7, 10)
        if not star_out:
            rows.append({"player_id": "star", "team": "LAL", "date": d, "min": 36, "pts": 28})
        rows.append({"player_id": "bench", "team": "LAL", "date": d,
                     "min": 32 if star_out else 22, "pts": 18 if star_out else 9})
        rows.append({"player_id": "big", "team": "LAL", "date": d, "min": 30, "pts": 12})
    logs = pd.DataFrame(rows).assign(season=2027)
    usage = logs[["player_id", "date"]].assign(usg=0.2)
    oo = onoff.OnOff(logs, usage, 2027)
    assert len(oo.absent_dates("star", "LAL")) == 3
    assert oo.ahead_of("star", "bench", "LAL") and not oo.ahead_of("bench", "star", "LAL")
    s = oo.split("bench", "LAL", ["star"], "pts")
    assert s["without"]["n"] == 3 and s["without"]["min"] == 32 and s["with"]["min"] == 22
    assert abs(s["weight"] - 3 / 7) < 1e-9
    assert s["without"]["rate"] > s["with"]["rate"]
