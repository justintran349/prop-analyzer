"""The daily job - run every morning (scripts/install_daily_job.sh sets that up).

    python scripts/daily.py              # everything
    python scripts/daily.py --no-update  # skip the data refresh
    python scripts/daily.py --retrain    # also retrain the models (default: Mondays)

Steps
  1. refresh game logs + schedule (NBA.com / Basketball-Reference)
  2. grade yesterday's tracked picks
  3. pull the ESPN injury report
  4. pull sportsbook lines + game totals (if ODDS_API_KEY is set) -> saved for training
  5. score today's slate -> data/slates/YYYY-MM-DD.csv
  6. auto-track the model's best picks (paper trading) so the tracker fills itself
  7. phone alerts for big edges and line moves (if NTFY_TOPIC is set)
  8. retrain models on Mondays
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from props import alerts, config, context, odds, slate, tracker  # noqa: E402

AUTO_TRACK_MIN_EDGE = 0.05
AUTO_TRACK_MAX = 15


def step(msg):
    print(f"\n== {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-auto-track", action="store_true")
    args = ap.parse_args()
    today = pd.Timestamp.today().normalize()
    print(f"Daily run {pd.Timestamp.now():%Y-%m-%d %H:%M}")

    if not args.no_update:
        step("Refreshing data")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "update_data.py")], check=False)

    step("Injury report")
    try:
        inj = context.fetch_injuries()
        print(f"{len(inj)} players listed")
    except Exception as e:
        print(f"ESPN injuries failed: {e}")

    lines = odds.load_manual_lines()
    if odds.odds_api_available():
        step("Sportsbook lines")
        try:
            gl, _ = odds.fetch_game_lines()
            api, remaining = odds.fetch_odds_api_lines()
            print(f"{len(api)} prop lines, {len(gl)} game lines (API credits left: {remaining})")
            lines = pd.concat([lines, api], ignore_index=True)
        except Exception as e:
            print(f"Odds API failed: {e}")

    step("oddsscraper lines")
    try:
        from props import scraped_lines
        if scraped_lines.installed():
            ok, msg = scraped_lines.run_scrape()
            print(("scrape ok " if ok else "scrape failed: ") + msg)
        print(scraped_lines.status())
        nba_sc = scraped_lines.prop_lines("NBA")
        if not nba_sc.empty:
            odds.record_history(nba_sc)
            lines = pd.concat([lines, nba_sc], ignore_index=True)
        nfl_sc = scraped_lines.prop_lines("NFL")
        if not nfl_sc.empty:
            from props import nfl_data
            h = nfl_data.LINE_HISTORY_CSV
            keep = ["date", "player", "stat", "line", "over_odds", "under_odds", "book"]
            allh = pd.concat([pd.read_csv(h, parse_dates=["date"]), nfl_sc[keep]]) if h.exists() else nfl_sc[keep]
            allh.drop_duplicates(["date", "player", "stat", "book"], keep="last").to_csv(h, index=False)
        print(f"{len(nba_sc)} NBA + {len(nfl_sc)} NFL prop lines saved for training")
    except Exception as e:
        print(f"oddsscraper step skipped: {e}")

    lg = slate.League()

    step("Grading tracked picks")
    print(f"{tracker.grade(lg.logs)} picks graded")
    s = tracker.summary()
    if s["graded"]:
        print(f"Record {s['wins']}-{s['losses']}, win rate {s['win_rate']:.1%}, "
              f"profit {s['profit']:+.2f}u, ROI {s['roi']:+.1%}")

    step("Scoring today's slate")
    lines = lines[pd.to_datetime(lines["date"]) >= today] if not lines.empty else lines
    board = slate.score_lines(lg, lines) if not lines.empty else pd.DataFrame()
    games = slate.todays_games(lg, today)
    print(f"{len(games)} games today, {len(board)} lines scored")
    if not board.empty:
        board.to_csv(config.SLATES_DIR / f"{today:%Y-%m-%d}.csv", index=False)
        print(board.head(10)[["player", "stat", "line", "pick", "prob_pick", "edge", "book"]]
              .to_string(index=False))
        if not args.no_auto_track:
            best = (board[(board["edge"] >= AUTO_TRACK_MIN_EDGE) & (board["date"] == today)]
                    .drop_duplicates(["player_id", "stat"]).head(AUTO_TRACK_MAX))
            added = sum(tracker.add(r.date, r.player, r.player_id, r.stat, r.line, r.pick,
                                    r.pick_odds, r.prob_pick, r.book, source="auto")
                        for r in best.itertuples())
            print(f"auto-tracked {added} picks")

    step("NFL")
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "update_nfl_data.py")], check=False)
        from props import nfl_slate
        if odds.odds_api_available() and today.dayofweek in (3, 6, 0):   # Thu / Sun / Mon game days
            nfl_api, _ = nfl_slate.fetch_odds_api_lines()
            print(f"{len(nfl_api)} NFL prop lines saved")
        nlg = nfl_slate.NFLLeague()
        print(f"{tracker.grade_nfl(nlg.pg)} NFL picks graded")
        nfl_lines = nfl_slate.load_lines()
        nfl_board = nlg.score_lines(nfl_lines) if not nfl_lines.empty else pd.DataFrame()
        if not nfl_board.empty:
            nfl_board.to_csv(config.SLATES_DIR / f"nfl_{today:%Y-%m-%d}.csv", index=False)
            print(nfl_board.head(10)[["player", "stat", "line", "pick", "prob_pick", "edge", "weather"]]
                  .to_string(index=False))
        if today.dayofweek == 1:   # Tuesday: retrain with last week's games
            subprocess.run([sys.executable, str(ROOT / "scripts" / "train_nfl_model.py")], check=False)
    except Exception as e:
        nfl_board = pd.DataFrame()
        print(f"NFL step failed: {e}")

    if alerts.enabled():
        step("Alerts")
        msg = alerts.edge_alerts(board)
        if msg:
            alerts.send(f"🏀 {msg.count(chr(10)) + 1} props with edge today", msg, priority="high")
        if not nfl_board.empty:
            hot = nfl_board[nfl_board["edge"] >= config.ALERT_MIN_EDGE].head(10)
            if not hot.empty:
                alerts.send(f"🏈 {len(hot)} NFL props with edge", "\n".join(
                    f"{r.player} {r.pick} {r.line} {r.stat} {r.game} ({r.prob_pick:.0%}) {r.weather}"
                    for r in hot.itertuples()), tags="football")
        moves = alerts.move_alerts(odds.line_moves())
        if moves:
            alerts.send("📉 Line moves", moves)
        print("sent")

    step("Grade pick history")
    try:
        from props import history
        print(f"{history.grade_history()} logged Top Picks graded (see the Results page)")
    except Exception as e:
        print(f"History grading failed: {e}")

    step("Top picks")
    try:
        from props import top_picks
        tp = top_picks.build(log=print)
        print(f"{len(tp)} picks ranked -> data/top_picks.csv (shown when you open the app)")
    except Exception as e:
        print(f"Top picks failed: {e}")

    if args.retrain or today.dayofweek == 0:
        step("Retraining models")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "train_model.py")], check=False)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "backtest.py")], check=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
