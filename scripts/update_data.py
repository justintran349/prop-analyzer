"""Download/refresh NBA data.

    python scripts/update_data.py                   # fast: NBA.com game logs (~1 minute)
    python scripts/update_data.py --source bref     # slow: Basketball-Reference only (~35 min)

Either way, positions (PG/SG/SF/PF/C) and the schedule come from
Basketball-Reference (about 10 pages, cached).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from props import config, scraper  # noqa: E402


def from_nba_com(seasons):
    from props import nba_com
    bref = pd.concat([scraper.get_players(s) for s in seasons], ignore_index=True)
    raws = []
    for s in seasons:
        print(f"NBA.com game log {nba_com.season_str(s)} ...", end=" ", flush=True)
        raw = nba_com.fetch_league_gamelog(s)
        print(f"{len(raw):,} rows")
        raws.append(raw)
    logs, players = nba_com.convert(pd.concat(raws, ignore_index=True), bref)
    missing = logs.loc[logs["player_id"].str.startswith("nba"), "player_id"].nunique()
    if missing:
        print(f"  {missing} players couldn't be matched to a BRef position "
              "(add them to NICKNAMES in props/nba_com.py)")
    return logs, players


def from_bref(seasons, min_mpg, min_games):
    from tqdm import tqdm
    all_players, all_logs = [], []
    for season in seasons:
        players = scraper.get_players(season)
        all_players.append(players)
        todo = players[(players["mpg"] >= min_mpg) & (players["games"] >= min_games)]
        print(f"{season}: scraping {len(todo)} player game logs")
        for pid in tqdm(todo["player_id"], unit="player"):
            try:
                gl = scraper.get_gamelog(pid, season)
                if not gl.empty:
                    all_logs.append(gl)
            except RuntimeError as e:   # rate limited - save what we have
                print(f"\n{e}")
                break
            except Exception as e:
                print(f"\n  skipped {pid}: {e}")
    return pd.concat(all_logs, ignore_index=True), pd.concat(all_players, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["nba", "bref"], default="nba")
    ap.add_argument("--seasons", type=int, nargs="+", default=config.TRAINING_SEASONS)
    ap.add_argument("--min-mpg", type=float, default=15.0, help="bref source only")
    ap.add_argument("--min-games", type=int, default=3, help="bref source only")
    ap.add_argument("--no-schedule", action="store_true")
    args = ap.parse_args()

    if args.source == "nba":
        try:
            logs, players = from_nba_com(args.seasons)
        except Exception as e:
            sys.exit(f"NBA.com request failed ({e}).\n"
                     "NBA.com sometimes blocks requests - try again, or use --source bref")
    else:
        logs, players = from_bref(args.seasons, args.min_mpg, args.min_games)

    players.to_csv(config.PLAYERS_CSV, index=False)
    logs.to_csv(config.GAMELOGS_CSV, index=False)
    print(f"Saved {len(logs):,} player-games (latest {pd.to_datetime(logs['date']).max():%b %d})")

    if not args.no_schedule:
        sched = scraper.get_schedule(config.SEASON)
        sched.to_csv(config.SCHEDULE_CSV, index=False)
        print(f"Saved {len(sched):,} scheduled games")
    print("Next: python scripts/train_model.py")


if __name__ == "__main__":
    main()
