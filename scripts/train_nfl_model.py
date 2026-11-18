"""Train one NFL over/under model per stat and print how each did.

    python scripts/train_nfl_model.py
    python scripts/train_nfl_model.py passing_yards receptions
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from props import nfl_data as nd, nfl_model as nm  # noqa: E402


def main():
    stats = sys.argv[1:] or list(nd.STATS)
    games, pg, _ = nd.load()
    lines = None
    if nd.LINE_HISTORY_CSV.exists():
        lines = pd.read_csv(nd.LINE_HISTORY_CSV, parse_dates=["date"])
        from props.nfl_slate import NFLLeague  # map names -> ids
        lg = NFLLeague()
        lines["player_id"] = lines["player"].map(lg.player_id)
    res = nm.train_all(pg, games, stats, lines)
    print(f"\n{'Stat':<18}{'Accuracy':>9}{'AUC':>7}{'Brier':>8}{'Baseline':>9}{'>=60% sure':>12}{'Bad weather':>13}")
    for s in stats:
        m = res[s]
        ca = f"{m['confident_accuracy']:.1%}" if m["confident_accuracy"] is not None else "-"
        bw = f"{m['bad_weather_accuracy']:.1%}" if m["bad_weather_accuracy"] is not None else "-"
        print(f"{nd.LABEL[s]:<18}{m['accuracy']:>9.1%}{m['auc']:>7.3f}{m['brier']:>8.4f}"
              f"{m['baseline_brier']:>9.4f}{ca:>12}{bw:>13}")
    print("\nResults are vs ESTIMATED lines until real lines are collected (Odds API).")


if __name__ == "__main__":
    main()
