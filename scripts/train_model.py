"""Train the minutes model + one over/under model per stat, and print how they did.

    python scripts/train_model.py            # all stats
    python scripts/train_model.py pts ast    # just some
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from props import config, model  # noqa: E402
from props.analysis import attach_positions, load_data  # noqa: E402


def main():
    stats = sys.argv[1:] or list(config.STATS)
    logs, players = load_data()
    if logs.empty:
        sys.exit("No data yet - run: python scripts/update_data.py")
    logs = attach_positions(logs, players)
    lines = (pd.read_csv(config.LINE_HISTORY_CSV, parse_dates=["date"])
             if config.LINE_HISTORY_CSV.exists() else None)

    res = model.train_all(logs, stats, lines)
    print(f"\n{'Stat':<16}{'Accuracy':>9}{'AUC':>7}{'Brier':>8}{'Baseline':>9}{'>=60% sure':>12}{'picks':>7}")
    for stat in stats:
        m = res[stat]
        ca = f"{m['confident_accuracy']:.1%}" if m["confident_accuracy"] is not None else "-"
        print(f"{config.STATS[stat]:<16}{m['accuracy']:>9.1%}{m['auc']:>7.3f}{m['brier']:>8.4f}"
              f"{m['baseline_brier']:>9.4f}{ca:>12}{m['confident_picks']:>7}")
    real = max(res[s]["real_line_rows"] for s in stats)
    print(f"\nReal sportsbook lines used: {real:,}"
          + ("  (results are vs ESTIMATED lines - real accuracy will be lower)" if real < 1000 else ""))
    print("At -110 odds you need to win 52.4% of bets just to break even.")


if __name__ == "__main__":
    main()
