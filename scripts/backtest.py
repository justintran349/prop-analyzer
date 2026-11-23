"""Re-play this NBA season out-of-sample for the Results page.

    python scripts/backtest.py                 # from opening night
    python scripts/backtest.py 2026-11-01      # from a date

Models are trained only on games before the start date, then every rotation
player's game since is scored at the line Top Picks would have used and graded.
Takes a few minutes. Writes data/backtest_picks.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from props import history  # noqa: E402

if __name__ == "__main__":
    bt = history.backtest_nba(sys.argv[1] if len(sys.argv) > 1 else None)
    for label, d in (("All picks", bt), ("Confidence 60%+", bt[bt["confidence"] >= 0.6]),
                     ("Top 10 a day (60%+)", history.top_per_day(bt[bt["confidence"] >= 0.6], 10))):
        r = history.record(d)
        print(f"{label:<22} {r['wins']}-{r['losses']}  {r['hit_rate']:.1%}  {r['units']:+.1f} units")
    print("Note: graded vs ESTIMATED lines at -110 - real sportsbook results will be lower.")
