"""Pull fresh sportsbook + DFS lines with oddsscraper (one pass), then show what came in.

    python scripts/scrape_lines.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from props import scraped_lines as sl  # noqa: E402


def main():
    print("Scraping DraftKings, FanDuel, Bovada, PrizePicks, Underdog (a few minutes)...", flush=True)
    ok, msg = sl.run_scrape()
    print(("OK " if ok else "FAILED ") + msg)
    print(sl.status())
    for r in sl.scrape_health():
        mark = "ok " if r["ok"] else "ERR"
        print(f"  {mark} {r['source']:<11} {r['kind']:<22} {r['sport_key'] or '':<22} "
              f"{r['n_events'] or 0:>4} games  {(r['error'] or '')[:80]}")
    for sport in ("NBA", "NFL"):
        df = sl.prop_lines(sport)
        books = ", ".join(f"{b} {n}" for b, n in df["book"].value_counts().items()) if len(df) else "none"
        print(f"{sport}: {len(df)} upcoming prop lines ({books})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
