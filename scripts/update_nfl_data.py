"""Download/refresh NFL data (nflverse) + game weather (Open-Meteo), then rebuild tables.

    python scripts/update_nfl_data.py            # ~1-2 minutes
    python scripts/update_nfl_data.py --no-weather
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from props import nfl_data as nd, nfl_weather as nw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-weather", action="store_true")
    args = ap.parse_args()
    nd.download()
    if not args.no_weather:
        try:
            nw.update()
        except Exception as e:
            print(f"Weather update failed ({e}) - using what's saved.")
    nd.build_all()
    print("Next: python scripts/train_nfl_model.py")


if __name__ == "__main__":
    main()
