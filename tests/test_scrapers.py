from io import StringIO
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scrapers.common import analyze_props_vs_history, stat_column_map
from scrapers.csv_import import latest_player_props, normalize_csv_rows


CSV_HEADER = (
    "sport,league,player_name,team,opponent,game_start_time,stat_type,line,"
    "over_odds,under_odds,source,captured_at\n"
)


class OddsCsvBackendTests(unittest.TestCase):
    def test_player_lookup_reads_updated_backend_file_each_time(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "odds.csv"
            row = (
                "NBA,NBA,Jayson Tatum,BOS,NYK,2030-01-01T00:00:00Z,Points,"
                "25.5,,,PrizePicks,2026-09-30T12:00:00Z\n"
            )
            csv_path.write_text(CSV_HEADER + row, encoding="utf-8")

            first = latest_player_props("nba", "Jayson Tatum", csv_path)
            csv_path.write_text(
                CSV_HEADER + row.replace(",25.5,", ",27.5,").replace(
                    "2026-09-30T12:00:00Z", "2026-09-30T13:00:00Z"
                ),
                encoding="utf-8",
            )
            second = latest_player_props("nba", "Jayson Tatum", csv_path)

            self.assertEqual(first.iloc[0]["line"], 25.5)
            self.assertEqual(second.iloc[0]["line"], 27.5)
            self.assertEqual(second.iloc[0]["stat_type"], "points")

    def test_invalid_rows_are_reported_and_skipped(self):
        source = pd.read_csv(
            StringIO(
                CSV_HEADER
                + "NFL,NFL,Player One,BUF,NYJ,2030-01-01T00:00:00Z,Passing Yards,"
                "not-a-number,,,PrizePicks,\n"
            )
        )

        records, errors = normalize_csv_rows(source)

        self.assertEqual(records, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("line must be numeric", errors[0]["Error"])

    def test_hit_rates_compare_latest_games_to_imported_line(self):
        props = pd.DataFrame(
            [{"sport": "nba", "stat_type": "points", "line": 20.0}]
        )
        logs = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=5),
                "points": [30, 20, 15, 10, 25],
            }
        )

        result = analyze_props_vs_history(
            props,
            logs,
            stat_column_map({"Points": "points"}),
            lookback=5,
        ).iloc[0]

        self.assertEqual(result["games_checked"], 5)
        self.assertEqual(result["over_hits"], 2)
        self.assertEqual(result["under_hits"], 2)
        self.assertEqual(result["pushes"], 1)
        self.assertEqual(result["over_rate"], 50.0)
        self.assertEqual(result["under_rate"], 50.0)


if __name__ == "__main__":
    unittest.main()