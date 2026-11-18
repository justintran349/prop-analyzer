from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

import pandas as pd

ODDS_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "odds.csv"
SNAPSHOT_COLUMNS = [
    "sport",
    "league",
    "player_name",
    "team",
    "opponent",
    "game_start_time",
    "stat_type",
    "line",
    "over_odds",
    "under_odds",
    "source",
    "captured_at",
]

STAT_ALIASES = {
    "points_rebounds_assists": "pra",
    "pts_reb_ast": "pra",
    "points_plus_rebounds_plus_assists": "pra",
    "points_plus_rebounds": "points_rebounds",
    "points_plus_assists": "points_assists",
    "rebounds_plus_assists": "rebounds_assists",
    "hits_plus_runs_plus_rbis": "hits_runs_rbi",
    "hits_plus_runs_plus_rbi": "hits_runs_rbi",
    "3pm": "three_pointers_made",
    "3_pt_made": "three_pointers_made",
    "3_pointers_made": "three_pointers_made",
    "three_pointers": "three_pointers_made",
    "3pa": "three_pointers_attempted",
    "3_pt_attempted": "three_pointers_attempted",
    "3_pointers_attempted": "three_pointers_attempted",
    "fgm": "field_goals_made",
    "fga": "field_goals_attempted",
    "ftm": "free_throws_made",
    "fta": "free_throws_attempted",
    "oreb": "offensive_rebounds",
    "dreb": "defensive_rebounds",
    "steals_plus_blocks": "stocks",
    "blocks_plus_steals": "stocks",
    "fantasy_points": "fantasy_score",
    "hitter_fantasy_score": "hitter_fantasy_score",
    "pitcher_fantasy_score": "pitcher_fantasy_score",
    "pitcher_strikeouts": "strikeouts",
    "batter_strikeouts": "strikeouts",
    "pass_yards": "passing_yards",
    "pass_tds": "passing_tds",
    "pass_touchdowns": "passing_tds",
    "rush_yards": "rushing_yards",
    "rush_tds": "rushing_tds",
    "rush_touchdowns": "rushing_tds",
    "receiving_tds": "receiving_tds",
    "receiving_touchdowns": "receiving_tds",
    "pass_plus_rush_yards": "pass_rush_yards",
    "rushing_plus_receiving_yards": "rush_receive_yards",
    "passing_plus_rushing_plus_receiving_tds": "total_tds",
    "passing_rushing_receiving_tds": "total_tds",
    "receptions": "receptions",
}


def utc_timestamp(value: object | None = None) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.Timestamp(datetime.now(timezone.utc))
    return timestamp.isoformat().replace("+00:00", "Z")


def normalize_stat_type(value: object) -> str:
    text = str(value).strip().lower().replace("&", " plus ")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return STAT_ALIASES.get(text, text)


def stat_column_map(options: dict[str, str]) -> dict[str, str]:
    columns: dict[str, str] = {}
    for label, column in options.items():
        columns[normalize_stat_type(label)] = column
        columns[normalize_stat_type(column)] = column
    return columns


def analyze_props_vs_history(
    props: pd.DataFrame,
    game_logs: pd.DataFrame,
    stat_columns: dict[str, str],
    lookback: int | None = 5,
) -> pd.DataFrame:
    results = []
    ordered_logs = game_logs.copy()
    if "date" in ordered_logs:
        ordered_logs = ordered_logs.sort_values("date", ascending=False)

    for prop in props.to_dict("records"):
        stat_type = normalize_stat_type(prop["stat_type"])
        column = stat_columns.get(stat_type)
        values = pd.Series(dtype="float64")
        if column in ordered_logs:
            values = pd.to_numeric(ordered_logs[column], errors="coerce").dropna()
            if lookback is not None:
                values = values.head(lookback)

        line = float(prop["line"])
        over_count = int(values.gt(line).sum())
        under_count = int(values.lt(line).sum())
        push_count = int(values.eq(line).sum())
        decided = over_count + under_count
        row = {column_name: prop.get(column_name) for column_name in SNAPSHOT_COLUMNS}
        row.update(
            {
                "games_checked": len(values),
                "over_hits": over_count,
                "under_hits": under_count,
                "pushes": push_count,
                "over_rate": 100 * over_count / decided if decided else 0.0,
                "under_rate": 100 * under_count / decided if decided else 0.0,
                "average": float(values.mean()) if len(values) else None,
                "history_stat_found": column in ordered_logs,
            }
        )
        results.append(row)
    return pd.DataFrame(results)