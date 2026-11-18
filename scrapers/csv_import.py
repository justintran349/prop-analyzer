from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path

import pandas as pd

from scrapers.common import (
    ODDS_CSV_PATH,
    SNAPSHOT_COLUMNS,
    normalize_stat_type,
    utc_timestamp,
)

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = {
    "sport",
    "league",
    "player_name",
    "team",
    "opponent",
    "game_start_time",
    "stat_type",
    "line",
    "source",
}
SUPPORTED_SPORTS = {
    "mlb": "mlb",
    "baseball": "mlb",
    "nba": "nba",
    "basketball": "nba",
    "nfl": "nfl",
    "football": "nfl",
}


def normalize_csv_rows(
    source: pd.DataFrame,
    captured_at_default: str | None = None,
) -> tuple[list[dict], list[dict]]:
    frame = source.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    aliases = {"sportsbook": "source", "player": "player_name", "stat": "stat_type"}
    frame = frame.rename(columns=aliases)
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing_columns))

    captured_at_default = captured_at_default or utc_timestamp(
        datetime.now(timezone.utc)
    )
    records = []
    errors = []
    for index, row in frame.iterrows():
        row_number = int(index) + 2
        try:
            values = {column: row.get(column) for column in REQUIRED_COLUMNS}
            missing_values = [
                column
                for column, value in values.items()
                if pd.isna(value) or not str(value).strip()
            ]
            if missing_values:
                raise ValueError("blank required fields: " + ", ".join(missing_values))

            sport_input = str(row["sport"]).strip().lower()
            if sport_input not in SUPPORTED_SPORTS:
                raise ValueError(f"unsupported sport {sport_input!r}")

            game_start = pd.to_datetime(row["game_start_time"], utc=True, errors="coerce")
            if pd.isna(game_start):
                raise ValueError("invalid game_start_time")
            line = pd.to_numeric(pd.Series([row["line"]]), errors="coerce").iloc[0]
            if pd.isna(line):
                raise ValueError("line must be numeric")

            captured = row.get("captured_at")
            captured_at = (
                captured_at_default
                if captured is None or pd.isna(captured) or not str(captured).strip()
                else utc_timestamp(captured)
            )
            if pd.isna(pd.to_datetime(captured_at, utc=True, errors="coerce")):
                raise ValueError("invalid captured_at")

            normalized = {
                "sport": SUPPORTED_SPORTS[sport_input],
                "league": str(row["league"]).strip(),
                "player_name": str(row["player_name"]).strip(),
                "team": str(row["team"]).strip(),
                "opponent": str(row["opponent"]).strip(),
                "game_start_time": game_start.isoformat().replace("+00:00", "Z"),
                "stat_type": normalize_stat_type(row["stat_type"]),
                "line": float(line),
                "source": str(row["source"]).strip(),
                "captured_at": captured_at,
            }
            if not normalized["stat_type"]:
                raise ValueError("stat_type cannot be blank")

            for odds_column in ("over_odds", "under_odds"):
                odds = row.get(odds_column)
                if odds is None or pd.isna(odds) or not str(odds).strip():
                    normalized[odds_column] = None
                else:
                    numeric_odds = pd.to_numeric(
                        pd.Series([odds]), errors="coerce"
                    ).iloc[0]
                    if pd.isna(numeric_odds):
                        raise ValueError(f"{odds_column} must be numeric when supplied")
                    normalized[odds_column] = float(numeric_odds)

            records.append({column: normalized.get(column) for column in SNAPSHOT_COLUMNS})
        except (TypeError, ValueError) as error:
            LOGGER.warning("Skipping invalid odds CSV row %s: %s", row_number, error)
            errors.append({"CSV row": row_number, "Error": str(error)})

    return records, errors


def read_backend_odds_csv(
    csv_path: Path | str = ODDS_CSV_PATH,
) -> tuple[pd.DataFrame, list[dict]]:
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS), []
    if path.stat().st_size > 15 * 1024 * 1024:
        raise ValueError("Backend odds.csv exceeds the 15 MB read limit")

    try:
        source = pd.read_csv(path)
        file_timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        records, errors = normalize_csv_rows(
            source,
            captured_at_default=utc_timestamp(file_timestamp),
        )
    except Exception:
        LOGGER.exception("Failed to read backend odds CSV %s", path)
        raise

    return pd.DataFrame(records, columns=SNAPSHOT_COLUMNS), errors


# ---------------------------------------------------------------------------
# Optional second source: the local oddsscraper database (~/Desktop/oddsscraper).
# Read-only; nothing is scraped from here. Rows are converted to the same
# snapshot columns as odds.csv so the pages treat them identically.
ODDSSCRAPER_SPORTS = {"nba": "basketball_nba", "nfl": "americanfootball_nfl", "mlb": "baseball_mlb"}
MARKET_TO_STAT = {
    "pass_yds": "passing_yards", "pass_tds": "passing_tds", "pass_completions": "completions",
    "pass_attempts": "attempts", "pass_interceptions": "interceptions", "rush_yds": "rushing_yards",
    "rush_attempts": "carries", "reception_yds": "receiving_yards", "receptions": "receptions",
    "rush_reception_yds": "rush_receive_yards", "threes": "three_pointers_made",
    "blocks_steals": "stocks", "points_rebounds_assists": "pra", "points_rebounds": "points_rebounds",
    "points_assists": "points_assists", "rebounds_assists": "rebounds_assists",
    "hits_runs_rbis": "hits_runs_rbi", "rbis": "rbis",
}


def _oddsscraper_db() -> Path:
    import os
    return Path(os.environ.get("ODDSSCRAPER_DB") or "~/Desktop/oddsscraper/oddsscraper.db").expanduser()


def _name_key(name: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(text.replace(".", "").replace("'", "").casefold().split())


def oddsscraper_props(sport: str, player_name: str) -> pd.DataFrame:
    """Upcoming lines for one player from the oddsscraper database (empty if not set up)."""
    import sqlite3
    path, key = _oddsscraper_db(), ODDSSCRAPER_SPORTS.get(sport.strip().lower())
    if key is None or not path.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        raw = pd.read_sql_query(
            "SELECT commence_time, home_team, away_team, bookmaker, market, outcome, player, point, "
            "price_american, last_update FROM v_odds WHERE sport_key = ? AND market_group = 'player_props'",
            con, params=(key,))
        con.close()
    except Exception as error:
        LOGGER.warning("Could not read oddsscraper database %s: %s", path, error)
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    raw = raw[raw["player"].map(_name_key) == _name_key(player_name)]
    raw = raw[raw["outcome"].str.lower().isin(["over", "under"]) & raw["point"].notna()]
    if raw.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    wide = raw.pivot_table(index=["commence_time", "home_team", "away_team", "bookmaker", "market",
                                  "player", "point"], columns=raw["outcome"].str.lower(),
                           values="price_american", aggfunc="first").reset_index()
    stamp = raw.groupby(["bookmaker", "market"])["last_update"].max()
    rows = []
    for r in wide.to_dict("records"):
        market = r["market"]
        for prefix in ("player_", "batter_", "pitcher_"):
            if market.startswith(prefix):
                market = market[len(prefix):]
                break
        rows.append({
            "sport": sport.lower(), "league": sport.upper(), "player_name": r["player"], "team": "",
            "opponent": f"{r['away_team']} @ {r['home_team']}",
            "game_start_time": utc_timestamp(r["commence_time"]),
            "stat_type": normalize_stat_type(MARKET_TO_STAT.get(market, market)),
            "line": float(r["point"]), "over_odds": r.get("over"), "under_odds": r.get("under"),
            "source": r["bookmaker"], "captured_at": utc_timestamp(stamp.get((r["bookmaker"], r["market"]))),
        })
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)


def latest_player_props(
    sport: str,
    player_name: str,
    csv_path: Path | str = ODDS_CSV_PATH,
    include_oddsscraper: bool = True,
) -> pd.DataFrame:
    props, _ = read_backend_odds_csv(csv_path)
    if include_oddsscraper:
        extra = oddsscraper_props(sport, player_name)
        if not extra.empty:
            props = pd.concat([props, extra], ignore_index=True)
            props = props[props["player_name"].map(_name_key) == _name_key(player_name)].assign(
                player_name=player_name)
    if props.empty:
        return props

    current_time = pd.Timestamp.now(tz="UTC")
    game_starts = pd.to_datetime(props["game_start_time"], utc=True, errors="coerce")
    matches = props[
        props["sport"].str.casefold().eq(sport.strip().casefold())
        & props["player_name"].str.strip().str.casefold().eq(player_name.strip().casefold())
        & game_starts.ge(current_time)
    ].copy()
    if matches.empty:
        return matches

    matches = matches.sort_values("captured_at", ascending=False)
    market_columns = [
        "sport",
        "league",
        "player_name",
        "team",
        "opponent",
        "game_start_time",
        "stat_type",
        "source",
    ]
    matches["_player_key"] = matches["player_name"].str.strip().str.casefold()
    market_columns[2] = "_player_key"
    return (
        matches.drop_duplicates(market_columns, keep="first")
        .drop(columns="_player_key")
        .reset_index(drop=True)
    )