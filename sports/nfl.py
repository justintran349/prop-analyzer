from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import streamlit as st

from scrapers.common import (
    analyze_props_vs_history,
    stat_column_map,
)
from scrapers.csv_import import latest_player_props


CURRENT_YEAR = 2026
DATA_DIR = Path("data") / "nfl"
PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
    "play_by_play_{year}.csv.gz"
)
PLAYERS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"
)
ROSTER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/"
    "roster_{year}.csv"
)

TEAM_ALIASES = {
    "Arizona Cardinals": {"ARI"}, "Atlanta Falcons": {"ATL"},
    "Baltimore Ravens": {"BAL"}, "Buffalo Bills": {"BUF"},
    "Carolina Panthers": {"CAR"}, "Chicago Bears": {"CHI"},
    "Cincinnati Bengals": {"CIN"}, "Cleveland Browns": {"CLE"},
    "Dallas Cowboys": {"DAL"}, "Denver Broncos": {"DEN"},
    "Detroit Lions": {"DET"}, "Green Bay Packers": {"GB"},
    "Houston Texans": {"HOU"}, "Indianapolis Colts": {"IND"},
    "Jacksonville Jaguars": {"JAX"}, "Kansas City Chiefs": {"KC"},
    "Las Vegas Raiders": {"LV", "OAK"}, "Los Angeles Chargers": {"LAC", "SD"},
    "Los Angeles Rams": {"LA", "STL"}, "Miami Dolphins": {"MIA"},
    "Minnesota Vikings": {"MIN"}, "New England Patriots": {"NE"},
    "New Orleans Saints": {"NO"}, "New York Giants": {"NYG"},
    "New York Jets": {"NYJ"}, "Philadelphia Eagles": {"PHI"},
    "Pittsburgh Steelers": {"PIT"}, "San Francisco 49ers": {"SF"},
    "Seattle Seahawks": {"SEA"}, "Tampa Bay Buccaneers": {"TB"},
    "Tennessee Titans": {"TEN"}, "Washington Commanders": {"WAS", "WSH"},
}

STAT_OPTIONS = {
    "QB": {
        "Passing Yards": "passing_yards", "Rushing Yards": "rushing_yards",
        "Passing Touchdowns": "passing_tds", "Rushing Touchdowns": "rushing_tds",
        "Passing + Rushing + Receiving TDs": "total_tds",
        "Pass Completions": "completions", "Pass Attempts": "pass_attempts",
        "Pass + Rush Yards": "pass_rush_yards", "Interceptions": "interceptions",
        "Rush Attempts": "rush_attempts", "Longest Completion": "longest_completion",
        "Sacks Taken": "sacks_taken", "Longest Rush": "longest_rush",
        "Completion Percentage": "completion_percentage", "QB Fantasy Score": "fantasy_score",
    },
    "RB": {
        "Rushing Yards": "rushing_yards", "Receiving Yards": "receiving_yards",
        "Rushing + Receiving Yards": "rush_receive_yards",
        "Rushing Touchdowns": "rushing_tds", "Receiving Touchdowns": "receiving_tds",
        "Passing + Rushing + Receiving TDs": "total_tds", "Receptions": "receptions",
        "Receiving Targets": "targets", "Rush Attempts": "rush_attempts",
        "Longest Rush": "longest_rush", "Longest Reception": "longest_reception",
        "RB Fantasy Score": "fantasy_score",
    },
    "WR/TE": {
        "Receiving Yards": "receiving_yards", "Receptions": "receptions",
        "Receiving Touchdowns": "receiving_tds", "Rushing Yards": "rushing_yards",
        "Rushing Touchdowns": "rushing_tds", "Receiving Targets": "targets",
        "Longest Reception": "longest_reception", "Longest Rush": "longest_rush",
        "Passing + Rushing + Receiving TDs": "total_tds", "WR/TE Fantasy Score": "fantasy_score",
    },
    "K": {
        "Field Goals Made": "field_goals_made", "Total Kicking Points": "kicking_points",
        "Kicker Fantasy Score": "fantasy_score",
    },
}


def series(frame: pd.DataFrame, name: str, default: int | float = 0) -> pd.Series:
    """Return a numeric column even when an older data file lacks it."""
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce").fillna(default)


def player_mask(frame: pd.DataFrame, column: str, player_id: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna("").astype(str).eq(str(player_id))


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        partial.replace(destination)
    except requests.RequestException:
        partial.unlink(missing_ok=True)
        raise


@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    path = DATA_DIR / "players.csv"
    if not path.exists():
        download(PLAYERS_URL, path)
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def load_pbp(year: int) -> pd.DataFrame:
    path = DATA_DIR / f"play_by_play_{year}.csv.gz"
    if not path.exists():
        download(PBP_URL.format(year=year), path)
    return pd.read_csv(path, compression="gzip", low_memory=False)


@st.cache_data(show_spinner=False)
def load_roster(year: int) -> pd.DataFrame:
    path = DATA_DIR / f"roster_{year}.csv"
    if not path.exists():
        download(ROSTER_URL.format(year=year), path)
    return pd.read_csv(path, low_memory=False)


def player_label(player: pd.Series) -> str:
    latest_team = player.get("latest_team")
    team_abbr = player.get("team_abbr")
    team = latest_team if pd.notna(latest_team) and str(latest_team) else team_abbr
    team = team if pd.notna(team) and str(team) else "No current team"
    return f"{player['display_name']} — {player['position']} · {team} · {player['gsis_id']}"


def player_candidates(players: pd.DataFrame, query: str, group: str) -> pd.DataFrame:
    positions = {"QB"} if group == "QB" else {"RB"} if group == "RB" else {"WR", "TE"} if group == "WR/TE" else {"K"}
    names = players.get("display_name", pd.Series("", index=players.index)).fillna("")
    matches = players[names.str.contains(query, case=False, regex=False) & players["position"].isin(positions)].copy()
    return matches.sort_values("display_name").drop_duplicates("gsis_id")


def context_rows(pbp: pd.DataFrame, player_id: str) -> pd.DataFrame:
    roles = (
        player_mask(pbp, "passer_player_id", player_id)
        | player_mask(pbp, "rusher_player_id", player_id)
        | player_mask(pbp, "receiver_player_id", player_id)
        | player_mask(pbp, "kicker_player_id", player_id)
    )
    columns = [column for column in ["game_id", "game_date", "posteam", "defteam", "home_team"] if column in pbp]
    return pbp.loc[roles, columns].drop_duplicates("game_id")


def grouped_sum(pbp: pd.DataFrame, mask: pd.Series, column: str) -> pd.Series:
    return series(pbp.loc[mask], column).groupby(pbp.loc[mask, "game_id"]).sum()


def grouped_max(pbp: pd.DataFrame, mask: pd.Series, column: str) -> pd.Series:
    return series(pbp.loc[mask], column).groupby(pbp.loc[mask, "game_id"]).max()


def game_stats(pbp: pd.DataFrame, player_id: str) -> pd.DataFrame:
    """Aggregate player game logs directly from play-by-play data."""
    context = context_rows(pbp, player_id).set_index("game_id")
    if context.empty:
        return pd.DataFrame()

    passing = player_mask(pbp, "passer_player_id", player_id)
    rushing = player_mask(pbp, "rusher_player_id", player_id)
    receiving = player_mask(pbp, "receiver_player_id", player_id)
    kicking = player_mask(pbp, "kicker_player_id", player_id)
    index = context.index

    def sums(mask: pd.Series, field: str) -> pd.Series:
        return grouped_sum(pbp, mask, field).reindex(index, fill_value=0)

    def maximum(mask: pd.Series, field: str) -> pd.Series:
        return grouped_max(pbp, mask, field).reindex(index).fillna(0)

    output = context.rename(columns={"game_date": "date", "posteam": "team", "defteam": "opponent"}).copy()
    output["home"] = output["team"].eq(output["home_team"])
    output["completions"] = sums(passing, "complete_pass")
    output["pass_attempts"] = sums(passing, "pass_attempt")
    output["passing_yards"] = sums(passing, "passing_yards")
    output["passing_tds"] = sums(passing, "pass_touchdown")
    output["interceptions"] = sums(passing, "interception")
    output["sacks_taken"] = sums(passing, "sack")
    output["longest_completion"] = maximum(passing & series(pbp, "complete_pass").eq(1), "passing_yards")
    output["rush_attempts"] = sums(rushing, "rush_attempt")
    output["rushing_yards"] = sums(rushing, "rushing_yards")
    output["rushing_tds"] = sums(rushing, "rush_touchdown")
    output["longest_rush"] = maximum(rushing & series(pbp, "rush_attempt").eq(1), "rushing_yards")
    output["receptions"] = sums(receiving, "reception")
    output["targets"] = sums(receiving, "target")
    output["receiving_yards"] = sums(receiving, "receiving_yards")
    # A receiving touchdown is recorded as pass_touchdown on the play-by-play
    # row; the receiver role isolates it to the receiving player.
    output["receiving_tds"] = sums(receiving, "pass_touchdown")
    output["longest_reception"] = maximum(receiving & series(pbp, "reception").eq(1), "receiving_yards")

    fumble_lost = series(pbp, "fumble_lost").eq(1)
    fumbled = player_mask(pbp, "fumbled_1_player_id", player_id) | player_mask(pbp, "fumbled_2_player_id", player_id)
    output["fumbles_lost"] = sums(fumble_lost & fumbled, "fumble_lost")

    field_goal = kicking & series(pbp, "field_goal_attempt").eq(1)
    fg_result = pbp.get("field_goal_result", pd.Series("", index=pbp.index)).fillna("")
    made_fg = field_goal & fg_result.eq("made")
    missed_fg = field_goal & fg_result.isin(["missed", "blocked"])
    distance = series(pbp, "kick_distance")
    output["fg_0_39"] = sums(made_fg & distance.lt(40), "field_goal_attempt")
    output["fg_40_49"] = sums(made_fg & distance.between(40, 49), "field_goal_attempt")
    output["fg_50_plus"] = sums(made_fg & distance.ge(50), "field_goal_attempt")
    output["field_goals_made"] = sums(made_fg, "field_goal_attempt")
    output["missed_field_goals"] = sums(missed_fg, "field_goal_attempt")
    extra = kicking & series(pbp, "extra_point_attempt").eq(1)
    xp_result = pbp.get("extra_point_result", pd.Series("", index=pbp.index)).fillna("")
    output["pat_made"] = sums(extra & xp_result.eq("good"), "extra_point_attempt")
    output["missed_pat"] = sums(extra & xp_result.isin(["failed", "blocked"]), "extra_point_attempt")

    output["total_tds"] = output[["passing_tds", "rushing_tds", "receiving_tds"]].sum(axis=1)
    output["pass_rush_yards"] = output["passing_yards"] + output["rushing_yards"]
    output["rush_receive_yards"] = output["rushing_yards"] + output["receiving_yards"]
    output["completion_percentage"] = (100 * output["completions"] / output["pass_attempts"]).where(output["pass_attempts"].gt(0), 0)
    output["kicking_points"] = 3 * output["field_goals_made"] + output["pat_made"]
    output["qb_fantasy"] = (.04 * output["passing_yards"] + 4 * output["passing_tds"] - output["interceptions"] + .1 * output["rushing_yards"] + 6 * output["rushing_tds"] - 2 * output["fumbles_lost"])
    output["skill_fantasy"] = (.1 * (output["rushing_yards"] + output["receiving_yards"]) + output["receptions"] + 6 * (output["rushing_tds"] + output["receiving_tds"]) - 2 * output["fumbles_lost"])
    output["kicker_fantasy"] = (3 * output["fg_0_39"] + 4 * output["fg_40_49"] + 5 * output["fg_50_plus"] + output["pat_made"] - output["missed_field_goals"] - output["missed_pat"])
    return output.reset_index()


def player_logs(player_id: str, years: Iterable[int]) -> pd.DataFrame:
    frames = []
    for year in sorted(set(years), reverse=True):
        pbp = load_pbp(year)
        regular = pbp.get("season_type", pd.Series("REG", index=pbp.index)).eq("REG")
        stats = game_stats(pbp.loc[regular].copy(), player_id)
        if not stats.empty:
            stats["season"] = year
            frames.append(stats)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date", ascending=False)


def career_start(player: pd.Series) -> int:
    value = pd.to_numeric(pd.Series([player.get("rookie_season")]), errors="coerce").iloc[0]
    return int(value) if pd.notna(value) else max(1999, CURRENT_YEAR - 10)


def current_injuries(team: str) -> pd.DataFrame:
    try:
        roster = load_roster(CURRENT_YEAR)
    except requests.RequestException:
        return pd.DataFrame()
    team_column = "team" if "team" in roster else "team_abbr"
    injury_column = "injury_status" if "injury_status" in roster else "status"
    name_column = "full_name" if "full_name" in roster else "player_name"
    if not {team_column, injury_column, name_column}.issubset(roster.columns):
        return pd.DataFrame()
    result = roster[roster[team_column].eq(team) & roster[injury_column].notna()][[name_column, injury_column]].copy()
    return result.rename(columns={name_column: "Player", injury_column: "Status"})


def add_line_results(frame: pd.DataFrame, stat: str, line: float) -> pd.DataFrame:
    result = frame.copy()
    result["value"] = result[stat]
    result["result"] = result["value"].map(lambda value: "Over" if value > line else "Under" if value < line else "Push")
    return result


def summary(frame: pd.DataFrame) -> dict[str, float | int]:
    counts = frame["result"].value_counts()
    over, under, push = int(counts.get("Over", 0)), int(counts.get("Under", 0)), int(counts.get("Push", 0))
    decided = over + under
    return {"games": len(frame), "over": over, "under": under, "push": push, "over_rate": 100 * over / decided if decided else 0, "average": frame["value"].mean() if len(frame) else 0, "median": frame["value"].median() if len(frame) else 0}


st.title("NFL Player Stat Research")
st.caption("NFL game logs come from nflverse play-by-play data. First-time season downloads are cached locally; career-against-team queries can take longer.")

with st.sidebar:
    st.header("NFL research inputs")
    position_group = st.radio("Position", list(STAT_OPTIONS), horizontal=True)
    player_query = st.text_input("Player name", placeholder="e.g., Josh Allen")
    default_years = list(range(CURRENT_YEAR, CURRENT_YEAR - 3, -1))
    selected_years = st.multiselect("Seasons", list(range(CURRENT_YEAR, 1998, -1)), default=default_years)
    stat_label = st.selectbox("Statistic", list(STAT_OPTIONS[position_group]))
    line = st.number_input("Your line", min_value=0.0, value=1.5, step=0.5)

if not player_query:
    st.info("Search for a quarterback, running back, receiver/tight end, or kicker to begin.")
    st.stop()

try:
    players = load_players()
except requests.RequestException as error:
    st.error(f"Could not download NFL player information: {error}")
    st.stop()

matches = player_candidates(players, player_query, position_group)
if matches.empty:
    st.warning("No matching NFL players were found for that position group.")
    st.stop()

lookup = {player_label(row): row for _, row in matches.iterrows()}
chosen_label = st.selectbox("Choose player", list(lookup), help="Type in this dropdown to narrow the alphabetical matches.")
player = lookup[chosen_label]
stat = STAT_OPTIONS[position_group][stat_label]

if not selected_years:
    st.warning("Select at least one season.")
    st.stop()

try:
    with st.spinner("Loading NFL game logs. New seasons download once, then are reused from your local cache..."):
        frame = player_logs(str(player["gsis_id"]), selected_years)
except requests.RequestException as error:
    st.error(f"Could not download NFL data: {error}")
    st.stop()

if frame.empty:
    st.warning("No regular-season game logs were returned for this player and season selection.")
    st.stop()

filter_one, filter_two, filter_three = st.columns(3)
with filter_one:
    team_lookup = {"All teams": None, **{name: aliases for name, aliases in sorted(TEAM_ALIASES.items())}}
    opponent_label = st.selectbox("Opponent", list(team_lookup), help="Selecting a team loads this player's complete NFL career against that franchise.")
with filter_two:
    venue = st.selectbox("Venue", ["All venues", "Home", "Away"])
with filter_three:
    lookback = st.selectbox("Games", ["Last 5", "Last 10", "Last 15", "All games"])

selected_opponent = team_lookup[opponent_label]
if selected_opponent:
    try:
        with st.spinner(f"Loading {player['display_name']}'s NFL career against {opponent_label}..."):
            frame = player_logs(str(player["gsis_id"]), range(career_start(player), CURRENT_YEAR + 1))
    except requests.RequestException as error:
        st.error(f"Could not download career NFL data: {error}")
        st.stop()
    filtered = frame[frame["opponent"].isin(selected_opponent)].copy()
else:
    filtered = frame.copy()

if venue == "Home":
    filtered = filtered[filtered["home"]]
elif venue == "Away":
    filtered = filtered[~filtered["home"]]

odds_history = filtered.copy()
if position_group == "QB":
    odds_history["fantasy_score"] = odds_history["qb_fantasy"]
elif position_group == "K":
    odds_history["fantasy_score"] = odds_history["kicker_fantasy"]
else:
    odds_history["fantasy_score"] = odds_history["skill_fantasy"]

player_props = latest_player_props("nfl", player["display_name"])
if not player_props.empty:
    st.subheader("Current imported player props")
    odds_lookback_label = st.selectbox(
        "Hit-rate window",
        ["Last 5", "Last 10", "Last 15", "All games"],
        key="nfl_odds_hit_window",
    )
    odds_lookback = (
        None if odds_lookback_label == "All games"
        else int(odds_lookback_label.split()[-1])
    )
    prop_analysis = analyze_props_vs_history(
        player_props,
        odds_history,
        stat_column_map(STAT_OPTIONS[position_group]),
        odds_lookback,
    )
    prop_columns = [
        "opponent", "game_start_time", "stat_type", "line", "over_odds",
        "under_odds", "source", "captured_at", "games_checked", "over_hits",
        "over_rate", "under_hits", "under_rate", "pushes", "average",
    ]
    st.dataframe(
        prop_analysis[prop_columns],
        use_container_width=True,
        hide_index=True,
    )

if lookback != "All games":
    filtered = filtered.head(int(lookback.split()[-1]))

if position_group == "QB":
    filtered["fantasy_score"] = filtered["qb_fantasy"]
elif position_group == "K":
    filtered["fantasy_score"] = filtered["kicker_fantasy"]
else:
    filtered["fantasy_score"] = filtered["skill_fantasy"]

analysis = add_line_results(filtered, stat, line)
metrics = summary(analysis)
st.subheader(f"{player['display_name']} — {stat_label}")
columns = st.columns(7)
for column, (label, value) in zip(columns, [("Games", metrics["games"]), ("Over", metrics["over"]), ("Under", metrics["under"]), ("Push", metrics["push"]), ("Over rate", f"{metrics['over_rate']:.1f}%"), ("Average", f"{metrics['average']:.2f}"), ("Median", f"{metrics['median']:.2f}")]):
    column.metric(label, value)
st.caption("Over and under rates exclude pushes. NFL stats use regular-season final game results.")

base_columns = ["date", "season", "team", "opponent", "home", "value", "result"]
if position_group == "QB":
    detail_columns = ["passing_yards", "passing_tds", "interceptions", "completions", "pass_attempts", "completion_percentage", "rushing_yards", "rushing_tds", "rush_attempts", "pass_rush_yards", "total_tds", "longest_completion", "longest_rush", "sacks_taken", "fumbles_lost", "qb_fantasy"]
elif position_group == "K":
    detail_columns = ["field_goals_made", "fg_0_39", "fg_40_49", "fg_50_plus", "pat_made", "missed_field_goals", "missed_pat", "kicking_points", "kicker_fantasy"]
else:
    detail_columns = ["rushing_yards", "rushing_tds", "rush_attempts", "receiving_yards", "receiving_tds", "receptions", "targets", "rush_receive_yards", "total_tds", "longest_rush", "longest_reception", "fumbles_lost", "skill_fantasy"]
display = analysis[base_columns + detail_columns].copy()
display["date"] = pd.to_datetime(display["date"], errors="coerce").dt.strftime("%Y-%m-%d")
st.dataframe(display, use_container_width=True, hide_index=True)

latest_team = player.get("latest_team")
team_abbr = player.get("team_abbr")
team = latest_team if pd.notna(latest_team) and str(latest_team) else team_abbr
if isinstance(team, str) and team:
    st.subheader("Current injured teammates")
    injuries = current_injuries(team)
    if injuries.empty:
        st.caption("No current injury entries were returned for this team.")
    else:
        st.dataframe(injuries, use_container_width=True, hide_index=True)
