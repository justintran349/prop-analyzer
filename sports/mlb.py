from __future__ import annotations

from collections import defaultdict
from datetime import date

import pandas as pd
import requests
import streamlit as st

from scrapers.common import (
    analyze_props_vs_history,
    stat_column_map,
)
from scrapers.csv_import import latest_player_props

BASE_URL = "https://statsapi.mlb.com/api/v1"
SUPPORTED_BVP_STATS = {
    "hits",
    "singles",
    "doubles",
    "total_bases",
    "rbi",
    "home_runs",
    "walks",
    "strikeouts",
}

HITTER_STAT_OPTIONS = {
    "Hits": "hits",
    "Singles": "singles",
    "Doubles": "doubles",
    "Total Bases": "total_bases",
    "Runs": "runs",
    "RBIs": "rbi",
    "Hits + Runs + RBIs": "hits_runs_rbi",
    "Home Runs": "home_runs",
    "Walks": "walks",
    "Batter Strikeouts": "strikeouts",
    "Stolen Bases": "stolen_bases",
    "Hitter Fantasy Score": "hitter_fantasy_score",
}

PITCHER_STAT_OPTIONS = {
    "Pitcher Fantasy Score": "pitcher_fantasy_score",
    "Strikeouts": "strikeouts",
    "Pitching Outs": "outs",
    "Hits Allowed": "hits_allowed",
    "Pitches Thrown": "pitches_thrown",
    "Earned Runs Allowed": "earned_runs",
    "Walks Allowed": "walks_allowed",
}


@st.cache_data(ttl=3600, show_spinner=False)
def mlb_get(path: str, **params) -> dict:
    response = requests.get(
        f"{BASE_URL}{path}",
        params=params,
        timeout=30,
        headers={"User-Agent": "sports-stats-db/0.1"},
    )
    response.raise_for_status()
    return response.json()


def search_players(name: str, pitcher_only: bool = False) -> list[dict]:
    people = mlb_get("/people/search", names=name).get("people", [])
    if pitcher_only:
        people = [
            player
            for player in people
            if player.get("primaryPosition", {}).get("code") == "1"
        ]
    # The search endpoint frequently omits currentTeam. Add the more complete
    # player profile so active players do not all appear as "No current team".
    enriched_people = []
    for player in people:
        try:
            profile = player_profile(player["id"])
        except requests.RequestException:
            # A player can still be selected if one optional profile lookup
            # temporarily fails.
            profile = {}
        enriched_people.append({**player, **profile})

    return sorted(
        enriched_people,
        key=lambda player: (player.get("fullName") or "").casefold(),
    )


@st.cache_data(ttl=86400, show_spinner=False)
def player_profile(player_id: int) -> dict:
    """Return the authoritative player record, including currentTeam when set."""
    people = mlb_get(f"/people/{player_id}").get("people", [])
    return people[0] if people else {}


def player_label(player: dict) -> str:
    """Create one clear, searchable label for a player-match dropdown."""
    team_name = player.get("currentTeam", {}).get("name")
    position = player.get("primaryPosition", {}).get("abbreviation")
    details = " · ".join(
        str(part) for part in (position, team_name, player.get("id")) if part
    )
    return f"{player['fullName']} — {details}" if details else player["fullName"]


def date_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep a table's game date readable without displaying midnight time."""
    output = frame.copy()
    parsed_dates = pd.to_datetime(output["date"], errors="coerce")
    output["date"] = parsed_dates.dt.strftime("%Y-%m-%d").fillna(
        output["date"].astype(str)
    )
    return output


def get_game_logs(
    player_id: int,
    seasons: list[int],
    stat_group: str,
) -> list[dict]:
    logs = []

    for season in seasons:
        payload = mlb_get(
            f"/people/{player_id}/stats",
            stats="gameLog",
            group=stat_group,
            season=season,
            sportIds=1,
        )
        logs.extend(payload.get("stats", [{}])[0].get("splits", []))

    return logs


def stat_number(stat: dict, field: str) -> int:
    try:
        return int(stat.get(field, 0) or 0)
    except (TypeError, ValueError):
        return 0


def normalize_logs(logs: list[dict], stat_group: str) -> pd.DataFrame:
    rows = []

    for log in logs:
        stat = log.get("stat", {})

        row = {
            "date": log.get("date"),
            "game_pk": log.get("game", {}).get("gamePk"),
            "team": log.get("team", {}).get("name"),
            "opponent": log.get("opponent", {}).get("name"),
            "opponent_id": log.get("opponent", {}).get("id"),
            "home": bool(log.get("isHome")),
        }

        if stat_group == "hitting":
            singles = (
                stat_number(stat, "hits")
                - stat_number(stat, "doubles")
                - stat_number(stat, "triples")
                - stat_number(stat, "homeRuns")
            )

            row.update(
                {
                    "plate_appearances": stat_number(stat, "plateAppearances"),
                    "at_bats": stat_number(stat, "atBats"),
                    "hits": stat_number(stat, "hits"),
                    "singles": singles,
                    "doubles": stat_number(stat, "doubles"),
                    "runs": stat_number(stat, "runs"),
                    "rbi": stat_number(stat, "rbi"),
                    "home_runs": stat_number(stat, "homeRuns"),
                    "walks": stat_number(stat, "baseOnBalls"),
                    "hit_by_pitch": stat_number(stat, "hitByPitch"),
                    "strikeouts": stat_number(stat, "strikeOuts"),
                    "stolen_bases": stat_number(stat, "stolenBases"),
                    "total_bases": (
                        stat_number(stat, "hits")
                        + stat_number(stat, "doubles")
                        + 2 * stat_number(stat, "triples")
                        + 3 * stat_number(stat, "homeRuns")
                    ),
                    "hits_runs_rbi": (
                        stat_number(stat, "hits")
                        + stat_number(stat, "runs")
                        + stat_number(stat, "rbi")
                    ),
                    "hitter_fantasy_score": (
                        3 * singles
                        + 5 * stat_number(stat, "doubles")
                        + 8 * stat_number(stat, "triples")
                        + 10 * stat_number(stat, "homeRuns")
                        + 2 * stat_number(stat, "runs")
                        + 2 * stat_number(stat, "rbi")
                        + 2 * stat_number(stat, "baseOnBalls")
                        + 2 * stat_number(stat, "hitByPitch")
                        + 5 * stat_number(stat, "stolenBases")
                    ),
                }
            )

        else:
            wins = stat_number(stat, "wins")
            earned_runs = stat_number(stat, "earnedRuns")
            strikeouts = stat_number(stat, "strikeOuts")
            outs = stat_number(stat, "outs")
            # The game-log endpoint does not reliably provide qualityStarts,
            # so calculate the standard definition from the individual game.
            quality_starts = int(
                stat_number(stat, "gamesStarted") == 1
                and outs >= 18
                and earned_runs <= 3
            )

            row.update(
                {
                    "wins": wins,
                    "quality_starts": quality_starts,
                    "earned_runs": earned_runs,
                    "strikeouts": strikeouts,
                    "outs": outs,
                    "hits_allowed": stat_number(stat, "hits"),
                    "pitches_thrown": stat_number(stat, "numberOfPitches"),
                    "walks_allowed": stat_number(stat, "baseOnBalls"),
                    "innings_pitched": stat.get("inningsPitched", "0.0"),
                    "pitcher_fantasy_score": (
                        6 * wins
                        + 4 * quality_starts
                        - 3 * earned_runs
                        + 3 * strikeouts
                        + outs
                    ),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def add_line_results(frame: pd.DataFrame, stat: str, line: float) -> pd.DataFrame:
    result = frame.copy()
    result["value"] = result[stat]
    result["result"] = result["value"].apply(
        lambda value: "Over" if value > line else "Under" if value < line else "Push"
    )
    return result


def calculate_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "games": 0,
            "over": 0,
            "under": 0,
            "push": 0,
            "over_rate": 0.0,
            "under_rate": 0.0,
            "average": 0.0,
            "median": 0.0,
        }

    counts = frame["result"].value_counts()
    over = int(counts.get("Over", 0))
    under = int(counts.get("Under", 0))
    push = int(counts.get("Push", 0))
    decided_games = over + under

    return {
        "games": len(frame),
        "over": over,
        "under": under,
        "push": push,
        "over_rate": 100 * over / decided_games if decided_games else 0.0,
        "under_rate": 100 * under / decided_games if decided_games else 0.0,
        "average": float(frame["value"].mean()),
        "median": float(frame["value"].median()),
    }


def current_injuries(team_id: int) -> list[dict]:
    try:
        roster = mlb_get(f"/teams/{team_id}/roster", rosterType="injuredList")
        return [
            {
                "Player": item["person"]["fullName"],
                "Status": item.get("status", {}).get("description", "Injured list"),
            }
            for item in roster.get("roster", [])
        ]
    except requests.RequestException:
        return []


def batter_vs_pitcher(
    batter_id: int, pitcher_id: int, game_pks: list[int]
) -> dict[int, dict]:
    results = defaultdict(
        lambda: {
            "plate_appearances": 0,
            "at_bats": 0,
            "hits": 0,
            "singles": 0,
            "doubles": 0,
            "total_bases": 0,
            "home_runs": 0,
            "rbi": 0,
            "walks": 0,
            "strikeouts": 0,
        }
    )

    for game_pk in game_pks:
        feed = mlb_get(f"/game/{game_pk}/feed/live")

        for play in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
            matchup = play.get("matchup", {})

            if (
                matchup.get("batter", {}).get("id") != batter_id
                or matchup.get("pitcher", {}).get("id") != pitcher_id
                or not play.get("about", {}).get("isComplete")
            ):
                continue

            result = results[game_pk]
            event = play.get("result", {}).get("eventType", "")
            result["plate_appearances"] += 1
            result["rbi"] += int(play.get("result", {}).get("rbi", 0) or 0)

            if event in {"single", "double", "triple", "home_run"}:
                bases = {
                    "single": 1,
                    "double": 2,
                    "triple": 3,
                    "home_run": 4,
                }[event]
                result["hits"] += 1
                result["total_bases"] += bases
                result["at_bats"] += 1

                if event == "single":
                    result["singles"] += 1
                elif event == "double":
                    result["doubles"] += 1

                if event == "home_run":
                    result["home_runs"] += 1

            elif event in {"walk", "intent_walk"}:
                result["walks"] += 1

            elif event == "strikeout":
                result["strikeouts"] += 1
                result["at_bats"] += 1

            elif event not in {
                "hit_by_pitch",
                "catcher_interf",
                "sac_bunt",
                "sac_fly",
            }:
                result["at_bats"] += 1

    return dict(results)

@st.cache_data(ttl=86400, show_spinner=False)
def player_debut_year(player_id: int) -> int:
    payload = mlb_get(f"/people/{player_id}")
    debut_date = payload["people"][0].get("mlbDebutDate")

    if not debut_date:
        return date.today().year - 5

    return int(debut_date[:4])


@st.cache_data(ttl=86400, show_spinner=False)
def mlb_teams() -> list[dict]:
    payload = mlb_get("/teams", sportId=1)
    return sorted(payload.get("teams", []), key=lambda team: team["name"])

st.title("MLB Player Stat Research")
st.caption(
    "Enter a line from PrizePicks or another source. "
    "This app analyzes MLB game data and does not scrape sportsbook odds."
)

with st.sidebar:
    st.header("Research inputs")

    player_type = st.radio(
        "Player type",
        ["Hitter", "Pitcher"],
        horizontal=True,
    )

    stat_options = (
        HITTER_STAT_OPTIONS
        if player_type == "Hitter"
        else PITCHER_STAT_OPTIONS
    )

    player_search = st.text_input(
        f"{player_type} name",
        placeholder=(
            "Example: Shohei Ohtani"
            if player_type == "Hitter"
            else "Example: Paul Skenes"
        ),
    )

    current_year = date.today().year

    seasons = st.multiselect(
        "Seasons",
        options=list(range(current_year, current_year - 5, -1)),
        default=list(range(current_year, current_year - 5, -1)),
    )

    stat_label = st.selectbox("Statistic", list(stat_options))
    line = st.number_input("Your line", min_value=0.0, value=1.5, step=0.5)

    if st.button("Refresh MLB data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared. Fresh MLB data will load now.")

if not player_search:
    st.info(f"Search for a {player_type.lower()} to begin.")
    st.stop()

try:
    player_matches = search_players(
        player_search,
        pitcher_only=(player_type == "Pitcher"),
    )
except requests.RequestException as error:
    st.error(f"Could not reach MLB data: {error}")
    st.stop()

if not player_matches:
    st.warning("No MLB players matched that name.")
    st.stop()

player_lookup = {
    player_label(player): player
    for player in player_matches
}
selected_label = st.selectbox(
    f"Choose {player_type.lower()}",
    list(player_lookup),
    help="Type in this dropdown to narrow the alphabetical list of matching names.",
)
player = player_lookup[selected_label]

if not seasons:
    st.warning("Select at least one season.")
    st.stop()

stat_group = "hitting" if player_type == "Hitter" else "pitching"

with st.spinner("Loading MLB game logs..."):
    logs = get_game_logs(player["id"], seasons, stat_group)

frame = normalize_logs(logs, stat_group)

if frame.empty:
    st.warning(f"No completed {player_type.lower()} game logs were found.")
    st.stop()

frame["date"] = pd.to_datetime(frame["date"])
frame = frame.sort_values("date", ascending=False)
stat = stat_options[stat_label]

st.subheader(f"{player['fullName']} — {stat_label}")

filter_one, filter_two, filter_three = st.columns(3)

with filter_one:
    teams = mlb_teams()
    team_lookup = {"All teams": None}
    team_lookup.update({team["name"]: team for team in teams})

    opponent_filter = st.selectbox(
        "Opponent",
        list(team_lookup),
        help="Selecting a team automatically loads the hitter's full MLB career against that team.",
    )

with filter_two:
    venue_filter = st.selectbox("Venue", ["All venues", "Home", "Away"])

with filter_three:
    game_filter = st.selectbox("Games", ["Last 5", "Last 10", "Last 15", "All games"])

selected_team = team_lookup[opponent_filter]

if selected_team:
    with st.spinner(
        f"Loading {player['fullName']}'s full MLB career against {selected_team['name']}..."
    ):
        debut_year = player_debut_year(player["id"])
        career_seasons = list(range(debut_year, current_year + 1))
        career_logs = get_game_logs(player["id"], career_seasons, stat_group)
        frame = normalize_logs(career_logs, stat_group)

        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame.sort_values("date", ascending=False)

    filtered = frame[frame["opponent_id"] == selected_team["id"]].copy()
else:
    filtered = frame.copy()

if venue_filter == "Home":
    filtered = filtered[filtered["home"]]
elif venue_filter == "Away":
    filtered = filtered[~filtered["home"]]

odds_history = filtered.copy()
player_props = latest_player_props("mlb", player["fullName"])
if not player_props.empty:
    st.subheader("Current imported player props")
    odds_lookback_label = st.selectbox(
        "Hit-rate window",
        ["Last 5", "Last 10", "Last 15", "All games"],
        key="mlb_odds_hit_window",
    )
    odds_lookback = (
        None if odds_lookback_label == "All games"
        else int(odds_lookback_label.split()[-1])
    )
    prop_stat_columns = stat_column_map(stat_options)
    prop_stat_columns["fantasy_score"] = stat_options.get(
        "Hitter Fantasy Score" if player_type == "Hitter" else "Pitcher Fantasy Score"
    )
    prop_analysis = analyze_props_vs_history(
        player_props, odds_history, prop_stat_columns, odds_lookback
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

if game_filter != "All games":
    filtered = filtered.head(int(game_filter.split()[-1]))

analysis = add_line_results(filtered, stat, line)
stats = calculate_summary(analysis)

metrics = st.columns(7)
metrics[0].metric("Games", stats["games"])
metrics[1].metric("Over", stats["over"])
metrics[2].metric("Under", stats["under"])
metrics[3].metric("Push", stats["push"])
metrics[4].metric("Over rate", f"{stats['over_rate']:.1f}%")
metrics[5].metric("Average", f"{stats['average']:.2f}")
metrics[6].metric("Median", f"{stats['median']:.2f}")

st.caption("Over and under rates exclude pushes from the denominator.")

if player_type == "Hitter":
    table_columns = [
        "date", "team", "opponent", "home", "value", "result",
        "plate_appearances", "at_bats", "hits", "singles", "doubles",
        "total_bases", "runs",
        "rbi", "hits_runs_rbi", "home_runs", "walks", "hit_by_pitch",
        "strikeouts", "stolen_bases", "hitter_fantasy_score",
    ]
else:
    table_columns = [
        "date", "team", "opponent", "home", "value", "result",
        "innings_pitched", "outs", "strikeouts", "hits_allowed",
        "pitches_thrown", "earned_runs", "walks_allowed", "wins",
        "quality_starts", "pitcher_fantasy_score",
    ]

display_analysis = date_only(analysis[table_columns])
st.dataframe(
    display_analysis,
    use_container_width=True,
    hide_index=True,
)

if player_type == "Hitter":
    st.subheader("Batter versus pitcher")
    pitcher_search = st.text_input(
        "Pitcher name",
        placeholder="Example: Clayton Kershaw",
    )
else:
    pitcher_search = ""

if pitcher_search:
    pitcher_matches = search_players(pitcher_search, pitcher_only=True)

    if not pitcher_matches:
        st.warning("No pitchers matched that name.")
    else:
        pitcher_lookup = {
            player_label(pitcher): pitcher
            for pitcher in pitcher_matches
        }
        pitcher_name = st.selectbox(
            "Choose pitcher",
            list(pitcher_lookup),
            help="Type in this dropdown to narrow the alphabetical list of matching names.",
        )
        pitcher = pitcher_lookup[pitcher_name]

        if stat not in SUPPORTED_BVP_STATS:
            st.info(
                "Exact batter-versus-pitcher analysis supports Hits, Singles, Doubles, "
                "Total Bases, RBIs, Home Runs, Walks, and Batter Strikeouts."
            )
        elif st.button("Analyze batter versus pitcher"):
            game_pks = filtered["game_pk"].dropna().astype(int).tolist()

            with st.spinner("Checking plate appearances in selected games..."):
                matchups = batter_vs_pitcher(
                    player["id"],
                    pitcher["id"],
                    game_pks,
                )

            matchup_frame = filtered[
                filtered["game_pk"].isin(matchups.keys())
            ].copy()

            if matchup_frame.empty:
                st.info(
                    "No completed plate appearances between these players "
                    "were found in the selected games."
                )
            else:
                for column in SUPPORTED_BVP_STATS:
                    matchup_frame[column] = matchup_frame["game_pk"].map(
                        lambda game_pk: matchups[int(game_pk)][column]
                    )

                matchup_frame["plate_appearances"] = matchup_frame["game_pk"].map(
                    lambda game_pk: matchups[int(game_pk)]["plate_appearances"]
                )
                matchup_frame["at_bats"] = matchup_frame["game_pk"].map(
                    lambda game_pk: matchups[int(game_pk)]["at_bats"]
                )

                matchup_analysis = add_line_results(matchup_frame, stat, line)

                st.dataframe(
                    date_only(matchup_analysis[
                        [
                            "date",
                            "team",
                            "opponent",
                            "value",
                            "result",
                            "plate_appearances",
                            "at_bats",
                            "hits",
                            "singles",
                            "doubles",
                            "total_bases",
                            "home_runs",
                            "rbi",
                            "walks",
                            "strikeouts",
                        ]
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )

team_id = player.get("currentTeam", {}).get("id")

if team_id:
    st.subheader("Current injured teammates")
    injuries = current_injuries(team_id)

    if injuries:
        st.dataframe(pd.DataFrame(injuries), use_container_width=True, hide_index=True)
    else:
        st.caption("No injured-list entries were returned for this team.")
    
