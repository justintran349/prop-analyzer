"""Player-vs-defender matchups (EXPERIMENTAL).

Basketball-Reference doesn't track who guards whom, so this uses NBA.com's
tracking data through the `nba_api` package. NBA.com sometimes blocks or slows
requests - if it hangs, try again later or from a different network.

What you get: for one offensive player, every defender on a given opponent who
guarded him this season, with points/assists per 100 possessions in that
matchup compared with the player's overall rate.
"""
from __future__ import annotations

import pandas as pd

from . import config

# BRef code -> NBA.com abbreviation
BREF_TO_NBA = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}


def _season_str(season: int = config.SEASON) -> str:
    return f"{season - 1}-{str(season)[-2:]}"   # 2027 -> "2026-27"


def _nba_player_id(name: str):
    from nba_api.stats.static import players
    hits = players.find_players_by_full_name(name)
    active = [h for h in hits if h.get("is_active")] or hits
    return active[0]["id"] if active else None


def _roster_ids(bref_team: str, season: int):
    from nba_api.stats.endpoints import commonteamroster
    from nba_api.stats.static import teams
    abbr = BREF_TO_NBA.get(bref_team, bref_team)
    t = teams.find_team_by_abbreviation(abbr)
    if not t:
        return set()
    roster = commonteamroster.CommonTeamRoster(team_id=t["id"], season=_season_str(season),
                                               timeout=30).get_data_frames()[0]
    return set(roster["PLAYER_ID"])


def matchups_vs_team(player_name: str, opp: str, season: int = config.SEASON) -> pd.DataFrame:
    """Defenders on `opp` who have guarded `player_name`, with per-100 rates."""
    from nba_api.stats.endpoints import leagueseasonmatchups

    pid = _nba_player_id(player_name)
    if pid is None:
        raise RuntimeError(f"NBA.com doesn't recognize '{player_name}'.")
    df = leagueseasonmatchups.LeagueSeasonMatchups(
        season=_season_str(season), off_player_id_nullable=pid,
        per_mode_simple="Totals", timeout=30,
    ).get_data_frames()[0]
    if df.empty:
        return df

    poss = df["PARTIAL_POSS"].sum()
    overall = {
        "pts": df["PLAYER_PTS"].sum() / poss * 100 if poss else None,
        "ast": df["MATCHUP_AST"].sum() / poss * 100 if poss and "MATCHUP_AST" in df else None,
    }
    roster = _roster_ids(opp, season)
    mine = df[df["DEF_PLAYER_ID"].isin(roster)].copy()
    if mine.empty:
        return mine
    mine["pts_per100"] = mine["PLAYER_PTS"] / mine["PARTIAL_POSS"] * 100
    mine["pts_vs_overall"] = mine["pts_per100"] - overall["pts"]
    if "MATCHUP_AST" in mine:
        mine["ast_per100"] = mine["MATCHUP_AST"] / mine["PARTIAL_POSS"] * 100
        mine["ast_vs_overall"] = mine["ast_per100"] - overall["ast"]
    keep = [c for c in ["DEF_PLAYER_NAME", "GP", "MATCHUP_MIN", "PARTIAL_POSS",
                        "PLAYER_PTS", "pts_per100", "pts_vs_overall",
                        "MATCHUP_AST", "ast_per100", "ast_vs_overall",
                        "MATCHUP_FG_PCT"] if c in mine.columns]
    return mine[keep].sort_values("PARTIAL_POSS", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ caching
def cached_matchups(player_name: str, opp: str, season: int = config.SEASON,
                    max_age_hours: float = 24) -> pd.DataFrame:
    """matchups_vs_team, saved to data/matchups.csv so each pair is only fetched once a day."""
    p = config.MATCHUPS_CSV
    now = pd.Timestamp.now()
    if p.exists():
        cache = pd.read_csv(p, parse_dates=["fetched"])
        hit = cache[(cache["player"] == player_name) & (cache["opp"] == opp)
                    & (cache["season"] == season)]
        if not hit.empty and (now - hit["fetched"].max()).total_seconds() < max_age_hours * 3600:
            return hit.drop(columns=["player", "opp", "season", "fetched"]).reset_index(drop=True)
    else:
        cache = pd.DataFrame()
    df = matchups_vs_team(player_name, opp, season)
    if not cache.empty:
        cache = cache[~((cache["player"] == player_name) & (cache["opp"] == opp)
                        & (cache["season"] == season))]
    new = df.assign(player=player_name, opp=opp, season=season, fetched=now)
    pd.concat([cache, new], ignore_index=True).to_csv(p, index=False)
    return df


def likely_defender(player_name: str, opp: str, season: int = config.SEASON) -> dict | None:
    """The opponent defender who has guarded this player the most this season
    (falls back to last season), and how the player did against him."""
    for s in (season, season - 1):
        try:
            df = cached_matchups(player_name, opp, s)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            top = df.iloc[0].to_dict()
            top["season"] = s
            return top
    return None
