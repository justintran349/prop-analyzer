"""Team names and abbreviations. Basketball-Reference, sportsbooks and NBA.com
all spell teams differently, so everything gets normalized to the BRef code."""

TEAMS = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BRK": "Brooklyn Nets",
    "CHO": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

NAME_TO_CODE = {name.lower(): code for code, name in TEAMS.items()}
NAME_TO_CODE["la clippers"] = "LAC"

# Abbreviations other sites use -> BRef code
ALIASES = {"BKN": "BRK", "CHA": "CHO", "PHX": "PHO", "NO": "NOP", "NY": "NYK",
           "GS": "GSW", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


def to_code(team: str) -> str:
    """Turn any team name/abbreviation into the Basketball-Reference code."""
    if not team:
        return ""
    t = team.strip()
    up = t.upper()
    if up in TEAMS:
        return up
    if up in ALIASES:
        return ALIASES[up]
    return NAME_TO_CODE.get(t.lower(), up)
