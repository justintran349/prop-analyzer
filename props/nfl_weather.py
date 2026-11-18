"""Game-time weather for NFL games from Open-Meteo (free, no key).

* Past games  -> historical archive (so the model can learn what wind/rain/snow/cold do)
* Upcoming    -> 16-day forecast, refreshed every time you update
Values are averaged over the first 3 hours after kickoff (precipitation/snow summed).
Domes and closed roofs are treated as 70°F, no wind, no precipitation.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from . import nfl_data as nd

STADIUMS = {  # stadium_id -> (lat, lon)
    "ATL97": (33.7554, -84.4009), "BAL00": (39.2780, -76.6227), "BOS00": (42.0909, -71.2643),
    "BUF00": (42.7738, -78.7870), "BUF01": (42.7738, -78.7870), "CAR00": (35.2258, -80.8528),
    "CHI98": (41.8623, -87.6167), "CIN00": (39.0954, -84.5160), "CLE00": (41.5061, -81.6995),
    "DAL00": (32.7473, -97.0945), "DEN00": (39.7439, -105.0201), "DET00": (42.3400, -83.0456),
    "FRA00": (50.0686, 8.6455), "GER00": (48.2188, 11.6247), "GNB00": (44.5013, -88.0622),
    "HOU00": (29.6847, -95.4107), "IND00": (39.7601, -86.1639), "JAX00": (30.3239, -81.6373),
    "KAN00": (39.0489, -94.4839), "LAX01": (33.9535, -118.3392), "LON00": (51.5560, -0.2796),
    "LON02": (51.6043, -0.0664), "MAD01": (40.4531, -3.6883), "MEL00": (-37.8200, 144.9834),
    "MEX00": (19.3029, -99.1505), "MIA00": (25.9580, -80.2389), "MIN01": (44.9737, -93.2581),
    "MUN01": (48.2188, 11.6247), "NAS00": (36.1665, -86.7713), "NOR00": (29.9511, -90.0812),
    "NYC01": (40.8135, -74.0745), "PAR00": (48.9245, 2.3602), "PHI00": (39.9008, -75.1675),
    "PHO00": (33.5276, -112.2626), "PIT00": (40.4468, -80.0158), "RIO00": (-22.9122, -43.2302),
    "SAO00": (-23.5453, -46.4742), "SEA00": (47.5952, -122.3316), "SFO01": (37.4030, -121.9700),
    "TAM00": (27.9759, -82.5033), "VEG00": (36.0909, -115.1833), "WAS00": (38.9076, -76.8645),
}
VARS = "temperature_2m,precipitation,snowfall,wind_speed_10m,wind_gusts_10m"
UNITS = dict(temperature_unit="fahrenheit", wind_speed_unit="mph", precipitation_unit="inch",
             timezone="America/New_York")
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
COLUMNS = ["game_id", "source", "wx_temp", "wx_wind", "wx_gust", "wx_precip", "wx_snow"]


def _get(url, params, tries=4):
    """Open-Meteo allows ~600 calls/minute, but a multi-year request counts as many.
    On 'limit exceeded' wait a minute and retry."""
    for _ in range(tries):
        r = requests.get(url, params=params, timeout=60)
        j = r.json()
        if "hourly" in j:
            return j["hourly"]
        if "limit" in str(j.get("reason", "")).lower():
            time.sleep(65)
            continue
        raise RuntimeError(j.get("reason", "Open-Meteo error"))
    raise RuntimeError("Open-Meteo rate limit - try again later")


def _window(hourly, day: str, hour: int, gid: str, source: str):
    key = f"{day}T{int(hour):02d}:00"
    try:
        i = hourly["time"].index(key)
    except ValueError:
        return None

    def vals(k):
        return [v for v in hourly[k][i:i + 3] if v is not None]

    def mean(k):
        v = vals(k)
        return float(np.mean(v)) if v else np.nan

    def total(k):
        v = vals(k)
        return float(np.sum(v)) if v else np.nan

    gust = vals("wind_gusts_10m")
    return {"game_id": gid, "source": source, "wx_temp": round(mean("temperature_2m")),
            "wx_wind": round(mean("wind_speed_10m"), 1), "wx_gust": max(gust) if gust else np.nan,
            "wx_precip": round(total("precipitation"), 2), "wx_snow": round(total("snowfall"), 2)}


def update(games_raw: pd.DataFrame | None = None, refresh_forecast_days: int = 16, log=print) -> pd.DataFrame:
    """Add weather for outdoor games that don't have it yet, and refresh all forecasts."""
    g = games_raw if games_raw is not None else pd.read_csv(nd.RAW / "games.csv", low_memory=False)
    g = g[(g["season"] >= min(nd.TRAINING_SEASONS)) & (~g["roof"].isin(["dome", "closed"]))].copy()
    g["gameday"] = g["gameday"].astype(str)
    g["hour"] = pd.to_numeric(g["gametime"].astype(str).str.split(":").str[0], errors="coerce").fillna(13)
    today = pd.Timestamp.today().normalize()
    archive_cut = (today - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    horizon = (today + pd.Timedelta(days=refresh_forecast_days - 1)).strftime("%Y-%m-%d")
    g = g[g["gameday"] <= horizon]

    have = pd.read_csv(nd.WEATHER_CSV) if nd.WEATHER_CSV.exists() else pd.DataFrame(columns=COLUMNS)
    done_archive = set(have.loc[have["source"] == "archive", "game_id"])
    need_past = g[(g["gameday"] <= archive_cut) & ~g["game_id"].isin(done_archive)]
    need_fut = g[g["gameday"] > archive_cut]

    rows = []
    for st, grp in need_past.groupby("stadium_id"):
        if st not in STADIUMS:
            continue
        lat, lon = STADIUMS[st]
        hourly = _get(ARCHIVE, dict(latitude=lat, longitude=lon, start_date=grp["gameday"].min(),
                                    end_date=grp["gameday"].max(), hourly=VARS, **UNITS))
        rows += [r for r in (_window(hourly, x.gameday, x.hour, x.game_id, "archive")
                             for x in grp.itertuples()) if r]
        time.sleep(1)
    for st, grp in need_fut.groupby("stadium_id"):
        if st not in STADIUMS:
            continue
        lat, lon = STADIUMS[st]
        hourly = _get(FORECAST, dict(latitude=lat, longitude=lon, past_days=7,
                                     forecast_days=refresh_forecast_days, hourly=VARS, **UNITS))
        rows += [r for r in (_window(hourly, x.gameday, x.hour, x.game_id, "forecast")
                             for x in grp.itertuples()) if r]
        time.sleep(0.5)
    if rows:
        new = pd.DataFrame(rows, columns=COLUMNS)
        have = pd.concat([have, new], ignore_index=True).drop_duplicates("game_id", keep="last")
        have.to_csv(nd.WEATHER_CSV, index=False)
    log(f"weather: {len(need_past)} past games added, {len(need_fut)} forecasts refreshed")
    return have


def describe(row) -> str:
    """'🏟️ Dome' or '🌧️ 41°F, wind 17 mph (gusts 28), rain 0.12in'."""
    if row.get("indoor", 0) == 1:
        return "🏟️ Indoors"
    t, w = row.get("game_temp"), row.get("game_wind")
    p, s, gust = row.get("wx_precip", 0) or 0, row.get("wx_snow", 0) or 0, row.get("wx_gust")
    icon = "❄️" if s and s > 0.05 else "🌧️" if p and p > 0.02 else "💨" if w and w >= 15 else "🥶" if t and t < 32 else "☀️"
    parts = [f"{t:.0f}°F" if pd.notna(t) else "?°F", f"wind {w:.0f} mph" if pd.notna(w) else ""]
    if pd.notna(gust) and gust >= 20:
        parts[-1] += f" (gusts {gust:.0f})"
    if p and p > 0.02:
        parts.append(f"rain {p:.2f}in")
    if s and s > 0.05:
        parts.append(f"snow {s:.1f}in")
    return f"{icon} " + ", ".join(x for x in parts if x)


def bad_weather(row) -> bool:
    """Wind 15+, real precipitation, or freezing - the conditions that hurt passing/kicking."""
    if row.get("indoor", 0) == 1:
        return False
    return bool((row.get("game_wind") or 0) >= 15 or (row.get("wx_precip") or 0) > 0.05
                or (row.get("wx_snow") or 0) > 0.05 or (row.get("game_temp") or 60) < 32)
