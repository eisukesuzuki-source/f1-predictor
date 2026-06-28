from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import fastf1
import numpy as np
import pandas as pd
import requests
import streamlit as st

logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


# ─── helpers ────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _safe_pos(val) -> Optional[int]:
    try:
        f = float(val)
        return None if np.isnan(f) else int(f)
    except (TypeError, ValueError):
        return None


# ─── schedule ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_schedule(year: int):
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        return schedule[schedule["RoundNumber"] > 0].reset_index(drop=True)
    except Exception:
        return None


# ─── session results ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_session_results(year: int, round_number: int, session_type: str):
    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        if session.results is None or len(session.results) == 0:
            return None
        return session.results.reset_index(drop=True)
    except Exception:
        return None


# ─── qualifying data (with laps for compound) ───────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_quali_data(year: int, round_number: int) -> dict:
    """Load qualifying session with laps to extract compound and Q3 gap.

    Returns dict: abbr -> {grid_pos, compound, gap_to_pole_pct, headshot_url}
    Empty dict on failure.
    """
    try:
        session = fastf1.get_session(year, round_number, "Q")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception:
        return {}

    results = session.results
    if results is None or len(results) == 0:
        return {}

    try:
        laps = session.laps
    except Exception:
        laps = None

    # Build per-driver fastest lap compound and time
    driver_compound: dict[str, str] = {}
    driver_best_time: dict[str, float] = {}  # in seconds

    if laps is not None and len(laps) > 0:
        for abbr in results["Abbreviation"].values:
            abbr = str(abbr)
            drv_laps = laps[laps["Driver"] == abbr]
            if drv_laps.empty:
                continue

            # Use pick_fastest() which works across Q1/Q2/Q3 segments
            try:
                fastest_lap = drv_laps.pick_fastest()
                cmp = str(fastest_lap.get("Compound", "UNKNOWN") or "UNKNOWN").upper()
                lt = fastest_lap["LapTime"]
                t = lt.total_seconds() if hasattr(lt, "total_seconds") else float("inf")
            except Exception:
                # Manual fallback: get fastest valid lap
                valid = drv_laps.dropna(subset=["LapTime"])
                if valid.empty:
                    continue
                fastest_lap = valid.loc[valid["LapTime"].idxmin()]
                cmp = str(fastest_lap.get("Compound", "UNKNOWN") or "UNKNOWN").upper()
                lt = fastest_lap["LapTime"]
                t = lt.total_seconds() if hasattr(lt, "total_seconds") else float("inf")

            driver_compound[abbr] = cmp
            if t < float("inf"):
                driver_best_time[abbr] = t

    # Compute pole time (best Q3 time among all drivers)
    q3_times: list[float] = []
    for abbr, results_row in results.set_index("Abbreviation").iterrows():
        abbr = str(abbr)
        t = None
        for col in ["Q3", "Q2", "Q1"]:
            val = results_row.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                try:
                    t = pd.Timedelta(val).total_seconds()
                    if t > 0:
                        break
                except Exception:
                    pass
        if t and t > 0:
            q3_times.append(t)
            driver_best_time.setdefault(abbr, t)

    pole_time = min(q3_times) if q3_times else None

    # Build output dict
    out: dict[str, dict] = {}
    for _, row in results.iterrows():
        abbr = str(row.get("Abbreviation") or "")
        if not abbr:
            continue

        grid_pos = _safe_pos(row.get("Position"))
        headshot = str(row.get("HeadshotUrl") or "")

        # Best time from results columns if not captured from laps
        if abbr not in driver_best_time:
            for col in ["Q3", "Q2", "Q1"]:
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    try:
                        t = pd.Timedelta(val).total_seconds()
                        if t > 0:
                            driver_best_time[abbr] = t
                            break
                    except Exception:
                        pass

        best_t = driver_best_time.get(abbr)
        if pole_time and best_t and pole_time > 0:
            gap_pct = (best_t - pole_time) / pole_time * 100.0
        else:
            gap_pct = 0.0

        out[abbr] = {
            "grid_pos": grid_pos if grid_pos is not None else 20,
            "compound": driver_compound.get(abbr, "UNKNOWN"),
            "gap_to_pole_pct": max(0.0, gap_pct),
            "headshot_url": headshot,
            "full_name": str(row.get("FullName") or row.get("BroadcastName") or abbr),
            "team": str(row.get("TeamName") or ""),
        }

    return out


# ─── weather ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_weather_for_race(lat: float, lon: float, race_date: str) -> dict:
    """Fetch daily weather for race_date (YYYY-MM-DD) from Open-Meteo.

    Uses forecast endpoint for dates within 7 days from today,
    archive endpoint for past dates.

    Returns dict with air_temp_max, air_temp_min, rain_probability,
    rain_mm, wind_speed.
    """
    import datetime as dt

    _defaults = {
        "air_temp_max": 25.0,
        "air_temp_min": 15.0,
        "rain_probability": 10.0,
        "rain_mm": 0.0,
        "wind_speed": 10.0,
    }

    try:
        race_dt = dt.date.fromisoformat(race_date)
        today = dt.date.today()
        days_diff = (race_dt - today).days

        daily_params = (
            "temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max,precipitation_sum,windspeed_10m_max"
        )

        # Forecast API: future dates (within ~16 days)
        # Archive API: past dates (precipitation_probability not available in archive)
        is_future = days_diff >= 0
        if is_future:
            base_url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": daily_params,
                "timezone": "auto",
                "start_date": race_date,
                "end_date": race_date,
            }
        else:
            base_url = "https://archive-api.open-meteo.com/v1/archive"
            # Archive API does not support precipitation_probability_max
            archive_params = (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,windspeed_10m_max"
            )
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": archive_params,
                "timezone": "auto",
                "start_date": race_date,
                "end_date": race_date,
            }

        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})

        def _first(key: str, fallback: float) -> float:
            vals = daily.get(key, [])
            if vals and vals[0] is not None:
                return float(vals[0])
            return fallback

        rain_mm = _first("precipitation_sum", _defaults["rain_mm"])
        if is_future:
            rain_prob = _first("precipitation_probability_max", _defaults["rain_probability"])
        else:
            # Estimate rain probability from precipitation amount for archive data
            rain_prob = min(95.0, rain_mm * 20.0) if rain_mm > 0 else 5.0

        return {
            "air_temp_max": _first("temperature_2m_max", _defaults["air_temp_max"]),
            "air_temp_min": _first("temperature_2m_min", _defaults["air_temp_min"]),
            "rain_probability": rain_prob,
            "rain_mm": rain_mm,
            "wind_speed": _first("windspeed_10m_max", _defaults["wind_speed"]),
        }

    except Exception:
        return dict(_defaults)


# ─── wet skill ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_wet_skill(year: int, round_number: int) -> dict:
    """Compute per-driver wet-race advantage over the last 3 years.

    Wet races are identified by session.weather_data['Rainfall'].any().
    For each driver: avg(GridPosition - Position) in wet vs dry races.
    Positive score = driver improves more in wet conditions.

    Returns dict: abbr -> wet_advantage_score (float)
    """
    wet_deltas: dict[str, list] = {}
    dry_deltas: dict[str, list] = {}

    for past_year in range(year - 3, year):
        if past_year < 2019:
            continue
        sched = load_schedule(past_year)
        if sched is None:
            continue

        max_round = int(sched["RoundNumber"].max())
        for rnd in range(1, max_round + 1):
            try:
                session = fastf1.get_session(past_year, rnd, "R")
                session.load(
                    laps=False,
                    telemetry=False,
                    weather=True,
                    messages=False,
                )
            except Exception:
                continue

            is_wet = False
            try:
                if (
                    session.weather_data is not None
                    and not session.weather_data.empty
                    and "Rainfall" in session.weather_data.columns
                ):
                    is_wet = bool(session.weather_data["Rainfall"].any())
            except Exception:
                pass

            results = session.results
            if results is None or len(results) == 0:
                continue

            for _, row in results.iterrows():
                abbr = str(row.get("Abbreviation") or "")
                if not abbr:
                    continue
                grid = _safe_pos(row.get("GridPosition"))
                finish = _safe_pos(row.get("Position"))
                if grid is None or finish is None or grid == 0:
                    continue
                delta = float(grid) - float(finish)  # positive = gained positions
                if is_wet:
                    wet_deltas.setdefault(abbr, []).append(delta)
                else:
                    dry_deltas.setdefault(abbr, []).append(delta)

    all_abbrs = set(wet_deltas.keys()) | set(dry_deltas.keys())
    result: dict[str, float] = {}
    for abbr in all_abbrs:
        wet_avg = float(np.mean(wet_deltas[abbr])) if abbr in wet_deltas else 0.0
        dry_avg = float(np.mean(dry_deltas[abbr])) if abbr in dry_deltas else 0.0
        result[abbr] = wet_avg - dry_avg  # positive = relatively better in wet

    return result


# ─── recent form ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_recent_form(year: int, round_number: int, n_races: int = 5):
    """Direct N races back: return (driver_form, team_form) dicts.

    driver_form: avg finishing position (lower = better; DNF → 20)
    team_form: avg points
    """
    driver_pos: dict[str, list] = {}
    team_pts: dict[str, list] = {}

    collected = 0
    curr_year, curr_round = year, round_number - 1

    while collected < n_races and curr_year >= 2020:
        if curr_round < 1:
            curr_year -= 1
            if curr_year < 2020:
                break
            sched = load_schedule(curr_year)
            if sched is None:
                break
            curr_round = int(sched["RoundNumber"].max())

        results = load_session_results(curr_year, curr_round, "R")
        if results is not None and len(results) > 0:
            for _, row in results.iterrows():
                abbr = str(row.get("Abbreviation") or "")
                team = str(row.get("TeamName") or "")
                pos = _safe_pos(row.get("Position"))
                pts = _safe_float(row.get("Points"), 0.0)
                if abbr:
                    driver_pos.setdefault(abbr, []).append(
                        float(pos) if pos is not None else 20.0
                    )
                if team:
                    team_pts.setdefault(team, []).append(pts)
            collected += 1

        curr_round -= 1

    return (
        {k: float(np.mean(v)) for k, v in driver_pos.items()},
        {k: float(np.mean(v)) for k, v in team_pts.items()},
    )


# ─── circuit history ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_circuit_history(year: int, round_number: int, n_years: int = 3) -> dict:
    """Return per-driver average finish position at the same circuit over past n years."""
    schedule = load_schedule(year)
    if schedule is None:
        return {}

    event_rows = schedule[schedule["RoundNumber"] == round_number]
    if len(event_rows) == 0:
        return {}

    target_location = str(event_rows.iloc[0].get("Location") or "")
    target_country = str(event_rows.iloc[0].get("Country") or "")

    driver_positions: dict[str, list] = {}

    for past_year in range(year - 1, max(year - n_years - 1, 2019), -1):
        past_schedule = load_schedule(past_year)
        if past_schedule is None:
            continue

        if "Location" in past_schedule.columns:
            loc_match = (
                past_schedule["Location"].astype(str).str.lower() == target_location.lower()
            )
        else:
            loc_match = pd.Series([False] * len(past_schedule))

        if "Country" in past_schedule.columns:
            country_match = (
                past_schedule["Country"].astype(str).str.lower() == target_country.lower()
            )
        else:
            country_match = pd.Series([False] * len(past_schedule))

        matches = past_schedule[loc_match | country_match]
        if len(matches) == 0:
            continue

        past_round = int(matches.iloc[0]["RoundNumber"])
        results = load_session_results(past_year, past_round, "R")

        if results is not None and len(results) > 0:
            for _, row in results.iterrows():
                abbr = str(row.get("Abbreviation") or "")
                pos = _safe_pos(row.get("Position"))
                if abbr:
                    driver_positions.setdefault(abbr, []).append(
                        float(pos) if pos is not None else 20.0
                    )

    return {k: float(np.mean(v)) for k, v in driver_positions.items()}


# ─── recent form detail ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_recent_form_detail(year: int, round_number: int, n_races: int = 5) -> dict:
    """Per-race breakdown of recent form for chart display.

    Returns: {abbr: {"races": [...], "points": [...], "positions": [...], "team": str}}
    Ordered oldest-first.
    """
    races_collected: list = []  # [(label, {abbr: (pts, pos, team)})]

    collected = 0
    curr_year, curr_round = year, round_number - 1

    while collected < n_races and curr_year >= 2020:
        if curr_round < 1:
            curr_year -= 1
            if curr_year < 2020:
                break
            sched = load_schedule(curr_year)
            if sched is None:
                break
            curr_round = int(sched["RoundNumber"].max())

        sched = load_schedule(curr_year)
        race_label = f"Rd.{curr_round}"
        if sched is not None:
            matching = sched[sched["RoundNumber"] == curr_round]
            if len(matching) > 0:
                gp = str(matching.iloc[0].get("EventName", ""))
                words = gp.replace("Grand Prix", "GP").split()
                race_label = " ".join(words[:2])[:12] if words else race_label

        results = load_session_results(curr_year, curr_round, "R")
        if results is not None and len(results) > 0:
            race_data: dict = {}
            for _, row in results.iterrows():
                abbr = str(row.get("Abbreviation") or "")
                team = str(row.get("TeamName") or "")
                pts = _safe_float(row.get("Points"), 0.0)
                pos_raw = _safe_pos(row.get("Position"))
                pos = float(pos_raw) if pos_raw is not None else 20.0
                if abbr:
                    race_data[abbr] = (pts, pos, team)
            races_collected.append((race_label, race_data))
            collected += 1

        curr_round -= 1

    races_collected.reverse()  # oldest first

    all_abbrs: set = set()
    for _, rd in races_collected:
        all_abbrs.update(rd.keys())

    result: dict = {}
    for abbr in all_abbrs:
        races_list: list = []
        pts_list: list = []
        pos_list: list = []
        team = ""
        for label, rd in races_collected:
            if abbr in rd:
                p, pos, t = rd[abbr]
                races_list.append(label)
                pts_list.append(p)
                pos_list.append(pos)
                if not team:
                    team = t
        if races_list:
            result[abbr] = {
                "races": races_list,
                "points": pts_list,
                "positions": pos_list,
                "team": team,
            }

    return result


# ─── race tire strategy ──────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_tire_strategy_data(year: int, round_number: int, n_years: int = 3) -> dict:
    """Historical race tire strategy metrics at this circuit.

    Loads race laps for the same circuit over the past n_years and extracts
    per-driver stint information.  Stint boundaries are detected by Compound
    changes (reliable across FastF1 versions).

    Returns: {
        abbr: {
            "avg_stops":        float,    # average pit stops (stints - 1)
            "tire_mgmt_index":  float,    # avg stint len / field avg  (>1 = good)
            "typical_compounds": [str],   # compound sequence from most recent year
        }
    }
    """
    schedule = load_schedule(year)
    if schedule is None:
        return {}

    event_rows = schedule[schedule["RoundNumber"] == round_number]
    if len(event_rows) == 0:
        return {}

    target_location = str(event_rows.iloc[0].get("Location") or "")
    target_country  = str(event_rows.iloc[0].get("Country") or "")

    driver_stops:    dict[str, list] = {}
    driver_stints:   dict[str, list] = {}   # list of (compound, length) tuples
    driver_compounds: dict[str, list] = {}  # most recent year wins
    # field stint lengths grouped by compound for per-compound normalisation
    field_by_compound: dict[str, list] = {}

    for past_year in range(year - 1, max(year - n_years - 1, 2019), -1):
        past_schedule = load_schedule(past_year)
        if past_schedule is None:
            continue

        loc_match = (
            past_schedule["Location"].astype(str).str.lower() == target_location.lower()
            if "Location" in past_schedule.columns
            else pd.Series([False] * len(past_schedule))
        )
        cty_match = (
            past_schedule["Country"].astype(str).str.lower() == target_country.lower()
            if "Country" in past_schedule.columns
            else pd.Series([False] * len(past_schedule))
        )
        matches = past_schedule[loc_match | cty_match]
        if len(matches) == 0:
            continue

        past_round = int(matches.iloc[0]["RoundNumber"])
        try:
            session = fastf1.get_session(past_year, past_round, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = session.laps
        except Exception:
            continue

        if laps is None or len(laps) == 0:
            continue

        # Minimum laps threshold: exclude DNF/early retirement (< 30% of race distance)
        max_laps = int(laps["LapNumber"].max()) if len(laps) > 0 else 1
        min_laps_threshold = max(5, int(max_laps * 0.30))

        for abbr in laps["Driver"].unique():
            abbr = str(abbr)
            drv = laps[laps["Driver"] == abbr].sort_values("LapNumber")
            if drv.empty:
                continue

            # Skip drivers who retired early (DNF, crash, etc.)
            if len(drv) < min_laps_threshold:
                continue

            # Detect stints by Compound changes
            stints: list = []     # list of (compound, length)
            cur_comp: Optional[str] = None
            cur_len = 0
            for _, lap in drv.iterrows():
                c = str(lap.get("Compound", "") or "").strip().upper()
                if c in ("", "UNKNOWN", "NAN", "NONE"):
                    c = cur_comp or "UNKNOWN"
                if c != cur_comp:
                    if cur_comp is not None and cur_len > 0:
                        stints.append((cur_comp, cur_len))
                    cur_comp = c
                    cur_len  = 1
                else:
                    cur_len += 1
            if cur_comp and cur_len > 0:
                stints.append((cur_comp, cur_len))

            if not stints:
                continue

            n_stops = len(stints) - 1
            compounds = [s[0] for s in stints if s[0] != "UNKNOWN"]

            driver_stops.setdefault(abbr, []).append(n_stops)
            driver_stints.setdefault(abbr, []).extend(stints)
            # Keep most recent year's compound list (loop goes newest→oldest)
            if compounds and abbr not in driver_compounds:
                driver_compounds[abbr] = compounds
            # Accumulate field stint lengths per compound
            for comp, length in stints:
                if comp != "UNKNOWN" and length > 0:
                    field_by_compound.setdefault(comp, []).append(length)

    # Per-compound field average; fallback to overall average for unknowns
    field_avg_by_compound = {
        c: float(np.mean(lens))
        for c, lens in field_by_compound.items() if lens
    }
    all_field_lens = [l for ls in field_by_compound.values() for l in ls]
    field_avg_overall = float(np.mean(all_field_lens)) if all_field_lens else 20.0

    result: dict = {}
    all_abbrs = set(driver_stops) | set(driver_stints)
    for abbr in all_abbrs:
        avg_stops = float(np.mean(driver_stops[abbr])) if abbr in driver_stops else 1.5
        # tire_mgmt_index: average of (stint_length / field_avg_for_that_compound)
        ratios = [
            length / field_avg_by_compound.get(comp, field_avg_overall)
            for comp, length in driver_stints.get(abbr, [])
            if length > 0 and comp != "UNKNOWN"
        ]
        tire_mgmt_index = float(np.mean(ratios)) if ratios else 1.0
        result[abbr] = {
            "avg_stops":         round(avg_stops, 2),
            "tire_mgmt_index":   round(tire_mgmt_index, 3),
            "typical_compounds": driver_compounds.get(abbr, ["SOFT", "MEDIUM"]),
        }

    return result


# ─── circuit history detail ──────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_circuit_history_detail(year: int, round_number: int, n_years: int = 3) -> dict:
    """Per-year finishing position at this circuit for chart display.

    Returns: {abbr: {"years": [2023, 2022, ...], "positions": [3.0, 5.0, ...]}}
    Ordered oldest-first.
    """
    schedule = load_schedule(year)
    if schedule is None:
        return {}

    event_rows = schedule[schedule["RoundNumber"] == round_number]
    if len(event_rows) == 0:
        return {}

    target_location = str(event_rows.iloc[0].get("Location") or "")
    target_country = str(event_rows.iloc[0].get("Country") or "")

    year_results: list = []  # [(past_year, {abbr: pos})]

    for past_year in range(year - 1, max(year - n_years - 1, 2019), -1):
        past_schedule = load_schedule(past_year)
        if past_schedule is None:
            continue

        if "Location" in past_schedule.columns:
            loc_match = past_schedule["Location"].astype(str).str.lower() == target_location.lower()
        else:
            loc_match = pd.Series([False] * len(past_schedule))

        if "Country" in past_schedule.columns:
            country_match = past_schedule["Country"].astype(str).str.lower() == target_country.lower()
        else:
            country_match = pd.Series([False] * len(past_schedule))

        matches = past_schedule[loc_match | country_match]
        if len(matches) == 0:
            continue

        past_round = int(matches.iloc[0]["RoundNumber"])
        results = load_session_results(past_year, past_round, "R")

        if results is not None and len(results) > 0:
            yr_data: dict = {}
            for _, row in results.iterrows():
                abbr = str(row.get("Abbreviation") or "")
                pos = _safe_pos(row.get("Position"))
                if abbr:
                    yr_data[abbr] = float(pos) if pos is not None else 20.0
            year_results.append((past_year, yr_data))

    year_results.reverse()  # oldest first

    all_abbrs: set = set()
    for _, yd in year_results:
        all_abbrs.update(yd.keys())

    result: dict = {}
    for abbr in all_abbrs:
        years: list = []
        positions: list = []
        for past_year, yd in year_results:
            if abbr in yd:
                years.append(past_year)
                positions.append(yd[abbr])
        if years:
            result[abbr] = {"years": years, "positions": positions}

    return result


# ─── driver metadata ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_driver_metadata(year: int, round_number: int) -> dict:
    """Return abbr -> {full_name, team, headshot_url} from the most recent race.

    Walks backwards from round_number - 1 to find the latest available race
    result so we can populate driver names/teams even for future rounds.
    """
    curr_year, curr_round = year, round_number - 1

    while curr_year >= 2020:
        if curr_round < 1:
            curr_year -= 1
            if curr_year < 2020:
                break
            sched = load_schedule(curr_year)
            if sched is None:
                break
            curr_round = int(sched["RoundNumber"].max())

        results = load_session_results(curr_year, curr_round, "R")
        if results is not None and len(results) > 0:
            meta: dict = {}
            for _, row in results.iterrows():
                abbr = str(row.get("Abbreviation") or "")
                if not abbr:
                    continue
                meta[abbr] = {
                    "full_name": str(
                        row.get("FullName") or row.get("BroadcastName") or abbr
                    ),
                    "team": str(row.get("TeamName") or ""),
                    "headshot_url": str(row.get("HeadshotUrl") or ""),
                }
            if meta:
                return meta

        curr_round -= 1

    return {}


# ─── previous year's race start compound (for race tire prediction) ─────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_prev_year_start_compounds(year: int, round_number: int) -> dict:
    """Return {abbr: compound} — each driver's race STARTING compound at this
    circuit in the previous year (year-1).  Only looks at year-1; if that race
    is unavailable returns empty dict so caller falls back to MEDIUM.
    """
    schedule = load_schedule(year)
    if schedule is None:
        return {}

    event_rows = schedule[schedule["RoundNumber"] == round_number]
    if len(event_rows) == 0:
        return {}

    target_location = str(event_rows.iloc[0].get("Location") or "")
    target_country  = str(event_rows.iloc[0].get("Country") or "")

    past_year = year - 1
    past_schedule = load_schedule(past_year)
    if past_schedule is None:
        return {}

    loc_match = (
        past_schedule["Location"].astype(str).str.lower() == target_location.lower()
        if "Location" in past_schedule.columns
        else pd.Series([False] * len(past_schedule))
    )
    cty_match = (
        past_schedule["Country"].astype(str).str.lower() == target_country.lower()
        if "Country" in past_schedule.columns
        else pd.Series([False] * len(past_schedule))
    )
    matches = past_schedule[loc_match | cty_match]
    if len(matches) == 0:
        return {}

    past_round = int(matches.iloc[0]["RoundNumber"])
    try:
        session = fastf1.get_session(past_year, past_round, "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
    except Exception:
        return {}

    if laps is None or len(laps) == 0:
        return {}

    result: dict = {}
    for abbr in laps["Driver"].unique():
        abbr = str(abbr)
        drv = laps[laps["Driver"] == abbr].sort_values("LapNumber")
        if drv.empty:
            continue
        # First valid compound = starting compound
        for _, lap in drv.iterrows():
            c = str(lap.get("Compound", "") or "").strip().upper()
            if c not in ("", "UNKNOWN", "NAN", "NONE"):
                result[abbr] = c
                break

    return result


# ─── actual race tire compounds ──────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_race_tire_compounds(year: int, round_number: int) -> dict:
    """Return {abbr: [compound, ...]} with each distinct tire used in stint order.

    Returns empty dict if the race hasn't happened yet or data is unavailable.
    """
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
    except Exception:
        return {}

    if laps is None or len(laps) == 0:
        return {}

    result: dict = {}
    for abbr in laps["Driver"].unique():
        abbr = str(abbr)
        drv = laps[laps["Driver"] == abbr].sort_values("LapNumber")
        if drv.empty:
            continue

        compounds: list = []
        cur_comp: Optional[str] = None
        for _, lap in drv.iterrows():
            c = str(lap.get("Compound", "") or "").strip().upper()
            if c in ("", "UNKNOWN", "NAN", "NONE"):
                continue
            if c != cur_comp:
                compounds.append(c)
                cur_comp = c

        if compounds:
            result[abbr] = compounds

    return result
