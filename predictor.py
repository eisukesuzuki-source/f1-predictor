from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─── DriverScore dataclass ──────────────────────────────────────────────────

@dataclass
class DriverScore:
    abbreviation: str
    full_name: str
    team: str

    # Qualifying
    grid_position: Optional[int] = None
    grid_score: float = 0.0
    compound: str = "UNKNOWN"
    compound_reason: str = ""
    gap_to_pole_pct: float = 0.0
    headshot_url: str = ""

    # Sub-scores
    car_pace_score: float = 0.0
    driver_form_avg_pts: float = 0.0
    driver_form_score: float = 0.0
    team_form_avg_pts: float = 0.0
    team_form_score: float = 0.0
    circuit_avg_pos: Optional[float] = None
    circuit_score: float = 0.0
    tire_score: float = 0.0
    tire_strategy_score: float = 0.0
    weather_score: float = 0.0
    circuit_char_score: float = 0.0
    temp_score: float = 0.0

    # Environmental context
    air_temp: float = 25.0
    rain_probability: float = 0.0

    total_score: float = 0.0
    predicted_position: int = 0
    explanation: str = ""
    has_quali: bool = False


# ─── helpers ────────────────────────────────────────────────────────────────

def _normalize(values: list, higher_is_better: bool = True) -> list:
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.5] * len(values)
    norm = [(v - mn) / (mx - mn) for v in values]
    return norm if higher_is_better else [1.0 - v for v in norm]


def _safe_int(val) -> Optional[int]:
    try:
        f = float(val)
        return None if np.isnan(f) else int(f)
    except (TypeError, ValueError):
        return None


def _predict_race_compound(
    abbr: str,
    prev_year_compounds: Optional[dict],
    rain_prob: float,
) -> tuple:
    """Return (compound, reason) for predicted race start tire.

    Priority:
    1. Heavy rain (≥70%) → WET
    2. Moderate rain (≥40%) → INTERMEDIATE
    3. Previous year's race start compound at this circuit
    4. Default → MEDIUM
    """
    VALID = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
    # Heavy rain
    if rain_prob >= 70:
        return ("WET", "降水確率70%以上のため大雨想定")
    # Moderate rain
    if rain_prob >= 50:
        return ("INTERMEDIATE", "降水確率50%以上のため雨想定")
    # Previous year's race start compound at this exact circuit
    prev = (prev_year_compounds or {}).get(abbr, "")
    if prev.upper() in VALID:
        return (prev.upper(), "前年同サーキットのスタートタイヤ実績")
    # Default
    return ("MEDIUM", "過去データなしのためデフォルト")


def _tire_score_for_driver(compound: str, tire_deg: int, air_temp_max: float) -> float:
    """Compute tire score (0–1) based on compound vs circuit deg and temperature."""
    compound = compound.upper()
    # Adjust effective deg for high temperature
    effective_deg = tire_deg
    if air_temp_max > 32:
        effective_deg = min(5, tire_deg + 1)

    if compound == "SOFT":
        if effective_deg >= 4:
            return 0.25   # penalised on high-deg track
        elif effective_deg <= 2:
            return 0.85   # ideal for low-deg: raw pace advantage
        else:
            return 0.55   # medium deg: slight edge in qualy pace
    elif compound == "MEDIUM":
        if effective_deg >= 4:
            return 0.75   # better race pace on high-deg
        elif effective_deg <= 2:
            return 0.65   # decent on low-deg
        else:
            return 0.70   # neutral/good
    elif compound == "HARD":
        if effective_deg >= 4:
            return 0.80   # race-pace king on high-deg
        elif effective_deg <= 2:
            return 0.50   # no pace advantage on low-deg
        else:
            return 0.65
    elif compound in ("INTERMEDIATE", "INTER"):
        return 0.50   # neutral (should not appear in Q3)
    elif compound == "WET":
        return 0.50
    else:
        return 0.50   # UNKNOWN


def _temp_score_for_driver(compound: str, air_temp_max: float) -> float:
    """Compound fit based on air temperature optimum windows."""
    compound = compound.upper()
    t = air_temp_max
    if compound == "SOFT":
        # optimal 15–25 °C
        if 15.0 <= t <= 25.0:
            return 0.85
        elif t < 15.0:
            return 0.85 - (15.0 - t) * 0.02  # gets colder → drops a little
        else:
            return 0.85 - (t - 25.0) * 0.04  # gets hotter → drops more
    elif compound == "MEDIUM":
        # optimal 20–32 °C
        if 20.0 <= t <= 32.0:
            return 0.80
        elif t < 20.0:
            return 0.80 - (20.0 - t) * 0.02
        else:
            return 0.80 - (t - 32.0) * 0.03
    elif compound == "HARD":
        # optimal > 28 °C
        if t >= 28.0:
            return 0.80
        else:
            return 0.80 - (28.0 - t) * 0.025
    else:
        return 0.50

    # Clamp to [0, 1]


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


# ─── main prediction function ────────────────────────────────────────────────

def predict_race(
    quali_data: dict,           # abbr -> {grid_pos, compound, gap_to_pole_pct, headshot_url, full_name, team}
    driver_form: dict,          # abbr -> avg pts
    team_form: dict,            # team_name -> avg pts
    circuit_history: dict,      # abbr -> avg finish pos
    circuit_info: dict,         # from get_circuit_info
    weather: dict,              # from get_weather_for_race
    wet_skills: dict,           # abbr -> wet_advantage_score
    tire_strategy: Optional[dict] = None,       # abbr -> {avg_stops, tire_mgmt_index, typical_compounds}
    prev_year_compounds: Optional[dict] = None,  # abbr -> start compound from prev year same circuit
) -> list:
    """Compute DriverScore list sorted by predicted position."""

    has_quali = len(quali_data) > 0

    air_temp_max = float(weather.get("air_temp_max", 25.0))
    rain_prob = float(weather.get("rain_probability", 10.0))  # 0–100
    tire_deg = int(circuit_info.get("tire_deg", 3))
    altitude_m = float(circuit_info.get("altitude_m", 50))
    pu_sensitivity = float(circuit_info.get("pu_sensitivity", 3))
    overtaking = int(circuit_info.get("overtaking", 3))

    # ── Build driver list ────────────────────────────────────────────────────
    drivers: list = []

    if has_quali:
        for abbr, qdata in quali_data.items():
            team = str(qdata.get("team", ""))
            ds = DriverScore(
                abbreviation=abbr,
                full_name=str(qdata.get("full_name", abbr)),
                team=team,
                grid_position=int(qdata.get("grid_pos", 20)),
                compound=str(qdata.get("compound", "UNKNOWN")).upper(),
                gap_to_pole_pct=float(qdata.get("gap_to_pole_pct", 0.0)),
                headshot_url=str(qdata.get("headshot_url", "")),
                has_quali=True,
                driver_form_avg_pts=float(driver_form.get(abbr, 0.0)),
                team_form_avg_pts=float(team_form.get(team, 0.0)),
                circuit_avg_pos=circuit_history.get(abbr),
                air_temp=air_temp_max,
                rain_probability=rain_prob,
            )
            drivers.append(ds)
    else:
        all_abbrs = set(driver_form.keys())
        for abbr in sorted(all_abbrs):
            ds = DriverScore(
                abbreviation=abbr,
                full_name=abbr,
                team="",
                has_quali=False,
                driver_form_avg_pts=float(driver_form.get(abbr, 0.0)),
                circuit_avg_pos=circuit_history.get(abbr),
                air_temp=air_temp_max,
                rain_probability=rain_prob,
            )
            drivers.append(ds)

    if not drivers:
        return []

    # ── Weight definitions ────────────────────────────────────────────────────
    has_tire_strat = bool(tire_strategy)

    if has_quali:
        W_QUALI          = 0.30
        W_CAR_PACE       = 0.12
        W_DRIVER_FORM    = 0.12
        W_TEAM_FORM      = 0.00
        W_CIRCUIT_HIST   = 0.08
        W_TIRE           = 0.07
        W_TIRE_STRATEGY  = 0.09
        W_WEATHER        = 0.09
        W_CIRCUIT_CHAR   = 0.07
        W_TEMP           = 0.06
        # Total = 0.30+0.12+0.12+0.08+0.07+0.09+0.09+0.07+0.06 = 1.00

        # Hard-overtaking circuits (Monaco etc.): boost quali weight so grid
        # position actually dominates.  Multiply the weight, NOT the score, to
        # avoid clamping artifacts that collapse top-3 grids to the same value.
        if overtaking <= 2:
            boost = 1.0 + (3 - overtaking) * 0.12   # 1.24 for overtaking=1
            W_QUALI *= boost
            other_sum = (W_CAR_PACE + W_DRIVER_FORM + W_CIRCUIT_HIST
                         + W_TIRE + W_TIRE_STRATEGY + W_WEATHER + W_CIRCUIT_CHAR + W_TEMP)
            scale = (1.0 - W_QUALI) / other_sum
            W_CAR_PACE      *= scale
            W_DRIVER_FORM   *= scale
            W_CIRCUIT_HIST  *= scale
            W_TIRE          *= scale
            W_TIRE_STRATEGY *= scale
            W_WEATHER       *= scale
            W_CIRCUIT_CHAR  *= scale
            W_TEMP          *= scale
    else:
        W_QUALI          = 0.00
        W_CAR_PACE       = 0.00
        W_DRIVER_FORM    = 0.21
        W_TEAM_FORM      = 0.19
        W_CIRCUIT_HIST   = 0.11
        W_TIRE           = 0.07
        W_TIRE_STRATEGY  = 0.10
        W_WEATHER        = 0.12
        W_CIRCUIT_CHAR   = 0.10
        W_TEMP           = 0.10
        # Total = 0.21+0.19+0.11+0.07+0.10+0.12+0.10+0.10 = 1.00

    # ── quali_score ───────────────────────────────────────────────────────────
    if has_quali:
        grid_vals = [float(d.grid_position or 20) for d in drivers]
        grid_norms = _normalize(grid_vals, higher_is_better=False)
        for d, s in zip(drivers, grid_norms):
            d.grid_score = s   # plain 0-1; weight carries the overtaking adjustment

    # ── car_pace_score (normalized gap-to-pole; smaller gap → higher score) ──
    if has_quali:
        gap_vals = [d.gap_to_pole_pct for d in drivers]
        gap_norms = _normalize(gap_vals, higher_is_better=False)
        for d, s in zip(drivers, gap_norms):
            d.car_pace_score = s

    # ── driver_form_score (avg position: lower = better) ──────────────────────
    form_norms = _normalize([d.driver_form_avg_pts for d in drivers], higher_is_better=False)
    for d, s in zip(drivers, form_norms):
        d.driver_form_score = s

    # ── team_form_score ───────────────────────────────────────────────────────
    team_norms = _normalize([d.team_form_avg_pts for d in drivers])
    for d, s in zip(drivers, team_norms):
        d.team_form_score = s

    # ── circuit_hist_score ────────────────────────────────────────────────────
    known_pos = [d.circuit_avg_pos for d in drivers if d.circuit_avg_pos is not None]
    if len(known_pos) >= 2:
        mn_p, mx_p = min(known_pos), max(known_pos)
        for d in drivers:
            if d.circuit_avg_pos is not None and mx_p != mn_p:
                d.circuit_score = 1.0 - (d.circuit_avg_pos - mn_p) / (mx_p - mn_p)
            else:
                d.circuit_score = 0.5
    else:
        for d in drivers:
            d.circuit_score = 0.5

    # ── Predict race start compound (overrides quali compound) ───────────────
    for d in drivers:
        cmp, reason = _predict_race_compound(d.abbreviation, prev_year_compounds, rain_prob)
        d.compound = cmp
        d.compound_reason = reason

    # ── tire_score (predicted race compound × circuit degradation fit) ────────
    for d in drivers:
        d.tire_score = _clamp01(_tire_score_for_driver(d.compound, tire_deg, air_temp_max))

    # ── tire_strategy_score (race stint management from historical data) ──────
    # Drivers with no historical data at this circuit get 0.5 (neutral).
    # Only drivers WITH data are normalized together so new drivers aren't
    # dragged to 0 simply because veterans have higher mgmt_index values.
    if has_tire_strat:
        has_data = [
            d.abbreviation in (tire_strategy or {})
            for d in drivers
        ]
        indices_with_data = [
            float((tire_strategy or {})[d.abbreviation]["tire_mgmt_index"])
            for d, ok in zip(drivers, has_data) if ok
        ]
        if indices_with_data:
            normed_with_data = _normalize(indices_with_data, higher_is_better=True)
        else:
            normed_with_data = []
        data_iter = iter(normed_with_data)
        for d, ok in zip(drivers, has_data):
            if ok:
                d.tire_strategy_score = next(data_iter)
            else:
                d.tire_strategy_score = 0.5   # no historical data → neutral
    else:
        for d in drivers:
            d.tire_strategy_score = 0.5   # neutral when no historical race data

    # ── weather_score ─────────────────────────────────────────────────────────
    # Normalize wet_skills to 0-1
    wet_vals = list(wet_skills.values())
    if len(wet_vals) >= 2:
        ws_min, ws_max = min(wet_vals), max(wet_vals)
        ws_range = ws_max - ws_min if ws_max != ws_min else 1.0
        ws_normalized = {k: (v - ws_min) / ws_range for k, v in wet_skills.items()}
    elif len(wet_vals) == 1:
        ws_normalized = {k: 0.5 for k in wet_skills}
    else:
        ws_normalized = {}

    for d in drivers:
        if rain_prob < 50.0:
            d.weather_score = 0.5  # doesn't matter much in dry conditions
        else:
            wet_norm = ws_normalized.get(d.abbreviation, 0.5)
            d.weather_score = _clamp01(
                (rain_prob / 100.0) * wet_norm + (1.0 - rain_prob / 100.0) * 0.5
            )

    # ── circuit_char_score ────────────────────────────────────────────────────
    # altitude penalty: higher altitude slightly hurts PU-sensitive teams
    altitude_penalty = 1.0 - max(0.0, (altitude_m - 500.0) / 4000.0)
    altitude_penalty = _clamp01(altitude_penalty)

    # pu_score: use car_pace_score as proxy for car's PU quality, scaled by sensitivity
    # Normalize car_pace_score across drivers first (already 0-1)
    for d in drivers:
        car_pace_proxy = d.car_pace_score if has_quali else d.team_form_score
        pu_score = car_pace_proxy * (pu_sensitivity / 5.0)
        d.circuit_char_score = _clamp01(0.6 * altitude_penalty + 0.4 * pu_score)

    # ── temp_score ────────────────────────────────────────────────────────────
    for d in drivers:
        raw = _temp_score_for_driver(d.compound, air_temp_max)
        d.temp_score = _clamp01(raw)

    # ── total_score ───────────────────────────────────────────────────────────
    for d in drivers:
        d.total_score = (
            W_QUALI          * d.grid_score
            + W_CAR_PACE     * d.car_pace_score
            + W_DRIVER_FORM  * d.driver_form_score
            + W_TEAM_FORM    * d.team_form_score
            + W_CIRCUIT_HIST * d.circuit_score
            + W_TIRE         * d.tire_score
            + W_TIRE_STRATEGY * d.tire_strategy_score
            + W_WEATHER      * d.weather_score
            + W_CIRCUIT_CHAR * d.circuit_char_score
            + W_TEMP         * d.temp_score
        )

    # ── rank ──────────────────────────────────────────────────────────────────
    drivers.sort(key=lambda d: d.total_score, reverse=True)
    for i, d in enumerate(drivers):
        d.predicted_position = i + 1

    # ── explanation ───────────────────────────────────────────────────────────
    _generate_explanations(
        drivers,
        has_quali=has_quali,
        circuit_info=circuit_info,
        weather=weather,
        wet_skills=ws_normalized,
        mode="race",
    )

    return drivers


# ─── qualifying prediction function ─────────────────────────────────────────

def predict_qualifying(
    driver_form: dict,
    team_form: dict,
    circuit_history: dict,
    circuit_info: dict,
    weather: dict,
    wet_skills: dict,
    driver_metadata: Optional[dict] = None,
) -> list:
    """Predict qualifying order when no official quali data is available.

    driver_metadata: optional dict abbr -> {full_name, team, headshot_url}
    from get_driver_metadata(). If provided, populates name/team/headshot.

    Weights focus on single-lap pace:
      W_TEAM_FORM    = 0.35  (car pace dominates in single lap)
      W_DRIVER_FORM  = 0.20
      W_CIRCUIT_HIST = 0.18  (position history at this circuit)
      W_CIRCUIT_CHAR = 0.15  (PU sensitivity, altitude)
      W_WEATHER      = 0.07
      W_TEMP         = 0.05
    """
    W_TEAM_FORM    = 0.35
    W_DRIVER_FORM  = 0.20
    W_CIRCUIT_HIST = 0.18
    W_CIRCUIT_CHAR = 0.15
    W_WEATHER      = 0.07
    W_TEMP         = 0.05

    air_temp_max = float(weather.get("air_temp_max", 25.0))
    rain_prob = float(weather.get("rain_probability", 10.0))
    tire_deg = int(circuit_info.get("tire_deg", 3))
    altitude_m = float(circuit_info.get("altitude_m", 50))
    pu_sensitivity = float(circuit_info.get("pu_sensitivity", 3))

    # Build driver list from driver_form; assign team via team_form
    # We need to reverse-map driver -> team. Since team_form is {team: avg_pts},
    # we attempt to infer team per driver from any available source.
    # If a driver->team mapping exists in driver_form values (dict of dicts), use it;
    # otherwise leave team blank and use team_form mean as placeholder.
    drivers: list = []
    all_abbrs = set(driver_form.keys())

    for abbr in sorted(all_abbrs):
        val = driver_form.get(abbr, 0.0)
        # driver_form may be {abbr: float} or {abbr: {avg_pts, team, ...}}
        if isinstance(val, dict):
            avg_pts = float(val.get("avg_pts", 0.0))
            team = str(val.get("team", ""))
        else:
            avg_pts = float(val)
            team = ""

        # Enrich with driver_metadata if available
        headshot_url = ""
        if isinstance(val, dict):
            headshot_url = str(val.get("headshot_url", ""))
        if driver_metadata and abbr in driver_metadata:
            meta = driver_metadata[abbr]
            full_name = str(meta.get("full_name", abbr))
            if not team:
                team = str(meta.get("team", ""))
            if not headshot_url:
                headshot_url = str(meta.get("headshot_url", ""))
        else:
            full_name = abbr

        team_avg_pts = float(team_form.get(team, 0.0)) if team else 0.0

        ds = DriverScore(
            abbreviation=abbr,
            full_name=full_name,
            team=team,
            compound="SOFT",  # qualifying default
            gap_to_pole_pct=0.0,
            headshot_url=headshot_url,
            has_quali=False,
            driver_form_avg_pts=avg_pts,
            team_form_avg_pts=team_avg_pts,
            circuit_avg_pos=circuit_history.get(abbr),
            air_temp=air_temp_max,
            rain_probability=rain_prob,
        )
        drivers.append(ds)

    if not drivers:
        return []

    # ── driver_form_score (avg position: lower = better) ──────────────────────
    form_norms = _normalize([d.driver_form_avg_pts for d in drivers], higher_is_better=False)
    for d, s in zip(drivers, form_norms):
        d.driver_form_score = s

    # ── team_form_score ───────────────────────────────────────────────────────
    team_norms = _normalize([d.team_form_avg_pts for d in drivers])
    for d, s in zip(drivers, team_norms):
        d.team_form_score = s

    # ── circuit_hist_score ────────────────────────────────────────────────────
    known_pos = [d.circuit_avg_pos for d in drivers if d.circuit_avg_pos is not None]
    if len(known_pos) >= 2:
        mn_p, mx_p = min(known_pos), max(known_pos)
        for d in drivers:
            if d.circuit_avg_pos is not None and mx_p != mn_p:
                d.circuit_score = 1.0 - (d.circuit_avg_pos - mn_p) / (mx_p - mn_p)
            else:
                d.circuit_score = 0.5
    else:
        for d in drivers:
            d.circuit_score = 0.5

    # ── tire_score (SOFT default for qualifying) ──────────────────────────────
    for d in drivers:
        d.tire_score = _clamp01(_tire_score_for_driver("SOFT", tire_deg, air_temp_max))

    # ── weather_score ─────────────────────────────────────────────────────────
    wet_vals = list(wet_skills.values())
    if len(wet_vals) >= 2:
        ws_min, ws_max = min(wet_vals), max(wet_vals)
        ws_range = ws_max - ws_min if ws_max != ws_min else 1.0
        ws_normalized = {k: (v - ws_min) / ws_range for k, v in wet_skills.items()}
    elif len(wet_vals) == 1:
        ws_normalized = {k: 0.5 for k in wet_skills}
    else:
        ws_normalized = {}

    for d in drivers:
        if rain_prob < 20.0:
            d.weather_score = 0.5
        else:
            wet_norm = ws_normalized.get(d.abbreviation, 0.5)
            d.weather_score = _clamp01(
                (rain_prob / 100.0) * wet_norm + (1.0 - rain_prob / 100.0) * 0.5
            )

    # ── circuit_char_score ────────────────────────────────────────────────────
    altitude_penalty = 1.0 - max(0.0, (altitude_m - 500.0) / 4000.0)
    altitude_penalty = _clamp01(altitude_penalty)

    for d in drivers:
        pu_score = d.team_form_score * (pu_sensitivity / 5.0)
        d.circuit_char_score = _clamp01(0.6 * altitude_penalty + 0.4 * pu_score)

    # ── temp_score (SOFT default) ─────────────────────────────────────────────
    for d in drivers:
        d.temp_score = _clamp01(_temp_score_for_driver("SOFT", air_temp_max))

    # ── total_score ───────────────────────────────────────────────────────────
    for d in drivers:
        d.total_score = (
            W_TEAM_FORM    * d.team_form_score
            + W_DRIVER_FORM * d.driver_form_score
            + W_CIRCUIT_HIST * d.circuit_score
            + W_CIRCUIT_CHAR * d.circuit_char_score
            + W_WEATHER    * d.weather_score
            + W_TEMP       * d.temp_score
        )

    # ── rank ──────────────────────────────────────────────────────────────────
    drivers.sort(key=lambda d: d.total_score, reverse=True)
    for i, d in enumerate(drivers):
        d.predicted_position = i + 1

    # ── explanation ───────────────────────────────────────────────────────────
    _generate_explanations(
        drivers,
        has_quali=False,
        circuit_info=circuit_info,
        weather=weather,
        wet_skills=ws_normalized,
        mode="quali",
    )

    return drivers


# ─── quali_predictions_to_data helper ────────────────────────────────────────

def quali_predictions_to_data(preds: list) -> dict:
    """Convert predict_qualifying output to the quali_data format expected by predict_race."""
    result: dict = {}
    for d in preds:
        abbr = d.abbreviation
        result[abbr] = {
            "grid_pos": d.predicted_position,
            "compound": "SOFT",
            "gap_to_pole_pct": max(0.0, (d.predicted_position - 1) * 0.12),
            "headshot_url": d.headshot_url,
            "full_name": d.full_name,
            "team": d.team,
        }
    return result


# ─── explanation generator ───────────────────────────────────────────────────

def _generate_explanations(
    drivers: list,
    has_quali: bool,
    circuit_info: dict,
    weather: dict,
    wet_skills: dict,
    mode: str = "race",
) -> None:
    air_temp_max = float(weather.get("air_temp_max", 25.0))
    rain_prob = float(weather.get("rain_probability", 10.0))
    altitude_m = float(circuit_info.get("altitude_m", 50))
    pu_sensitivity = float(circuit_info.get("pu_sensitivity", 3))
    tire_deg = int(circuit_info.get("tire_deg", 3))
    circuit_type = str(circuit_info.get("type", "permanent"))

    for d in drivers:
        parts: list = []

        if mode == "quali":
            # ── Qualifying-specific explanations ──────────────────────────────

            # 1. Team/car pace in recent sessions
            team_pts = d.team_form_avg_pts
            if team_pts >= 30:
                parts.append(f"チームの直近平均ポイントは{team_pts:.1f}ptと圧倒的な戦闘力を誇り、予選ペースでも最前線")
            elif team_pts >= 20:
                parts.append(f"チームの直近平均{team_pts:.1f}ptと高い車両性能を維持しており、シングルラップペースに期待")
            elif team_pts >= 10:
                parts.append(f"チームの直近平均{team_pts:.1f}ptと中上位の車両戦闘力を持つ")
            elif team_pts >= 3:
                parts.append(f"チームの直近平均{team_pts:.1f}ptとやや厳しい車両ペース")
            else:
                parts.append("チームの最近の予選ペースは苦しい状況")

            # 2. Driver single-lap reputation
            drv_pts = d.driver_form_avg_pts
            if drv_pts >= 15:
                parts.append(f"ドライバー自身も直近5戦平均{drv_pts:.1f}ptと絶好調で、シングルラップのアタックに強みがある")
            elif drv_pts >= 8:
                parts.append(f"ドライバーの直近平均{drv_pts:.1f}ptと安定した予選パフォーマンス")
            elif drv_pts >= 3:
                parts.append(f"ドライバーの直近平均{drv_pts:.1f}ptとやや波のある状態")
            else:
                parts.append("直近のドライバーポイントが少なく、予選アタックに課題あり")

            # 3. Circuit characteristics (PU sensitivity, altitude, circuit type)
            if altitude_m > 800:
                alt_jp = f"標高{altitude_m:.0f}mの高地でエンジン出力が制限され"
                pu_jp = {1: "極めて低い", 2: "低い", 3: "中程度", 4: "高い", 5: "非常に高い"}.get(
                    int(pu_sensitivity), "中程度"
                )
                parts.append(f"{alt_jp}、PU依存度{pu_jp}のサーキット特性が予選タイムに影響する")
            elif pu_sensitivity >= 4:
                pu_jp = {4: "高い", 5: "非常に高い"}.get(int(pu_sensitivity), "高い")
                parts.append(f"PU依存度が{pu_jp}サーキットで、エンジンパワーが予選タイムに直結する")
            elif pu_sensitivity <= 2:
                parts.append("低速テクニカルコースで空力・メカニカルグリップがシングルラップタイムを左右する")

            if circuit_type == "street":
                parts.append("市街地サーキットは壁への距離が極めて近く、アタックラップのリスクが高い")

            # 4. Circuit history
            if d.circuit_avg_pos is not None:
                avg_p = d.circuit_avg_pos
                if avg_p <= 3:
                    parts.append(f"このサーキットでの過去平均{avg_p:.1f}位と非常に相性が良く、予選でも上位が期待できる")
                elif avg_p <= 6:
                    parts.append(f"過去平均{avg_p:.1f}位と相性の良いサーキット")
                elif avg_p <= 10:
                    parts.append(f"過去平均{avg_p:.1f}位と中程度の相性")
                else:
                    parts.append(f"過去平均{avg_p:.1f}位と苦手意識のあるサーキット傾向")
            else:
                parts.append("このサーキットでの過去データがなく相性は未知数")

            # 5. Weather for wet qualifying (no tire strategy language)
            if rain_prob >= 60:
                wet_norm = wet_skills.get(d.abbreviation, 0.5)
                if wet_norm >= 0.7:
                    parts.append(f"ウェット予選（降水確率{rain_prob:.0f}%）はこのドライバーの得意とするシチュエーション")
                elif wet_norm <= 0.35:
                    parts.append(f"ウェット予選（降水確率{rain_prob:.0f}%）はやや苦手で、アタックに影響する可能性")
                else:
                    parts.append(f"雨の可能性が高く（{rain_prob:.0f}%）、ウェットコンディションでの予選となる可能性")
            elif rain_prob >= 20:
                parts.append(f"雨の可能性（{rain_prob:.0f}%）があり、予選タイミングで天候が変化するリスクがある")
            else:
                parts.append(f"ドライコンディション見込み（降水確率{rain_prob:.0f}%）で、純粋な車速勝負となる")

        else:
            # ── Race-specific explanations ────────────────────────────────────

            # 1. Qualifying position
            if has_quali and d.grid_position:
                pos = d.grid_position
                if pos == 1:
                    parts.append("ポールポジションからのスタートで最大のアドバンテージを持つ")
                elif pos <= 3:
                    parts.append(f"予選{pos}番手という最前列近くからのスタート")
                elif pos <= 6:
                    parts.append(f"予選{pos}番手から好位置スタート")
                elif pos <= 10:
                    parts.append(f"予選{pos}番手と中団前方からのスタート")
                else:
                    parts.append(f"予選{pos}番手と後方からの追い上げが必要")

            # 2. Car pace (gap to pole)
            if has_quali and d.gap_to_pole_pct > 0:
                if d.gap_to_pole_pct < 0.15:
                    parts.append(f"ポールからのギャップは{d.gap_to_pole_pct:.2f}%と非常に僅差で高い車両ペースを示す")
                elif d.gap_to_pole_pct < 0.5:
                    parts.append(f"ポールとの差は{d.gap_to_pole_pct:.2f}%と競争力あり")
                else:
                    parts.append(f"ポールとの差{d.gap_to_pole_pct:.2f}%は比較的大きく車両ペースに課題")

            # 3. Tire compound vs circuit deg
            cmp = d.compound.upper()
            cmp_jp = {"SOFT": "ソフト", "MEDIUM": "ミディアム", "HARD": "ハード",
                      "INTERMEDIATE": "インター", "WET": "ウェット"}.get(cmp, cmp)
            effective_deg = min(5, tire_deg + 1) if air_temp_max > 32 else tire_deg
            deg_jp = {1: "非常に低い", 2: "低い", 3: "中程度", 4: "高い", 5: "非常に高い"}.get(effective_deg, "中程度")
            if cmp in ("SOFT", "MEDIUM", "HARD"):
                if effective_deg >= 4 and cmp == "SOFT":
                    parts.append(
                        f"{cmp_jp}タイヤは当サーキットのタイヤ摩耗（{deg_jp}）に対して不利で、レース中盤での劣化が懸念される"
                    )
                elif effective_deg >= 4 and cmp in ("MEDIUM", "HARD"):
                    parts.append(
                        f"{cmp_jp}タイヤは摩耗が激しい当サーキット（{deg_jp}）に適しており、レースペースで有利"
                    )
                elif effective_deg <= 2 and cmp == "SOFT":
                    parts.append(
                        f"摩耗の少ない当サーキット（{deg_jp}）でのソフトタイヤは速さをそのまま活かせる"
                    )
                else:
                    parts.append(f"予選タイヤは{cmp_jp}でサーキット特性（摩耗：{deg_jp}）に対してニュートラル")
            else:
                parts.append("タイヤコンパウンド情報なし")

            # 4. Weather condition + wet skill
            if rain_prob >= 60:
                rain_note = "雨の可能性が非常に高く"
                wet_norm = wet_skills.get(d.abbreviation, 0.5)
                if wet_norm >= 0.7:
                    parts.append(f"{rain_note}、このドライバーはウェットコンディションが得意で有利に働く")
                elif wet_norm <= 0.35:
                    parts.append(f"{rain_note}、このドライバーはウェットコンディションをやや苦手としており不利になり得る")
                else:
                    parts.append(f"{rain_note}、ウェットスキルは平均的")
            elif rain_prob >= 20:
                parts.append(f"雨の可能性（{rain_prob:.0f}%）があり、天候次第で展開が変わる")
            else:
                parts.append(f"ドライコンディションが見込まれ（降水確率{rain_prob:.0f}%）、タイヤ戦略が鍵となる")

            # 5. Circuit characteristics — altitude & PU sensitivity
            if altitude_m > 800:
                alt_jp = f"標高{altitude_m:.0f}mの高地でパワーユニットの出力が制限される"
                pu_jp = {1: "極めて低い", 2: "低い", 3: "中程度", 4: "高い", 5: "非常に高い"}.get(
                    int(pu_sensitivity), "中程度"
                )
                parts.append(f"{alt_jp}。PU依存度は{pu_jp}コースで{d.team}の特性が大きく影響")
            elif pu_sensitivity >= 4:
                pu_jp = {4: "高い", 5: "非常に高い"}.get(int(pu_sensitivity), "高い")
                parts.append(f"PU依存度が{pu_jp}サーキットで、エンジン性能が順位に直結する傾向")
            elif pu_sensitivity <= 2:
                parts.append("低速テクニカルコースでPU依存度が低く、ドライバースキルとメカニカルグリップが重要")

            if circuit_type == "street":
                parts.append("市街地サーキットはウォールとの距離が近く、ミスが致命的になりやすい")

            # 6. Circuit history
            if d.circuit_avg_pos is not None:
                avg_p = d.circuit_avg_pos
                if avg_p <= 3:
                    parts.append(f"過去の同サーキットでの平均フィニッシュ{avg_p:.1f}位と極めて得意なコース")
                elif avg_p <= 6:
                    parts.append(f"過去平均{avg_p:.1f}位と相性の良いサーキット")
                elif avg_p <= 10:
                    parts.append(f"過去平均{avg_p:.1f}位と中程度の相性")
                else:
                    parts.append(f"過去平均{avg_p:.1f}位とやや苦手なサーキット傾向")
            else:
                parts.append("このサーキットでの過去データがなく相性は未知数（新人または初参戦の可能性）")

            # 7. Tire strategy
            strat_score = d.tire_strategy_score
            if strat_score >= 0.7:
                parts.append("このサーキットで長いスティントを維持できる傾向があり、タイヤ管理に優れる")
            elif strat_score <= 0.35:
                parts.append("このサーキットでタイヤの消耗が速い傾向があり、ピット戦略が鍵となる")

            # 8. Driver form
            pts = d.driver_form_avg_pts
            if pts >= 15:
                parts.append(f"直近5戦の平均{pts:.1f}ptと圧倒的な好調を維持")
            elif pts >= 10:
                parts.append(f"直近5戦平均{pts:.1f}ptと高水準のパフォーマンス")
            elif pts >= 5:
                parts.append(f"直近5戦平均{pts:.1f}ptと安定した成績")
            elif pts >= 1:
                parts.append(f"直近5戦平均{pts:.1f}ptとやや低調")
            else:
                parts.append("直近のポイント獲得が少ない状況")

        d.explanation = "。".join(parts) + "。"
