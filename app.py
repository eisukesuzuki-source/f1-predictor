from __future__ import annotations

import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from circuit_data import get_circuit_info
from data_loader import (
    get_circuit_history,
    get_circuit_history_detail,
    get_driver_metadata,
    get_prev_year_start_compounds,
    get_quali_data,
    get_race_tire_compounds,
    get_recent_form,
    get_recent_form_detail,
    get_tire_strategy_data,
    get_weather_for_race,
    get_wet_skill,
    load_schedule,
    load_session_results,
)
from predictor import DriverScore, predict_race, predict_qualifying, quali_predictions_to_data

# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 レース順位予測",
    page_icon="F1",
    layout="wide",
)

# ─── team colors ─────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "RB": "#6692FF",
    "Racing Bulls": "#6692FF",
    "Haas F1 Team": "#B6BABD",
    "Haas": "#B6BABD",
    "Kick Sauber": "#52E252",
    "Sauber": "#52E252",
    "AlphaTauri": "#5F8FAA",
    "Alfa Romeo": "#B12039",
    "Renault": "#FFF500",
}
DEFAULT_TEAM_COLOR = "#555566"

# ─── CSS injection ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stApp"] { background-color: #0d1117; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { background-color: #161b22; border-bottom: 1px solid #30363d; }
    .stTabs [data-baseweb="tab"] { background-color: #161b22; color: #e6edf3; border: none; padding: 8px 20px; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #161b22; color: #f0c040; border-bottom: 3px solid #f0c040; }
    .stTabs [data-baseweb="tab-panel"] { background-color: #0d1117; }
    .stExpander { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; }
    h1, h2, h3, h4, h5 { color: #e6edf3 !important; }
    p, li, label { color: #e6edf3; }
    .compound-badge-soft { background:#cc0000;color:white;padding:2px 7px;border-radius:4px;font-size:0.78rem;font-weight:bold;display:inline-block; }
    .compound-badge-medium { background:#ffe400;color:black;padding:2px 7px;border-radius:4px;font-size:0.78rem;font-weight:bold;display:inline-block; }
    .compound-badge-hard { background:#e8e8e8;color:black;padding:2px 7px;border-radius:4px;font-size:0.78rem;font-weight:bold;display:inline-block; }
    .compound-badge-inter { background:#39b54a;color:white;padding:2px 7px;border-radius:4px;font-size:0.78rem;font-weight:bold;display:inline-block; }
    .compound-badge-wet { background:#0066ff;color:white;padding:2px 7px;border-radius:4px;font-size:0.78rem;font-weight:bold;display:inline-block; }
    .compound-badge-unknown { background:#333;color:#e6edf3;padding:2px 7px;border-radius:4px;font-size:0.78rem;font-weight:bold;display:inline-block; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── helpers ──────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font={"color": "#e6edf3", "size": 11},
)


def _compound_badge_html(compound: str) -> str:
    c = compound.upper()
    cls = {"SOFT": "compound-badge-soft", "MEDIUM": "compound-badge-medium",
           "HARD": "compound-badge-hard", "INTERMEDIATE": "compound-badge-inter",
           "INTER": "compound-badge-inter", "WET": "compound-badge-wet"}.get(c, "compound-badge-unknown")
    lbl = {"SOFT": "SOFT", "MEDIUM": "MED", "HARD": "HARD",
           "INTERMEDIATE": "INTER", "INTER": "INTER", "WET": "WET"}.get(c, c)
    return (
        f'<span class="{cls}" style="display:inline-block;width:44px;'
        f'text-align:center;box-sizing:border-box;white-space:nowrap">{lbl}</span>'
    )


def _stars(rating: int, max_stars: int = 5) -> str:
    return (
        f'<span style="color:#f0c040">{"★" * rating}</span>'
        f'<span style="color:#444">{"☆" * (max_stars - rating)}</span>'
    )


def _rain_label(prob: float) -> str:
    return "RAIN" if prob >= 60 else ("MIXED" if prob >= 20 else "DRY")


def _race_date_str(event) -> str:
    for col in ["EventDate", "Session5Date", "Session5DateUtc", "date"]:
        val = event.get(col)
        if val is not None:
            try:
                if isinstance(val, (datetime.date, datetime.datetime)):
                    return val.strftime("%Y-%m-%d")
                if isinstance(val, str):
                    return val[:10]
                return pd.Timestamp(val).strftime("%Y-%m-%d")
            except Exception:
                continue
    return datetime.date.today().strftime("%Y-%m-%d")


def _headshot_html(d: DriverScore, tc: str, size: int = 42) -> str:
    if d.headshot_url:
        return (
            f'<img src="{d.headshot_url}" style="width:{size}px;height:{size}px;border-radius:50%;'
            f'border:2px solid {tc};object-fit:cover;margin:0 12px;flex-shrink:0"'
            f' onerror="this.style.display=\'none\'">'
        )
    abbr = d.abbreviation[:3].upper()
    return (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:{tc}33;'
        f'border:2px solid {tc};display:flex;align-items:center;justify-content:center;'
        f'margin:0 12px;font-size:0.7rem;color:{tc};font-weight:bold;flex-shrink:0">{abbr}</div>'
    )


def _actual_badge_html(predicted_pos: int, actual_pos: int) -> str:
    diff = abs(predicted_pos - actual_pos)
    color = "#3fb950" if diff == 0 else ("#f0c040" if diff <= 3 else "#ff6b6b")
    return f'<div style="min-width:60px;text-align:center;font-size:0.85rem;color:{color};font-weight:bold">P{actual_pos}</div>'


def _score_color(score: float) -> str:
    if score >= 0.7:
        return "#3fb950"
    elif score >= 0.4:
        return "#f0c040"
    return "#ff6b6b"


# ─── scoreboard ───────────────────────────────────────────────────────────────

def render_scoreboard(
    drivers: list,
    actual_pos_map: dict,
    mode: str,
    team_colors: dict,
    actual_tire_compounds: Optional[dict] = None,
) -> None:
    header = (
        '<div style="display:flex;align-items:center;background:#010409;border-bottom:2px solid #e8002d;'
        'padding:8px 14px;border-radius:4px 4px 0 0;margin-bottom:4px">'
        '<div style="min-width:52px;font-size:0.72rem;color:#e6edf3;font-weight:bold">POS</div>'
        '<div style="width:42px;margin:0 12px"></div>'
        '<div style="flex:1;font-size:0.72rem;color:#e6edf3;font-weight:bold">DRIVER</div>'
        '<div style="min-width:180px;margin:0 14px;font-size:0.72rem;color:#e6edf3;font-weight:bold">SCORE</div>'
        '<div style="min-width:56px;text-align:center;font-size:0.72rem;color:#e6edf3;font-weight:bold">TYRE</div>'
        '<div style="width:56px;min-width:56px;max-width:56px;text-align:center;font-size:0.72rem;color:#8b949e;font-weight:bold;flex-shrink:0">ACTUAL TYRE</div>'
        '<div style="min-width:72px;text-align:center;font-size:0.72rem;color:#e6edf3;font-weight:bold">GRID / △</div>'
        '<div style="min-width:72px;text-align:center;font-size:0.72rem;color:#8b949e;font-weight:bold">ACTUAL / △</div>'
        '</div>'
    )
    st.markdown(header, unsafe_allow_html=True)

    parts = ['<div style="margin-bottom:12px">']
    for d in drivers:
        tc = team_colors.get(d.team, DEFAULT_TEAM_COLOR)
        score_pct = min(100.0, d.total_score * 100.0)
        cmp = _compound_badge_html(d.compound)
        hs = _headshot_html(d, tc)
        score_val = d.total_score * 100.0

        # ── ACTUAL TYRE cell ────────────────────────────────────────────────
        act_tire_list = (actual_tire_compounds or {}).get(d.abbreviation, [])
        if act_tire_list:
            inner = "".join(
                f'<div style="line-height:1.6">{_compound_badge_html(c)}</div>'
                for c in act_tire_list
            )
            actual_tyre_cell = (
                f'<div style="width:56px;min-width:56px;max-width:56px;'
                f'text-align:center;flex-shrink:0">{inner}</div>'
            )
        else:
            actual_tyre_cell = (
                '<div style="width:56px;min-width:56px;max-width:56px;'
                'text-align:center;flex-shrink:0;color:#8b949e;font-size:0.8rem">—</div>'
            )

        # ── GRID / △ cell ───────────────────────────────────────────────────
        grid_pos = d.grid_position if d.grid_position else None
        if grid_pos:
            gd = grid_pos - d.predicted_position
            if gd > 0:
                gd_html = f'<span style="color:#3fb950;font-size:0.72rem;font-weight:700">▲{gd}</span>'
            elif gd < 0:
                gd_html = f'<span style="color:#ff6b6b;font-size:0.72rem;font-weight:700">▼{abs(gd)}</span>'
            else:
                gd_html = '<span style="color:#8b949e;font-size:0.72rem">—</span>'
            grid_cell = (
                f'<div style="min-width:72px;text-align:center">'
                f'<div style="font-size:0.88rem;font-weight:700;color:#e6edf3">P{grid_pos}</div>'
                f'<div style="margin-top:2px">{gd_html}</div>'
                f'</div>'
            )
        else:
            grid_cell = '<div style="min-width:72px;text-align:center;color:#8b949e;font-size:0.8rem">—</div>'

        # ── ACTUAL / △ cell ─────────────────────────────────────────────────
        actual_pos = actual_pos_map.get(d.abbreviation) if actual_pos_map else None
        if actual_pos:
            ad = d.predicted_position - actual_pos   # positive = actual better than predicted
            if ad > 0:
                ad_html = f'<span style="color:#3fb950;font-size:0.72rem;font-weight:700">▲{ad}</span>'
            elif ad < 0:
                ad_html = f'<span style="color:#ff6b6b;font-size:0.72rem;font-weight:700">▼{abs(ad)}</span>'
            else:
                ad_html = '<span style="color:#f0c040;font-size:0.72rem;font-weight:700">=</span>'
            actual_cell = (
                f'<div style="min-width:72px;text-align:center">'
                f'<div style="font-size:0.88rem;font-weight:700;color:#e6edf3">P{actual_pos}</div>'
                f'<div style="margin-top:2px">{ad_html}</div>'
                f'</div>'
            )
        else:
            actual_cell = '<div style="min-width:72px;text-align:center;color:#8b949e;font-size:0.8rem">—</div>'

        row = (
            f'<div style="display:flex;align-items:center;padding:10px 14px;margin:3px 0;'
            f'background:#1c2128;border-radius:4px;border-left:5px solid {tc}">'
            f'<div style="min-width:52px;font-size:1.3rem;font-weight:900;color:#f0c040">P{d.predicted_position}</div>'
            f'{hs}'
            f'<div style="flex:1;min-width:0">'
            f'<div style="font-size:0.95rem;font-weight:700;color:#e6edf3;text-transform:uppercase">{d.full_name.upper()}</div>'
            f'<div style="font-size:0.75rem;color:{tc};margin-top:2px">{d.team}</div>'
            f'</div>'
            f'<div style="min-width:180px;margin:0 14px">'
            f'<div style="height:5px;background:rgba(255,255,255,0.12);border-radius:2px">'
            f'<div style="height:5px;width:{score_pct:.1f}%;background:{tc};border-radius:2px"></div>'
            f'</div>'
            f'<div style="font-size:0.72rem;color:#e6edf3;margin-top:3px">{score_val:.1f} / 100</div>'
            f'</div>'
            f'<div style="min-width:56px;text-align:center">{cmp}</div>'
            f'{actual_tyre_cell}'
            f'{grid_cell}'
            f'{actual_cell}'
            f'</div>'
        )
        parts.append(row)
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ドライバー詳細")
    for d in drivers[:10]:
        tc = team_colors.get(d.team, DEFAULT_TEAM_COLOR)
        with st.expander(f"P{d.predicted_position}  {d.full_name}  —  {d.team}"):
            col_img, col_text, col_metrics = st.columns([1, 3, 2])
            with col_img:
                if d.headshot_url:
                    st.image(d.headshot_url, width=100)
            with col_text:
                compound_label = {"SOFT": "SOFT", "MEDIUM": "MEDIUM", "HARD": "HARD",
                                  "INTERMEDIATE": "INTER", "WET": "WET"}.get(d.compound, d.compound)
                st.markdown(f"**タイヤ:** {compound_label}")
                if hasattr(d, "gap_to_pole_pct") and d.gap_to_pole_pct is not None and d.gap_to_pole_pct >= 0:
                    st.markdown("**ポールタイム差:** POLE" if d.gap_to_pole_pct == 0 else f"**ポールタイム差:** +{d.gap_to_pole_pct:.3f}%")
                st.markdown(f"**予測根拠:**\n{d.explanation}")
            with col_metrics:
                if d.grid_position:
                    st.metric("予選順位", f"P{d.grid_position}")
                st.metric("ドライバーフォーム", f"{d.driver_form_avg_pts:.1f} 位/戦")
                st.metric("チーム直近平均", f"{d.team_form_avg_pts:.1f} pt/戦")
                if d.circuit_avg_pos is not None:
                    st.metric("サーキット過去平均", f"{d.circuit_avg_pos:.1f}位")
                actual_pos_for_driver = actual_pos_map.get(d.abbreviation) if actual_pos_map else None
                if actual_pos_for_driver:
                    diff = abs(d.predicted_position - actual_pos_for_driver)
                    st.metric("実際の順位", f"P{actual_pos_for_driver}",
                              delta=f"誤差{diff}位" if diff > 0 else "完全一致",
                              delta_color="normal" if diff <= 2 else "inverse")


# ─── factor analysis chart ────────────────────────────────────────────────────

def render_factor_analysis(predictions: list, has_quali: bool, quali_data_state: dict) -> None:
    top_drivers = predictions[:min(10, len(predictions))]
    labels = [f"P{d.predicted_position} {d.abbreviation}" for d in top_drivers]
    fig = go.Figure()

    if has_quali:
        traces = [
            ("予選順位 (30%)",        [d.grid_score          * 0.30 * 100 for d in top_drivers], "#E8002D"),
            ("車両ペース (12%)",       [d.car_pace_score      * 0.12 * 100 for d in top_drivers], "#FF8000"),
            ("ドライバーフォーム (12%)",[d.driver_form_score   * 0.12 * 100 for d in top_drivers], "#1E90FF"),
        ]
    else:
        traces = [
            ("ドライバーフォーム (21%)",[d.driver_form_score   * 0.21 * 100 for d in top_drivers], "#1E90FF"),
            ("チーム戦闘力 (19%)",     [d.team_form_score     * 0.19 * 100 for d in top_drivers], "#6699FF"),
        ]

    traces += [
        (f"サーキット相性 ({8 if has_quali else 11}%)",  [d.circuit_score         * (0.08 if has_quali else 0.11) * 100 for d in top_drivers], "#32CD32"),
        (f"タイヤ適性 (7%)",                              [d.tire_score            * 0.07 * 100                        for d in top_drivers], "#FFD700"),
        (f"タイヤ戦略 ({9 if has_quali else 10}%)",      [d.tire_strategy_score   * (0.09 if has_quali else 0.10) * 100 for d in top_drivers], "#FFA500"),
        (f"天気/ウェット ({9 if has_quali else 12}%)",   [d.weather_score         * (0.09 if has_quali else 0.12) * 100 for d in top_drivers], "#00BFFF"),
        (f"サーキット特性 ({7 if has_quali else 10}%)",  [d.circuit_char_score    * (0.07 if has_quali else 0.10) * 100 for d in top_drivers], "#9B59B6"),
        (f"気温 ({6 if has_quali else 10}%)",             [d.temp_score            * (0.06 if has_quali else 0.10) * 100 for d in top_drivers], "#FF69B4"),
    ]

    for name, x_vals, color in traces:
        fig.add_trace(go.Bar(name=name, y=labels, x=x_vals, orientation="h",
                             marker_color=color, hovertemplate=f"%{{y}}: %{{x:.1f}}<extra>{name}</extra>"))

    fig.update_layout(**CHART_LAYOUT, barmode="stack", height=460,
                      xaxis_title="スコア（合計100点満点）",
                      yaxis={"categoryorder": "total ascending"},
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      margin=dict(l=20, r=20, t=60, b=20))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)", color="#e6edf3")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", color="#e6edf3")
    st.plotly_chart(fig, use_container_width=True)

    if has_quali and quali_data_state:
        st.markdown("#### 予選データ")
        rows_html = (
            '<table style="width:100%;border-collapse:collapse;color:#e6edf3;font-size:0.85rem">'
            '<tr style="background:#010409;border-bottom:2px solid #e8002d">'
            '<th style="padding:6px 10px;text-align:left">グリッド</th>'
            '<th style="padding:6px 10px;text-align:left">ドライバー</th>'
            '<th style="padding:6px 10px;text-align:left">チーム</th>'
            '<th style="padding:6px 10px;text-align:center">タイヤ</th>'
            '<th style="padding:6px 10px;text-align:right">ポールとの差</th></tr>'
        )
        for abbr, qd in sorted(quali_data_state.items(), key=lambda x: x[1].get("grid_pos", 99)):
            badge = _compound_badge_html(str(qd.get("compound", "UNKNOWN")))
            gap = float(qd.get("gap_to_pole_pct", 0.0))
            gap_str = "POLE" if gap == 0.0 else f"+{gap:.3f}%"
            team = qd.get("team", "")
            tc = TEAM_COLORS.get(team, DEFAULT_TEAM_COLOR)
            rows_html += (
                f'<tr style="background:#1c2128;border-left:4px solid {tc}">'
                f'<td style="padding:6px 10px">P{qd.get("grid_pos", "-")}</td>'
                f'<td style="padding:6px 10px;font-weight:bold">{qd.get("full_name", abbr)}</td>'
                f'<td style="padding:6px 10px;color:{tc}">{team}</td>'
                f'<td style="padding:6px 10px;text-align:center">{badge}</td>'
                f'<td style="padding:6px 10px;text-align:right">{gap_str}</td></tr>'
            )
        rows_html += "</table>"
        st.markdown(rows_html, unsafe_allow_html=True)


# ─── driver analysis ─────────────────────────────────────────────────────────

def render_driver_analysis(
    predictions: list,
    form_detail: dict,
    circuit_detail: dict,
    circuit_info: dict,
    weather: dict,
    wet_skills: dict,
    tire_strategy: dict,
    has_quali: bool,
    team_colors: dict,
) -> None:
    """Per-driver factor trends, score breakdown, and calculation methods."""

    # ── Selector ──────────────────────────────────────────────────────────────
    options = [f"P{d.predicted_position}  {d.full_name}  ({d.team})" for d in predictions]
    sel = st.selectbox("分析するドライバー", range(len(options)), format_func=lambda i: options[i])
    d = predictions[sel]
    tc = team_colors.get(d.team, DEFAULT_TEAM_COLOR)

    # ── Header card ───────────────────────────────────────────────────────────
    if d.headshot_url:
        hs = (f'<img src="{d.headshot_url}" style="width:64px;height:64px;border-radius:50%;'
              f'border:3px solid {tc};object-fit:cover;margin-right:16px;flex-shrink:0"'
              f' onerror="this.style.display=\'none\'">')
    else:
        hs = (f'<div style="width:64px;height:64px;border-radius:50%;background:{tc}33;'
              f'border:3px solid {tc};display:flex;align-items:center;justify-content:center;'
              f'margin-right:16px;font-size:0.85rem;color:{tc};font-weight:bold;flex-shrink:0">'
              f'{d.abbreviation}</div>')

    st.markdown(
        f'<div style="display:flex;align-items:center;background:#1c2128;border-left:5px solid {tc};'
        f'border-radius:6px;padding:16px 20px;margin-bottom:20px">'
        f'{hs}'
        f'<div style="flex:1">'
        f'<div style="font-size:1.3rem;font-weight:900;color:#e6edf3;text-transform:uppercase">{d.full_name}</div>'
        f'<div style="color:{tc};font-size:0.9rem;margin-top:2px">{d.team}</div>'
        f'</div>'
        f'<div style="text-align:center;margin-left:24px">'
        f'<div style="font-size:2rem;font-weight:900;color:#f0c040">P{d.predicted_position}</div>'
        f'<div style="font-size:0.72rem;color:#e6edf3">予測順位</div>'
        f'</div>'
        f'<div style="text-align:center;margin-left:32px">'
        f'<div style="font-size:1.6rem;font-weight:700;color:#e6edf3">{d.total_score * 100:.1f}</div>'
        f'<div style="font-size:0.72rem;color:#e6edf3">スコア / 100</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Factor scores chart + formula table ───────────────────────────────────
    col_chart, col_table = st.columns([5, 6])

    with col_chart:
        st.markdown("##### 因子スコア一覧")
        if has_quali:
            factor_list = [
                ("気温",             d.temp_score,           0.06),
                ("サーキット特性",   d.circuit_char_score,   0.07),
                ("天気/ウェット",    d.weather_score,        0.09),
                ("タイヤ適性",       d.tire_score,           0.07),
                ("タイヤ戦略",       d.tire_strategy_score,  0.09),
                ("サーキット相性",   d.circuit_score,        0.08),
                ("ドライバーフォーム", d.driver_form_score,  0.12),
                ("車両ペース",       d.car_pace_score,       0.12),
                ("予選順位",         d.grid_score,           0.30),
            ]
        else:
            factor_list = [
                ("気温",             d.temp_score,           0.10),
                ("サーキット特性",   d.circuit_char_score,   0.10),
                ("天気/ウェット",    d.weather_score,        0.12),
                ("タイヤ適性",       d.tire_score,           0.07),
                ("タイヤ戦略",       d.tire_strategy_score,  0.10),
                ("サーキット相性",   d.circuit_score,        0.11),
                ("チームフォーム",   d.team_form_score,      0.19),
                ("ドライバーフォーム", d.driver_form_score,  0.21),
            ]

        bar_labels = [f"{name}  ({w * 100:.0f}%)" for name, _, w in factor_list]
        bar_scores = [s * 100 for _, s, _ in factor_list]
        bar_colors = [_score_color(s) for _, s, _ in factor_list]
        bar_text = [f"{s * 100:.0f}" for _, s, _ in factor_list]

        fig_scores = go.Figure(go.Bar(
            y=bar_labels, x=bar_scores, orientation="h",
            marker_color=bar_colors,
            text=bar_text, textposition="auto", textfont={"color": "#000000", "size": 11},
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        ))
        fig_scores.update_layout(
            **CHART_LAYOUT, height=280, showlegend=False,
            xaxis={"range": [0, 100], "gridcolor": "rgba(255,255,255,0.1)", "color": "#e6edf3"},
            yaxis={"gridcolor": "rgba(255,255,255,0.05)", "color": "#e6edf3"},
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_scores, use_container_width=True)

    with col_table:
        st.markdown("##### 各因子の計算方法")

        alt_m = float(circuit_info.get("altitude_m", 50))
        tire_deg = int(circuit_info.get("tire_deg", 3))
        pu_sens = int(circuit_info.get("pu_sensitivity", 3))

        circ_avg_str = f"{d.circuit_avg_pos:.1f}位" if d.circuit_avg_pos is not None else "データなし"

        strat_data = tire_strategy.get(d.abbreviation, {})
        avg_stops_v = strat_data.get("avg_stops")
        mgmt_idx_v  = strat_data.get("tire_mgmt_index")
        typical_comps = strat_data.get("typical_compounds", [])
        strat_input = (
            f"平均{avg_stops_v:.1f}回停車 / スティント比率{mgmt_idx_v:.2f}"
            if avg_stops_v is not None else "過去データなし"
        )
        comps_str = " → ".join(typical_comps) if typical_comps else "不明"

        tire_input = f"{d.compound}（{d.compound_reason}）/ 摩耗{tire_deg}/5"
        if has_quali:
            rows = [
                ("予選順位",          f"{d.grid_position or '?'}番手",              "グリッドを1位=1.0, 20位=0.0に線形変換し全体正規化",           d.grid_score),
                ("車両ペース",         f"ポール差 {d.gap_to_pole_pct:.3f}%",         "ポールタイム差の小ささを正規化 (差0%→スコア最大)",             d.car_pace_score),
                ("ドライバーフォーム", f"直近5戦平均 {d.driver_form_avg_pts:.1f}位", "全ドライバーの平均順位を相対正規化（順位が低い方が良い）", d.driver_form_score),
                ("サーキット相性",     circ_avg_str,                                  "過去3年の平均フィニッシュ位置を逆数正規化 (低位→高スコア)",   d.circuit_score),
                ("タイヤ適性",         tire_input,                                    "コンパウンド×摩耗度の適性テーブル参照 (SOFT×低摩耗→0.85)",   d.tire_score),
                ("タイヤ戦略",         strat_input,                                   "過去スティント長÷全体平均を全体正規化 (長い→タイヤ管理良好)", d.tire_strategy_score),
                ("天気/ウェット",      f"降水{d.rain_probability:.0f}%",              "雨確率<50%→0.5固定, 以上→ウェットスキルと加重平均",           d.weather_score),
                ("サーキット特性",     f"標高{alt_m:.0f}m / PU感度{pu_sens}/5",      "0.6×高度補正 + 0.4×(車両ペース×PU感度/5)",                  d.circuit_char_score),
                ("気温",               f"{d.air_temp:.0f}°C / {d.compound}",         "コンパウンド最適温度域との乖離をペナルティ化",                  d.temp_score),
            ]
        else:
            rows = [
                ("ドライバーフォーム", f"直近5戦平均 {d.driver_form_avg_pts:.1f}位", "全ドライバーの平均順位を相対正規化（順位が低い方が良い）", d.driver_form_score),
                ("チームフォーム",     f"チーム平均 {d.team_form_avg_pts:.1f}pt",    "全チーム間で正規化",                                            d.team_form_score),
                ("サーキット相性",     circ_avg_str,                                  "過去3年の平均フィニッシュ位置を逆数正規化",                     d.circuit_score),
                ("タイヤ適性",         tire_input,                                    "コンパウンド×摩耗度の適性テーブル参照",                         d.tire_score),
                ("タイヤ戦略",         strat_input,                                   "過去スティント長÷全体平均を全体正規化 (長い→タイヤ管理良好)", d.tire_strategy_score),
                ("天気/ウェット",      f"降水{d.rain_probability:.0f}%",              "雨確率<50%→0.5固定, 以上→ウェットスキルと加重平均",           d.weather_score),
                ("サーキット特性",     f"標高{alt_m:.0f}m / PU感度{pu_sens}/5",      "0.6×高度補正 + 0.4×(チーム戦闘力×PU感度/5)",                d.circuit_char_score),
                ("気温",               f"{d.air_temp:.0f}°C / {d.compound}",         "コンパウンド最適温度域との乖離をペナルティ化",                  d.temp_score),
            ]

        tbl = (
            '<table style="width:100%;border-collapse:collapse;font-size:0.78rem;color:#e6edf3">'
            '<tr style="background:#010409;border-bottom:2px solid #e8002d">'
            '<th style="padding:5px 8px;text-align:left">因子</th>'
            '<th style="padding:5px 8px;text-align:left">入力値</th>'
            '<th style="padding:5px 8px;text-align:left">計算方法</th>'
            '<th style="padding:5px 8px;text-align:center;min-width:48px">スコア</th>'
            '</tr>'
        )
        for name, input_val, formula, score in rows:
            sc = _score_color(score)
            tbl += (
                f'<tr style="background:#1c2128;border-bottom:1px solid #30363d">'
                f'<td style="padding:5px 8px;font-weight:bold;white-space:nowrap">{name}</td>'
                f'<td style="padding:5px 8px;color:#e6edf3">{input_val}</td>'
                f'<td style="padding:5px 8px;color:#e6edf3">{formula}</td>'
                f'<td style="padding:5px 8px;text-align:center;font-weight:bold;color:{sc}">{score:.2f}</td>'
                f'</tr>'
            )
        tbl += '</table>'
        st.markdown(tbl, unsafe_allow_html=True)

    st.markdown("---")

    # ── Trend charts ─────────────────────────────────────────────────────────
    drv_detail = form_detail.get(d.abbreviation, {})
    races = drv_detail.get("races", [])
    positions = drv_detail.get("positions", [])
    points = drv_detail.get("points", [])

    col1, col2 = st.columns(2)

    _AXIS_STYLE = {"gridcolor": "rgba(255,255,255,0.2)", "color": "#e6edf3",
                   "showline": True, "linecolor": "rgba(255,255,255,0.3)"}
    _CHART_MARGIN = dict(l=48, r=16, t=36, b=40)

    with col1:
        st.markdown("##### ドライバーフォーム推移（直近5戦）")
        if races and positions:
            avg_pos = sum(positions) / len(positions)
            fig_form = go.Figure()
            fig_form.add_trace(go.Scatter(
                x=races, y=positions,
                mode="lines+markers+text",
                line={"color": tc, "width": 2},
                marker=dict(color=tc, size=10, line=dict(color="#ffffff", width=1.5)),
                name="フィニッシュ順位",
                text=[f"P{int(p)}" for p in positions],
                textposition="top center",
                textfont=dict(size=12, color="#e6edf3"),
                hovertemplate="%{x}: P%{y}<extra></extra>",
            ))
            fig_form.add_trace(go.Scatter(
                x=races, y=[avg_pos] * len(races), mode="lines",
                line={"color": "#f0c040", "dash": "dash", "width": 2},
                name=f"平均 {avg_pos:.1f}位",
            ))
            fig_form.update_layout(
                **CHART_LAYOUT, height=300, showlegend=True,
                margin=_CHART_MARGIN,
                legend=dict(x=0, y=1.12, orientation="h", font=dict(size=11)),
                yaxis={**_AXIS_STYLE, "title": "順位", "autorange": "reversed",
                       "tickvals": list(range(1, 21)), "dtick": 1},
                xaxis={**_AXIS_STYLE, "tickfont": dict(size=11)},
            )
            st.plotly_chart(fig_form, use_container_width=True)
            st.markdown(
                f'<div style="background:#161b22;border-radius:4px;padding:8px 12px;'
                f'font-size:0.78rem;color:#e6edf3">'
                f'<b>計算式:</b> average({", ".join(str(int(p)) for p in positions)}) = {avg_pos:.1f}位 '
                f'→ 全ドライバー間で正規化（順位が低い方が良い）→ スコア {d.driver_form_score:.2f}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("フォームデータが見つかりませんでした")

    with col2:
        st.markdown("##### チームフォーム推移（直近5戦）")
        # Aggregate team points per race from form_detail
        team_race_pts: dict = {}
        for abbr, fd in form_detail.items():
            if fd.get("team") == d.team:
                for race, pts_val in zip(fd.get("races", []), fd.get("points", [])):
                    team_race_pts.setdefault(race, []).append(pts_val)

        if team_race_pts:
            t_races = sorted(team_race_pts.keys(), key=lambda r: list(team_race_pts.keys()).index(r))
            t_pts = [sum(team_race_pts[r]) for r in t_races]
            t_avg = sum(t_pts) / len(t_pts)
            fig_team = go.Figure()
            fig_team.add_trace(go.Bar(
                x=t_races, y=t_pts,
                marker=dict(color=tc, opacity=0.85, line=dict(color="rgba(255,255,255,0.3)", width=1)),
                name="チーム合計ポイント",
                text=[f"{int(p)}pt" for p in t_pts],
                textposition="outside",
                textfont=dict(size=12, color="#e6edf3"),
                hovertemplate="%{x}: %{y}pt<extra></extra>",
            ))
            fig_team.add_trace(go.Scatter(
                x=t_races, y=[t_avg] * len(t_races), mode="lines",
                line={"color": "#f0c040", "dash": "dash", "width": 2},
                name=f"平均 {t_avg:.1f}pt",
            ))
            fig_team.update_layout(
                **CHART_LAYOUT, height=300, showlegend=True,
                margin=_CHART_MARGIN,
                legend=dict(x=0, y=1.12, orientation="h", font=dict(size=11)),
                yaxis={**_AXIS_STYLE, "title": "合計ポイント"},
                xaxis={**_AXIS_STYLE, "tickfont": dict(size=11)},
                bargap=0.35,
            )
            st.plotly_chart(fig_team, use_container_width=True)
            st.markdown(
                f'<div style="background:#161b22;border-radius:4px;padding:8px 12px;'
                f'font-size:0.78rem;color:#e6edf3">'
                f'<b>計算式:</b> チーム直近5戦平均 {d.team_form_avg_pts:.1f}pt '
                f'→ 全チーム間で正規化 → スコア {d.team_form_score:.2f}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("チームフォームデータが見つかりませんでした")

    # ── Circuit history ───────────────────────────────────────────────────────
    st.markdown("##### このサーキットでの過去成績（過去3年）")
    circ_hist = circuit_detail.get(d.abbreviation, {})
    years = circ_hist.get("years", [])
    positions = circ_hist.get("positions", [])

    if years:
        avg_pos = sum(positions) / len(positions)
        fig_circ = go.Figure()
        fig_circ.add_trace(go.Scatter(
            x=[str(y) for y in years], y=positions,
            mode="lines+markers+text",
            line={"color": tc, "width": 2},
            marker=dict(color=tc, size=10, line=dict(color="#ffffff", width=1.5)),
            name="フィニッシュ位置",
            text=[f"P{int(p)}" for p in positions],
            textposition="top center",
            textfont=dict(size=13, color="#e6edf3"),
            hovertemplate="%{x}年: P%{y}<extra></extra>",
        ))
        fig_circ.add_trace(go.Scatter(
            x=[str(y) for y in years], y=[avg_pos] * len(years), mode="lines",
            line={"color": "#f0c040", "dash": "dash", "width": 2},
            name=f"平均 {avg_pos:.1f}位",
        ))
        fig_circ.update_layout(
            **CHART_LAYOUT, height=300, showlegend=True,
            margin=dict(l=48, r=16, t=36, b=40),
            legend=dict(x=0, y=1.12, orientation="h", font=dict(size=11)),
            yaxis={"title": "フィニッシュ位置", "autorange": "reversed",
                   "gridcolor": "rgba(255,255,255,0.2)", "color": "#e6edf3",
                   "showline": True, "linecolor": "rgba(255,255,255,0.3)",
                   "tickvals": list(range(1, 21)), "dtick": 1},
            xaxis={"color": "#e6edf3", "showline": True,
                   "linecolor": "rgba(255,255,255,0.3)", "tickfont": dict(size=12)},
        )
        st.plotly_chart(fig_circ, use_container_width=True)
        pos_strs = ", ".join(f"P{int(p)}" for p in positions)
        st.markdown(
            f'<div style="background:#161b22;border-radius:4px;padding:8px 12px;'
            f'font-size:0.78rem;color:#e6edf3">'
            f'<b>計算式:</b> average({pos_strs}) = {avg_pos:.1f}位 '
            f'→ 全ドライバー間で逆数正規化 (低い順位→高スコア) → スコア {d.circuit_score:.2f}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#1c2128;border:1px solid #30363d;border-radius:6px;'
            'padding:12px 16px;color:#e6edf3;font-size:0.85rem">'
            'このサーキットでの過去データが見つかりませんでした（新参戦または初参加の可能性）</div>',
            unsafe_allow_html=True,
        )

    # ── Tire strategy history ─────────────────────────────────────────────────
    st.markdown("##### タイヤ戦略（過去レース実績）")
    if strat_data:
        info_parts = [
            f'<div style="display:inline-block;background:#1c2128;border:1px solid #30363d;'
            f'border-radius:6px;padding:10px 18px;margin:4px 8px 4px 0;text-align:center">'
            f'<div style="font-size:1.3rem;font-weight:900;color:#f0c040">{avg_stops_v:.1f}</div>'
            f'<div style="font-size:0.72rem;color:#e6edf3">平均ピット回数</div></div>',
            f'<div style="display:inline-block;background:#1c2128;border:1px solid #30363d;'
            f'border-radius:6px;padding:10px 18px;margin:4px 8px 4px 0;text-align:center">'
            f'<div style="font-size:1.3rem;font-weight:900;color:{_score_color(d.tire_strategy_score)}">{mgmt_idx_v:.2f}</div>'
            f'<div style="font-size:0.72rem;color:#e6edf3">スティント比率 (vs 全体平均)</div></div>',
            f'<div style="display:inline-block;background:#1c2128;border:1px solid #30363d;'
            f'border-radius:6px;padding:10px 18px;margin:4px 0;text-align:center">'
            f'<div style="font-size:1.1rem;font-weight:700;color:#e6edf3">{comps_str}</div>'
            f'<div style="font-size:0.72rem;color:#e6edf3">典型コンパウンド列</div></div>',
        ]
        st.markdown("".join(info_parts), unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:#161b22;border-radius:4px;padding:8px 12px;'
            f'font-size:0.78rem;color:#e6edf3;margin-top:8px">'
            f'<b>計算式:</b> スティント比率 = ドライバー平均スティント長 ÷ 全体平均スティント長。'
            f'1.0より大きければフィールド平均より長くタイヤを持たせられることを示す。'
            f'全ドライバー間で正規化 → スコア {d.tire_strategy_score:.2f}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#1c2128;border:1px solid #30363d;border-radius:6px;'
            'padding:12px 16px;color:#e6edf3;font-size:0.85rem">'
            'このサーキットでのタイヤ戦略履歴データが見つかりませんでした。'
            'タイヤ戦略スコアは0.5（中立）として計算されています。</div>',
            unsafe_allow_html=True,
        )


# ─── accuracy summary ─────────────────────────────────────────────────────────

def render_accuracy_summary(predictions: list, actual_pos_map: dict) -> None:
    if not actual_pos_map or not predictions:
        return
    st.markdown("#### 予測精度サマリー")
    pred_top5 = {d.abbreviation for d in predictions[:5]}
    actual_top5 = {abbr for abbr, pos in actual_pos_map.items() if pos <= 5}
    top5_correct = len(pred_top5 & actual_top5)
    errors = [abs(d.predicted_position - actual_pos_map[d.abbreviation])
              for d in predictions if d.abbreviation in actual_pos_map]
    avg_error = sum(errors) / len(errors) if errors else 0.0
    perfect = sum(1 for e in errors if e == 0)
    col1, col2, col3 = st.columns(3)
    col1.metric("トップ5的中数", f"{top5_correct}/5", delta=f"{top5_correct * 20}%")
    col2.metric("平均誤差", f"{avg_error:.1f}位")
    col3.metric("完全一致", f"{perfect}件")


# ─── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="background:#010409;padding:12px 16px;'
        'border-bottom:3px solid #e8002d;margin:-1rem -1rem 1rem -1rem">'
        '<span style="color:#e6edf3;font-size:1.1rem;font-weight:bold;letter-spacing:2px">F1 PREDICTOR</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**レース選択**")

    current_year = datetime.datetime.now().year
    year = st.selectbox("シーズン", list(range(current_year, 2019, -1)))

    schedule = load_schedule(year)
    if schedule is None:
        st.error(f"{year}年のスケジュール取得に失敗しました")
        st.stop()

    race_labels = [f"Rd.{int(r['RoundNumber'])}: {r['EventName']}" for _, r in schedule.iterrows()]
    selected_idx = st.selectbox("グランプリ", range(len(race_labels)), format_func=lambda x: race_labels[x])

    selected_event = schedule.iloc[selected_idx]
    round_number = int(selected_event["RoundNumber"])
    gp_name = str(selected_event["EventName"])
    event_location = str(selected_event.get("Location") or "")

    cinfo_sb = get_circuit_info(event_location, gp_name)
    st.markdown("---")
    circuit_type_jp = "市街地" if cinfo_sb["type"] == "street" else "パーマネント"
    st.markdown(
        f'<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;'
        f'padding:12px;color:#e6edf3;font-size:0.8rem;line-height:1.9">'
        f'<div style="font-weight:bold;font-size:0.9rem;color:#f0c040;margin-bottom:4px">{cinfo_sb["name"]}</div>'
        f'<div>タイプ: <span style="color:#e6edf3">{circuit_type_jp}</span></div>'
        f'<div>標高: <span style="color:#e6edf3">{cinfo_sb["altitude_m"]}m</span></div>'
        f'<div>タイヤ摩耗: {_stars(cinfo_sb["tire_deg"])}</div>'
        f'<div>オーバーテイク: {_stars(cinfo_sb["overtaking"])}</div>'
        f'<div>PU感度: {_stars(cinfo_sb["pu_sensitivity"])}</div>'
        f'<div>DRSゾーン数: <span style="color:#e6edf3">{cinfo_sb["drs_zones"]}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    predict_btn = st.button("予測を実行", type="primary", use_container_width=True)


# ─── initial screen ──────────────────────────────────────────────────────────
if not predict_btn and "predictions" not in st.session_state:
    st.markdown(
        '<div style="text-align:center;padding:40px 0 20px 0">'
        '<h1 style="color:#e6edf3;letter-spacing:3px">F1 RACE PREDICTOR</h1>'
        '<p style="color:#e6edf3;font-size:1rem">左のサイドバーでレースを選んで「予測を実行」ボタンを押してください</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("データソース", "FastF1")
    col2.metric("予測因子", "9指標")
    col3.metric("フォーム参照", "直近5戦")
    col4.metric("相性参照", "過去3年")
    st.stop()


# ─── prediction execution ────────────────────────────────────────────────────
if predict_btn:
    with st.spinner(f"{gp_name} の予選データを取得中..."):
        quali_data = get_quali_data(year, round_number)

    with st.spinner("ドライバーの直近フォームを分析中..."):
        driver_form, team_form = get_recent_form(year, round_number, n_races=5)

    with st.spinner("サーキット相性を計算中..."):
        circuit_history = get_circuit_history(year, round_number, n_years=3)

    with st.spinner("ドライバーフォーム詳細を取得中..."):
        form_detail = get_recent_form_detail(year, round_number, n_races=5)

    with st.spinner("サーキット過去成績詳細を取得中..."):
        circuit_detail = get_circuit_history_detail(year, round_number, n_years=3)

    with st.spinner("タイヤ戦略データを取得中..."):
        tire_strategy = get_tire_strategy_data(year, round_number, n_years=3)

    with st.spinner("前年のスタートタイヤを取得中..."):
        prev_year_compounds = get_prev_year_start_compounds(year, round_number)

    circuit_info = get_circuit_info(event_location, gp_name)
    race_date_str = _race_date_str(selected_event)

    with st.spinner("天気データを取得中..."):
        weather = get_weather_for_race(
            lat=float(circuit_info["lat"]),
            lon=float(circuit_info["lon"]),
            race_date=race_date_str,
        )

    with st.spinner("ウェットスキルを計算中..."):
        wet_skills = get_wet_skill(year, round_number)

    has_quali = len(quali_data) > 0
    quali_preds: Optional[list] = None

    if has_quali:
        race_predictions = predict_race(
            quali_data=quali_data, driver_form=driver_form, team_form=team_form,
            circuit_history=circuit_history, circuit_info=circuit_info,
            weather=weather, wet_skills=wet_skills, tire_strategy=tire_strategy,
            prev_year_compounds=prev_year_compounds,
        )
    else:
        with st.spinner("ドライバー情報を取得中..."):
            driver_meta = get_driver_metadata(year, round_number)

        quali_preds = predict_qualifying(
            driver_form=driver_form, team_form=team_form, circuit_history=circuit_history,
            circuit_info=circuit_info, weather=weather, wet_skills=wet_skills,
            driver_metadata=driver_meta,
        )
        synthetic_quali_data = quali_predictions_to_data(quali_preds)
        race_predictions = predict_race(
            quali_data=synthetic_quali_data, driver_form=driver_form, team_form=team_form,
            circuit_history=circuit_history, circuit_info=circuit_info,
            weather=weather, wet_skills=wet_skills, tire_strategy=tire_strategy,
            prev_year_compounds=prev_year_compounds,
        )

    with st.spinner("実際のレース結果を取得中..."):
        actual_results = load_session_results(year, round_number, "R")

    with st.spinner("実際のタイヤデータを取得中..."):
        actual_tire_compounds = get_race_tire_compounds(year, round_number)

    st.session_state.predictions = race_predictions
    st.session_state.quali_preds = quali_preds
    st.session_state.actual_results = actual_results
    st.session_state.actual_tire_compounds = actual_tire_compounds
    st.session_state.gp_name = gp_name
    st.session_state.year = year
    st.session_state.has_quali = has_quali
    st.session_state.quali_data = quali_data
    st.session_state.circuit_info = circuit_info
    st.session_state.weather = weather
    st.session_state.wet_skills = wet_skills
    st.session_state.form_detail = form_detail
    st.session_state.circuit_detail = circuit_detail
    st.session_state.tire_strategy = tire_strategy


# ─── results display ─────────────────────────────────────────────────────────
if "predictions" not in st.session_state:
    st.stop()

predictions: list = st.session_state.predictions
quali_preds_state: Optional[list] = st.session_state.get("quali_preds")
actual_results = st.session_state.actual_results
gp_name_state: str = st.session_state.gp_name
has_quali: bool = st.session_state.has_quali
disp_year: int = st.session_state.year
quali_data_state: dict = st.session_state.get("quali_data", {})
circuit_info_state: dict = st.session_state.get("circuit_info", {})
weather_state: dict = st.session_state.get("weather", {})
wet_skills_state: dict = st.session_state.get("wet_skills", {})
form_detail_state: dict = st.session_state.get("form_detail", {})
circuit_detail_state: dict = st.session_state.get("circuit_detail", {})
tire_strategy_state: dict = st.session_state.get("tire_strategy", {})
actual_tire_state: dict = st.session_state.get("actual_tire_compounds", {})

if not predictions:
    st.error("予測データを生成できませんでした。別のレースを選択してください。")
    st.stop()

# ── actual position map ──────────────────────────────────────────────────────
actual_pos_map: dict = {}
if actual_results is not None:
    for _, row in actual_results.iterrows():
        abbr = str(row.get("Abbreviation") or "")
        try:
            pos = int(float(row.get("Position") or 0))
            if abbr and pos > 0:
                actual_pos_map[abbr] = pos
        except (TypeError, ValueError):
            pass

# ── weather values ────────────────────────────────────────────────────────────
rain_prob = float(weather_state.get("rain_probability", 10))
air_max = float(weather_state.get("air_temp_max", 25))
wind_spd = float(weather_state.get("wind_speed", 10))
altitude_m = float(circuit_info_state.get("altitude_m", 50))

# ── race info header bar ──────────────────────────────────────────────────────
circuit_type_badge = (
    '<span style="background:#e8002d;color:white;padding:2px 8px;border-radius:3px;font-size:0.75rem;font-weight:bold">STREET</span>'
    if circuit_info_state.get("type") == "street" else
    '<span style="background:#3671C6;color:white;padding:2px 8px;border-radius:3px;font-size:0.75rem;font-weight:bold">PERMANENT</span>'
)
rain_label = _rain_label(rain_prob)
rain_color = "#00BFFF" if rain_prob >= 60 else ("#f0c040" if rain_prob >= 20 else "#3fb950")
alt_html = (f'<span style="color:#e6edf3;margin-left:16px">ALT {altitude_m:.0f}m</span>' if altitude_m > 500 else "")

st.markdown(
    f'<div style="background:#010409;border-bottom:3px solid #e8002d;padding:14px 20px;'
    f'border-radius:6px;margin-bottom:16px;display:flex;align-items:center;gap:16px;flex-wrap:wrap">'
    f'<div style="font-size:1.3rem;font-weight:bold;color:#e6edf3;letter-spacing:1px">{disp_year} {gp_name_state.upper()}</div>'
    f'{circuit_type_badge}'
    f'<span style="color:#e6edf3">{air_max:.1f}°C</span>'
    f'<span style="color:{rain_color};font-weight:bold">{rain_label} {rain_prob:.0f}%</span>'
    f'<span style="color:#e6edf3">WIND {wind_spd:.1f} km/h</span>'
    f'{alt_html}</div>',
    unsafe_allow_html=True,
)

if actual_results is not None:
    st.success("レース結果も取得済みです。予測と実際の結果を比較できます。")

# ── tabs ──────────────────────────────────────────────────────────────────────
if not has_quali and quali_preds_state:
    tab_quali, tab_race, tab_analysis, tab_driver = st.tabs(
        ["予選予測（推定）", "レース予測", "因子分析", "ドライバー分析"]
    )

    with tab_quali:
        st.markdown(
            '<div style="background:#1c2128;border:1px solid #f0c040;border-radius:4px;'
            'padding:10px 14px;margin-bottom:12px;color:#f0c040;font-size:0.85rem">'
            'PREDICTED GRID — 公式予選データなし。フォームとサーキットデータから推定。</div>',
            unsafe_allow_html=True,
        )
        render_scoreboard(quali_preds_state, {}, "quali", TEAM_COLORS)

    with tab_race:
        st.markdown(
            '<div style="background:#1c2128;border:1px solid #30363d;border-radius:4px;'
            'padding:10px 14px;margin-bottom:12px;color:#e6edf3;font-size:0.85rem">'
            '上記の推定グリッドをベースにレース予測を算出しています。</div>',
            unsafe_allow_html=True,
        )
        render_scoreboard(predictions, actual_pos_map, "race", TEAM_COLORS,
                          actual_tire_compounds=actual_tire_state)
        if actual_pos_map:
            render_accuracy_summary(predictions, actual_pos_map)

    with tab_analysis:
        render_factor_analysis(predictions, has_quali, quali_data_state)

    with tab_driver:
        render_driver_analysis(predictions, form_detail_state, circuit_detail_state,
                               circuit_info_state, weather_state, wet_skills_state,
                               tire_strategy_state, has_quali, TEAM_COLORS)

else:
    tab_race, tab_analysis, tab_driver = st.tabs(["レース予測", "因子分析", "ドライバー分析"])

    with tab_race:
        render_scoreboard(predictions, actual_pos_map, "race", TEAM_COLORS,
                          actual_tire_compounds=actual_tire_state)
        if actual_pos_map:
            render_accuracy_summary(predictions, actual_pos_map)

    with tab_analysis:
        render_factor_analysis(predictions, has_quali, quali_data_state)

    with tab_driver:
        render_driver_analysis(predictions, form_detail_state, circuit_detail_state,
                               circuit_info_state, weather_state, wet_skills_state,
                               tire_strategy_state, has_quali, TEAM_COLORS)
