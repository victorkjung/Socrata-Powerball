from __future__ import annotations

import math
import os
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="NY Powerball Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATASET_ID = "d6yy-54nr"
ENDPOINT = f"https://data.ny.gov/resource/{DATASET_ID}.json"

# Current Powerball rules (used for UI + simulator defaults)
WHITE_POOL_SIZE = 69
PB_POOL_SIZE = 26
WHITE_BALLS_PER_DRAW = 5

# API settings
MAX_ROWS = 50_000
API_CHUNK_SIZE = 5_000
API_TIMEOUT_SECONDS = 30

# Cache settings
DEFAULT_TTL_HOURS = 6
MIN_TTL_HOURS = 1
MAX_TTL_HOURS = 48

# UI settings
MOBILE_BREAKPOINT = 768
DEFAULT_CHART_HEIGHT = 650
MOBILE_CHART_HEIGHT = 360


def get_app_token() -> Optional[str]:
    token = None
    if hasattr(st, "secrets"):
        token = st.secrets.get("SOCRATA_APP_TOKEN")
    return token or os.environ.get("SOCRATA_APP_TOKEN")


APP_TOKEN = get_app_token()

# -----------------------------
# Theme Configuration
# -----------------------------
@dataclass(frozen=True)
class ThemeColors:
    bg: str
    panel: str
    text: str
    subtle: str
    border: str


DARK_THEME = ThemeColors(
    bg="#0e1117",
    panel="#161b22",
    text="#e6edf3",
    subtle="#9da7b1",
    border="#30363d",
)

LIGHT_THEME = ThemeColors(
    bg="#ffffff",
    panel="#f6f8fa",
    text="#111827",
    subtle="#4b5563",
    border="#e5e7eb",
)


def inject_responsive_css(dark: bool) -> None:
    colors = DARK_THEME if dark else LIGHT_THEME
    css = f"""
<style>
.stApp {{
  background: {colors.bg};
  color: {colors.text};
}}
section.main > div {{
  padding-top: 0.75rem;
}}
section[data-testid="stSidebar"] > div {{
  background: {colors.panel};
  border-right: 1px solid {colors.border};
}}
html, body, [class*="css"] {{
  color: {colors.text};
}}
small, .stCaption, .stMarkdown p, .stMarkdown li {{
  color: {colors.subtle};
}}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {{
  background: {colors.panel};
  color: {colors.text};
  border-color: {colors.border};
}}
div[data-testid="stDataFrame"] {{
  border: 1px solid {colors.border};
  border-radius: 10px;
  overflow: hidden;
}}

@media (max-width: {MOBILE_BREAKPOINT}px) {{
  section.main > div {{
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
  }}
  div[data-testid="stHorizontalBlock"] {{
    flex-direction: column !important;
    gap: 0.75rem !important;
  }}
  .js-plotly-plot, .plotly, .plot-container {{
    height: {MOBILE_CHART_HEIGHT}px !important;
  }}
  div[data-testid="stMetric"] {{
    padding: 0.75rem !important;
  }}
}}
</style>
<meta name="theme-color" content="{colors.bg}">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
"""
    st.markdown(css, unsafe_allow_html=True)


def apply_plotly_theme(fig: "Figure", dark: bool, compact: bool) -> "Figure":
    template = "plotly_dark" if dark else "plotly_white"
    font_size = 11 if compact else 13
    title_size = 14 if compact else 18
    fig.update_layout(
        template=template,
        margin=dict(l=16, r=16, t=56, b=24),
        font=dict(size=font_size),
        title=dict(font=dict(size=title_size), x=0.02, xanchor="left"),
        legend=dict(
            orientation="h" if compact else "v",
            yanchor="bottom" if compact else "top",
            y=-0.35 if compact else 1,
            x=0.5 if compact else 1,
            xanchor="center" if compact else "right",
            font=dict(size=10 if compact else 12),
        ),
    )
    if dark:
        fig.update_traces(line=dict(width=2.5), selector=dict(type="scatter"))
    return fig


# -----------------------------
# API & Data Fetching
# -----------------------------
class SocrataAPIError(Exception):
    pass


def _socrata_get(params: dict) -> list[dict]:
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}
    try:
        resp = requests.get(
            ENDPOINT,
            params=params,
            headers=headers,
            timeout=API_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout as e:
        raise SocrataAPIError("API request timed out. Please try again.") from e
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status == 429:
            raise SocrataAPIError(
                "Rate limit exceeded (HTTP 429). Add SOCRATA_APP_TOKEN in Streamlit Secrets to increase limits."
            ) from e
        raise SocrataAPIError(f"API error (HTTP {status}).") from e
    except requests.exceptions.RequestException as e:
        raise SocrataAPIError(f"Network error: {e}") from e


def _parse_winning_numbers(s: object) -> Optional[list[int]]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    parts = str(s).strip().split()
    if len(parts) != 6:
        return None
    try:
        return [int(x) for x in parts]
    except ValueError:
        return None


@st.cache_data(show_spinner=False)
def fetch_draws_from_api_cached(ttl_seconds: int) -> pd.DataFrame:
    """
    Cached API pull. ttl_seconds is included so cache key changes with TTL selection.
    """
    _ = ttl_seconds

    all_rows: list[dict] = []
    offset = 0
    select_fields = "draw_date,winning_numbers,multiplier"

    while True:
        params = {
            "$select": select_fields,
            "$order": "draw_date ASC",
            "$limit": API_CHUNK_SIZE,
            "$offset": offset,
        }
        chunk = _socrata_get(params)
        if not chunk:
            break

        all_rows.extend(chunk)
        offset += API_CHUNK_SIZE

        if len(all_rows) >= MAX_ROWS:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Parse dates to naive datetime (no timezone) for easier charting
    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce", utc=True).dt.tz_convert(None)
    df["multiplier"] = pd.to_numeric(df.get("multiplier"), errors="coerce")

    df["nums"] = df["winning_numbers"].apply(_parse_winning_numbers)
    df = df.dropna(subset=["draw_date", "nums"]).reset_index(drop=True)
    if df.empty:
        return df

    df["white"] = df["nums"].apply(lambda x: x[:5])
    df["pb"] = df["nums"].apply(lambda x: x[5])
    df["year_month"] = df["draw_date"].dt.to_period("M").astype(str)

    return df


# -----------------------------
# Analytics Helpers
# -----------------------------
def explode_numbers(df: pd.DataFrame) -> pd.DataFrame:
    white_df = df[["draw_date", "year_month", "white"]].copy()
    white_df = white_df.explode("white").rename(columns={"white": "number"})
    white_df["is_powerball"] = False
    white_df["number"] = white_df["number"].astype(int)

    pb_df = df[["draw_date", "year_month", "pb"]].copy().rename(columns={"pb": "number"})
    pb_df["is_powerball"] = True
    pb_df["number"] = pb_df["number"].astype(int)

    return pd.concat([white_df, pb_df], ignore_index=True)


def build_heatmap(long_df: pd.DataFrame, is_powerball: bool) -> pd.DataFrame:
    filtered = long_df[long_df["is_powerball"] == is_powerball]
    pivot = (
        filtered.groupby(["year_month", "number"])
        .size()
        .reset_index(name="count")
        .pivot(index="year_month", columns="number", values="count")
        .fillna(0)
        .astype(int)
    )
    return pivot.sort_index().reindex(sorted(pivot.columns), axis=1)


def compute_win_percentage(long_df: pd.DataFrame, numbers: list[int], is_powerball: bool) -> pd.DataFrame:
    filtered = long_df[long_df["is_powerball"] == is_powerball].copy()
    month_totals = filtered.groupby("year_month")["draw_date"].nunique()

    number_data = filtered[filtered["number"].isin(numbers)]
    hits = (
        number_data.groupby(["year_month", "number"])["draw_date"]
        .nunique()
        .reset_index(name="hits")
    )

    hits["total"] = hits["year_month"].map(month_totals)
    hits["win_pct"] = (hits["hits"] / hits["total"] * 100).fillna(0.0)
    return hits[["year_month", "number", "win_pct"]]


def compute_top_bottom_numbers(long_df: pd.DataFrame) -> tuple[list[int], list[int], list[int], list[int]]:
    white_counts = (
        long_df[~long_df["is_powerball"]]["number"]
        .value_counts()
        .sort_values(ascending=False)
    )
    pb_counts = (
        long_df[long_df["is_powerball"]]["number"]
        .value_counts()
        .sort_values(ascending=False)
    )
    return (
        white_counts.head(5).index.tolist(),
        white_counts.tail(5).index.tolist(),
        pb_counts.head(1).index.tolist(),
        pb_counts.tail(1).index.tolist(),
    )


def compute_hot_cold_scores(long_df: pd.DataFrame, is_powerball: bool, window_draws: int = 100) -> pd.DataFrame:
    filtered = long_df[long_df["is_powerball"] == is_powerball].copy()
    draws = filtered["draw_date"].drop_duplicates().sort_values()

    if draws.empty or len(draws) < window_draws:
        return pd.DataFrame()

    recent_dates = set(draws.tail(window_draws))
    total_draws = len(draws)
    recent_draws = len(recent_dates)

    total_counts = filtered["number"].value_counts()
    recent_counts = filtered[filtered["draw_date"].isin(recent_dates)]["number"].value_counts()

    all_numbers = total_counts.index
    recent_counts = recent_counts.reindex(all_numbers, fill_value=0)

    total_rate = total_counts / total_draws
    recent_rate = recent_counts / recent_draws

    p = total_rate.clip(1e-9, 1 - 1e-9)
    se = np.sqrt(p * (1 - p) / max(recent_draws, 1))
    z = (recent_rate - total_rate) / se

    result = pd.DataFrame(
        {
            "number": all_numbers.astype(int),
            "recent_rate": recent_rate.values,
            "base_rate": total_rate.values,
            "z_score": z.values,
        }
    ).replace([np.inf, -np.inf], np.nan)

    result["score"] = result["z_score"].clip(-6, 6).fillna(0.0)
    return result.sort_values("score", ascending=False).reset_index(drop=True)


# -----------------------------
# Probability Helpers
# -----------------------------
def compute_exact_probabilities(white_pool: int, pb_pool: int) -> pd.DataFrame:
    denom = math.comb(white_pool, 5) * pb_pool
    rows = []
    for w in range(0, 6):
        ways_white = math.comb(5, w) * math.comb(white_pool - 5, 5 - w)
        for pb_match in (0, 1):
            ways_pb = 1 if pb_match else (pb_pool - 1)
            p = (ways_white * ways_pb) / denom
            rows.append((w, pb_match, p))

    df = pd.DataFrame(rows, columns=["white_matches", "pb_match", "probability"])
    df["pb_match"] = df["pb_match"].map({0: "No PB", 1: "PB"})
    df["odds_1_in"] = (1 / df["probability"]).replace([np.inf], np.nan)
    return df.sort_values(["white_matches", "pb_match"], ascending=[False, False])


def simulate_lottery_sessions(
    white_pool: int,
    pb_pool: int,
    num_draws: int,
    num_sessions: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    if seed is None:
        seed = secrets.randbelow(10**9)
    rng = np.random.default_rng(seed)

    results = []
    for _ in range(num_sessions):
        ticket_white = set(rng.choice(np.arange(1, white_pool + 1), size=5, replace=False))
        ticket_pb = int(rng.integers(1, pb_pool + 1))

        best = (0, 0)
        for _d in range(num_draws):
            draw_white = set(rng.choice(np.arange(1, white_pool + 1), size=5, replace=False))
            draw_pb = int(rng.integers(1, pb_pool + 1))
            wm = len(ticket_white & draw_white)
            pm = 1 if draw_pb == ticket_pb else 0
            if (wm, pm) > best:
                best = (wm, pm)

        results.append(best)

    sim_df = pd.DataFrame(results, columns=["white_matches", "pb_match"])
    sim_df["pb_match"] = sim_df["pb_match"].map({0: "No PB", 1: "PB"})
    return sim_df


# -----------------------------
# Input Validation (Pickers)
# -----------------------------
def validate_white_balls(nums: list[int], max_white: int) -> list[int]:
    if len(nums) != 5:
        raise ValueError("Please pick exactly 5 white ball numbers.")
    if len(set(nums)) != 5:
        raise ValueError("White balls must be 5 distinct numbers (no duplicates).")
    if not all(1 <= n <= max_white for n in nums):
        raise ValueError(f"White balls must be between 1 and {max_white}.")
    return sorted(nums)


def validate_powerball(pb: int, max_pb: int) -> int:
    if not (1 <= pb <= max_pb):
        raise ValueError(f"Powerball must be between 1 and {max_pb}.")
    return pb


# -----------------------------
# UI Components
# -----------------------------
def render_latest_draw_summary(df: pd.DataFrame) -> None:
    latest_row = (
        df.dropna(subset=["draw_date", "white", "pb"])
        .sort_values("draw_date", ascending=False)
        .iloc[0]
    )
    latest_date = latest_row["draw_date"].date()
    latest_white = sorted(int(x) for x in latest_row["white"])
    latest_pb = int(latest_row["pb"])
    latest_mult = latest_row.get("multiplier")

    c1, c2, c3 = st.columns([1.2, 2.6, 1.2])
    with c1:
        st.metric("Most Recent Drawing", str(latest_date))
    with c2:
        st.markdown(
            f"**Winning Numbers (5 White + 1 Powerball)**  \n"
            f"White balls: **{latest_white}**  \n"
            f"Powerball: **{latest_pb}**"
        )
    with c3:
        mult_display = f"{int(latest_mult)}x" if pd.notna(latest_mult) else "—"
        st.metric("Multiplier", mult_display)

    st.caption(f"Rows: **{len(df):,}** | Latest draw in dataset: **{latest_date}**")


def render_download_buttons(df: pd.DataFrame, long_df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇ Download Draw Data (CSV)",
            data=df.to_csv(index=False),
            file_name="powerball_draws.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "⬇ Download Long Format (CSV)",
            data=long_df.to_csv(index=False),
            file_name="powerball_numbers_long.csv",
            mime="text/csv",
        )


# -----------------------------
# Sidebar (Mobile Pickers + Clear/Reset)
# -----------------------------
def render_sidebar() -> tuple[bool, int, str, list[int], int, bool, bool]:
    DEFAULT_WHITES = [1, 2, 3, 4, 5]
    DEFAULT_PB = 6

    # Initialize picker state once
    for i, v in enumerate(DEFAULT_WHITES, start=1):
        st.session_state.setdefault(f"white_{i}", v)
    st.session_state.setdefault("pb_pick", DEFAULT_PB)

    with st.sidebar:
        st.header("Controls")

        st.subheader("Appearance")
        dark_mode = st.toggle("🌙 Dark mode", value=True)
        compact_mode = st.toggle("📱 Compact charts (mobile-style)", value=False)

        if not APP_TOKEN:
            st.error(
                "Missing SOCRATA_APP_TOKEN.\n\n"
                "The app will still run, but you may hit Socrata rate limits.\n\n"
                "Streamlit Cloud → Settings → Secrets:\n"
                'SOCRATA_APP_TOKEN = "YOUR_TOKEN"'
            )

        ttl_hours = st.slider("Cache TTL (hours)", MIN_TTL_HOURS, MAX_TTL_HOURS, DEFAULT_TTL_HOURS)

        number_mode = st.radio(
            "Show analytics for:",
            ["Both", "White balls only", "Powerball only"],
            index=0,
        )

        st.divider()
        st.subheader("Mock Drawing")
        st.caption("Tap to pick 5 white balls + 1 Powerball.")

        # Clear / Reset
        if st.button("🧹 Clear / Reset numbers", use_container_width=True):
            for i, v in enumerate(DEFAULT_WHITES, start=1):
                st.session_state[f"white_{i}"] = v
            st.session_state["pb_pick"] = DEFAULT_PB
            st.toast("Numbers reset.", icon="🧹")
            st.rerun()

        st.markdown("**White balls (pick 5):**")
        w_cols = st.columns(5)

        white_picks: list[int] = []
        for i, col in enumerate(w_cols, start=1):
            with col:
                white_picks.append(
                    int(
                        st.number_input(
                            f"W{i}",
                            min_value=1,
                            max_value=WHITE_POOL_SIZE,
                            step=1,
                            key=f"white_{i}",
                            label_visibility="collapsed",
                        )
                    )
                )

        st.markdown("**Powerball (pick 1):**")
        pb_pick = int(
            st.number_input(
                "PB",
                min_value=1,
                max_value=PB_POOL_SIZE,
                step=1,
                key="pb_pick",
                label_visibility="collapsed",
            )
        )

        st.divider()
        refresh_now = st.button("🔄 Force refresh (clear cache + pull API)")

    return dark_mode, ttl_hours, number_mode, white_picks, pb_pick, refresh_now, compact_mode


# -----------------------------
# Tabs
# -----------------------------
def render_tab_launcher() -> None:
    st.subheader("Mobile Launcher")
    st.caption("Quick launchpad. Add to Home Screen for an app-like experience.")
    cols = st.columns(3)
    cards = [
        ("🔥 Heat Maps", "Monthly frequency by number"),
        ("🧠 Hot vs Cold", "Rolling trends vs baseline"),
        ("🧮 Simulator", "Odds + Monte Carlo"),
    ]
    for col, (label, desc) in zip(cols, cards):
        with col:
            st.button(label, use_container_width=True, disabled=True)
            st.markdown(f"**{label}**: {desc}")

    st.divider()
    st.markdown("### Add to Home Screen")
    st.markdown(
        "- **iOS (Safari):** Share → *Add to Home Screen*\n"
        "- **Android (Chrome):** Menu → *Add to Home screen*"
    )


def render_tab_heatmaps(long_df: pd.DataFrame, number_mode: str, dark_mode: bool, compact: bool) -> None:
    st.subheader("Heat Map of Winning Number Frequency (by Month)")
    col_left, col_right = st.columns(2)

    with col_left:
        if number_mode in ["Both", "White balls only"]:
            pivot = build_heatmap(long_df, is_powerball=False)
            fig = px.imshow(
                pivot,
                aspect="auto",
                labels=dict(x="Number", y="Year-Month", color="Count"),
                title="White Ball Frequency Heatmap",
            )
            fig.update_layout(height=DEFAULT_CHART_HEIGHT)
            apply_plotly_theme(fig, dark_mode, compact)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Toggle is set to Powerball only.")

    with col_right:
        if number_mode in ["Both", "Powerball only"]:
            pivot = build_heatmap(long_df, is_powerball=True)
            fig = px.imshow(
                pivot,
                aspect="auto",
                labels=dict(x="Number", y="Year-Month", color="Count"),
                title="Powerball Frequency Heatmap",
            )
            fig.update_layout(height=DEFAULT_CHART_HEIGHT)
            apply_plotly_theme(fig, dark_mode, compact)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Toggle is set to White balls only.")


def render_tab_top_bottom(
    long_df: pd.DataFrame,
    number_mode: str,
    dark_mode: bool,
    compact: bool,
) -> None:
    top5_white, bot5_white, top1_pb, bot1_pb = compute_top_bottom_numbers(long_df)

    st.subheader("Top 6 and Bottom 6 Numbers (5 White + 1 Powerball) + Win % Over Time")
    st.info("Win % = % of draws in a month where the number appeared (white) or matched PB (powerball).")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Top 6")
        st.write("White (Top 5):", top5_white)
        st.write("Powerball (Top 1):", top1_pb)
    with col2:
        st.markdown("### ❄️ Bottom 6")
        st.write("White (Bottom 5):", bot5_white)
        st.write("Powerball (Bottom 1):", bot1_pb)

    if number_mode in ["Both", "White balls only"]:
        st.markdown("### White Balls — Win % by Month")
        for nums, title in [(top5_white, "Top 5 White Balls"), (bot5_white, "Bottom 5 White Balls")]:
            series = compute_win_percentage(long_df, nums, is_powerball=False)
            fig = px.line(series, x="year_month", y="win_pct", color="number", title=title)
            fig.update_layout(height=320, xaxis_title="Year-Month", yaxis_title="Win %")
            apply_plotly_theme(fig, dark_mode, compact)
            st.plotly_chart(fig, use_container_width=True)

    if number_mode in ["Both", "Powerball only"]:
        st.markdown("### Powerball — Win % by Month")
        for nums, title in [(top1_pb, "Top 1 Powerball"), (bot1_pb, "Bottom 1 Powerball")]:
            series = compute_win_percentage(long_df, nums, is_powerball=True)
            fig = px.line(series, x="year_month", y="win_pct", color="number", title=title)
            fig.update_layout(height=280, xaxis_title="Year-Month", yaxis_title="Win %")
            apply_plotly_theme(fig, dark_mode, compact)
            st.plotly_chart(fig, use_container_width=True)


def render_tab_mock_checker(df: pd.DataFrame, white_picks: list[int], pb_pick: int) -> None:
    st.subheader("Mock Drawing Checker (Exact Match)")
    st.caption("Pick your numbers in the sidebar, then click **Check my numbers**.")

    check = st.button("✅ Check my numbers", type="primary")
    if not check:
        st.info("Select 5 white balls and 1 Powerball in the sidebar, then click **Check my numbers**.")
        return

    try:
        user_white = validate_white_balls(white_picks, max_white=WHITE_POOL_SIZE)
        user_pb = validate_powerball(pb_pick, max_pb=PB_POOL_SIZE)

        st.write(f"**Your Pick:** White balls: **{user_white}** | Powerball: **{user_pb}**")

        df_check = df.copy()
        df_check["white_key"] = df_check["white"].apply(lambda x: tuple(sorted(int(i) for i in x)))
        df_check["pb_key"] = df_check["pb"].astype(int)

        matches = df_check[(df_check["white_key"] == tuple(user_white)) & (df_check["pb_key"] == user_pb)]

        if matches.empty:
            st.warning("No exact historical match found in this dataset.")
        else:
            st.success(f"✅ Found **{len(matches)}** exact match(es)!")
            out = matches[["draw_date", "winning_numbers", "multiplier"]].copy()
            out["draw_date"] = out["draw_date"].dt.date
            out = out.sort_values("draw_date", ascending=False)
            st.dataframe(out, use_container_width=True)

    except ValueError as e:
        st.error(str(e))


def render_tab_simulator(long_df: pd.DataFrame, dark_mode: bool, compact: bool) -> None:
    st.subheader("🧮 Probability Simulator")
    st.caption("Exact odds + Monte Carlo simulation.")

    white_pool = WHITE_POOL_SIZE
    pb_pool = PB_POOL_SIZE

    col1, col2, col3 = st.columns(3)
    jackpot_odds = 1 / (math.comb(white_pool, 5) * pb_pool)
    col1.metric("White pool", white_pool)
    col2.metric("PB pool", pb_pool)
    col3.metric("Jackpot odds", f"1 in {int(1/jackpot_odds):,}")

    st.divider()
    st.markdown("### Exact probabilities by match pattern")
    prob_df = compute_exact_probabilities(white_pool, pb_pool)
    st.dataframe(prob_df, use_container_width=True)

    st.divider()
    st.markdown("### Monte Carlo: play the same ticket for N draws")

    c1, c2, c3 = st.columns(3)
    num_draws = c1.number_input("Draws", 1, 5000, 365, step=1)
    num_sessions = c2.number_input("Sessions", 100, 20000, 5000, step=100)
    seed = c3.number_input("Seed", 0, 10**9, 42, step=1)

    if st.button("Run simulation", type="primary"):
        with st.spinner("Simulating…"):
            sim_df = simulate_lottery_sessions(white_pool, pb_pool, int(num_draws), int(num_sessions), int(seed))

        summary = sim_df.value_counts().reset_index(name="count")
        summary["pct"] = (summary["count"] / summary["count"].sum() * 100).round(2)

        fig = px.bar(summary, x="white_matches", y="pct", color="pb_match", title="Best match achieved per session")
        fig.update_layout(height=360, xaxis_title="White matches", yaxis_title="% of sessions")
        apply_plotly_theme(fig, dark_mode, compact)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            summary.sort_values(["white_matches", "pb_match"], ascending=[False, False]),
            use_container_width=True,
        )


def render_tab_hot_cold(long_df: pd.DataFrame, dark_mode: bool, compact: bool) -> None:
    st.subheader("🧠 Hot vs Cold Trend Scoring")
    st.caption("Compares the last N draws to the long-run baseline using a z-score-like signal.")

    window = st.slider("Rolling window (draws)", 20, 400, 100, step=10)
    ball_type = st.radio("Analyze", ["White balls", "Powerball"], horizontal=True)

    is_pb = (ball_type == "Powerball")
    scores = compute_hot_cold_scores(long_df, is_powerball=is_pb, window_draws=int(window))

    if scores.empty:
        st.info("Not enough data to compute scores.")
        return

    hot = scores.head(10)
    cold = scores.tail(10).sort_values("score", ascending=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔥 Hottest (Top 10)")
        fig = px.bar(hot, x="number", y="score", title="Hot numbers (score)")
        fig.update_layout(height=340, xaxis_title="Number", yaxis_title="Score")
        apply_plotly_theme(fig, dark_mode, compact)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🧊 Coldest (Bottom 10)")
        fig = px.bar(cold, x="number", y="score", title="Cold numbers (score)")
        fig.update_layout(height=340, xaxis_title="Number", yaxis_title="Score")
        apply_plotly_theme(fig, dark_mode, compact)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Full table")
    show_n = st.slider("Show top N by absolute score", 20, 200, 60, step=10)

    scores = scores.copy()
    scores["abs_score"] = scores["score"].abs()
    view = scores.sort_values("abs_score", ascending=False).head(int(show_n)).drop(columns=["abs_score"])
    st.dataframe(view, use_container_width=True)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    st.title("🎯 NY Powerball Analyzer (Socrata API)")

    dark_mode, ttl_hours, number_mode, white_picks, pb_pick, force_refresh, compact_mode = render_sidebar()
    inject_responsive_css(dark_mode)

    if force_refresh:
        st.cache_data.clear()
        st.toast("Cache cleared — fetching fresh data from API…", icon="🔄")

    ttl_seconds = int(ttl_hours * 3600)

    try:
        with st.spinner("Loading draws from data.ny.gov…"):
            df = fetch_draws_from_api_cached(ttl_seconds=ttl_seconds)
    except SocrataAPIError as e:
        st.error(f"API Error: {e}")
        st.stop()

    if df.empty:
        st.error("No data returned from the API. Try refreshing or check connectivity.")
        st.stop()

    render_latest_draw_summary(df)

    long_df = explode_numbers(df)

    render_download_buttons(df, long_df)

    tabs = st.tabs(["📲 Launcher", "🔥 Heat Maps", "📈 Top/Bottom", "✅ Mock Checker", "🧮 Simulator", "🧠 Hot vs Cold"])

    with tabs[0]:
        render_tab_launcher()

    with tabs[1]:
        render_tab_heatmaps(long_df, number_mode, dark_mode, compact_mode)

    with tabs[2]:
        render_tab_top_bottom(long_df, number_mode, dark_mode, compact_mode)

    with tabs[3]:
        render_tab_mock_checker(df, white_picks, pb_pick)

    with tabs[4]:
        render_tab_simulator(long_df, dark_mode, compact_mode)

    with tabs[5]:
        render_tab_hot_cold(long_df, dark_mode, compact_mode)

    with st.expander("Notes"):
        st.markdown(
            "- Cache is managed by Streamlit via `@st.cache_data`.\n"
            "- Use **Force refresh** to clear the cache and pull fresh API data.\n"
            "- Hot/Cold scoring is descriptive (trend signal), not predictive.\n"
            "- Lottery outcomes are independent events."
        )


if __name__ == "__main__":
    main()
