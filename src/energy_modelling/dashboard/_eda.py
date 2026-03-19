"""Tab 1 -- Exploratory Data Analysis.

Renders 24 EDA sections (12 original + 6 Phase 2 + 6 Phase 6):
price time-series, generation mix, load, neighbour prices, weather,
correlations, distributions, negative-price analysis, heatmap, scatter,
plus price changes, autocorrelation, forecast errors, feature importance,
volatility/regimes, residual load, day-of-week edge stability,
feature drift, quarterly patterns, volatility regime performance,
wind quintile analysis, and strategy correlation insights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_modelling.dashboard.eda_analysis import (
    autocorrelation,
    clean_hourly_data,
    compute_daily_settlement,
    compute_direction_streaks,
    compute_forecast_errors,
    compute_price_changes,
    compute_residual_load,
    day_of_week_edge_by_year,
    direction_base_rates,
    direction_by_group,
    feature_drift,
    lagged_direction_correlation,
    quarterly_direction_rates,
    rolling_volatility,
    volatility_regime_performance,
    wind_quintile_analysis,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/processed/dataset_de_lu.parquet")
CHALLENGE_DATA_PATH = Path("data/challenge/daily_public.csv")
PRICE_COL = "price_eur_mwh"

GEN_COLS_DISPLAY = {
    "gen_biomass_mw": "Biomass",
    "gen_fossil_brown_coal_lignite_mw": "Lignite",
    "gen_fossil_coal_derived_gas_mw": "Coal Gas",
    "gen_fossil_gas_mw": "Natural Gas",
    "gen_fossil_hard_coal_mw": "Hard Coal",
    "gen_fossil_oil_mw": "Oil",
    "gen_geothermal_mw": "Geothermal",
    "gen_hydro_pumped_storage_mw": "Pumped Storage",
    "gen_hydro_run_of_river_and_poundage_mw": "Run of River",
    "gen_hydro_water_reservoir_mw": "Water Reservoir",
    "gen_nuclear_mw": "Nuclear",
    "gen_other_mw": "Other",
    "gen_other_renewable_mw": "Other Renewable",
    "gen_solar_mw": "Solar",
    "gen_waste_mw": "Waste",
    "gen_wind_offshore_mw": "Wind Offshore",
    "gen_wind_onshore_mw": "Wind Onshore",
}
WEATHER_COLS_DISPLAY = {
    "weather_temperature_2m_degc": "Temperature (2m) [C]",
    "weather_relative_humidity_2m_pct": "Humidity (2m) [%]",
    "weather_wind_speed_10m_kmh": "Wind Speed (10m) [km/h]",
    "weather_wind_speed_100m_kmh": "Wind Speed (100m) [km/h]",
    "weather_shortwave_radiation_wm2": "GHI [W/m2]",
    "weather_direct_normal_irradiance_wm2": "DNI [W/m2]",
    "weather_precipitation_mm": "Precipitation [mm]",
}
RENEWABLE_COLS = [
    "gen_solar_mw",
    "gen_wind_onshore_mw",
    "gen_wind_offshore_mw",
    "gen_hydro_run_of_river_and_poundage_mw",
    "gen_hydro_water_reservoir_mw",
    "gen_biomass_mw",
    "gen_other_renewable_mw",
    "gen_geothermal_mw",
]
FOSSIL_COLS = [
    "gen_fossil_gas_mw",
    "gen_fossil_brown_coal_lignite_mw",
    "gen_fossil_hard_coal_mw",
    "gen_fossil_oil_mw",
    "gen_fossil_coal_derived_gas_mw",
]
NEIGHBOUR_PRICE_COLS = {
    "price_fr_eur_mwh": "France",
    "price_nl_eur_mwh": "Netherlands",
    "price_at_eur_mwh": "Austria",
    "price_pl_eur_mwh": "Poland",
    "price_cz_eur_mwh": "Czech Republic",
    "price_dk_1_eur_mwh": "Denmark West",
    "price_dk_2_eur_mwh": "Denmark East",
    "price_be_eur_mwh": "Belgium",
    "price_se_4_eur_mwh": "Sweden South",
}
GEN_COLORS = {
    "Solar": "#FFD700",
    "Wind Onshore": "#1E90FF",
    "Wind Offshore": "#4169E1",
    "Lignite": "#8B4513",
    "Hard Coal": "#333333",
    "Natural Gas": "#FF6347",
    "Nuclear": "#9370DB",
    "Biomass": "#228B22",
    "Run of River": "#00CED1",
    "Water Reservoir": "#20B2AA",
    "Pumped Storage": "#48D1CC",
    "Oil": "#696969",
    "Coal Gas": "#A0522D",
    "Waste": "#808000",
    "Other": "#C0C0C0",
    "Other Renewable": "#32CD32",
    "Geothermal": "#FF8C00",
}

_DISPLAY_ORDER = [
    "Solar",
    "Wind Onshore",
    "Wind Offshore",
    "Run of River",
    "Water Reservoir",
    "Biomass",
    "Other Renewable",
    "Geothermal",
    "Natural Gas",
    "Lignite",
    "Hard Coal",
    "Oil",
    "Coal Gas",
    "Nuclear",
    "Pumped Storage",
    "Waste",
    "Other",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@st.cache_data
def _load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = clean_hourly_data(df)
    df["hour"] = df.index.hour  # type: ignore[union-attr]
    df["dayofweek"] = df.index.dayofweek  # type: ignore[union-attr]
    df["month"] = df.index.month  # type: ignore[union-attr]
    df["year"] = df.index.year  # type: ignore[union-attr]
    df["date"] = df.index.date  # type: ignore[union-attr]

    gen_cols = [c for c in df.columns if c.startswith("gen_") and c.endswith("_mw")]
    df["total_generation_mw"] = df[gen_cols].sum(axis=1)
    df["renewable_generation_mw"] = df[[c for c in RENEWABLE_COLS if c in df.columns]].sum(axis=1)
    df["fossil_generation_mw"] = df[[c for c in FOSSIL_COLS if c in df.columns]].sum(axis=1)
    df["renewable_share_pct"] = (
        df["renewable_generation_mw"] / df["total_generation_mw"] * 100
    ).clip(0, 100)
    return df


@st.cache_data
def _load_challenge_data() -> pd.DataFrame | None:
    """Load the daily challenge CSV for Phase 6 feedback sections."""
    if not CHALLENGE_DATA_PATH.exists():
        return None
    df = pd.read_csv(CHALLENGE_DATA_PATH, parse_dates=["delivery_date"])
    df["year"] = df["delivery_date"].dt.year
    return df


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

_AGG_FREQ = {"Daily": "1D", "Weekly": "1W", "Monthly": "1ME"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the EDA tab contents."""

    if not DATA_PATH.exists():
        st.error(
            f"Dataset not found at `{DATA_PATH}`. "
            "Run `uv run collect-data --step all --kaggle --years 2019 ... --years 2025` first."
        )
        return

    df = _load_data()

    # --- Year filter (local to this tab) ---
    years = sorted(df["year"].unique())
    year_range = st.slider(
        "Year range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years))),
        key="eda_year_range",
    )
    mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
    dff = df[mask]

    _section_overview(dff)
    _section_price_ts(dff)
    _section_generation(dff)
    _section_load(dff)
    _section_neighbours(dff)
    _section_commodities(dff)
    _section_weather(dff)
    _section_correlations(dff)
    _section_distributions(dff)
    _section_negative(dff)
    _section_heatmap(dff)
    _section_scatter(dff)

    # --- Phase 2: Trading-focused analyses ---
    st.divider()
    st.subheader("Trading Signal Analysis")
    st.caption(
        "The sections below analyse daily price *changes* (settlement - last settlement) "
        "— the actual quantity that trading strategies must predict."
    )
    _section_price_changes(dff)
    _section_autocorrelation(dff)
    _section_forecast_errors(dff)
    _section_feature_importance(dff)
    _section_volatility_regimes(dff)
    _section_residual_load(dff)

    # --- Phase 6: Strategy feedback analyses ---
    challenge_df = _load_challenge_data()
    if challenge_df is not None:
        # Apply same year filter
        cdf = challenge_df[
            (challenge_df["year"] >= year_range[0]) & (challenge_df["year"] <= year_range[1])
        ]
        if not cdf.empty:
            st.divider()
            st.subheader("Strategy Feedback Analysis (Phase 6)")
            st.caption(
                "The sections below feed Phase 5 strategy performance insights "
                "back into deeper EDA — using the daily challenge dataset."
            )
            _section_dow_edge_stability(cdf)
            _section_feature_drift(cdf)
            _section_quarterly_patterns(cdf)
            _section_volatility_regime_performance(cdf)
            _section_wind_quintile(cdf)
            _section_strategy_correlation_insights(cdf)


# ---------------------------------------------------------------------------
# Section renderers (private)
# ---------------------------------------------------------------------------


def _section_overview(dff: pd.DataFrame) -> None:
    st.header("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(dff):,}")
    c2.metric("Columns", f"{len(dff.columns)}")
    c3.metric(
        "Date range",
        f"{dff.index.min().strftime('%Y-%m-%d')} to {dff.index.max().strftime('%Y-%m-%d')}",
    )
    c4.metric("Hourly frequency", "1h")

    with st.expander("Descriptive statistics"):
        show_cols = [PRICE_COL] + list(GEN_COLS_DISPLAY) + list(WEATHER_COLS_DISPLAY)
        show_cols = [c for c in show_cols if c in dff.columns]
        st.dataframe(
            dff[show_cols].describe().round(2).T.style.format("{:.2f}"),
            use_container_width=True,
        )

    with st.expander("Missing data per column"):
        missing = dff.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values in the filtered data.")
        else:
            st.dataframe(
                pd.DataFrame(
                    {"Missing": missing, "Pct": (missing / len(dff) * 100).round(2)}
                ).sort_values("Pct", ascending=False)
            )


def _section_price_ts(dff: pd.DataFrame) -> None:
    st.header("Day-Ahead Price Time Series")
    agg = st.radio(
        "Aggregation",
        ["Hourly (raw)", "Daily mean", "Weekly mean", "Monthly mean"],
        horizontal=True,
        key="eda_price_agg",
        index=3,
    )
    freq_map = {
        "Hourly (raw)": None,
        "Daily mean": "1D",
        "Weekly mean": "1W",
        "Monthly mean": "1ME",
    }
    freq = freq_map[agg]
    ts = dff[[PRICE_COL]] if freq is None else dff[[PRICE_COL]].resample(freq).mean()
    fig = px.line(
        ts.reset_index(),
        x="timestamp_utc",
        y=PRICE_COL,
        labels={"timestamp_utc": "", PRICE_COL: "EUR/MWh"},
        title=f"Day-Ahead Price ({agg})",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean", f"{dff[PRICE_COL].mean():.1f} EUR/MWh")
    c2.metric("Median", f"{dff[PRICE_COL].median():.1f} EUR/MWh")
    c3.metric("Std Dev", f"{dff[PRICE_COL].std():.1f}")
    c4.metric("Min", f"{dff[PRICE_COL].min():.1f} EUR/MWh")
    c5.metric("Max", f"{dff[PRICE_COL].max():.1f} EUR/MWh")


def _section_generation(dff: pd.DataFrame) -> None:
    st.header("Generation Mix")
    gen_agg = st.radio(
        "Aggregation",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="eda_gen_agg",
        index=2,
    )
    gen_freq = _AGG_FREQ[gen_agg]
    gen_cols = [c for c in GEN_COLS_DISPLAY if c in dff.columns]
    gen_rs = dff[gen_cols].resample(gen_freq).mean().rename(columns=GEN_COLS_DISPLAY)

    fig = go.Figure()
    for fuel in _DISPLAY_ORDER:
        if fuel in gen_rs.columns:
            fig.add_trace(
                go.Scatter(
                    x=gen_rs.index,
                    y=gen_rs[fuel],
                    name=fuel,
                    mode="lines",
                    stackgroup="one",
                    line={"width": 0},
                    fillcolor=GEN_COLORS.get(fuel, "#888888"),
                )
            )
    fig.update_layout(
        title=f"Generation Mix ({gen_agg} Average, MW)",
        yaxis_title="MW",
        height=500,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Renewable Share Over Time")
    ren = dff[["renewable_share_pct"]].resample(gen_freq).mean()
    fig2 = px.line(
        ren.reset_index(),
        x="timestamp_utc",
        y="renewable_share_pct",
        labels={"timestamp_utc": "", "renewable_share_pct": "Renewable %"},
    )
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)


def _section_load(dff: pd.DataFrame) -> None:
    st.header("Load & Forecasts")
    load_cols = [c for c in ["load_actual_mw", "load_forecast_mw"] if c in dff.columns]
    fc_cols = [c for c in dff.columns if c.startswith("forecast_") and c.endswith("_mw")]
    if not load_cols and not fc_cols:
        st.info("No load or forecast data available.")
        return

    agg = st.radio(
        "Aggregation",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="eda_load_agg",
        index=2,
    )
    freq = _AGG_FREQ[agg]

    if load_cols:
        ts = dff[load_cols].resample(freq).mean()
        fig = px.line(
            ts.reset_index(),
            x="timestamp_utc",
            y=load_cols,
            labels={"timestamp_utc": "", "value": "MW", "variable": "Series"},
            title=f"Total Load: Actual vs Forecast ({agg})",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    if fc_cols:
        ts = dff[fc_cols].resample(freq).mean()
        fig = px.line(
            ts.reset_index(),
            x="timestamp_utc",
            y=fc_cols,
            labels={"timestamp_utc": "", "value": "MW", "variable": "Source"},
            title=f"Wind & Solar DA Forecast ({agg})",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def _section_neighbours(dff: pd.DataFrame) -> None:
    st.header("Neighbour Prices & Cross-Border Flows")
    nb_avail = {k: v for k, v in NEIGHBOUR_PRICE_COLS.items() if k in dff.columns}
    flow_cols = [c for c in dff.columns if c.startswith("flow_") and c.endswith("_net_import_mw")]

    col_l, col_r = st.columns(2)
    with col_l:
        if nb_avail:
            agg = st.radio(
                "Aggregation",
                ["Daily", "Weekly", "Monthly"],
                horizontal=True,
                key="eda_nb_agg",
                index=2,
            )
            freq = _AGG_FREQ[agg]
            plot_cols = [PRICE_COL] + list(nb_avail)
            labels = {"price_eur_mwh": "DE-LU", **nb_avail}
            ts = dff[plot_cols].resample(freq).mean().rename(columns=labels)
            fig = px.line(
                ts.reset_index(),
                x="timestamp_utc",
                y=list(labels.values()),
                labels={"timestamp_utc": "", "value": "EUR/MWh", "variable": "Zone"},
                title=f"Day-Ahead Prices: DE-LU vs Neighbours ({agg})",
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No neighbour price data available.")

    with col_r:
        if flow_cols:
            agg = st.radio(
                "Aggregation",
                ["Daily", "Weekly", "Monthly"],
                horizontal=True,
                key="eda_flow_agg",
                index=2,
            )
            freq = _AGG_FREQ[agg]
            ts = dff[flow_cols].resample(freq).mean()
            display = {
                c: c.replace("flow_", "").replace("_net_import_mw", "").upper() for c in flow_cols
            }
            ts = ts.rename(columns=display)
            fig = px.line(
                ts.reset_index(),
                x="timestamp_utc",
                y=list(display.values()),
                labels={"timestamp_utc": "", "value": "MW (+ = import)", "variable": "Border"},
                title=f"Net Cross-Border Flows ({agg})",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cross-border flow data available.")


def _section_commodities(dff: pd.DataFrame) -> None:
    st.header("Carbon & Gas Prices")
    cols = [c for c in ["carbon_price_usd", "gas_price_usd"] if c in dff.columns]
    if not cols:
        st.info("No carbon or gas price data available.")
        return
    agg = st.radio(
        "Aggregation",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="eda_comm_agg",
        index=2,
    )
    freq = _AGG_FREQ[agg]
    ts = dff[cols].resample(freq).mean()
    fig = px.line(
        ts.reset_index(),
        x="timestamp_utc",
        y=cols,
        labels={"timestamp_utc": "", "value": "USD", "variable": "Commodity"},
        title=f"Carbon & Gas Prices ({agg})",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def _section_weather(dff: pd.DataFrame) -> None:
    st.header("Weather Overview")
    avail = [c for c in WEATHER_COLS_DISPLAY if c in dff.columns]
    if not avail:
        st.info("No weather data available.")
        return
    var = st.selectbox(
        "Weather variable",
        avail,
        format_func=lambda x: WEATHER_COLS_DISPLAY[x],
        key="eda_weather_var",
    )
    weather_agg = st.radio(
        "Aggregation",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="eda_weather_agg",
        index=2,
    )
    weather_freq = _AGG_FREQ[weather_agg]
    ts = dff[[var]].resample(weather_freq).mean()
    fig = px.line(
        ts.reset_index(),
        x="timestamp_utc",
        y=var,
        labels={"timestamp_utc": "", var: WEATHER_COLS_DISPLAY[var]},
        title=f"{WEATHER_COLS_DISPLAY[var]} ({weather_agg} Mean)",
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def _section_correlations(dff: pd.DataFrame) -> None:
    st.header("Key Correlations with Price")
    gen_cols = [c for c in GEN_COLS_DISPLAY if c in dff.columns]
    weather_cols = [c for c in WEATHER_COLS_DISPLAY if c in dff.columns]
    corr_cols = (
        [
            "renewable_share_pct",
            "total_generation_mw",
            "fossil_generation_mw",
            "renewable_generation_mw",
        ]
        + [c for c in ["load_actual_mw", "carbon_price_usd", "gas_price_usd"] if c in dff.columns]
        + weather_cols
        + gen_cols
    )
    corr_cols = [c for c in corr_cols if c in dff.columns]
    corr = dff[corr_cols + [PRICE_COL]].corr()[PRICE_COL].drop(PRICE_COL).sort_values()

    fig = px.bar(
        x=corr.values,
        y=corr.index,
        orientation="h",
        labels={"x": "Correlation with Price", "y": ""},
        title="Feature Correlations with Day-Ahead Price",
        color=corr.values,
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
    )
    fig.update_layout(height=max(600, len(corr_cols) * 22), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _section_distributions(dff: pd.DataFrame) -> None:
    st.header("Price Distribution & Temporal Patterns")

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.histogram(
            dff,
            x=PRICE_COL,
            nbins=100,
            title="Price Distribution",
            labels={PRICE_COL: "EUR/MWh"},
            marginal="box",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig = px.box(
            dff.reset_index(),
            x="year",
            y=PRICE_COL,
            title="Price by Year",
            labels={"year": "Year", PRICE_COL: "EUR/MWh"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        hp = dff.groupby("hour")[PRICE_COL].agg(["mean", "median", "std"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hp["hour"], y=hp["mean"], name="Mean", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=hp["hour"], y=hp["median"], name="Median", mode="lines+markers"))
        fig.add_trace(
            go.Scatter(
                x=hp["hour"],
                y=hp["mean"] + hp["std"],
                fill=None,
                mode="lines",
                line={"dash": "dash", "color": "rgba(100,100,100,0.3)"},
                name="+1 Std",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hp["hour"],
                y=hp["mean"] - hp["std"],
                fill="tonexty",
                mode="lines",
                line={"dash": "dash", "color": "rgba(100,100,100,0.3)"},
                name="-1 Std",
            )
        )
        fig.update_layout(
            title="Hourly Price Profile",
            xaxis_title="Hour (UTC)",
            yaxis_title="EUR/MWh",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_r2:
        mp = dff.groupby("month")[PRICE_COL].agg(["mean", "median"]).reset_index()
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        mp["month_name"] = mp["month"].map(lambda m: month_names[m - 1])
        fig = px.bar(
            mp,
            x="month_name",
            y=["mean", "median"],
            barmode="group",
            title="Monthly Price Profile",
            labels={"value": "EUR/MWh", "month_name": "Month"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dp = dff.groupby("dayofweek")[PRICE_COL].agg(["mean", "median"]).reset_index()
    dp["day_name"] = dp["dayofweek"].map(lambda d: dow_names[d])
    fig = px.bar(
        dp,
        x="day_name",
        y=["mean", "median"],
        barmode="group",
        title="Day-of-Week Price Profile",
        labels={"value": "EUR/MWh", "day_name": ""},
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def _section_negative(dff: pd.DataFrame) -> None:
    st.header("Negative Price Analysis")
    neg = dff[dff[PRICE_COL] < 0]
    total_hours, neg_hours = len(dff), len(neg)

    c1, c2, c3 = st.columns(3)
    c1.metric("Negative price hours", f"{neg_hours:,}")
    c2.metric("Pct of total", f"{neg_hours / total_hours * 100:.2f}%")
    c3.metric("Most negative", f"{neg[PRICE_COL].min():.1f} EUR/MWh" if neg_hours else "N/A")

    if neg_hours == 0:
        return

    npy = neg.groupby("year").size().reset_index(name="count")
    fig = px.bar(
        npy,
        x="year",
        y="count",
        title="Negative Price Hours per Year",
        labels={"year": "Year", "count": "Hours"},
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    cl, cr = st.columns(2)
    with cl:
        nh = neg.groupby("hour").size().reset_index(name="count")
        fig = px.bar(
            nh,
            x="hour",
            y="count",
            title="Negative Prices by Hour (UTC)",
            labels={"hour": "Hour", "count": "Occurrences"},
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        st.markdown("**Renewable share comparison:**")
        ren_neg = neg["renewable_share_pct"].mean()
        ren_pos = dff[dff[PRICE_COL] >= 0]["renewable_share_pct"].mean()
        comp = pd.DataFrame(
            {"Condition": ["Negative", "Non-negative"], "Avg Renewable %": [ren_neg, ren_pos]}
        )
        fig = px.bar(
            comp,
            x="Condition",
            y="Avg Renewable %",
            title="Avg Renewable Share: Negative vs Non-Negative",
            color="Condition",
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def _section_heatmap(dff: pd.DataFrame) -> None:
    st.header("Correlation Heatmap")
    gen_cols = [c for c in GEN_COLS_DISPLAY if c in dff.columns]
    weather_cols = [c for c in WEATHER_COLS_DISPLAY if c in dff.columns]
    cols = [PRICE_COL] + gen_cols + weather_cols
    for extra in ["load_actual_mw", "carbon_price_usd", "gas_price_usd"]:
        if extra in dff.columns:
            cols.append(extra)
    labels = (
        ["Price"]
        + [GEN_COLS_DISPLAY.get(c, c) for c in gen_cols]
        + [WEATHER_COLS_DISPLAY.get(c, c) for c in weather_cols]
        + [c for c in ["load_actual_mw", "carbon_price_usd", "gas_price_usd"] if c in dff.columns]
    )
    corr = dff[cols].corr()
    fig = px.imshow(
        corr.values,
        x=labels,
        y=labels,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Full Correlation Matrix",
        aspect="auto",
    )
    fig.update_layout(height=800, width=900)
    st.plotly_chart(fig, use_container_width=True)


def _section_scatter(dff: pd.DataFrame) -> None:
    st.header("Renewable Share vs Price")
    sdf = dff[[PRICE_COL, "renewable_share_pct", "year"]].dropna()
    if len(sdf) > 10_000:
        sdf = sdf.sample(10_000, random_state=42)
    fig = px.scatter(
        sdf,
        x="renewable_share_pct",
        y=PRICE_COL,
        color="year",
        opacity=0.4,
        labels={"renewable_share_pct": "Renewable Share [%]", PRICE_COL: "EUR/MWh"},
        title="Day-Ahead Price vs Renewable Share (10k sample)",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Phase 2: Trading-focused section renderers
# ---------------------------------------------------------------------------


def _section_price_changes(dff: pd.DataFrame) -> None:
    """P1: Daily price change distribution — the actual trading signal."""
    st.header("13. Price Change Distribution")

    settlements = compute_daily_settlement(dff[PRICE_COL])
    changes = compute_price_changes(settlements)
    rates = direction_base_rates(changes)

    # --- Metric cards ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Up days", f"{rates['n_up']} ({rates['pct_up']:.1f}%)")
    c2.metric("Down days", f"{rates['n_down']} ({rates['pct_down']:.1f}%)")
    c3.metric("Mean up move", f"{rates['mean_up_move']:.2f} EUR")
    c4.metric("Mean down move", f"{rates['mean_down_move']:.2f} EUR")
    c5.metric("Median change", f"{rates['median_change']:.2f} EUR")

    c6, c7, c8 = st.columns(3)
    c6.metric("Skewness", f"{rates['skewness']:.3f}")
    c7.metric("Kurtosis", f"{rates['kurtosis']:.3f}")
    c8.metric("Total days", f"{rates['n_total']}")

    # --- Histogram ---
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.histogram(
            changes,
            nbins=80,
            title="Daily Price Change Distribution",
            labels={"value": "EUR/MWh", "count": "Days"},
            marginal="box",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Direction by month
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        months = pd.Series(
            changes.index.month.map(lambda m: month_names[m - 1]),
            index=changes.index,
            name="month",
        )
        monthly_dir = direction_by_group(changes, months)
        # Reindex to calendar order
        present = [m for m in month_names if m in monthly_dir.index]
        monthly_dir = monthly_dir.reindex(present)
        fig = px.bar(
            monthly_dir.reset_index(),
            x="group" if "group" in monthly_dir.reset_index().columns else monthly_dir.index.name,
            y="pct_up",
            title="% Up Days by Month",
            labels={"pct_up": "% Up", "month": ""},
        )
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Direction by day of week
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = pd.Series(
        changes.index.dayofweek.map(lambda d: dow_names[d]),
        index=changes.index,
        name="day",
    )
    dow_dir = direction_by_group(changes, dow)
    present_dow = [d for d in dow_names if d in dow_dir.index]
    dow_dir = dow_dir.reindex(present_dow)
    fig = px.bar(
        dow_dir.reset_index(),
        x="group" if "group" in dow_dir.reset_index().columns else dow_dir.index.name,
        y="pct_up",
        title="% Up Days by Day of Week",
        labels={"pct_up": "% Up", "day": ""},
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def _section_autocorrelation(dff: pd.DataFrame) -> None:
    """P2: Autocorrelation and direction persistence of price changes."""
    st.header("14. Autocorrelation & Direction Persistence")

    settlements = compute_daily_settlement(dff[PRICE_COL])
    changes = compute_price_changes(settlements)

    col_l, col_r = st.columns(2)

    with col_l:
        max_lag = min(40, len(changes) // 3)
        if max_lag < 1:
            st.info("Not enough data for autocorrelation analysis.")
            return
        acf = autocorrelation(changes, max_lag=max_lag)
        # Significance band: +/- 1.96/sqrt(n)
        n = len(changes)
        sig = 1.96 / np.sqrt(n)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=acf.index, y=acf.values, name="ACF"))
        fig.add_hline(y=sig, line_dash="dash", line_color="red", annotation_text="95% CI")
        fig.add_hline(y=-sig, line_dash="dash", line_color="red")
        fig.add_hline(y=0, line_color="black", line_width=0.5)
        fig.update_layout(
            title="Autocorrelation of Daily Price Changes",
            xaxis_title="Lag (days)",
            yaxis_title="ACF",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        streaks = compute_direction_streaks(changes)
        st.markdown("**Direction Streaks**")
        s1, s2 = st.columns(2)
        s1.metric("Max up streak", f"{streaks['max_up_streak']} days")
        s2.metric("Max down streak", f"{streaks['max_down_streak']} days")
        s3, s4 = st.columns(2)
        s3.metric("Mean up streak", f"{streaks['mean_up_streak']:.1f} days")
        s4.metric("Mean down streak", f"{streaks['mean_down_streak']:.1f} days")

        # Transition matrix
        directions = np.sign(changes.values)
        transitions = pd.DataFrame(
            index=["After Up", "After Down"],
            columns=["Next Up", "Next Down"],
            dtype=float,
        )
        for prev, label in [(1, "After Up"), (-1, "After Down")]:
            mask = directions[:-1] == prev
            next_dirs = directions[1:][mask]
            if len(next_dirs) > 0:
                transitions.loc[label, "Next Up"] = (next_dirs > 0).mean() * 100
                transitions.loc[label, "Next Down"] = (next_dirs < 0).mean() * 100
            else:
                transitions.loc[label] = [np.nan, np.nan]
        st.markdown("**Transition Probabilities (%)**")
        st.dataframe(transitions.style.format("{:.1f}"), use_container_width=True)


def _section_forecast_errors(dff: pd.DataFrame) -> None:
    """P3: Forecast error analysis for load, wind, solar."""
    st.header("15. Forecast Error Analysis")

    forecast_pairs = [
        ("load_actual_mw", "load_forecast_mw", "Load"),
        ("gen_solar_mw", "forecast_solar_mw", "Solar"),
        ("gen_wind_onshore_mw", "forecast_wind_onshore_mw", "Wind Onshore"),
        ("gen_wind_offshore_mw", "forecast_wind_offshore_mw", "Wind Offshore"),
    ]

    available_pairs = [
        (act, fc, name)
        for act, fc, name in forecast_pairs
        if act in dff.columns and fc in dff.columns
    ]

    if not available_pairs:
        st.info("No forecast/actual column pairs found in the data.")
        return

    selected = st.selectbox(
        "Select forecast type",
        [name for _, _, name in available_pairs],
        key="eda_forecast_error_select",
    )

    act_col, fc_col, _ = next((a, f, n) for a, f, n in available_pairs if n == selected)

    errors = compute_forecast_errors(dff[act_col], dff[fc_col])

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Error", f"{errors['error'].mean():.1f} MW")
    c2.metric("MAE", f"{errors['abs_error'].mean():.1f} MW")
    c3.metric("RMSE", f"{np.sqrt((errors['error'] ** 2).mean()):.1f} MW")
    c4.metric("Mean Bias", "Over" if errors["error"].mean() > 0 else "Under")

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.histogram(
            errors,
            x="error",
            nbins=80,
            title=f"{selected}: Forecast Error Distribution (forecast - actual)",
            labels={"error": "MW"},
            marginal="box",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Error by month
        monthly_error = errors.copy()
        monthly_error["month"] = monthly_error.index.month
        monthly_stats = monthly_error.groupby("month")["error"].agg(["mean", "std"]).reset_index()
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        monthly_stats["month_name"] = monthly_stats["month"].map(lambda m: month_names[m - 1])
        fig = px.bar(
            monthly_stats,
            x="month_name",
            y="mean",
            error_y="std",
            title=f"{selected}: Mean Forecast Error by Month",
            labels={"mean": "Mean Error (MW)", "month_name": ""},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: actual vs forecast
    sample = dff[[act_col, fc_col]].dropna()
    if len(sample) > 10_000:
        sample = sample.sample(10_000, random_state=42)
    fig = px.scatter(
        sample,
        x=act_col,
        y=fc_col,
        opacity=0.3,
        title=f"{selected}: Actual vs Forecast",
        labels={act_col: "Actual (MW)", fc_col: "Forecast (MW)"},
    )
    # Perfect prediction line
    min_val = min(sample[act_col].min(), sample[fc_col].min())
    max_val = max(sample[act_col].max(), sample[fc_col].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line={"dash": "dash", "color": "red"},
            name="Perfect",
        )
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


def _section_feature_importance(dff: pd.DataFrame) -> None:
    """P4: Lagged feature importance for price direction."""
    st.header("16. Feature Importance for Price Direction")

    settlements = compute_daily_settlement(dff[PRICE_COL])
    changes = compute_price_changes(settlements)
    direction = np.sign(changes)
    direction.name = "direction"

    # Build daily feature means (lagged by 1 day, matching challenge data)
    feature_cols = (
        [c for c in GEN_COLS_DISPLAY if c in dff.columns]
        + [c for c in WEATHER_COLS_DISPLAY if c in dff.columns]
        + [c for c in ["load_actual_mw", "carbon_price_usd", "gas_price_usd"] if c in dff.columns]
        + [
            c
            for c in dff.columns
            if c.startswith("price_") and c.endswith("_eur_mwh") and c != PRICE_COL
        ]
        + [c for c in dff.columns if c.startswith("flow_")]
    )
    feature_cols = list(dict.fromkeys(feature_cols))  # deduplicate preserving order

    if not feature_cols:
        st.info("No feature columns available for importance analysis.")
        return

    # Aggregate to daily and lag by 1 day
    daily_features = dff[feature_cols].groupby(dff.index.date).mean()
    daily_features.index = pd.DatetimeIndex(daily_features.index)
    daily_features_lagged = daily_features.shift(1)

    # Rename for readability
    rename_map = {**GEN_COLS_DISPLAY, **WEATHER_COLS_DISPLAY}
    rename_map.update(
        {
            "load_actual_mw": "Load Actual",
            "carbon_price_usd": "Carbon Price",
            "gas_price_usd": "Gas Price",
        }
    )
    for c in daily_features_lagged.columns:
        if c.startswith("price_") and c.endswith("_eur_mwh"):
            zone = c.replace("price_", "").replace("_eur_mwh", "").upper()
            rename_map[c] = f"Price {zone}"
        if c.startswith("flow_"):
            border = c.replace("flow_", "").replace("_net_import_mw", "").upper()
            rename_map[c] = f"Flow {border}"

    daily_features_lagged = daily_features_lagged.rename(columns=rename_map)

    # Align with direction
    common_idx = daily_features_lagged.index.intersection(direction.index)
    if len(common_idx) < 30:
        st.info("Not enough data points for feature importance analysis.")
        return

    features_aligned = daily_features_lagged.loc[common_idx].dropna(axis=1, how="all")
    direction_aligned = direction.loc[common_idx]

    # Drop rows with any NaN
    valid = features_aligned.dropna()
    direction_valid = direction_aligned.loc[valid.index]

    corr = lagged_direction_correlation(valid, direction_valid)

    fig = px.bar(
        x=corr.values,
        y=corr.index,
        orientation="h",
        labels={"x": "Correlation with Next-Day Direction", "y": ""},
        title="D-1 Feature Correlation with D Price Direction (lagged)",
        color=corr.values,
        color_continuous_scale="RdBu_r",
        range_color=[-0.3, 0.3],
    )
    fig.update_layout(
        height=max(500, len(corr) * 22),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Top predictive features"):
        top_n = min(10, len(corr))
        top = corr.head(top_n)
        st.dataframe(
            pd.DataFrame({"Feature": top.index, "Correlation": top.values}).style.format(
                {"Correlation": "{:.4f}"}
            ),
            use_container_width=True,
        )


def _section_volatility_regimes(dff: pd.DataFrame) -> None:
    """P5: Volatility clustering and regime detection."""
    st.header("17. Volatility & Regime Analysis")

    settlements = compute_daily_settlement(dff[PRICE_COL])
    changes = compute_price_changes(settlements)

    window = st.slider(
        "Rolling window (days)",
        min_value=7,
        max_value=90,
        value=30,
        key="eda_vol_window",
    )

    vol = rolling_volatility(changes, window=window)

    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=changes.index,
                y=changes.values,
                mode="lines",
                name="Daily Change",
                line={"color": "steelblue", "width": 0.8},
                opacity=0.6,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=vol.index,
                y=vol.values,
                mode="lines",
                name=f"{window}d Volatility",
                line={"color": "red", "width": 2},
                yaxis="y2",
            )
        )
        fig.update_layout(
            title=f"Price Changes & {window}-Day Rolling Volatility",
            yaxis={"title": "Price Change (EUR/MWh)"},
            yaxis2={
                "title": "Volatility (EUR/MWh)",
                "overlaying": "y",
                "side": "right",
            },
            height=450,
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Regime classification: high-vol vs low-vol
        vol_median = vol.dropna().median()
        vol_clean = vol.dropna()
        is_high_vol = vol_clean > vol_median

        # Direction base rates in each regime
        changes_aligned = changes.loc[vol_clean.index]
        high_vol_changes = changes_aligned[is_high_vol]
        low_vol_changes = changes_aligned[~is_high_vol]

        if len(high_vol_changes) > 0 and len(low_vol_changes) > 0:
            high_rates = direction_base_rates(high_vol_changes)
            low_rates = direction_base_rates(low_vol_changes)

            regime_df = pd.DataFrame(
                {
                    "Metric": [
                        "% Up days",
                        "Mean up move (EUR)",
                        "Mean down move (EUR)",
                        "Std dev (EUR)",
                        "N days",
                    ],
                    "High Volatility": [
                        f"{high_rates['pct_up']:.1f}%",
                        f"{high_rates['mean_up_move']:.2f}",
                        f"{high_rates['mean_down_move']:.2f}",
                        f"{high_vol_changes.std():.2f}",
                        str(high_rates["n_total"]),
                    ],
                    "Low Volatility": [
                        f"{low_rates['pct_up']:.1f}%",
                        f"{low_rates['mean_up_move']:.2f}",
                        f"{low_rates['mean_down_move']:.2f}",
                        f"{low_vol_changes.std():.2f}",
                        str(low_rates["n_total"]),
                    ],
                }
            )
            st.markdown("**Regime Comparison** (split at median volatility)")
            st.dataframe(regime_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data to split into regimes.")

    # Year-over-year volatility
    if "year" not in dff.columns:
        return
    yearly_vol = changes.groupby(changes.index.year).std().reset_index()
    yearly_vol.columns = ["year", "annual_std"]
    fig = px.bar(
        yearly_vol,
        x="year",
        y="annual_std",
        title="Price Change Volatility by Year",
        labels={"year": "Year", "annual_std": "Std Dev (EUR/MWh)"},
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def _section_residual_load(dff: pd.DataFrame) -> None:
    """P6: Residual load (load - renewables) analysis."""
    st.header("18. Residual Load Analysis")

    if "load_actual_mw" not in dff.columns:
        st.info("Load data not available for residual load analysis.")
        return

    ren_cols = [c for c in RENEWABLE_COLS if c in dff.columns]
    if not ren_cols:
        st.info("Renewable generation data not available.")
        return

    load = dff["load_actual_mw"]
    renewable = dff[ren_cols].sum(axis=1)
    residual = compute_residual_load(load, renewable)

    col_l, col_r = st.columns(2)

    with col_l:
        # Residual load vs price scatter (daily)
        daily_residual = residual.groupby(residual.index.date).mean()
        daily_residual.index = pd.DatetimeIndex(daily_residual.index)
        daily_price = dff[PRICE_COL].groupby(dff.index.date).mean()
        daily_price.index = pd.DatetimeIndex(daily_price.index)

        scatter_df = pd.DataFrame(
            {"Residual Load (MW)": daily_residual, "Price (EUR/MWh)": daily_price}
        ).dropna()

        if len(scatter_df) > 5_000:
            scatter_df = scatter_df.sample(5_000, random_state=42)

        fig = px.scatter(
            scatter_df,
            x="Residual Load (MW)",
            y="Price (EUR/MWh)",
            opacity=0.4,
            title="Daily Residual Load vs Price",
            trendline="ols",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Residual load CHANGE vs price CHANGE
        daily_residual_change = daily_residual.diff().dropna()
        settlements = compute_daily_settlement(dff[PRICE_COL])
        price_change = compute_price_changes(settlements)

        common = daily_residual_change.index.intersection(price_change.index)
        if len(common) > 10:
            delta_df = pd.DataFrame(
                {
                    "Residual Load Change (MW)": daily_residual_change.loc[common],
                    "Price Change (EUR/MWh)": price_change.loc[common],
                }
            ).dropna()

            fig = px.scatter(
                delta_df,
                x="Residual Load Change (MW)",
                y="Price Change (EUR/MWh)",
                opacity=0.4,
                title="Change in Residual Load vs Change in Price",
                trendline="ols",
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for residual load change analysis.")

    # Time series
    ts = (
        pd.DataFrame(
            {
                "Residual Load": residual,
                "Load": load,
                "Renewables": renewable,
            }
        )
        .resample("1W")
        .mean()
    )

    fig = px.line(
        ts.reset_index(),
        x="timestamp_utc",
        y=["Residual Load", "Load", "Renewables"],
        labels={"timestamp_utc": "", "value": "MW", "variable": ""},
        title="Weekly Average: Load, Renewables, and Residual Load",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Phase 6: Strategy Feedback Sections (19-24) — use daily challenge data
# ---------------------------------------------------------------------------

_DOW_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def _section_dow_edge_stability(cdf: pd.DataFrame) -> None:
    """Section 19: Day-of-week edge stability across years."""
    st.header("19. Day-of-Week Edge Stability")
    st.caption(
        "Is the Monday-long/Saturday-short edge stable, or is it decaying? "
        "Each cell shows the directional edge (up_rate - baseline) for a day-year pair."
    )

    result = day_of_week_edge_by_year(
        price_changes=cdf["price_change_eur_mwh"],
        dates=cdf["delivery_date"],
    )
    result["dow_name"] = result["dow"].map(_DOW_NAMES)

    # Pivot: rows = day-of-week, columns = year
    pivot = result.pivot_table(index="dow_name", columns="year", values="edge", aggfunc="first")
    # Reorder rows Mon-Sun
    row_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = pivot.reindex([d for d in row_order if d in pivot.index])

    # Heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=list(pivot.index),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot.values * 100, 1),
            texttemplate="%{text}%",
            colorbar=dict(title="Edge"),
        )
    )
    fig.update_layout(
        title="Day-of-Week Directional Edge by Year",
        xaxis_title="Year",
        yaxis_title="Day of Week",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    with st.expander("Raw edge data"):
        display = result[["year", "dow_name", "up_rate", "overall_up_rate", "edge"]].copy()
        display["edge_pct"] = (display["edge"] * 100).round(1)
        display["up_rate_pct"] = (display["up_rate"] * 100).round(1)
        st.dataframe(display, use_container_width=True)


def _section_feature_drift(cdf: pd.DataFrame) -> None:
    """Section 20: Feature distribution drift between train and validation."""
    st.header("20. Feature Drift (Train vs Validation)")
    st.caption(
        "How much do feature distributions shift between training (2019-2023) and "
        "validation (2024)? Large shifts can break threshold-based strategies."
    )

    train = cdf[cdf["split"] == "train"] if "split" in cdf.columns else cdf[cdf["year"] <= 2023]
    val = cdf[cdf["split"] == "validation"] if "split" in cdf.columns else cdf[cdf["year"] == 2024]

    if train.empty or val.empty:
        st.info("Need both train and validation data for drift analysis.")
        return

    # Exclude label/metadata columns
    exclude = {
        "delivery_date",
        "split",
        "year",
        "settlement_price",
        "price_change_eur_mwh",
        "target_direction",
        "pnl_long_eur",
        "pnl_short_eur",
        "last_settlement_price",
    }
    feat_cols = [
        c for c in train.columns if c not in exclude and train[c].dtype in ("float64", "int64")
    ]
    drift = feature_drift(train[feat_cols], val[feat_cols])

    # Sort by absolute shift
    drift_sorted = drift.sort_values("shift_pct", key=abs, ascending=False)

    # Bar chart of top shifts
    top_n = min(15, len(drift_sorted))
    top = drift_sorted.head(top_n).reset_index()
    fig = px.bar(
        top,
        x="feature",
        y="shift_pct",
        color="shift_pct",
        color_continuous_scale="RdYlBu_r",
        color_continuous_midpoint=0,
        title=f"Top {top_n} Feature Shifts (Train -> Validation, % change in mean)",
        labels={"shift_pct": "Shift %", "feature": ""},
    )
    fig.update_layout(height=450, xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Full table
    with st.expander("Full drift table"):
        styled = drift_sorted.style.format(
            {
                "train_mean": "{:.2f}",
                "val_mean": "{:.2f}",
                "shift_pct": "{:+.1f}%",
                "train_std": "{:.2f}",
                "val_std": "{:.2f}",
                "std_ratio": "{:.2f}",
            }
        )
        st.dataframe(styled, use_container_width=True)


def _section_quarterly_patterns(cdf: pd.DataFrame) -> None:
    """Section 21: Quarterly direction rates and volatility."""
    st.header("21. Quarterly Direction Rates")
    st.caption(
        "Is Q4 consistently best, or is the 2024 Q4 outperformance a volatility artefact? "
        "Shows up-rate and mean |price change| by year-quarter."
    )

    qdr = quarterly_direction_rates(
        price_changes=cdf["price_change_eur_mwh"],
        dates=cdf["delivery_date"],
    )

    col_l, col_r = st.columns(2)

    with col_l:
        # Heatmap: up_rate by year x quarter
        pivot_up = qdr.pivot_table(index="year", columns="quarter", values="up_rate")
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_up.values,
                x=[f"Q{c}" for c in pivot_up.columns],
                y=[str(r) for r in pivot_up.index],
                colorscale="RdYlGn",
                zmid=0.5,
                text=np.round(pivot_up.values * 100, 1),
                texttemplate="%{text}%",
                colorbar=dict(title="Up Rate"),
            )
        )
        fig.update_layout(title="Up Rate by Year-Quarter", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Heatmap: mean_abs_change by year x quarter
        pivot_vol = qdr.pivot_table(index="year", columns="quarter", values="mean_abs_change")
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_vol.values,
                x=[f"Q{c}" for c in pivot_vol.columns],
                y=[str(r) for r in pivot_vol.index],
                colorscale="YlOrRd",
                text=np.round(pivot_vol.values, 1),
                texttemplate="%{text}",
                colorbar=dict(title="EUR/MWh"),
            )
        )
        fig.update_layout(title="Mean |Price Change| by Year-Quarter", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Quarterly data table"):
        display = qdr.copy()
        display["up_rate_pct"] = (display["up_rate"] * 100).round(1)
        display["mean_abs_change"] = display["mean_abs_change"].round(2)
        st.dataframe(display, use_container_width=True)


def _section_volatility_regime_performance(cdf: pd.DataFrame) -> None:
    """Section 22: Direction stats conditioned on volatility regime."""
    st.header("22. Volatility Regime Performance")
    st.caption(
        "Do high-volatility regimes favour longs or shorts? "
        "Shows direction stats within low/mid/high 30-day rolling-volatility regimes."
    )

    vrp = volatility_regime_performance(
        price_changes=cdf["price_change_eur_mwh"],
        window=30,
        n_regimes=3,
    )

    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.bar(
            vrp,
            x="regime",
            y="up_rate",
            color="regime",
            color_discrete_map={"low": "#2ecc71", "mid": "#f1c40f", "high": "#e74c3c"},
            title="Up Rate by Volatility Regime",
            labels={"up_rate": "Up Rate", "regime": "Regime"},
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.bar(
            vrp,
            x="regime",
            y="mean_abs_change",
            color="regime",
            color_discrete_map={"low": "#2ecc71", "mid": "#f1c40f", "high": "#e74c3c"},
            title="Mean |Price Change| by Regime",
            labels={"mean_abs_change": "EUR/MWh", "regime": "Regime"},
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        vrp.style.format(
            {"up_rate": "{:.1%}", "mean_abs_change": "{:.2f}", "mean_change": "{:+.2f}"}
        ),
        use_container_width=True,
    )


def _section_wind_quintile(cdf: pd.DataFrame) -> None:
    """Section 23: Wind quintile direction analysis."""
    st.header("23. Wind Quintile Analysis")
    st.caption(
        "The wind signal is the second-most reliable after Monday. "
        "Low wind -> up (62%), High wind -> down (68%). Is this stable?"
    )

    # Compute combined wind (use forecast columns if available, else generation)
    wind_cols: list[str] = []
    for col in ("forecast_wind_offshore_mw_mean", "forecast_wind_onshore_mw_mean"):
        if col in cdf.columns:
            wind_cols.append(col)
    if not wind_cols:
        for col in ("gen_wind_offshore_mw_mean", "gen_wind_onshore_mw_mean"):
            if col in cdf.columns:
                wind_cols.append(col)

    if not wind_cols or "target_direction" not in cdf.columns:
        st.info("Wind or direction columns not available.")
        return

    combined_wind = cdf[wind_cols].sum(axis=1)
    wqa = wind_quintile_analysis(
        combined_wind=combined_wind,
        direction=cdf["target_direction"],
        price_changes=cdf["price_change_eur_mwh"],
    )

    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.bar(
            wqa,
            x="wind_bin",
            y="up_rate",
            color="up_rate",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0.5,
            title="Up Rate by Wind Power Quintile",
            labels={"up_rate": "Up Rate", "wind_bin": "Wind Quintile (Q1=Low)"},
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.bar(
            wqa,
            x="wind_bin",
            y="mean_price_change",
            color="mean_price_change",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title="Mean Price Change by Wind Quintile",
            labels={"mean_price_change": "EUR/MWh", "wind_bin": "Wind Quintile (Q1=Low)"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Year-by-year stability
    st.subheader("Wind Quintile Stability by Year")
    years = sorted(cdf["year"].unique())
    year_data: list[pd.DataFrame] = []
    for yr in years:
        yr_df = cdf[cdf["year"] == yr]
        if len(yr_df) < 30:
            continue
        yr_wind = yr_df[wind_cols].sum(axis=1)
        try:
            yr_wqa = wind_quintile_analysis(
                combined_wind=yr_wind,
                direction=yr_df["target_direction"],
                price_changes=yr_df["price_change_eur_mwh"],
            )
            yr_wqa["year"] = yr
            year_data.append(yr_wqa)
        except Exception:
            continue

    if year_data:
        all_years = pd.concat(year_data)
        pivot = all_years.pivot_table(index="wind_bin", columns="year", values="up_rate")
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=list(pivot.index),
                colorscale="RdYlGn",
                zmid=0.5,
                text=np.round(pivot.values * 100, 1),
                texttemplate="%{text}%",
                colorbar=dict(title="Up Rate"),
            )
        )
        fig.update_layout(
            title="Wind Quintile Up Rate by Year",
            xaxis_title="Year",
            yaxis_title="Wind Quintile",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Wind quintile data"):
        st.dataframe(wqa, use_container_width=True)


def _section_strategy_correlation_insights(cdf: pd.DataFrame) -> None:
    """Section 24: Composite signal decomposition and strategy insights."""
    st.header("24. Strategy Correlation Insights")
    st.caption(
        "Which features drive direction on ambiguous days (Wed/Thu)? "
        "And how do key signals correlate with each other?"
    )

    # Feature-direction correlations for Wed/Thu vs all days
    feature_cols = [
        c
        for c in cdf.columns
        if c
        not in {
            "delivery_date",
            "split",
            "year",
            "settlement_price",
            "price_change_eur_mwh",
            "target_direction",
            "pnl_long_eur",
            "pnl_short_eur",
            "last_settlement_price",
        }
        and cdf[c].dtype in ("float64", "int64")
    ]

    if "target_direction" not in cdf.columns or not feature_cols:
        st.info("Required columns not available for correlation analysis.")
        return

    dow = cdf["delivery_date"].dt.dayofweek
    wed_thu = cdf[dow.isin([2, 3])]
    all_corr = cdf[feature_cols].corrwith(cdf["target_direction"]).rename("all_days")
    wt_corr = wed_thu[feature_cols].corrwith(wed_thu["target_direction"]).rename("wed_thu")

    corr_compare = pd.concat([all_corr, wt_corr], axis=1).dropna()
    corr_compare["diff"] = corr_compare["wed_thu"] - corr_compare["all_days"]
    corr_compare = corr_compare.sort_values("wed_thu", key=abs, ascending=False)

    col_l, col_r = st.columns(2)

    with col_l:
        top_wt = corr_compare.head(10).reset_index()
        top_wt = top_wt.rename(columns={"index": "feature"})
        fig = px.bar(
            top_wt,
            x="feature",
            y=["all_days", "wed_thu"],
            barmode="group",
            title="Top Feature-Direction Correlations: All Days vs Wed/Thu",
            labels={"value": "Correlation", "feature": ""},
        )
        fig.update_layout(height=400, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Key signal correlations (between features themselves)
        key_signals = [
            "load_forecast_mw_mean",
            "forecast_wind_offshore_mw_mean",
            "forecast_wind_onshore_mw_mean",
            "gen_fossil_gas_mw_mean",
            "gen_wind_onshore_mw_mean",
            "forecast_solar_mw_mean",
        ]
        available = [c for c in key_signals if c in cdf.columns]
        if len(available) >= 2:
            sig_corr = cdf[available].corr()
            # Short names for display
            short_names = {c: c.replace("_mw_mean", "").replace("_mean", "") for c in available}
            sig_corr = sig_corr.rename(index=short_names, columns=short_names)

            fig = go.Figure(
                data=go.Heatmap(
                    z=sig_corr.values,
                    x=list(sig_corr.columns),
                    y=list(sig_corr.index),
                    colorscale="RdBu_r",
                    zmid=0,
                    text=np.round(sig_corr.values, 2),
                    texttemplate="%{text}",
                    colorbar=dict(title="Corr"),
                )
            )
            fig.update_layout(
                title="Key Signal Inter-Correlations",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full correlation comparison"):
        st.dataframe(
            corr_compare.style.format("{:.3f}"),
            use_container_width=True,
        )
