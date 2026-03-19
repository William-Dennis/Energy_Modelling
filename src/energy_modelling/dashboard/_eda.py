"""Tab 1 -- Exploratory Data Analysis.

Renders 12 EDA sections (price time-series, generation mix, load,
neighbour prices, weather, correlations, distributions, negative-price
analysis, heatmap, scatter) from the processed Parquet dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/processed/dataset_de_lu.parquet")
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
        "Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True, key="eda_gen_agg"
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
        "Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True, key="eda_load_agg"
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
                "Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True, key="eda_nb_agg"
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
                "Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True, key="eda_flow_agg"
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
        "Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True, key="eda_comm_agg"
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
    ts = dff[[var]].resample("1D").mean()
    fig = px.line(
        ts.reset_index(),
        x="timestamp_utc",
        y=var,
        labels={"timestamp_utc": "", var: WEATHER_COLS_DISPLAY[var]},
        title=f"{WEATHER_COLS_DISPLAY[var]} (Daily Mean)",
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
