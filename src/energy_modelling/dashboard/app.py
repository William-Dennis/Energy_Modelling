"""Streamlit EDA dashboard for the DE-LU energy market dataset.

Launch with:
    uv run streamlit run src/energy_modelling/dashboard/app.py

Sections
--------
1. Dataset overview & descriptive statistics
2. Price time-series (daily, weekly, monthly aggregation)
3. Generation mix (stacked area chart)
4. Load & forecasts
5. Neighbour prices & cross-border flows
6. Carbon & gas prices
7. Weather overview
8. Price-feature correlations
9. Price distribution & hourly/monthly patterns
10. Negative price analysis
11. Correlation heatmap
12. Renewable share vs price scatter
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

# Column groupings — all now with unit suffixes
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

# Neighbour zone price columns
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

# Colour scheme (roughly matches conventional energy colours)
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the processed Parquet and add derived time columns."""
    df = pd.read_parquet(DATA_PATH)
    df["hour"] = df.index.hour  # type: ignore[union-attr]
    df["dayofweek"] = df.index.dayofweek  # type: ignore[union-attr]
    df["month"] = df.index.month  # type: ignore[union-attr]
    df["year"] = df.index.year  # type: ignore[union-attr]
    df["date"] = df.index.date  # type: ignore[union-attr]

    # Derived features
    gen_cols = [c for c in df.columns if c.startswith("gen_") and c.endswith("_mw")]
    df["total_generation_mw"] = df[gen_cols].sum(axis=1)
    df["renewable_generation_mw"] = df[[c for c in RENEWABLE_COLS if c in df.columns]].sum(axis=1)
    df["fossil_generation_mw"] = df[[c for c in FOSSIL_COLS if c in df.columns]].sum(axis=1)
    df["renewable_share_pct"] = (
        df["renewable_generation_mw"] / df["total_generation_mw"] * 100
    ).clip(0, 100)
    return df


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DE-LU Energy Market EDA",
    page_icon="",
    layout="wide",
)

st.title("DE-LU Day-Ahead Electricity Market -- Exploratory Data Analysis")
st.markdown(
    "**Dataset:** ENTSO-E (prices, generation, load, forecasts, neighbours, flows, NTC) "
    "+ Open-Meteo/ERA5 (weather) + Yahoo Finance (carbon, gas) for the "
    "Germany-Luxembourg bidding zone, 2019-2025. All timestamps UTC."
)

if not DATA_PATH.exists():
    st.error(
        f"Dataset not found at `{DATA_PATH}`. "
        "Run `uv run collect-data --step all --kaggle --years 2019 ... --years 2025` first."
    )
    st.stop()

df = load_data()

# ===== Sidebar filters =====
st.sidebar.header("Filters")
years = sorted(df["year"].unique())
year_range = st.sidebar.slider(
    "Year range",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=(int(min(years)), int(max(years))),
)
mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
dff = df[mask]

# ======================================================================
# 1. DATASET OVERVIEW
# ======================================================================
st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(dff):,}")
col2.metric("Columns", f"{len(dff.columns)}")
col3.metric(
    "Date range",
    f"{dff.index.min().strftime('%Y-%m-%d')} to {dff.index.max().strftime('%Y-%m-%d')}",
)
col4.metric("Hourly frequency", "1h")

with st.expander("Descriptive statistics"):
    show_cols = [PRICE_COL] + list(GEN_COLS_DISPLAY.keys()) + list(WEATHER_COLS_DISPLAY.keys())
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
                {
                    "Missing": missing,
                    "Pct": (missing / len(dff) * 100).round(2),
                }
            ).sort_values("Pct", ascending=False)
        )

# ======================================================================
# 2. PRICE TIME-SERIES
# ======================================================================
st.header("2. Day-Ahead Price Time Series")

agg_option = st.radio(
    "Aggregation",
    ["Hourly (raw)", "Daily mean", "Weekly mean", "Monthly mean"],
    horizontal=True,
    key="price_agg",
)

if agg_option == "Hourly (raw)":
    price_ts = dff[[PRICE_COL]]
elif agg_option == "Daily mean":
    price_ts = dff[[PRICE_COL]].resample("1D").mean()
elif agg_option == "Weekly mean":
    price_ts = dff[[PRICE_COL]].resample("1W").mean()
else:
    price_ts = dff[[PRICE_COL]].resample("1ME").mean()

fig_price = px.line(
    price_ts.reset_index(),
    x="timestamp_utc",
    y=PRICE_COL,
    labels={"timestamp_utc": "", PRICE_COL: "EUR/MWh"},
    title=f"Day-Ahead Price ({agg_option})",
)
fig_price.update_layout(height=400)
st.plotly_chart(fig_price, use_container_width=True)

# Key stats
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Mean", f"{dff[PRICE_COL].mean():.1f} EUR/MWh")
col2.metric("Median", f"{dff[PRICE_COL].median():.1f} EUR/MWh")
col3.metric("Std Dev", f"{dff[PRICE_COL].std():.1f}")
col4.metric("Min", f"{dff[PRICE_COL].min():.1f} EUR/MWh")
col5.metric("Max", f"{dff[PRICE_COL].max():.1f} EUR/MWh")

# ======================================================================
# 3. GENERATION MIX
# ======================================================================
st.header("3. Generation Mix")

gen_agg = st.radio(
    "Aggregation",
    ["Daily", "Weekly", "Monthly"],
    horizontal=True,
    key="gen_agg",
)
gen_freq = {"Daily": "1D", "Weekly": "1W", "Monthly": "1ME"}[gen_agg]

gen_cols_available = [c for c in GEN_COLS_DISPLAY if c in dff.columns]
gen_resampled = dff[gen_cols_available].resample(gen_freq).mean()
gen_resampled = gen_resampled.rename(columns=GEN_COLS_DISPLAY)

# Stacked area chart
fig_gen = go.Figure()
display_order = [
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
for fuel in display_order:
    if fuel in gen_resampled.columns:
        fig_gen.add_trace(
            go.Scatter(
                x=gen_resampled.index,
                y=gen_resampled[fuel],
                name=fuel,
                mode="lines",
                stackgroup="one",
                line={"width": 0},
                fillcolor=GEN_COLORS.get(fuel, "#888888"),
            )
        )
fig_gen.update_layout(
    title=f"Generation Mix ({gen_agg} Average, MW)",
    yaxis_title="MW",
    height=500,
    legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
)
st.plotly_chart(fig_gen, use_container_width=True)

# Renewable share over time
st.subheader("Renewable Share Over Time")
ren_share_ts = dff[["renewable_share_pct"]].resample(gen_freq).mean()
fig_ren = px.line(
    ren_share_ts.reset_index(),
    x="timestamp_utc",
    y="renewable_share_pct",
    labels={"timestamp_utc": "", "renewable_share_pct": "Renewable %"},
)
fig_ren.update_layout(height=300)
st.plotly_chart(fig_ren, use_container_width=True)

# ======================================================================
# 4. LOAD & FORECASTS
# ======================================================================
st.header("4. Load & Forecasts")

load_cols_available = [c for c in ["load_actual_mw", "load_forecast_mw"] if c in dff.columns]
forecast_cols_available = [
    c for c in dff.columns if c.startswith("forecast_") and c.endswith("_mw")
]

if load_cols_available or forecast_cols_available:
    load_agg = st.radio(
        "Aggregation",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="load_agg",
    )
    load_freq = {"Daily": "1D", "Weekly": "1W", "Monthly": "1ME"}[load_agg]

    if load_cols_available:
        load_ts = dff[load_cols_available].resample(load_freq).mean()
        fig_load = px.line(
            load_ts.reset_index(),
            x="timestamp_utc",
            y=load_cols_available,
            labels={"timestamp_utc": "", "value": "MW", "variable": "Series"},
            title=f"Total Load: Actual vs Forecast ({load_agg})",
        )
        fig_load.update_layout(height=400)
        st.plotly_chart(fig_load, use_container_width=True)

    if forecast_cols_available:
        fc_ts = dff[forecast_cols_available].resample(load_freq).mean()
        fig_fc = px.line(
            fc_ts.reset_index(),
            x="timestamp_utc",
            y=forecast_cols_available,
            labels={"timestamp_utc": "", "value": "MW", "variable": "Source"},
            title=f"Wind & Solar DA Forecast ({load_agg})",
        )
        fig_fc.update_layout(height=400)
        st.plotly_chart(fig_fc, use_container_width=True)
else:
    st.info("No load or forecast data available. Run the full pipeline first.")

# ======================================================================
# 5. NEIGHBOUR PRICES & CROSS-BORDER FLOWS
# ======================================================================
st.header("5. Neighbour Prices & Cross-Border Flows")

neighbour_cols_available = {k: v for k, v in NEIGHBOUR_PRICE_COLS.items() if k in dff.columns}
flow_net_cols = [c for c in dff.columns if c.startswith("flow_") and c.endswith("_net_import_mw")]

col_left, col_right = st.columns(2)

with col_left:
    if neighbour_cols_available:
        nb_agg = st.radio(
            "Aggregation",
            ["Daily", "Weekly", "Monthly"],
            horizontal=True,
            key="nb_agg",
        )
        nb_freq = {"Daily": "1D", "Weekly": "1W", "Monthly": "1ME"}[nb_agg]
        # Plot DE_LU price alongside neighbours
        nb_plot_cols = [PRICE_COL] + list(neighbour_cols_available.keys())
        nb_labels = {"price_eur_mwh": "DE-LU"}
        nb_labels.update(neighbour_cols_available)
        nb_ts = dff[nb_plot_cols].resample(nb_freq).mean()
        nb_ts = nb_ts.rename(columns=nb_labels)
        fig_nb = px.line(
            nb_ts.reset_index(),
            x="timestamp_utc",
            y=list(nb_labels.values()),
            labels={"timestamp_utc": "", "value": "EUR/MWh", "variable": "Zone"},
            title=f"Day-Ahead Prices: DE-LU vs Neighbours ({nb_agg})",
        )
        fig_nb.update_layout(height=450)
        st.plotly_chart(fig_nb, use_container_width=True)
    else:
        st.info("No neighbour price data available.")

with col_right:
    if flow_net_cols:
        flow_agg = st.radio(
            "Aggregation",
            ["Daily", "Weekly", "Monthly"],
            horizontal=True,
            key="flow_agg",
        )
        flow_freq = {"Daily": "1D", "Weekly": "1W", "Monthly": "1ME"}[flow_agg]
        flow_ts = dff[flow_net_cols].resample(flow_freq).mean()
        # Clean column names for display
        flow_display = {
            c: c.replace("flow_", "").replace("_net_import_mw", "").upper() for c in flow_net_cols
        }
        flow_ts = flow_ts.rename(columns=flow_display)
        fig_flow = px.line(
            flow_ts.reset_index(),
            x="timestamp_utc",
            y=list(flow_display.values()),
            labels={
                "timestamp_utc": "",
                "value": "MW (+ = import)",
                "variable": "Border",
            },
            title=f"Net Cross-Border Flows ({flow_agg})",
        )
        fig_flow.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_flow.update_layout(height=450)
        st.plotly_chart(fig_flow, use_container_width=True)
    else:
        st.info("No cross-border flow data available.")

# ======================================================================
# 6. CARBON & GAS PRICES
# ======================================================================
st.header("6. Carbon & Gas Prices")

commodity_cols = [c for c in ["carbon_price_usd", "gas_price_usd"] if c in dff.columns]
if commodity_cols:
    comm_agg = st.radio(
        "Aggregation",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="comm_agg",
    )
    comm_freq = {"Daily": "1D", "Weekly": "1W", "Monthly": "1ME"}[comm_agg]
    comm_ts = dff[commodity_cols].resample(comm_freq).mean()
    fig_comm = px.line(
        comm_ts.reset_index(),
        x="timestamp_utc",
        y=commodity_cols,
        labels={"timestamp_utc": "", "value": "USD", "variable": "Commodity"},
        title=f"Carbon & Gas Prices ({comm_agg})",
    )
    fig_comm.update_layout(height=400)
    st.plotly_chart(fig_comm, use_container_width=True)
else:
    st.info("No carbon or gas price data available. Run the full pipeline first.")

# ======================================================================
# 7. WEATHER OVERVIEW
# ======================================================================
st.header("7. Weather Overview")

weather_cols_available = [c for c in WEATHER_COLS_DISPLAY if c in dff.columns]
if weather_cols_available:
    weather_var = st.selectbox(
        "Weather variable",
        weather_cols_available,
        format_func=lambda x: WEATHER_COLS_DISPLAY[x],
    )

    weather_daily = dff[[weather_var]].resample("1D").mean()
    fig_weather = px.line(
        weather_daily.reset_index(),
        x="timestamp_utc",
        y=weather_var,
        labels={
            "timestamp_utc": "",
            weather_var: WEATHER_COLS_DISPLAY[weather_var],
        },
        title=f"{WEATHER_COLS_DISPLAY[weather_var]} (Daily Mean)",
    )
    fig_weather.update_layout(height=350)
    st.plotly_chart(fig_weather, use_container_width=True)
else:
    st.info("No weather data available.")

# ======================================================================
# 8. PRICE-FEATURE CORRELATIONS
# ======================================================================
st.header("8. Key Correlations with Price")

corr_cols = (
    [
        "renewable_share_pct",
        "total_generation_mw",
        "fossil_generation_mw",
        "renewable_generation_mw",
    ]
    + [c for c in ["load_actual_mw", "carbon_price_usd", "gas_price_usd"] if c in dff.columns]
    + weather_cols_available
    + gen_cols_available
)
corr_cols = [c for c in corr_cols if c in dff.columns]
corr_with_price = dff[corr_cols + [PRICE_COL]].corr()[PRICE_COL].drop(PRICE_COL).sort_values()

fig_corr = px.bar(
    x=corr_with_price.values,
    y=corr_with_price.index,
    orientation="h",
    labels={"x": "Correlation with Price", "y": ""},
    title="Feature Correlations with Day-Ahead Price",
    color=corr_with_price.values,
    color_continuous_scale="RdBu_r",
    range_color=[-1, 1],
)
fig_corr.update_layout(height=max(600, len(corr_cols) * 22), showlegend=False)
st.plotly_chart(fig_corr, use_container_width=True)

# ======================================================================
# 9. PRICE DISTRIBUTION & PATTERNS
# ======================================================================
st.header("9. Price Distribution & Temporal Patterns")

col_left, col_right = st.columns(2)

with col_left:
    fig_dist = px.histogram(
        dff,
        x=PRICE_COL,
        nbins=100,
        title="Price Distribution",
        labels={PRICE_COL: "EUR/MWh"},
        marginal="box",
    )
    fig_dist.update_layout(height=400)
    st.plotly_chart(fig_dist, use_container_width=True)

with col_right:
    fig_yoy = px.box(
        dff.reset_index(),
        x="year",
        y=PRICE_COL,
        title="Price by Year",
        labels={"year": "Year", PRICE_COL: "EUR/MWh"},
    )
    fig_yoy.update_layout(height=400)
    st.plotly_chart(fig_yoy, use_container_width=True)

col_left2, col_right2 = st.columns(2)

with col_left2:
    hourly_profile = dff.groupby("hour")[PRICE_COL].agg(["mean", "median", "std"]).reset_index()
    fig_hourly = go.Figure()
    fig_hourly.add_trace(
        go.Scatter(
            x=hourly_profile["hour"],
            y=hourly_profile["mean"],
            name="Mean",
            mode="lines+markers",
        )
    )
    fig_hourly.add_trace(
        go.Scatter(
            x=hourly_profile["hour"],
            y=hourly_profile["median"],
            name="Median",
            mode="lines+markers",
        )
    )
    fig_hourly.add_trace(
        go.Scatter(
            x=hourly_profile["hour"],
            y=hourly_profile["mean"] + hourly_profile["std"],
            fill=None,
            mode="lines",
            line={"dash": "dash", "color": "rgba(100,100,100,0.3)"},
            name="+1 Std",
        )
    )
    fig_hourly.add_trace(
        go.Scatter(
            x=hourly_profile["hour"],
            y=hourly_profile["mean"] - hourly_profile["std"],
            fill="tonexty",
            mode="lines",
            line={"dash": "dash", "color": "rgba(100,100,100,0.3)"},
            name="-1 Std",
        )
    )
    fig_hourly.update_layout(
        title="Hourly Price Profile",
        xaxis_title="Hour of Day (UTC)",
        yaxis_title="EUR/MWh",
        height=400,
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

with col_right2:
    monthly_profile = dff.groupby("month")[PRICE_COL].agg(["mean", "median"]).reset_index()
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
    monthly_profile["month_name"] = monthly_profile["month"].map(lambda m: month_names[m - 1])
    fig_monthly = px.bar(
        monthly_profile,
        x="month_name",
        y=["mean", "median"],
        barmode="group",
        title="Monthly Price Profile",
        labels={"value": "EUR/MWh", "month_name": "Month"},
    )
    fig_monthly.update_layout(height=400)
    st.plotly_chart(fig_monthly, use_container_width=True)

# Day of week pattern
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_profile = dff.groupby("dayofweek")[PRICE_COL].agg(["mean", "median"]).reset_index()
dow_profile["day_name"] = dow_profile["dayofweek"].map(lambda d: dow_names[d])
fig_dow = px.bar(
    dow_profile,
    x="day_name",
    y=["mean", "median"],
    barmode="group",
    title="Day-of-Week Price Profile",
    labels={"value": "EUR/MWh", "day_name": ""},
)
fig_dow.update_layout(height=350)
st.plotly_chart(fig_dow, use_container_width=True)

# ======================================================================
# 10. NEGATIVE PRICE ANALYSIS
# ======================================================================
st.header("10. Negative Price Analysis")

neg = dff[dff[PRICE_COL] < 0]
total_hours = len(dff)
neg_hours = len(neg)

col1, col2, col3 = st.columns(3)
col1.metric("Negative price hours", f"{neg_hours:,}")
col2.metric("Pct of total", f"{neg_hours / total_hours * 100:.2f}%")
col3.metric(
    "Most negative",
    f"{neg[PRICE_COL].min():.1f} EUR/MWh" if neg_hours > 0 else "N/A",
)

if neg_hours > 0:
    neg_per_year = neg.groupby("year").size().reset_index(name="count")
    fig_neg_year = px.bar(
        neg_per_year,
        x="year",
        y="count",
        title="Negative Price Hours per Year",
        labels={"year": "Year", "count": "Hours"},
    )
    fig_neg_year.update_layout(height=300)
    st.plotly_chart(fig_neg_year, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        neg_hourly = neg.groupby("hour").size().reset_index(name="count")
        fig_neg_hour = px.bar(
            neg_hourly,
            x="hour",
            y="count",
            title="Negative Prices by Hour of Day (UTC)",
            labels={"hour": "Hour", "count": "Occurrences"},
        )
        fig_neg_hour.update_layout(height=300)
        st.plotly_chart(fig_neg_hour, use_container_width=True)

    with col_r:
        st.markdown("**Renewable share comparison:**")
        ren_neg = neg["renewable_share_pct"].mean()
        ren_pos = dff[dff[PRICE_COL] >= 0]["renewable_share_pct"].mean()
        comp_df = pd.DataFrame(
            {
                "Condition": ["Negative prices", "Non-negative prices"],
                "Avg Renewable %": [ren_neg, ren_pos],
            }
        )
        fig_comp = px.bar(
            comp_df,
            x="Condition",
            y="Avg Renewable %",
            title="Avg Renewable Share: Negative vs Non-Negative Price Hours",
            color="Condition",
        )
        fig_comp.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_comp, use_container_width=True)

# ======================================================================
# 11. CORRELATION HEATMAP
# ======================================================================
st.header("11. Correlation Heatmap")

heatmap_cols = [PRICE_COL] + gen_cols_available + weather_cols_available
# Add load/carbon/gas if present
for extra in ["load_actual_mw", "carbon_price_usd", "gas_price_usd"]:
    if extra in dff.columns:
        heatmap_cols.append(extra)
heatmap_labels = (
    ["Price"]
    + [GEN_COLS_DISPLAY.get(c, c) for c in gen_cols_available]
    + [WEATHER_COLS_DISPLAY.get(c, c) for c in weather_cols_available]
    + [c for c in ["load_actual_mw", "carbon_price_usd", "gas_price_usd"] if c in dff.columns]
)
corr_matrix = dff[heatmap_cols].corr()

fig_hm = px.imshow(
    corr_matrix.values,
    x=heatmap_labels,
    y=heatmap_labels,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Full Correlation Matrix",
    aspect="auto",
)
fig_hm.update_layout(height=800, width=900)
st.plotly_chart(fig_hm, use_container_width=True)

# ======================================================================
# 12. SCATTER: RENEWABLE SHARE vs PRICE
# ======================================================================
st.header("12. Renewable Share vs Price")

scatter_df = dff[[PRICE_COL, "renewable_share_pct", "year"]].dropna()
if len(scatter_df) > 10000:
    scatter_df = scatter_df.sample(10000, random_state=42)

fig_scatter = px.scatter(
    scatter_df,
    x="renewable_share_pct",
    y=PRICE_COL,
    color="year",
    opacity=0.4,
    labels={"renewable_share_pct": "Renewable Share [%]", PRICE_COL: "EUR/MWh"},
    title="Day-Ahead Price vs Renewable Share (10k sample)",
)
fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, use_container_width=True)


# ======================================================================
# FOOTER
# ======================================================================
st.divider()
st.markdown(
    """
    **Data sources:** ENTSO-E Transparency Platform (prices, generation, load,
    forecasts, neighbour prices, cross-border flows, NTC),
    Open-Meteo Archive API / ERA5 (weather),
    Yahoo Finance (carbon EUA proxy, gas price proxy).

    **License:** CC-BY-4.0 | **Bidding zone:** DE-LU (Germany/Luxembourg)
    """
)
