"""Streamlit EDA dashboard for the DE-LU energy market dataset.

Launch with:
    uv run streamlit run src/energy_modelling/dashboard/app.py

Sections
--------
1. Dataset overview & descriptive statistics
2. Price time-series (daily, weekly, monthly aggregation)
3. Generation mix (stacked area chart)
4. Weather overview
5. Price-feature correlations
6. Price distribution & hourly/monthly patterns
7. Negative price analysis
8. Correlation heatmap
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

# Column groupings
PRICE_COL = "price_eur_mwh"
GEN_COLS_DISPLAY = {
    "gen_biomass": "Biomass",
    "gen_fossil_brown_coal_lignite": "Lignite",
    "gen_fossil_coal_derived_gas": "Coal Gas",
    "gen_fossil_gas": "Natural Gas",
    "gen_fossil_hard_coal": "Hard Coal",
    "gen_fossil_oil": "Oil",
    "gen_geothermal": "Geothermal",
    "gen_hydro_pumped_storage": "Pumped Storage",
    "gen_hydro_run_of_river_and_poundage": "Run of River",
    "gen_hydro_water_reservoir": "Water Reservoir",
    "gen_nuclear": "Nuclear",
    "gen_other": "Other",
    "gen_other_renewable": "Other Renewable",
    "gen_solar": "Solar",
    "gen_waste": "Waste",
    "gen_wind_offshore": "Wind Offshore",
    "gen_wind_onshore": "Wind Onshore",
}
WEATHER_COLS_DISPLAY = {
    "weather_temperature_2m": "Temperature (2m) [C]",
    "weather_relative_humidity_2m": "Humidity (2m) [%]",
    "weather_wind_speed_10m": "Wind Speed (10m) [km/h]",
    "weather_wind_speed_100m": "Wind Speed (100m) [km/h]",
    "weather_shortwave_radiation": "GHI [W/m2]",
    "weather_direct_normal_irradiance": "DNI [W/m2]",
    "weather_precipitation": "Precipitation [mm]",
}

RENEWABLE_COLS = [
    "gen_solar",
    "gen_wind_onshore",
    "gen_wind_offshore",
    "gen_hydro_run_of_river_and_poundage",
    "gen_hydro_water_reservoir",
    "gen_biomass",
    "gen_other_renewable",
    "gen_geothermal",
]
FOSSIL_COLS = [
    "gen_fossil_gas",
    "gen_fossil_brown_coal_lignite",
    "gen_fossil_hard_coal",
    "gen_fossil_oil",
    "gen_fossil_coal_derived_gas",
]

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
    gen_cols = [
        c
        for c in df.columns
        if c.startswith("gen_") and c != "gen_hydro_pumped_storage_consumption"
    ]
    df["total_generation"] = df[gen_cols].sum(axis=1)
    df["renewable_generation"] = df[[c for c in RENEWABLE_COLS if c in df.columns]].sum(axis=1)
    df["fossil_generation"] = df[[c for c in FOSSIL_COLS if c in df.columns]].sum(axis=1)
    df["renewable_share"] = (df["renewable_generation"] / df["total_generation"] * 100).clip(0, 100)
    return df


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DE-LU Energy Market EDA",
    page_icon="",
    layout="wide",
)

st.title("DE-LU Day-Ahead Electricity Market — Exploratory Data Analysis")
st.markdown(
    "**Dataset:** ENTSO-E day-ahead prices + generation mix + ERA5 weather for the "
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

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
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
    # Only show main columns
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

# ═══════════════════════════════════════════════════════════════════════════
# 2. PRICE TIME-SERIES
# ═══════════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════════
# 3. GENERATION MIX
# ═══════════════════════════════════════════════════════════════════════════
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
# Order: renewables first, then fossil, then other
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
ren_share_ts = dff[["renewable_share"]].resample(gen_freq).mean()
fig_ren = px.line(
    ren_share_ts.reset_index(),
    x="timestamp_utc",
    y="renewable_share",
    labels={"timestamp_utc": "", "renewable_share": "Renewable %"},
)
fig_ren.update_layout(height=300)
st.plotly_chart(fig_ren, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 4. WEATHER OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
st.header("4. Weather Overview")

weather_cols_available = [c for c in WEATHER_COLS_DISPLAY if c in dff.columns]
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
    labels={"timestamp_utc": "", weather_var: WEATHER_COLS_DISPLAY[weather_var]},
    title=f"{WEATHER_COLS_DISPLAY[weather_var]} (Daily Mean)",
)
fig_weather.update_layout(height=350)
st.plotly_chart(fig_weather, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 5. PRICE-FEATURE CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════
st.header("5. Key Correlations with Price")

corr_cols = (
    ["renewable_share", "total_generation", "fossil_generation", "renewable_generation"]
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
fig_corr.update_layout(height=600, showlegend=False)
st.plotly_chart(fig_corr, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 6. PRICE DISTRIBUTION & PATTERNS
# ═══════════════════════════════════════════════════════════════════════════
st.header("6. Price Distribution & Temporal Patterns")

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
    # Year-over-year boxplot
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
    # Hourly profile
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
    # Monthly profile
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

# ═══════════════════════════════════════════════════════════════════════════
# 7. NEGATIVE PRICE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
st.header("7. Negative Price Analysis")

neg = dff[dff[PRICE_COL] < 0]
total_hours = len(dff)
neg_hours = len(neg)

col1, col2, col3 = st.columns(3)
col1.metric("Negative price hours", f"{neg_hours:,}")
col2.metric("Pct of total", f"{neg_hours / total_hours * 100:.2f}%")
col3.metric("Most negative", f"{neg[PRICE_COL].min():.1f} EUR/MWh" if neg_hours > 0 else "N/A")

if neg_hours > 0:
    # Count per year
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
        # When do negative prices happen? By hour
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
        # Renewable share during negative prices vs positive
        st.markdown("**Renewable share comparison:**")
        ren_neg = neg["renewable_share"].mean()
        ren_pos = dff[dff[PRICE_COL] >= 0]["renewable_share"].mean()
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

# ═══════════════════════════════════════════════════════════════════════════
# 8. CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
st.header("8. Correlation Heatmap")

heatmap_cols = [PRICE_COL] + gen_cols_available + weather_cols_available
heatmap_labels = (
    ["Price"]
    + [GEN_COLS_DISPLAY.get(c, c) for c in gen_cols_available]
    + [WEATHER_COLS_DISPLAY.get(c, c) for c in weather_cols_available]
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

# ═══════════════════════════════════════════════════════════════════════════
# 9. SCATTER: RENEWABLE SHARE vs PRICE
# ═══════════════════════════════════════════════════════════════════════════
st.header("9. Renewable Share vs Price")

# Subsample for performance (scatter with 60k points is heavy)
scatter_df = dff[[PRICE_COL, "renewable_share", "year"]].dropna()
if len(scatter_df) > 10000:
    scatter_df = scatter_df.sample(10000, random_state=42)

fig_scatter = px.scatter(
    scatter_df,
    x="renewable_share",
    y=PRICE_COL,
    color="year",
    opacity=0.4,
    labels={"renewable_share": "Renewable Share [%]", PRICE_COL: "EUR/MWh"},
    title="Day-Ahead Price vs Renewable Share (10k sample)",
)
fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    """
    **Data sources:** ENTSO-E Transparency Platform (prices & generation),
    Open-Meteo Archive API / ERA5 (weather).

    **License:** CC-BY-4.0 | **Bidding zone:** DE-LU (Germany/Luxembourg)
    """
)
