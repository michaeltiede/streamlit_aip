# app.py (Streamlit)
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import json
import os
import re
import pydeck as pdk
import altair as alt

# Import preprocessed objects and original function
from preprocessed_data_streamlit import df,features,sorted_states,sorted_counties,racial_features,industries_list,get_pool_and_scaled,combine_racial


# -------------------------------
# Load scaler and base weights
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
scaler_path = os.path.join(BASE_DIR, '../../pkl/scaler.pkl')
weights_path = os.path.join(BASE_DIR, '../../pkl/weights.pkl')
geojson_path = os.path.join(BASE_DIR, '../../data/geojson-counties-fips.json')

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(weights_path, 'rb') as f:
    all_weights = pickle.load(f)

with open(geojson_path, 'r') as f:
    counties = json.load(f)

# -------------------------------
# Helper
# -------------------------------
def make_compare_link(original_fips, compare_fips):
    original_fips = str(original_fips).zfill(5)
    compare_fips = str(compare_fips).zfill(5)
    return f"https://www.countyhealthrankings.org/health-data/compare-counties?year=2025&compareCounties={original_fips},{compare_fips}"

# -------------------------------
# Streamlit Layout / Sidebar
# -------------------------------
st.set_page_config(page_title="American Inequality Project", layout="wide")

col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://raw.githubusercontent.com/michaeltiede/american_inequality/main/AIP_logo.png", width=120)
with col2:
    st.title("The American Inequality Project: County Comparison Dashboard")

st.markdown(
    """
    [AmericanInequality.substack.com](https://www.americaninequality.substack.com/)  
    <span style='font-size:10px'>Developed by Michael Tiede</span>
    """,
    unsafe_allow_html=True
)

# Sidebar controls
st.sidebar.header("Compare Options")

st.sidebar.header("Choose State & County")
state_input = st.sidebar.selectbox("Select a State", sorted_states, index=sorted_states.index("Alabama"))
county_input = st.sidebar.selectbox("Select a County", df[df['State'] == state_input]['County'].unique())

# Toggle: use custom settings OR use original logic
use_custom_settings = st.sidebar.checkbox(
    "Turn off Algorithmic Assumptions (Manual Selection)",
    value=False
)

# If custom: show sliders
if use_custom_settings:
    st.sidebar.subheader("Mirror Population Size")
    k = st.sidebar.slider("Number of similar counties (k)", 1, 500, 200)
    st.sidebar.subheader("Custom Weights")
    race_weight_multiplier = st.sidebar.slider("Race Multiplier", 0.0, 10.0, 1.0)
    industry_weight_multiplier = st.sidebar.slider("Industry Multiplier", 0.0, 10.0, 1.0)
    st.sidebar.subheader("Population Filters")
    min_pop_slider = st.sidebar.number_input("Minimum Population", min_value=0, max_value=int(df['Population'].max()), value=0, step=50_000)
    max_pop_slider = st.sidebar.number_input("Maximum Population", min_value=0, max_value=int(df['Population'].max()), value=9_000_000, step=50_000)

else:
    # placeholders used only when custom settings are off
    k = 200
    min_pop_slider = None
    max_pop_slider = None
    race_weight_multiplier = None
    industry_weight_multiplier = None
    population_weight_multiplier = None

# -------------------------------
# Main Display: selected county
# -------------------------------
selected_row = df[(df['State'] == state_input) & (df['County'] == county_input)]
if selected_row.empty:
    st.error(f"No data found for {county_input}, {state_input}.")
    st.stop()

index = selected_row.index[0]
selected_scaled = scaler.transform(selected_row[features]).flatten()

population_value = int(selected_row['Population'].values[0])

# -------------------------------
# Compute df_pool, df_pool_scaled, weights
# Modes:
#  - If use_custom_settings == False -> call original get_pool_and_scaled (keeps original thresholds & weight adjustments)
#  - If use_custom_settings == True  -> build pool & weights based on sliders
# -------------------------------
if not use_custom_settings:
    # Use original logic from preprocess module (this preserves the thresholds/weight adjustments you coded)
    try:
        df_pool, df_scaled_pool, weights = get_pool_and_scaled(
            population_value, df, scaler, features, racial_features, all_weights
        )
    except TypeError:
        # Fallback if imported function signature differs: use original simple call
        df_pool, df_scaled_pool, weights = get_pool_and_scaled(population_value, df, scaler, features, racial_features, all_weights)
else:
    # Custom override: filter by sliders and apply multipliers to base weights
    df_pool = df.copy()
    # Apply population slider filter
    df_pool = df_pool[
        (df_pool['Population'] >= int(min_pop_slider)) &
        (df_pool['Population'] <= int(max_pop_slider))
    ].copy()

    if df_pool.empty:
        st.warning("Custom population filter returned no counties — falling back to the full dataset.")
        df_pool = df.copy()

    # Scale filtered pool
    df_scaled_pool = scaler.transform(df_pool[features])

    # Start from base weights and apply multipliers to racial and industry features
    weights = all_weights.copy().astype(float)  # ensure float copy
    for i, col in enumerate(features):
        if col in racial_features:
            weights[i] = weights[i] * float(race_weight_multiplier)
        elif col in industries_list:
            weights[i] = weights[i] * float(industry_weight_multiplier)

# -------------------------------
# Compute distances and k nearest neighbors
# -------------------------------
# Ensure df_scaled_pool shape matches expectation
if isinstance(df_scaled_pool, pd.DataFrame):
    pool_array = df_scaled_pool[features].values
else:
    pool_array = np.asarray(df_scaled_pool)

# Compute distances
# broadcast multiply: (n_pool, n_features) * (n_features,) -> (n_pool, n_features)
try:
    distances = np.linalg.norm((pool_array - selected_scaled) * weights, axis=1)
except Exception as e:
    st.error(f"Error computing distances: {e}")
    st.stop()
# ---------------------------------------
# Compute K nearest neighbors from pool
# ---------------------------------------

# Sort all counties by distance first
sorted_idx = np.argsort(distances)

# Pull K nearest (before any filters)
k_use = min(int(k), len(sorted_idx))
nearest_idx = sorted_idx[:k_use]

# Extract those counties + attach distance
similar = df_pool.iloc[nearest_idx].copy()
similar["distance"] = distances[nearest_idx]

# Remove the selected county (if present)
similar = similar[similar.index != index]

# --------------------------------------------------------------------
# INCOME FILTER
# Require a match to have income at least 10% greater than selected county
# --------------------------------------------------------------------
selected_income = float(selected_row.iloc[0]["Income"])
similar = similar[similar["Income"] >= selected_income * 1.10]

# --------------------------------------------------------------------
# POPULATION PERCENTILE FILTER (only for small counties ≤ 1M)
# --------------------------------------------------------------------
if population_value <= 1_000_000:
    sel_pct = selected_row.iloc[0]["population_percentile"]
    similar = similar[
        similar["population_percentile"].between(sel_pct - 3, sel_pct + 3)
    ]

# --------------------------------------------------------------------
# FINAL RANKING — always sort by distance after all filters
# --------------------------------------------------------------------
ranked_counties = similar.sort_values("distance", ascending=True)

# Top 10 matches (currently not use due to the scrolling feature)
top_10_counties = ranked_counties.head(10)


# Add clickable county links for More Info button
original_fips = selected_row['FIPS'].values[0]
top_10_counties = top_10_counties.copy()
top_10_counties['County_name'] = top_10_counties.apply(
    lambda row: f'<a href="{make_compare_link(original_fips, row["FIPS"])}" target="_blank">{row["County"]}</a>',
    axis=1
)

# Columns to display
display_columns = ['State', 'County', 'Population', 'Income', 'Primary Industry', 'Secondary Industry', 'Racial Breakdown']

# -------------------------------
# Sidebar Mirror Counties dropdown + button
# -------------------------------
st.sidebar.header("Mirror Counties")

variable_input = st.sidebar.selectbox(
    "Select a variable to compare",
    ["Income", "Life Expectancy", "Upward mobility"],
    index=0
)

top_10_options = top_10_counties.copy()

# top_10_options['display'] = top_10_options['State'] + " – " + top_10_options['County']

#using ranked_counties instead of top_10 above
ranked_counties['display'] = ranked_counties['State'] + " – " + ranked_counties['County']

selected_county_for_info = st.sidebar.selectbox(
    "Select a county for more info",
    ranked_counties['display'].tolist() if not ranked_counties.empty else []
)

if st.sidebar.button("More Info Here!"):
    if not top_10_options.empty:
        fips_selected = top_10_options.loc[
            top_10_options['display'] == selected_county_for_info, 'FIPS'
        ].values[0]
        compare_url = make_compare_link(original_fips, fips_selected)
        js = f"window.open('{compare_url}')"
        st.components.v1.html(f"<script>{js}</script>", height=0)

# -------------------------------
# Display DataFrames
# -------------------------------
# Number of rows to display
num_rows = len(ranked_counties)
display_height = min(num_rows + 1, 11) * 35  # ~35px per row including header

st.subheader(f"Selected County: {county_input}, {state_input}")
st.dataframe(selected_row[display_columns].reset_index(drop=True))

st.subheader(f"Top 10 Similar Counties to {county_input}, {state_input}")
top_10_display = top_10_counties.copy()
st.dataframe(ranked_counties[display_columns].reset_index(drop=True), height=display_height)


# -------------------------------
# Charts: Bar + Map
# -------------------------------
# Prepare plotting dataframe: include selected + top 10
plot_df = pd.concat([selected_row, top_10_counties], ignore_index=True).drop_duplicates(subset="FIPS")

selected_fips = selected_row["FIPS"].values[0]
plot_df["order"] = 0
plot_df.loc[plot_df["FIPS"] == selected_fips, "order"] = -1
plot_df = plot_df.sort_values(by=["order", variable_input], ascending=[True, False])

national_min = df[variable_input].min()
national_max = df[variable_input].max()
color_scale = alt.Scale(scheme='redblue', domain=[national_min, national_max])

# Y-axis formatting
if variable_input == "Life Expectancy":
    y_axis = alt.Y(variable_input, axis=alt.Axis(title=variable_input), scale=alt.Scale(domain=[0, 100]))
elif variable_input == "Upward mobility":
    y_axis = alt.Y(variable_input, axis=alt.Axis(title=variable_input), scale=alt.Scale(domain=[0, 80]))
else:
    y_axis = alt.Y(variable_input, axis=alt.Axis(title=variable_input))

bar_chart = alt.Chart(plot_df).mark_bar().encode(
    x=alt.X("County", sort=plot_df["County"].tolist(), axis=alt.Axis(title="County")),
    y=y_axis,
    color=alt.Color(variable_input, scale=color_scale),
    tooltip=["County", variable_input]
).properties(width=600)

national_avg = df[variable_input].mean()
avg_line = alt.Chart(pd.DataFrame({'y':[national_avg]})).mark_rule(color='black', strokeDash=[5,5]).encode(y='y:Q')
bar_chart_with_avg = bar_chart + avg_line

# Prepare GeoJSON for map
if not top_10_counties.empty:
    top_10_counties['FIPS_str'] = top_10_counties['FIPS'].astype(str).str.zfill(5)
else:
    top_10_counties['FIPS_str'] = []

valid_features = [feat for feat in counties['features'] if 'geometry' in feat and feat['geometry'] is not None]

fips_to_variable = dict(zip(top_10_counties['FIPS_str'], top_10_counties[variable_input])) if not top_10_counties.empty else {}
fips_to_county = dict(zip(top_10_counties['FIPS_str'], top_10_counties['County'])) if not top_10_counties.empty else {}

for feat in valid_features:
    fips = feat['id']
    original_name = feat['properties'].get('County') or feat['properties'].get('NAME') or 'Unknown'
    plain_name = re.sub(r'<.*?>', '', original_name)
    feat['properties']['County'] = plain_name
    feat['properties'][variable_input] = fips_to_variable.get(fips, 0)
    feat['properties']['County'] = fips_to_county.get(fips, plain_name)

# Map data
map_data = top_10_counties[['County', 'State', 'Latitude', 'Longitude']].copy() if not top_10_counties.empty else pd.DataFrame(columns=['County','State','Latitude','Longitude'])
map_data['color'] = [[255, 0, 0]] * len(map_data)
map_data['size'] = 50000

selected_data = df[df['FIPS'] == selected_fips].copy()
selected_data = selected_data[['County', 'State', 'Latitude', 'Longitude']]
selected_data['color'] = [[255, 215, 0]] * len(selected_data)
selected_data['size'] = 50000

map_df = pd.concat([map_data, selected_data], ignore_index=True)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position='[Longitude, Latitude]',
    get_color='color',
    get_radius="size",
    pickable=True
)

us_lat, us_lon = 37.0902, -95.7129
default_zoom = 3
includes_ak_hi = map_df['State'].isin(['Alaska', 'Hawaii']).any()

if includes_ak_hi:
    initial_lat = 48
    initial_lon = -110
    initial_zoom = 2
else:
    initial_lat = us_lat
    initial_lon = us_lon
    initial_zoom = default_zoom

initial_view = pdk.ViewState(latitude=initial_lat, longitude=initial_lon, zoom=initial_zoom, pitch=0)

deck = pdk.Deck(layers=[layer], initial_view_state=initial_view, map_style='dark', tooltip={"text": "{County}, {State}"})

# -------------------------------
# Display charts and map
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"{variable_input} compared to Mirror Counties")
    st.altair_chart(bar_chart_with_avg, use_container_width=True)
    if variable_input == "Income":
        avg_text = f"${national_avg:,.1f}"
    else:
        avg_text = f"{national_avg:,.1f}"
    st.markdown(f"*The dashed black line represents the national average for {variable_input}: {avg_text}.*")

with col2:
    st.subheader("Mirror Counties")
    st.pydeck_chart(deck)
