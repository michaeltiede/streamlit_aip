# dashboard.py (Streamlit version)

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import json
import os
import altair as alt
import plotly.express as px

from preprocessed_data_streamlit import df, features, sorted_states, sorted_counties, racial_features, get_pool_and_scaled

# -------------------------------
# Helper Function
# -------------------------------
def make_compare_link(original_fips, compare_fips):
    original_fips = str(original_fips).zfill(5)
    compare_fips = str(compare_fips).zfill(5)
    return f"https://www.countyhealthrankings.org/health-data/compare-counties?year=2025&compareCounties={original_fips},{compare_fips}"

# -------------------------------
# Load scaler and weights
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

scaler_path = os.path.join(BASE_DIR, '../pkl/scaler.pkl')
weights_path = os.path.join(BASE_DIR, '../pkl/weights.pkl')
geojson_path = os.path.join(BASE_DIR, '../data/geojson-counties-fips.json')

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(weights_path, 'rb') as f:
    all_weights = pickle.load(f)

with open(geojson_path, 'r') as f:
    counties = json.load(f)


# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="American Inequality Project", layout="wide")

# Header
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

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("Choose State & County")
state_input = st.sidebar.selectbox("Select a State", sorted_states, index=sorted_states.index("Alabama"))
county_input = st.sidebar.selectbox("Select a County", df[df['State'] == state_input]['County'].unique())

variable_input = st.sidebar.selectbox(
    "Select a variable to compare",
    ["Income", "Life Expectancy", "Upward mobility"],
    index=0
)


# -------------------------------
# Main Display
# -------------------------------
selected_row = df[(df['State'] == state_input) & (df['County'] == county_input)]

if selected_row.empty:
    st.error(f"No data found for {county_input}, {state_input}.")
    st.stop()

index = selected_row.index[0]

# Scale features
selected_scaled = scaler.transform(selected_row[features]).flatten()
df_scaled = scaler.transform(df[features])

# Get weighted pool
population_value = selected_row['Population'].values[0]
df_pool, df_scaled_pool, weights = get_pool_and_scaled(
    population_value, df, scaler, features, racial_features, all_weights
)

# Compute distances
distances = np.linalg.norm((df_scaled_pool - selected_scaled) * weights, axis=1)

# Get k nearest neighbors
k = 200
indices = np.argsort(distances)[:k]
similar_counties = df_pool.iloc[indices]
similar_counties = similar_counties[similar_counties['State'] != state_input]

# Percentile filter for small counties
if population_value <= 1_000_000:
    selected_percentile = selected_row['population_percentile'].values[0]
    percentile_min = selected_percentile - 3
    percentile_max = selected_percentile + 3
    similar_counties = similar_counties[
        (similar_counties['population_percentile'] >= percentile_min) &
        (similar_counties['population_percentile'] <= percentile_max)
    ]

# Rank by variable
ranked_counties = similar_counties.sort_values(by='Income', ascending=False)
ranked_counties = ranked_counties[ranked_counties.index != index]
top_10_counties = ranked_counties.head(10)

# Add clickable county links
original_fips = selected_row['FIPS'].values[0]
top_10_counties = top_10_counties.copy()
top_10_counties['County_name'] = top_10_counties.apply(
    lambda row: f'<a href="{make_compare_link(original_fips, row["FIPS"])}" target="_blank">{row["County"]}</a>',
    axis=1
)

# Columns to display
display_columns = [
    'State', 'County', 'Population', 'Income', 'Primary Industry',
    'Secondary Industry'
]

st.subheader(f"Top 10 Similar Counties to {county_input}, {state_input}")

# -------------------------------
# Sidebar Controls (Dropdown + Button)
# -------------------------------
st.sidebar.header("Mirror Counties")

# Prepare dropdown options: plain text (no HTML)
top_10_options = top_10_counties.copy()
top_10_options['display'] = top_10_options['State'] + " – " + top_10_options['County']

# Dropdown selection
selected_county_for_info = st.sidebar.selectbox(
    "Select a county for more info", 
    top_10_options['display'].tolist()
)

# Button to open the link
if st.sidebar.button("More Info Here!"):
    # Get FIPS for the selected county
    fips_selected = top_10_options.loc[
        top_10_options['display'] == selected_county_for_info, 'FIPS'
    ].values[0]

    # Generate URL
    compare_url = make_compare_link(original_fips, fips_selected)

    # Open URL in new tab
    js = f"window.open('{compare_url}')"
    st.components.v1.html(f"<script>{js}</script>", height=0)

# -------------------------------
# Display DataFrames
# -------------------------------
st.subheader(f"Selected County: {county_input}, {state_input}")
st.dataframe(selected_row[display_columns].reset_index(drop=True))

st.subheader(f"Mirror Counties for {county_input}, {state_input}")

# Display top 10 counties without HTML in the dataframe
top_10_display = top_10_counties.copy()
# top_10_display['County'] = top_10_display['County'].str.replace(r'<.*?>', '', regex=True)
st.dataframe(top_10_display[display_columns].reset_index(drop=True))

import pydeck as pdk

# -------------------------------
# Charts Section
# -------------------------------
import altair as alt
import pandas as pd

col1, col2 = st.columns(2)

# -------------------------------
# Charts: Bar + Map
# -------------------------------

import re

# Compute national min/max for the selected variable
national_min = df[variable_input].min()
national_max = df[variable_input].max()

# -------------------------------
# Prepare data for bar chart
# -------------------------------

# Include selected county + top 10 counties
plot_df = pd.concat([selected_row, top_10_counties], ignore_index=True)
plot_df = plot_df.drop_duplicates(subset="FIPS")

# Remove HTML tags for County names
# plot_df['County_name'] = plot_df['County'].str.replace(r'<.*?>', '', regex=True)

# Keep selected county first
selected_fips = selected_row["FIPS"].values[0]
plot_df["order"] = 0
plot_df.loc[plot_df["FIPS"] == selected_fips, "order"] = -1
plot_df = plot_df.sort_values(by=["order", variable_input], ascending=[True, False])
# -------------------------------
# National color scale
# -------------------------------
national_min = df[variable_input].min()
national_max = df[variable_input].max()

color_scale = alt.Scale(scheme='redblue', domain=[national_min, national_max])

# Bar Chart Formatting
# Y-axis encoding with conditional scales
if variable_input == "Life Expectancy":
    y_axis = alt.Y(
        variable_input,
        axis=alt.Axis(title=variable_input),
        scale=alt.Scale(domain=[0, 100])  # Life Expectancy range
    )
elif variable_input == "Upward mobility":
    y_axis = alt.Y(
        variable_input,
        axis=alt.Axis(title=variable_input),
        scale=alt.Scale(domain=[0, 80])  # Upward Mobility range
    )
else:
    y_axis = alt.Y(variable_input, axis=alt.Axis(title=variable_input))
# -------------------------------
# Bar chart
# -------------------------------
bar_chart = alt.Chart(plot_df).mark_bar().encode(
    x=alt.X("County", sort=plot_df["County"].tolist(), axis=alt.Axis(title="County")),
    y=y_axis,
    color=alt.Color(variable_input, scale=color_scale),
    tooltip=["County", variable_input]
).properties(width=600)

# -------------------------------
# Horizontal line for national average
# -------------------------------
national_avg = df[variable_input].mean()

avg_line = alt.Chart(pd.DataFrame({'y':[national_avg]})).mark_rule(
    color='black',
    strokeDash=[5,5]
).encode(
    y='y:Q'
)

# -------------------------------
# Combine bar chart + horizontal line
# -------------------------------
bar_chart_with_avg = bar_chart + avg_line

# -------------------------------
# Prepare GeoJSON for map
# -------------------------------
top_10_counties['FIPS_str'] = top_10_counties['FIPS'].astype(str).str.zfill(5)
selected_fips_str = str(selected_fips).zfill(5)

valid_features = [feat for feat in counties['features'] if 'geometry' in feat and feat['geometry'] is not None]

# Add County_name and variable to all features
fips_to_variable = dict(zip(top_10_counties['FIPS_str'], top_10_counties[variable_input]))
fips_to_county = dict(zip(top_10_counties['FIPS_str'], top_10_counties['County']))

for feat in valid_features:
    fips = feat['id']
    original_name = feat['properties'].get('County') or feat['properties'].get('NAME') or 'Unknown'
    plain_name = re.sub(r'<.*?>', '', original_name)
    feat['properties']['County'] = plain_name
    feat['properties'][variable_input] = fips_to_variable.get(fips, 0)
    feat['properties']['County'] = fips_to_county.get(fips, plain_name)

top10_fips_set = set(top_10_counties['FIPS_str'])

# -------------------------------
# PyDeck Map (Black background, Red Dots)
# -------------------------------

import pydeck as pdk

# Step 1: Create a plain-text version of County names
# top_10_counties['County_name'] = top_10_counties['County'].str.replace(r'<.*?>', '', regex=True)

# Step 2: Prepare map data using the new column
map_data = top_10_counties[['County', 'State', 'Latitude', 'Longitude']].copy()
map_data['color'] = [[255, 0, 0]] * len(map_data)  # red dots
map_data['size'] = 50000

# Step 3: Prepare selected county data
selected_data = df[df['FIPS'] == selected_fips].copy()
# selected_data['County_name'] = selected_data['County'].str.replace(r'<.*?>', '', regex=True)
selected_data = selected_data[['County', 'State', 'Latitude', 'Longitude']]
selected_data['color'] = [[255, 215, 0]] * len(selected_data)  # Gold dot
selected_data['size'] = 50000

# Step 4: Combine
map_df = pd.concat([map_data, selected_data], ignore_index=True)


# Define the layer for red points
layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position='[Longitude, Latitude]',
    get_color='color',
    get_radius="size",
    pickable=True
)


# Default US center
us_lat, us_lon = 37.0902, -95.7129
default_zoom = 3

# Check if Alaska or Hawaii are in the dataset
includes_ak_hi = map_df['State'].isin(['Alaska', 'Hawaii']).any()

if includes_ak_hi:
    # Expand the zoom to show Alaska/Hawaii; adjust latitude and longitude slightly
    # This centers roughly near the midpoint of Alaska, Hawaii, and contiguous US
    initial_lat = 48   # midpoint for visual fit
    initial_lon = -110
    initial_zoom = 2   # zoomed out to include all
else:
    initial_lat = us_lat
    initial_lon = us_lon
    initial_zoom = default_zoom

initial_view = pdk.ViewState(
    latitude=initial_lat,
    longitude=initial_lon,
    zoom=initial_zoom,
    pitch=0
)

# -------------------------------
# Deck with both layers
# -------------------------------
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=initial_view,
    map_style='dark',                 # built-in black background
    tooltip={"text": "{County}, {State}"}
)

# -------------------------------
# Display in Streamlit
# -------------------------------
col1, col2 = st.columns(2)

# add $ for Income in Note
if variable_input == "Income":
    avg_text = f"${national_avg:,.1f}"
else:
    avg_text = f"{national_avg:,.1f}"

with col1:
    st.subheader(f"{variable_input} compared to Mirror Counties")
    st.altair_chart(bar_chart_with_avg, use_container_width=True)
    st.markdown(
        f"*The dashed black line represents the national average for {variable_input}: {avg_text}.*"
    )

with col2:
    st.subheader("Mirror Counties")
    st.pydeck_chart(deck)
