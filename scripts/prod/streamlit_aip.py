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
from preprocessed_data_streamlit import df,features,sorted_states,racial_features,industries_list,get_pool_and_scaled

# -------------------------------
# Load scaler and base weights
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
scaler_path = os.path.join(BASE_DIR, '../../pkl/scaler.pkl')
weights_path = os.path.join(BASE_DIR, '../../pkl/weights.pkl')
geojson_path = os.path.join(BASE_DIR, '../../data/geojson-counties-fips.json')
logo_path = os.path.join(BASE_DIR, '..', 'AIP_logo.png')  # adjust path if needed

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
########################

col1, col2 = st.columns([1, 5])
with col1:
    st.image(
        "https://static.wixstatic.com/media/8ca3ee_77a2a3c2434c449b9303de5e88a91cb2~mv2.png/v1/crop/x_302,y_430,w_646,h_869/fill/w_40,h_55,al_c,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/logo.png",
        width=100
    )
with col2:
    st.title("American Inequality Mirror Counties")

st.markdown(
    """
    [Substack](https://americaninequality.substack.com/?utm_campaign=profile_chips)  
    """,
    unsafe_allow_html=True
)

# Sidebar controls
st.sidebar.header("Tools")

###Page Navigation###
page = st.sidebar.radio(
    "Select a Page",
    ["Mirror Counties", "Culture Analysis", "About"]
)

st.sidebar.header("Choose State & County")
state_input = st.sidebar.selectbox("Select a State", sorted_states, index=sorted_states.index("California"))
sorted_counties = df[df['State'] == state_input]['County'].unique()
county_input = st.sidebar.selectbox("Select a County", sorted_counties, index=list(sorted_counties).index("Imperial"))

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
        similar["population_percentile"].between(sel_pct - 5, sel_pct + 5)
    ]

# --------------------------------------------------------------------
# FINAL RANKING — always sort by distance after all filters
# --------------------------------------------------------------------
ranked_counties = similar.sort_values("distance", ascending=True)

if page == "Mirror Counties":

    # Add clickable county links for More Info button
    original_fips = selected_row['FIPS'].values[0]
    ranked_counties = ranked_counties.copy()
    ranked_counties['County_name'] = ranked_counties.apply(
        lambda row: f'<a href="{make_compare_link(original_fips, row["FIPS"])}" target="_blank">{row["County"]}</a>',
        axis=1
    )

    # Columns to display
    display_columns = ['State', 'County', 'Population', 'Income', 'Racial Breakdown','Top Industries']

    # -------------------------------
    # Sidebar Mirror Counties dropdown + button
    # -------------------------------
    st.sidebar.header("Mirror Counties")

    variable_input = st.sidebar.selectbox(
        "Select a variable to compare",
        ["Income", "Life Expectancy", "Upward mobility"],
        index=0
    )

    # Prepare display name for sidebar dropdown
    ranked_counties['display'] = ranked_counties['State'] + " – " + ranked_counties['County']

    selected_county_for_info = st.sidebar.selectbox(
        "Select a county for more info",
        ranked_counties['display'].tolist() if not ranked_counties.empty else []
    )


    if st.sidebar.button("More Info Here!"):
        if not ranked_counties.empty:
            fips_selected = ranked_counties.loc[
                ranked_counties['display'] == selected_county_for_info, 'FIPS'
            ].values[0]
            compare_url = make_compare_link(original_fips, fips_selected)
            js = f"window.open('{compare_url}')"
            st.components.v1.html(f"<script>{js}</script>", height=0)

    
    ## -------------------------------
    # Display DataFrames with aligned columns
    # -------------------------------
    header_height = 35
    row_height = 35
    visible_rows = 10
    max_scroll_rows = 15

    # Prepare copies for display with formatting
    selected_row_display = selected_row[display_columns].copy().reset_index(drop=True)
    ranked_display_df = ranked_counties[display_columns].head(max_scroll_rows).copy().reset_index(drop=True)

        
    #St.dataframe
    def display_aligned_df(df, height=None):
        styled_df = (
            df.style
            .set_table_styles([
                {'selector': 'th', 'props': [('min-width', '120px')]},
                {'selector': 'td', 'props': [('min-width', '120px')]},
            ])
            .format({
                "Population": "{:,}",
                "Income": "${:,.0f}"
            })
        )

        st.dataframe(styled_df, height=height)

    # Display selected county
    st.subheader(f"Selected County: {county_input}, {state_input}")
    display_aligned_df(selected_row_display, height=row_height + header_height)

    # Display mirror counties
    st.subheader(f"Mirror Counties to {county_input}, {state_input}")
    n_rows = min(len(ranked_display_df), max_scroll_rows)
    display_height = min(n_rows, visible_rows) * row_height + header_height
    display_aligned_df(ranked_display_df, height=display_height)


    # -------------------------------
    # Charts: Bar + Map
    # -------------------------------
    # Prepare plotting dataframe (keep numeric Income for charts)
    plot_df = pd.concat([selected_row, ranked_counties.head(max_scroll_rows)], ignore_index=True).drop_duplicates(subset="FIPS")
    plot_df['County_display'] = plot_df['County'] + ", " + plot_df['State']

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
        x=alt.X("County_display", sort=plot_df["County_display"].tolist(), axis=alt.Axis(title="County")),
        y=y_axis,
        color=alt.Color(variable_input, scale=color_scale),
        tooltip=["County_display", variable_input]
    ).properties(width=600)

    national_avg = df[variable_input].mean()
    avg_line = alt.Chart(pd.DataFrame({'y':[national_avg]})).mark_rule(color='black', strokeDash=[5,5]).encode(y='y:Q')
    bar_chart_with_avg = bar_chart + avg_line

    # Prepare GeoJSON for map
    ranked_counties['FIPS_str'] = ranked_counties['FIPS'].astype(str).str.zfill(5)
    valid_features = [feat for feat in counties['features'] if 'geometry' in feat and feat['geometry'] is not None]
    fips_to_variable = dict(zip(ranked_counties['FIPS_str'], ranked_counties[variable_input]))
    fips_to_county = dict(zip(ranked_counties['FIPS_str'], ranked_counties['County']))

    for feat in valid_features:
        fips = feat['id']
        original_name = feat['properties'].get('County') or feat['properties'].get('NAME') or 'Unknown'
        plain_name = re.sub(r'<.*?>', '', original_name)
        feat['properties']['County'] = plain_name
        feat['properties'][variable_input] = fips_to_variable.get(fips, 0)
        feat['properties']['County'] = fips_to_county.get(fips, plain_name)

    # Map data
    map_data = ranked_counties.head(max_scroll_rows)[['County', 'State', 'Latitude', 'Longitude']].copy() \
            if not ranked_counties.empty else pd.DataFrame(columns=['County','State','Latitude','Longitude'])
    map_data['color'] = [[50, 110, 200]] * len(map_data)
    map_data['size'] = 50000

    selected_data = df[df['FIPS'] == selected_fips].copy()
    selected_data = selected_data[['County', 'State', 'Latitude', 'Longitude']]
    selected_data['color'] = [[255, 0, 0]] * len(selected_data)
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
    deck = pdk.Deck(layers=[layer], initial_view_state=initial_view, map_style='light', tooltip={"text": "{County}, {State}"})

    # -------------------------------
    # Display charts and map
    # -------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{variable_input} Compared to Mirror Counties")
        st.altair_chart(bar_chart_with_avg, use_container_width=True)
        avg_text = f"${national_avg:,.0f}" if variable_input == "Income" else f"{national_avg:,.1f}"
        st.markdown(f"*Dashed line represents the national average for {variable_input}: {avg_text}.*")

    with col2:
        st.subheader("Mirror Location")
        st.pydeck_chart(deck)

# -------------------------------
# LLM Analysis Page
# -------------------------------

import openai
import time

if page == "Culture Analysis":
    st.title("Culture Analysis with LLM")
    st.markdown(
        f"Ask a question about {selected_row.iloc[0]['County']}, {selected_row.iloc[0]['State']} and its mirrors"
    )

    user_question = st.text_input(
        "Ask about a specific mirror county, historical context, cultural insights or demographic analysis",
        placeholder=f"Which mirror county is most similar to {selected_row.iloc[0]['County']}, {selected_row.iloc[0]['State']}?"
    )

    run_llm = st.button("Run Culture Analysis")

    if run_llm:
        # Hardcoded API key
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Summarize selected county
        sel = selected_row.iloc[0]
        sel = selected_row.iloc[0]
        selected_county = (
            f"{sel['County']}, {sel['State']}: Population ~{int(sel['Population']/1000)}k, "
            f"Income ~${int(sel['Income']/1000)}k, Life Exp {sel['Life Expectancy']}, "
            f"Rural {sel['% Rural']}%, "
            f"Race {sel.get('Racial Breakdown','N/A')}, Industries {sel.get('Top Industries','N/A')}"
        )


        # Top 10 mirrors summary
        top_mirrors = ranked_counties
        mirrors_summary = []
        for _, row in top_mirrors.iterrows():
            mirrors_summary.append(
                f"{row['County']}, {row['State']}: Pop ~{int(row['Population']/1000)}k, "
                f"Income ~${int(row['Income']/1000)}k, Life Exp {row['Life Expectancy']},"
                f"Rural {sel['% Rural']}%,"
            )
        mirrors_text = "\n".join(mirrors_summary)

        user_prompt = f"""
                Using the data below as context, compare the selected county with its mirror counties,
                focusing on cultural, institutional, economic, and historical similarities that shape lived experience, and answer the question below.

            Selected county:
            {selected_county}

            Mirror counties:
            {mirrors_text}

            Task:
            - You want to find a county to collaborate on policy solutions to improve outcomes in {selected_county}.
            - Focus on demographic composition, population characteristics, industry structure, and culture.
            - Large gaps in income and life expectancy are wanted to identify learning opportunities.
            - Highlight key similarities and meaningful differences.
            - Use historical, economic, and cultural context where relevant to make comparions, while still using data provided.

            """
        system_prompt = """
                You are a data analyst, public policy researcher, and social scientist.

                Quantitative Response Rules:
                - Use ONLY the data explicitly provided by the user for numeric analysis.
                - Do NOT invent, estimate, or infer numeric values.
                - Do NOT fabricate statistics, rankings, or causal claims.
                - Only compare counties to the selected county.
                - Synthesize numeric data into insights rather than listing values.
                - Identify mirror counties with higher income and life expectancy that could serve as benchmarks for the selected county.

                Qualitative Response Rules:
                - Provide qualitative context (culture, governance, history, social dynamics, economic structure, ethnic and religious composition) to support analysis.
                - Highlight recent historical events, migration patterns, population growth trends, and local cultural institutions that influence similarities or differences.
                - Clearly distinguish qualitative insights from data-driven comparisons.
                - Prioritize cultural similarities, social dynamics, and historical context when identifying a county as a good match.
                - You may answer detailed questions about qualitative aspects if prompted, in relation to the data provided.

                Response Format:
                - Keep answers concise, focused, and conversational.
                - No unnecessary preamble or conclusion.
                - Simple, readable text, do not use chat gpt italics.

                End with stating the best mirror county for collarboration and why in bullets
                Limit to 350 words
                """
        
        if user_question.strip():
            user_prompt += f"\n\nUser Question: {user_question.strip()}"

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )

            analysis_text = response.choices[0].message.content

            st.subheader("Mirror Analysis")

            # Typing effect
            text_container = st.empty()
            displayed_text = ""
            lines = analysis_text.strip().split("\n")

            for line in lines:
                # Clean each line
                line = line.replace("\u200b", "").replace("\xa0", " ").strip()
                displayed_text += line + "<br>"  # Use <br> instead of \n for HTML
                text_container.markdown(f'<div style="font-family: inherit;">{displayed_text}</div>', unsafe_allow_html=True)
                time.sleep(0.08)

        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")

if page == "About":
    st.markdown("""
# American Inequality Mirror Counties

**A tool for identifying demographically similar US counties to support policy research, advocacy, and community analysis.**

🔗 [Live App](https://americaninequalityproject.streamlit.app/) | 📰 [American Inequality on Substack](https://americaninequality.substack.com/)

---

## What This Is

Mirror Counties is an interactive data tool that finds US counties that are demographically and economically similar to any selected county. The goal is to help policymakers, researchers, and journalists identify communities that share comparable characteristics — and use those comparisons to surface learning opportunities, inform local policy, and understand inequality at a granular level.

If a county is struggling with low income, poor health outcomes, or limited upward mobility, finding a demographically similar county that is performing better can reveal concrete, transferable policy lessons. If a county has made meaningful progress, understanding its mirrors helps identify where those gains could be replicated.

The tool is built and maintained by [Michael Tiede](https://substack.com/@michaeltiede/posts) as part of the American Inequality Project, which publishes data-driven research on economic inequality, healthcare access, and social disparities in the United States.

---

## Features

- **Mirror County Matching** — Select any US county and find its closest demographic and economic matches nationwide
- **Comparison Charts** — Visualize how your selected county compares to its mirrors on income, life expectancy, and upward mobility
- **Interactive Map** — See where mirror counties are located geographically
- **Culture Analysis** — AI-powered qualitative analysis comparing the selected county and its mirrors across historical, cultural, and economic dimensions
- **Manual Controls** — Advanced users can override algorithmic assumptions and adjust population filters and feature weights manually

---

## How It Works

### Data Sources

The app uses a combination of US Census / American Community Survey (ACS) data and custom-compiled demographic and economic data including:

- **Demographic composition** — racial and ethnic population percentages at the county level
- **Economic indicators** — median household income, upward mobility scores
- **Health outcomes** — life expectancy
- **Industry composition** — primary and secondary industries by county
- **Geography** — rural/urban classification, latitude/longitude, FIPS codes

### Matching Algorithm

Counties are matched using a **weighted Euclidean distance** algorithm. For each county, we compute a feature vector across demographic, economic, and industry variables. The distance between two counties is calculated as:
```
distance = || (county_A - county_B) * weights ||
```

Lower distance = more similar counties.

The algorithm then ranks all counties in the candidate pool by distance and returns the closest matches after applying filters (see below).

### Feature Weighting

Not all features are treated equally. The weighting system reflects deliberate methodological choices about what constitutes meaningful similarity:

**Racial & Ethnic Demographics (highest weight)**  
Minority demographic groups — Black, Hispanic, Asian, American Indian/Alaska Native, and Native Hawaiian/Pacific Islander populations — are weighted at 50. White population is weighted at 20.

This is intentional. White population percentage is near-universal across US counties and carries less signal for identifying meaningfully distinct communities. Minority population shares are more predictive of shared lived experience, economic conditions, and policy context, and therefore receive higher weight in the matching process.

Dynamic adjustments are also applied based on the demographic profile of the selected county:
- A racial group making up 40%+ of the local population receives a 5x weight boost
- A racial group making up 20%+ receives a 3x boost
- A racial group making up 7% or less receives a negative weight adjustment, deprioritizing counties where that group is nearly absent and avoiding spurious matches based on marginal demographic overlap

**Industry Composition (moderate weight)**  
Primary and secondary industries are weighted at 10. Shared economic structure is an important dimension of community similarity — a manufacturing county and a tourism county may look demographically similar but have fundamentally different economic and social dynamics.

**Geography & Population (lower weight)**  
Rural/urban classification is weighted at 10. Raw population is weighted at 5, with additional threshold-based logic to ensure large metropolitan counties are compared against appropriately sized peers rather than being matched to small rural counties.

### Population Filtering

For counties with populations under 1 million, the algorithm restricts matches to counties within 5 percentile points of the selected county's population percentile. This prevents small rural counties from being matched to large urban ones solely on demographic grounds. For very large counties (over 1 million), explicit population range thresholds are applied to keep comparisons meaningful.

### Income Filter

Mirror counties are required to have a median household income at least 10% greater than the selected county. This is a deliberate design choice: the tool is oriented toward identifying counties that are performing *better* on economic outcomes, so that policymakers can study and learn from their approaches rather than simply confirming shared disadvantage.

---

## Culture Analysis

The Culture Analysis page uses a large language model (GPT-4o Mini) to generate qualitative analysis comparing the selected county and its mirrors across cultural, historical, economic, and social dimensions.

The LLM is instructed to:
- Use only the provided data for any quantitative claims
- Draw on historical, cultural, and economic context for qualitative analysis
- Identify the best mirror county for policy collaboration and explain why
- Avoid fabricating statistics or rankings not present in the data

---

## Manual Controls

Advanced users can disable the default algorithmic assumptions via the "Turn off Algorithmic Assumptions" toggle in the sidebar. This enables custom population filters, manual race and industry weight multipliers, and an adjustable number of mirror counties (k).

---

## Limitations & Caveats

- **Data recency** — reflects the most recent Census / ACS estimates; may not capture very recent population shifts
- **County-level aggregation** — averages can obscure significant within-county variation
- **Weighting is opinionated** — default weights prioritize minority demographic representation; use manual controls to explore alternatives
- **Income filter directionality** — surfaces upward comparisons only; not designed to find counties performing similarly or worse
- **LLM analysis** — AI-generated and should be treated as a starting point, not a definitive characterization

---

## About American Inequality

The American Inequality Project publishes data-driven research on economic inequality, healthcare access, demographic shifts, and social disparities in the United States. Founded by Jeremy Ney, the project's work has been featured in The New York Times, NPR, TIME, and BBC, and informs college curricula at Harvard, MIT, Columbia, and Georgetown.

📰 [Read on Substack](https://www.americaninequality.substack.com/)

---

## About the Project

**American Inequality** was founded by **Jeremy Ney**, economist and author. Jeremy is the primary author and researcher behind the project's publications on inequality, healthcare, and opportunity in America.

🔗 [Follow Jeremy on Substack](https://substack.com/@jeremybney) | [LinkedIn](https://www.linkedin.com/in/jeremy-ney/)

**Michael Tiede** is Head Data Scientist at American Inequality, where he designed and built the Mirror Counties app and matching algorithm. He also writes *The Dividing Line* series on geography, race, and economic opportunity.

🔗 [Read The Dividing Line](https://substack.com/@michaeltiede/posts) | [LinkedIn](https://www.linkedin.com/in/michaeltiede/)
    """)


st.sidebar.markdown(
    """
    Developed by [Michael Tiede](https://substack.com/@michaeltiede/posts)  
    """,
    unsafe_allow_html=True
)
