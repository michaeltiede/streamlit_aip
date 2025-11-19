# preprocessed_data_streamlit_dev.py

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from sklearn.preprocessing import StandardScaler
import pickle
import os

BASE_DIR = os.path.dirname(__file__)  # folder where this file lives
data_path = os.path.join(BASE_DIR, '../data/data.csv')

# -------------------------------
# Load Data
# -------------------------------
data = pd.read_csv(data_path)
df = data.copy()
df.columns = df.columns.str.strip()

# Rename columns
df = df.rename(columns={
    'Top 1 Industry String': 'Primary Industry',
    'Top 2 Industry String': 'Secondary Industry',
    '% Black': 'Black',
    '% American Indian or Alaska Native': 'AI/AN',
    '% Asian': 'Asian',
    '% Native Hawaiian or Other Pacific Islander': 'NH/PI',
    '% Hispanic': 'Hispanic',
    '% Non-Hispanic White': 'White'
})

# Columns to clean
columns_to_clean = ['Income', 'Upward mobility', 'Life Expectancy']
industries_list = df.columns[16:36].tolist()
industries_top2_list = ['Primary Industry', 'Secondary Industry']

# -------------------------------
# Clean numeric columns
# -------------------------------
for col in columns_to_clean:
    df[col] = df[col].astype(str)  # ensure everything is a string
    df[col] = df[col].str.replace(r'[^0-9.]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())  # fill NaN with mean

# Convert industry columns to numeric
for col in industries_list:
    df[col] = df[col].astype(str).str.strip().str.replace(r'[^0-9.]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# -------------------------------
# Population percentile
# -------------------------------
df['population_percentile'] = df['Population'].apply(lambda x: percentileofscore(df['Population'], x))

# -------------------------------
# Feature lists
# -------------------------------
racial_features = ['Black', 'AI/AN', 'Asian', 'NH/PI', 'Hispanic', 'White']

other_features = ['Population', '% Rural']
features = racial_features + other_features + industries_list

# Ensure numeric
df[features] = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)

# -------------------------------
# Scale features
# -------------------------------
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# -------------------------------
# Define weights
# -------------------------------
race_weights = {race: 50 for race in racial_features}
race_weights['White'] = 10

non_race_weights = {'Population': 5, '% Rural': 50}
industry_weights = {col: 50 for col in industries_list}

# Increase weight if racial category > 20%
for col in racial_features:
    if df[col].mean() >= 20:
        race_weights[col] *= 5

all_weights = np.array([
    race_weights.get(col, non_race_weights.get(col, industry_weights.get(col, 1)))
    for col in features
])

# -------------------------------
# Function: get_pool_and_scaled
# -------------------------------
def get_pool_and_scaled(
    selected_population, df, scaler, features, racial_features, weights,
    use_custom_limits=False, min_pop=None, max_pop=None
):
    """
    Returns df_pool, scaled features, and adjusted weights.
    
    If use_custom_limits=True, use min_pop/max_pop sliders instead of original thresholds.
    """
    weights = weights.copy()
    df_pool = df.copy()
    
    # Scale initial pool
    df_pool_scaled = scaler.transform(df_pool[features])
    
    # Original threshold logic for large counties
    '''
    Thresholds (Population boundary, Population Min, Population Max,Population weighting, race weighting)
    '''
    thresholds = [
        (1_800_000, 1_500_000, 10_000_000, 0.1, 1.8),
        (1_500_000, 1_000_000, 3_500_000, 0.1, 1.6),
        (1_300_000, 1_000_000, 1_700_000, 0.2, 1.4),
        (1_000_000, 700_000, 1_400_000, 0.3, 1.3)
    ]
    
    if use_custom_limits:
        # Apply slider-based population filtering
        if min_pop is not None:
            df_pool = df_pool[df_pool['Population'] >= min_pop]
        if max_pop is not None:
            df_pool = df_pool[df_pool['Population'] <= max_pop]
    else:
        # Apply original threshold logic
        for pop_thresh, pool_min, pool_max, pop_weight_factor, race_weight_factor in thresholds:
            if selected_population > pop_thresh:
                df_pool = df[(df['Population'] > pool_min) & (df['Population'] < pool_max)].copy()
                # Adjust weights
                pop_index = features.index('Population')
                weights[pop_index] *= pop_weight_factor
                for col in racial_features:
                    col_index = features.index(col)
                    weights[col_index] *= race_weight_factor
                break  # only first match
    
    df_pool_scaled = scaler.transform(df_pool[features])
    return df_pool, df_pool_scaled, weights

# -------------------------------
# Function: get_racial_breakdown
# -------------------------------
def combine_racial(df, racial_columns):
    """
    Combine racial demographic columns into one string column called 'Racial Breakdown'.
    Example output for a row: 'Black: 12%, Asian: 5%, Hispanic: 30%, ...'
    """
    def row_concat(row):
        parts = []
        for col in racial_columns:
            race_name = col.replace('% ', '')  # remove % from column name
            percent = row[col]
            parts.append(f"{race_name}: {percent:.1f}%")
        return ", ".join(parts)

    df['Racial Breakdown'] = df.apply(row_concat, axis=1)
    return df

df = combine_racial(df, racial_features)


# -------------------------------
# Sort states and counties
# -------------------------------
sorted_states = sorted(df['State'].unique())
sorted_counties = sorted(df[['County', 'State']].sort_values(by='County')['County'].unique())

df['More Info'] = ''

# -------------------------------
# Save scaler and weights
# -------------------------------
pkl_dir = os.path.join(BASE_DIR, '../pkl')
os.makedirs(pkl_dir, exist_ok=True)
if os.environ.get("STREAMLIT_SERVER", "") == "":
    with open(os.path.join(pkl_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(pkl_dir, 'weights.pkl'), 'wb') as f:
        pickle.dump(all_weights, f)

# -------------------------------
# Save preprocessed CSV
# -------------------------------
csv_path = os.path.join(BASE_DIR, '../data/preprocessed_data.csv')
df.to_csv(csv_path, index=False)

print("Preprocessing complete and saved.")
