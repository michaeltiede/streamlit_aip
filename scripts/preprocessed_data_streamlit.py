# preprocess_data.py

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from sklearn.preprocessing import StandardScaler
import pickle
import os

BASE_DIR = os.path.dirname(__file__)  # folder where preprocessed_data.py lives
data_path = os.path.join(BASE_DIR, '../data/data.csv')

data = pd.read_csv(data_path)
df = data
df.columns = df.columns.str.strip()


#rename columns
df = df.rename(columns={
    'Top 1 Industry String': 'Primary Industry',
    'Top 2 Industry String': 'Secondary Industry'
})

# List of columns to clean and convert
columns_to_clean = ['Income', 'Upward mobility', 'Life Expectancy']
industries_list = df.columns[16:36].tolist()
industries_top2_list = ['Primary Industry', 'Secondary Industry']


# Replace non-numeric characters and convert to float

for col in columns_to_clean:
    df[col] = pd.to_numeric(df[col].replace(r'[^0-9.]', '', regex=True), errors='coerce')

# Fill NaN values with the mean of the column, after ensuring the column is numeric
for col in columns_to_clean:
    df[col] = df[col].fillna(df[col].mean())

# Calculate population percentiles for each county

df['population_percentile'] = df['Population'].apply(lambda x: percentileofscore(df['Population'], x))

# List of features to normalize (racial demographics)
racial_features = ['% Black', 
                   '% American Indian or Alaska Native', '% Asian',
                   '% Native Hawaiian or Other Pacific Islander', '% Hispanic', '% Non-Hispanic White']

# Other features to consider (including non-racial features)
other_features = ['Population', '% Rural']
features = racial_features + other_features + industries_list
# print(features)
df[features] = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)


# Convert industry columns to numeric percentages
for col in industries_list:
    df[col] = df[col].astype(str).str.strip().str.replace(r'[^0-9.]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


# Calculate population percentiles if not already present
if 'population_percentile' not in df.columns:
    df['population_percentile'] = pd.qcut(df['Population'], q=100, labels=False)


# Scale the features in the DataFrame
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Define custom weights for features
race_weights ={'% Black': 50, '% American Indian or Alaska Native': 50, '% Asian': 50,
                   '% Native Hawaiian or Other Pacific Islander': 50, '% Hispanic': 50, '% Non-Hispanic White': 10}

non_race_weights = {'Population': 1, '% Rural': 50, 
                    # 'MinorityShare': 80
                    } 

# Define weights for industry features
industry_weights = {col: 50 for col in industries_list}  # set 10 as default weight

#Increase weight for any industry category greater than 20%
# for col in industries_list:
#     if df[col].mean() > 20:  # if county has >20% in this industry
#         industry_weights[col] *= 2


# Increase weight for any racial category greater than 20%
for col in racial_features:
    if df[col].mean() >= 20:
        race_weights[col] *= 2  # Double the weight

all_weights = np.array([race_weights.get(col, non_race_weights.get(col, industry_weights.get(col, 1))) for col in features])

# -------------------------------
# Function: get_weighted_pool
# -------------------------------
def get_pool_and_scaled(selected_population, df, scaler, features, racial_features, weights):
    """
    Returns a restricted df_pool, scaled features, and adjusted weights based on population.
    """
    
    # Make a copy of weights to avoid overwriting
    weights = weights.copy()

    # Define population thresholds and corresponding pool minimums and weight adjustments
    thresholds = [
        (1_800_000, 1_500_000,10_000_000, 0.1, 1.8),
        (1_500_000, 1_200_000,2_500_000, 0.1, 1.6),
        (1_300_000, 1_000_000,1_600_000, 0.2, 1.4),
        (1_000_000, 700_000,1_400_000, 0.3,1.3)
    ]

    
    # Default: use full dataset
    df_pool = df.copy()
    df_pool_scaled = scaler.transform(df_pool[features])

    for pop_thresh, pool_min,pool_max, pop_weight_factor, race_weight_factor in thresholds:
        if selected_population > pop_thresh:
            # Restrict pool to other large counties
            df_pool = df[(df['Population'] > pool_min) & (df['Population'] < pool_max)].copy()
            df_pool_scaled = scaler.transform(df_pool[features])
            
            # Adjust weights
            pop_index = features.index('Population')
            weights[pop_index] *= pop_weight_factor
            
            for col in racial_features: # add + ['MinorityShare'] inside the : if using MinorityShare 
                col_index = features.index(col)
                weights[col_index] *= race_weight_factor
            
            break  # Stop after the first matching threshold

    return df_pool, df_pool_scaled, weights


# -------------------------------
# Sort states and counties
# -------------------------------
sorted_states = sorted(df['State'].unique())
sorted_counties = sorted(df[['County', 'State']].sort_values(by='County')['County'].unique())

df['More Info'] = ''

# Save weights and scaler to pkl folder
pkl_dir = os.path.join(BASE_DIR, '../pkl')
os.makedirs(pkl_dir, exist_ok=True)  # create folder if missing

# Only save if running locally (not on Streamlit server)
if os.environ.get("STREAMLIT_SERVER", "") == "":
    with open(os.path.join(pkl_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(pkl_dir, 'weights.pkl'), 'wb') as f:
        pickle.dump(all_weights, f)


# Save preprocessed CSV to data folder
csv_path = os.path.join(BASE_DIR, '../data/preprocessed_data.csv')
df.to_csv(csv_path, index=False)

print("Preprocessing complete and saved.")
