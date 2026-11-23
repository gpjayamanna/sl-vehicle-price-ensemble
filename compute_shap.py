import joblib
import json
import shap
import os
import numpy as np
import pandas as pd

pipeline = joblib.load('models/vehicle_price_pipeline.joblib')
preprocessor = pipeline.named_steps['preprocessor']
model = pipeline.named_steps['regressor']

df = pd.read_csv('data/car_price_dataset.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
df.columns = [c.strip() for c in df.columns]

df = df[(df['YOM'] >= 1980) & (df['YOM'] <= 2025)]
df = df[(df['Price'] >= 3.0) & (df['Price'] <= 350.0)]
df = df[(df['Millage(KM)'] >= 1000) & (df['Millage(KM)'] <= 600000)]
df = df[(df['Engine (cc)'] >= 600) & (df['Engine (cc)'] <= 4500)]

df['Vehicle_Age'] = (2025 - df['YOM']).clip(lower=0)
df['Km_Per_Year'] = df['Millage(KM)'] / (df['Vehicle_Age'] + 1)
binary_cols = ['AIR CONDITION', 'POWER STEERING', 'POWER MIRROR', 'POWER WINDOW']
for col in binary_cols:
    df[col] = (df[col].str.strip().str.lower() == 'available').astype(int)
df['Feature_Score'] = df[binary_cols].sum(axis=1)

top_brands = df['Brand'].value_counts().head(20).index.tolist()
df['Brand_Cleaned'] = df['Brand'].apply(lambda x: x if x in top_brands else 'Other')
top_models = df['Model'].value_counts().head(60).index.tolist()
df['Model_Cleaned'] = df['Model'].apply(lambda x: x if x in top_models else 'Other')
top_towns = df['Town'].value_counts().head(30).index.tolist()
df['Town_Cleaned'] = df['Town'].apply(lambda x: x if x in top_towns else 'Other')

categorical_features = ['Brand_Cleaned', 'Model_Cleaned', 'Gear', 'Fuel Type', 'Town_Cleaned', 'Condition']
numeric_features = ['YOM', 'Vehicle_Age', 'Engine (cc)', 'Millage(KM)', 'Km_Per_Year', 
                    'AIR CONDITION', 'POWER STEERING', 'POWER MIRROR', 'POWER WINDOW', 'Feature_Score']

X = df[categorical_features + numeric_features]

# Sample 500 records for fast, accurate SHAP computation
sample_X = X.sample(n=500, random_state=42)
sample_X_prep = preprocessor.transform(sample_X)

cat_encoder = preprocessor.named_transformers_['cat']
cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
all_feature_names = numeric_features + cat_names

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_X_prep)

# Calculate mean absolute SHAP value per encoded feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Aggregate back to high-level buyer decision factors
feature_groups = {
    'Manufacture Year & Vehicle Age': 0.0,
    'Engine Capacity (cc)': 0.0,
    'Odometer Mileage & Annual Usage': 0.0,
    'Brand & Manufacturer Prestige': 0.0,
    'Vehicle Model & Trim Lineage': 0.0,
    'Transmission (Automatic vs Manual)': 0.0,
    'Fuel Type (Hybrid / Petrol / Diesel)': 0.0,
    'Power Options & Feature Equipment': 0.0,
    'Location / Geographic Region': 0.0,
    'Vehicle Condition (New vs Used)': 0.0
}

for feat_name, val in zip(all_feature_names, mean_abs_shap):
    if feat_name in ['YOM', 'Vehicle_Age']:
        feature_groups['Manufacture Year & Vehicle Age'] += val
    elif feat_name == 'Engine (cc)':
        feature_groups['Engine Capacity (cc)'] += val
    elif feat_name in ['Millage(KM)', 'Km_Per_Year']:
        feature_groups['Odometer Mileage & Annual Usage'] += val
    elif feat_name.startswith('Brand_Cleaned_'):
        feature_groups['Brand & Manufacturer Prestige'] += val
    elif feat_name.startswith('Model_Cleaned_'):
        feature_groups['Vehicle Model & Trim Lineage'] += val
    elif feat_name.startswith('Gear_'):
        feature_groups['Transmission (Automatic vs Manual)'] += val
    elif feat_name.startswith('Fuel Type_'):
        feature_groups['Fuel Type (Hybrid / Petrol / Diesel)'] += val
    elif feat_name in ['AIR CONDITION', 'POWER STEERING', 'POWER MIRROR', 'POWER WINDOW', 'Feature_Score']:
        feature_groups['Power Options & Feature Equipment'] += val
    elif feat_name.startswith('Town_Cleaned_'):
        feature_groups['Location / Geographic Region'] += val
    elif feat_name.startswith('Condition_'):
        feature_groups['Vehicle Condition (New vs Used)'] += val

total_impact = sum(feature_groups.values())
shap_summary = [
    {
        'Factor': k, 
        'Impact (Lakhs)': round(float(v), 2),
        'Relative Importance (%)': round(float(v / total_impact * 100), 1)
    } 
    for k, v in feature_groups.items()
]
shap_summary = sorted(shap_summary, key=lambda x: x['Impact (Lakhs)'], reverse=True)

with open('models/shap_summary.json', 'w') as f:
    json.dump(shap_summary, f, indent=4)

print('SHAP computation successful! Factors calculated:')
for r in shap_summary:
    print(f"{r['Factor']}: {r['Impact (Lakhs)']} Lakhs ({r['Relative Importance (%)']}%)")

