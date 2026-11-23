import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def mape_score(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    print("Loading dataset...")
    data_path = os.path.join("data", "car_price_dataset.csv")
    df = pd.read_csv(data_path)
    
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        
    df.columns = [c.strip() for c in df.columns]
    
    # Filter reasonable boundaries for Sri Lankan market
    df = df[(df["YOM"] >= 1980) & (df["YOM"] <= 2025)]
    df = df[(df["Price"] >= 3.0) & (df["Price"] <= 350.0)]
    df = df[(df["Millage(KM)"] >= 1000) & (df["Millage(KM)"] <= 600000)]
    df = df[(df["Engine (cc)"] >= 600) & (df["Engine (cc)"] <= 4500)]
    
    # Feature engineering
    df["Vehicle_Age"] = (2025 - df["YOM"]).clip(lower=0)
    df["Km_Per_Year"] = df["Millage(KM)"] / (df["Vehicle_Age"] + 1)
    
    binary_cols = ["AIR CONDITION", "POWER STEERING", "POWER MIRROR", "POWER WINDOW"]
    for col in binary_cols:
        df[col] = (df[col].str.strip().str.lower() == "available").astype(int)
        
    df["Feature_Score"] = df[binary_cols].sum(axis=1)
    
    top_brands = df["Brand"].value_counts().head(20).index.tolist()
    df["Brand_Cleaned"] = df["Brand"].apply(lambda x: x if x in top_brands else "Other")
    
    top_models = df["Model"].value_counts().head(60).index.tolist()
    df["Model_Cleaned"] = df["Model"].apply(lambda x: x if x in top_models else "Other")
    
    top_towns = df["Town"].value_counts().head(30).index.tolist()
    df["Town_Cleaned"] = df["Town"].apply(lambda x: x if x in top_towns else "Other")
    
    categorical_features = ["Brand_Cleaned", "Model_Cleaned", "Gear", "Fuel Type", "Town_Cleaned", "Condition"]
    numeric_features = ["YOM", "Vehicle_Age", "Engine (cc)", "Millage(KM)", "Km_Per_Year", 
                        "AIR CONDITION", "POWER STEERING", "POWER MIRROR", "POWER WINDOW", "Feature_Score"]
    
    X = df[categorical_features + numeric_features]
    y = df["Price"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    # Base models
    models = {
        "Random Forest (Bagging)": RandomForestRegressor(n_estimators=180, max_depth=20, random_state=42, n_jobs=-1),
        "Extra Trees (Bagging)": ExtraTreesRegressor(n_estimators=180, max_depth=20, random_state=42, n_jobs=-1),
        "Gradient Boosting (Boosting)": GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=6, random_state=42),
        "XGBoost (Boosting)": XGBRegressor(n_estimators=250, learning_rate=0.06, max_depth=6, random_state=42, n_jobs=-1),
        "LightGBM (Boosting)": LGBMRegressor(n_estimators=250, learning_rate=0.06, num_leaves=31, random_state=42, verbose=-1),
        "CatBoost (Boosting)": CatBoostRegressor(iterations=350, learning_rate=0.08, depth=6, random_seed=42, verbose=0)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_prep, y_train)
        y_pred = model.predict(X_test_prep)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mape_score(y_test, y_pred)
        
        trained_models[name] = model
        results.append({
            "Model": name,
            "Paradigm": "Bagging" if "Bagging" in name else "Boosting",
            "MAE (LKR Lakhs)": round(mae, 2),
            "RMSE (LKR Lakhs)": round(rmse, 2),
            "R2 Score": round(r2, 4),
            "MAPE (%)": round(mape, 2)
        })
    
    # Hybrid Stacking
    print("Training Stacking Regressor...")
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=120, max_depth=16, random_state=42, n_jobs=-1)),
        ("xgb", XGBRegressor(n_estimators=150, learning_rate=0.07, max_depth=6, random_state=42, n_jobs=-1)),
        ("cat", CatBoostRegressor(iterations=200, learning_rate=0.08, depth=6, random_seed=42, verbose=0))
    ]
    
    stacking_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=3,
        n_jobs=-1
    )
    
    stacking_reg.fit(X_train_prep, y_train)
    y_pred_stack = stacking_reg.predict(X_test_prep)
    
    mae_stack = mean_absolute_error(y_test, y_pred_stack)
    rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
    r2_stack = r2_score(y_test, y_pred_stack)
    mape_stack = mape_score(y_test, y_pred_stack)
    
    trained_models["Hybrid Stacking (RF + XGB + CatBoost)"] = stacking_reg
    results.append({
        "Model": "Hybrid Stacking (RF + XGB + CatBoost)",
        "Paradigm": "Hybrid Stacking",
        "MAE (LKR Lakhs)": round(mae_stack, 2),
        "RMSE (LKR Lakhs)": round(rmse_stack, 2),
        "R2 Score": round(r2_stack, 4),
        "MAPE (%)": round(mape_stack, 2)
    })
    
    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_json(os.path.join(output_dir, "benchmark_metrics.json"), orient="records", indent=4)
    
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", best_model)
    ])
    
    joblib.dump(final_pipeline, os.path.join(output_dir, "vehicle_price_pipeline.joblib"))
    
    metadata = {
        "brands": sorted(df["Brand_Cleaned"].unique().tolist()),
        "models_by_brand": {b: sorted(df[df["Brand_Cleaned"] == b]["Model_Cleaned"].unique().tolist()) for b in df["Brand_Cleaned"].unique()},
        "gears": sorted(df["Gear"].unique().tolist()),
        "fuel_types": sorted(df["Fuel Type"].unique().tolist()),
        "towns": sorted(df["Town_Cleaned"].unique().tolist()),
        "conditions": sorted(df["Condition"].unique().tolist()),
        "best_model_name": best_model_name
    }
    with open(os.path.join(output_dir, "ui_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nDone! Best model saved: {best_model_name}")

if __name__ == "__main__":
    main()
