# Sri Lankan Vehicle Price Prediction using Ensemble Learning

This project implements machine learning ensemble models (Bagging, Boosting, and Stacking) to predict used vehicle prices in Sri Lanka using real marketplace data scraped from ikman.lk and Riyasewana.lk.

## Project Structure
- `Sri_Lankan_Vehicle_Price_Ensemble_Modeling.ipynb` : Jupyter notebook with EDA, preprocessing, model training, evaluation, and comparisons.
- `train_models.py` : Python script to clean data, train models, and export the trained pipeline.
- `app.py` : Streamlit web dashboard for price prediction and model comparison.
- `data/car_price_dataset.csv` : Dataset containing Sri Lankan car listings.
- `models/` : Serialized model pipeline and evaluation benchmark metrics.

## Models Evaluated
- **Bagging**: Random Forest Regressor, Extra Trees Regressor
- **Boosting**: LightGBM Regressor, CatBoost Regressor, XGBoost Regressor, Gradient Boosting Regressor
- **Stacking / Hybrid**: Level-0 base models (Random Forest, XGBoost, CatBoost) with a RidgeCV meta-regressor

## Evaluation Results
| Model | Paradigm | MAE (Lakhs) | RMSE (Lakhs) | R² Score | MAPE (%) |
|---|---|---|---|---|---|
| LightGBM | Boosting | 6.15 | 13.46 | 0.9013 | 15.39% |
| Random Forest | Bagging | 5.75 | 14.24 | 0.8894 | 14.47% |
| CatBoost | Boosting | 6.57 | 14.28 | 0.8889 | 16.80% |
| Hybrid Stacking | Stacking | 6.38 | 14.33 | 0.8881 | 15.95% |
| XGBoost | Boosting | 6.45 | 14.60 | 0.8838 | 16.32% |
| Gradient Boosting | Boosting | 6.24 | 14.61 | 0.8837 | 16.04% |
| Extra Trees | Bagging | 5.85 | 15.41 | 0.8705 | 14.18% |

## How to Run

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Train models:
```bash
python train_models.py
```

3. Run the dashboard:
```bash
streamlit run app.py
```
