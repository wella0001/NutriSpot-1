# ML upgrade
This version:
- trains 3 models: Random Forest, XGBoost, LightGBM
- compares them with MAE, RMSE, and R²
- saves the best model automatically

## Run
pip install -r requirements.txt
python train_model.py

## Outputs
- best_nutrition_model.joblib
- model_comparison_results.json
