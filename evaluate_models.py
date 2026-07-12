"""
Evaluate all supported models (Random Forest, XGBoost, LightGBM),
compare metrics, save the best model and a JSON report.
"""

import json
import os
import sys

from meal_model import UserNutritionModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "nutrition_dataset.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "model_evaluation_results.json")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_nutrition_model.joblib")

MODEL_NAMES = ["random_forest", "xgboost", "lightgbm"]


def main() -> None:
    all_results = []

    for name in MODEL_NAMES:
        print(f"\n{'='*50}")
        print(f"  Evaluating: {name}")
        print(f"{'='*50}")
        try:
            model = UserNutritionModel(model_name=name)
            metrics = model.evaluate(CSV_PATH, cv=5)
            all_results.append(metrics)

            print(f"  Average R²: {metrics['average_r2']}")
            for target, vals in metrics["targets"].items():
                print(f"    {target:>10s}  MAE={vals['MAE']:>8.2f}  RMSE={vals['RMSE']:>8.2f}  R²={vals['R2']:>7.4f}")

        except ImportError as exc:
            print(f"  SKIPPED ({exc})")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not all_results:
        print("\nNo models could be evaluated!")
        sys.exit(1)

    # Pick best by average R²
    best = max(all_results, key=lambda m: m["average_r2"])
    print(f"\n{'='*50}")
    print(f"  Best model: {best['model_name']}  (avg R² = {best['average_r2']})")
    print(f"{'='*50}")

    # Train and save the best model
    print(f"\nTraining best model ({best['model_name']}) on full dataset ...")
    best_model = UserNutritionModel(model_name=best["model_name"])
    best_model.train(CSV_PATH)
    best_model.save(BEST_MODEL_PATH)
    print(f"Saved to {BEST_MODEL_PATH}")

    # Save JSON report
    report = {"models": all_results, "best_model": best["model_name"]}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
