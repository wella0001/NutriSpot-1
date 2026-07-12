import json
from meal_model import train_best_model


if __name__ == "__main__":
    report = train_best_model(
        csv_path="nutrition_dataset.csv",
        model_output_path="nutrition_model.joblib",
        cv=5,
    )

    with open("best_model_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Best model: {report['best_model']}")
    print("Saved model to nutrition_model.joblib")
    print("Saved evaluation report to best_model_report.json")
