import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ============================================================================
# UNIFIED CATEGORICAL VALUES
# ============================================================================

ALLERGIES = ["Lactose", "Gluten", "Nuts", "None"]
FITNESS_GOALS = ["Lose weight", "Gain weight", "Improve health", "Maintain weight", "Build muscle"]
ACTIVITY_LEVELS = ["Sedentary", "Light", "Moderate", "High"]
DIETARY_PREFERENCES = ["High protein", "Vegan", "Low carb", "Keto"]
HEALTH_CONDITIONS = ["High Blood Pressure", "Heart Disease", "Diabetes", "None"]
AVAILABLE_MODEL_NAMES = ["random_forest", "xgboost", "lightgbm"]

NUTRITION_LIMIT_RANGES = {
    "max_calories": (1500.0, 2500.0),
    "max_protein": (40.0, 200.0),
    "max_carbs": (130.0, 300.0),
    "max_fats": (30.0, 100.0),
}

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


TARGET_COLUMNS = ["Daily Calorie Target", "Protein", "Carbohydrates", "Fat"]
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Height",
    "Weight",
    "Activity Level",
    "Fitness Goal",
    "Dietary Preference",
]


@dataclass
class UserProfile:
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    activity_level: str
    fitness_goal: str
    dietary_preference: str
    allergies: List[str]
    health_conditions: List[str]
    meals_per_day: int = 3
    notes: str = ""
    max_calories: Optional[float] = None
    max_protein: Optional[float] = None
    max_carbs: Optional[float] = None
    max_fats: Optional[float] = None

    def __post_init__(self) -> None:
        if self.activity_level not in ACTIVITY_LEVELS:
            raise ValueError(f"Invalid activity_level: {self.activity_level}. Must be one of {ACTIVITY_LEVELS}")
        if self.fitness_goal not in FITNESS_GOALS:
            raise ValueError(f"Invalid fitness_goal: {self.fitness_goal}. Must be one of {FITNESS_GOALS}")
        if self.dietary_preference not in DIETARY_PREFERENCES:
            raise ValueError(
                f"Invalid dietary_preference: {self.dietary_preference}. Must be one of {DIETARY_PREFERENCES}"
            )
        if self.meals_per_day not in (2, 3):
            raise ValueError("meals_per_day must be 2 or 3")

        invalid_allergies = [a for a in self.allergies if a not in ALLERGIES]
        if invalid_allergies:
            raise ValueError(f"Invalid allergies: {invalid_allergies}. Must be from {ALLERGIES}")
        self.allergies = self.allergies or ["None"]

        invalid_conditions = [c for c in self.health_conditions if c not in HEALTH_CONDITIONS]
        if invalid_conditions:
            raise ValueError(f"Invalid health_conditions: {invalid_conditions}. Must be from {HEALTH_CONDITIONS}")
        self.health_conditions = self.health_conditions or ["None"]

        for field_name, (min_value, max_value) in NUTRITION_LIMIT_RANGES.items():
            field_value = getattr(self, field_name)
            if field_value is None:
                continue
            if not (min_value <= float(field_value) <= max_value):
                raise ValueError(f"{field_name} must be between {min_value:g} and {max_value:g}")


# ============================================================================
# TRAINING HELPERS
# ============================================================================


def normalize_recipe_lists(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize recipe payload for robust recipe-side matching only."""
    normalized = dict(recipe)
    for key in ["diet", "allergy", "diseases", "categories", "ingredients"]:
        value = normalized.get(key, [])
        if isinstance(value, list):
            normalized[key] = [str(v).strip().lower() for v in value]
        else:
            normalized[key] = []
    for key in ["name", "image"]:
        normalized[key] = str(normalized.get(key, ""))
    normalized["price"] = float(normalized.get("price", 0) or 0)
    normalized["discount"] = float(normalized.get("discount", 0) or 0)
    normalized["time"] = float(normalized.get("time", 0) or 0)
    return normalized


def _build_base_preprocessor() -> ColumnTransformer:
    numeric_features = ["Age", "Height", "Weight"]
    categorical_features = [
        "Gender",
        "Activity Level",
        "Fitness Goal",
        "Dietary Preference",
    ]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )


def build_regressor(model_name: str):
    model_name = model_name.strip().lower()

    if model_name == "random_forest":
        return MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=250,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        )

    if model_name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed. Install it with: pip install xgboost")
        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
        )

    if model_name == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed. Install it with: pip install lightgbm")
        return MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
        )

    raise ValueError("Unsupported model_name. Choose one of: random_forest, xgboost, lightgbm")


class UserNutritionModel:
    def __init__(self, model_name: str = "random_forest") -> None:
        self.model_name = model_name
        self.pipeline: Optional[Pipeline] = None

    @staticmethod
    def load_training_dataframe(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]

        missing = [c for c in FEATURE_COLUMNS + TARGET_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in nutrition dataset: {missing}")

        df = df[df["Age"].astype(str).str.lower() != "age"].copy()
        for col in ["Age", "Height", "Weight"] + TARGET_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Age", "Height", "Weight"] + TARGET_COLUMNS)
        return df.reset_index(drop=True)

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("preprocessor", _build_base_preprocessor()),
            ("model", build_regressor(self.model_name)),
        ])

    def train(self, csv_path: str) -> None:
        df = self.load_training_dataframe(csv_path)
        X = df[FEATURE_COLUMNS].copy()
        y = df[TARGET_COLUMNS].copy()
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)

    def evaluate(self, csv_path: str, cv: int = 5) -> Dict[str, Any]:
        df = self.load_training_dataframe(csv_path)
        X = df[FEATURE_COLUMNS].copy()
        y = df[TARGET_COLUMNS].copy()

        pipeline = self._build_pipeline()
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)

        target_names = ["calories", "protein", "carbs", "fats"]
        metrics: Dict[str, Any] = {"model_name": self.model_name, "cv_folds": cv, "targets": {}}
        overall_r2: List[float] = []
        overall_rmse: List[float] = []

        for idx, name in enumerate(target_names):
            y_true_col = y.iloc[:, idx].values
            y_pred_col = y_pred[:, idx]
            mae = float(mean_absolute_error(y_true_col, y_pred_col))
            rmse = float(np.sqrt(mean_squared_error(y_true_col, y_pred_col)))
            r2 = float(r2_score(y_true_col, y_pred_col))
            overall_r2.append(r2)
            overall_rmse.append(rmse)
            metrics["targets"][name] = {
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2": round(r2, 4),
            }

        metrics["average_r2"] = round(float(np.mean(overall_r2)), 4)
        metrics["average_rmse"] = round(float(np.mean(overall_rmse)), 2)
        return metrics

    def predict_daily_targets(self, user: UserProfile) -> Dict[str, float]:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained or loaded.")

        X_new = pd.DataFrame([
            {
                "Age": user.age,
                "Gender": user.gender,
                "Height": user.height_cm,
                "Weight": user.weight_kg,
                "Activity Level": user.activity_level,
                "Fitness Goal": user.fitness_goal,
                "Dietary Preference": user.dietary_preference,
            }
        ])

        pred = self.pipeline.predict(X_new)[0]
        return {
            "calories": max(float(pred[0]), 900.0),
            "protein": max(float(pred[1]), 20.0),
            "carbs": max(float(pred[2]), 30.0),
            "fats": max(float(pred[3]), 10.0),
        }

    def save(self, path: str) -> None:
        if self.pipeline is None:
            raise RuntimeError("No trained model to save.")
        joblib.dump({"model_name": self.model_name, "pipeline": self.pipeline}, path)

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        if isinstance(payload, dict) and "pipeline" in payload:
            self.model_name = payload.get("model_name", "random_forest")
            self.pipeline = payload["pipeline"]
        else:
            self.pipeline = payload
            self.model_name = "random_forest"


# ============================================================================
# MODEL SELECTION
# ============================================================================


def get_trainable_model_names() -> List[str]:
    available: List[str] = ["random_forest"]
    if XGBRegressor is not None:
        available.append("xgboost")
    if LGBMRegressor is not None:
        available.append("lightgbm")
    return available


def evaluate_all_models(csv_path: str, cv: int = 5) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for model_name in get_trainable_model_names():
        model = UserNutritionModel(model_name=model_name)
        results.append(model.evaluate(csv_path, cv=cv))
    results.sort(key=lambda item: (item["average_r2"], -item["average_rmse"]), reverse=True)
    return results


def train_best_model(csv_path: str, model_output_path: str, cv: int = 5) -> Dict[str, Any]:
    evaluation_results = evaluate_all_models(csv_path, cv=cv)
    best_result = evaluation_results[0]
    best_model = UserNutritionModel(model_name=best_result["model_name"])
    best_model.train(csv_path)
    best_model.save(model_output_path)
    return {
        "best_model": best_result["model_name"],
        "cv_folds": cv,
        "results": evaluation_results,
        "saved_model_path": model_output_path,
    }


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================


class RecipeRecommender:
    def __init__(self, recipes_path: str) -> None:
        with open(recipes_path, "r", encoding="utf-8") as f:
            self.recipes = [normalize_recipe_lists(r) for r in json.load(f)]

    @staticmethod
    def _contains_any(text: str, keywords: List[str]) -> bool:
        text = text.lower()
        return any(k in text for k in keywords)

    @staticmethod
    def _relative_match(actual: float, target: float) -> float:
        if target <= 0:
            return 0.0
        return max(0.0, 1.0 - abs(actual - target) / max(target, 1.0))

    @staticmethod
    def _category_name(recipe: Dict[str, Any]) -> str:
        categories = recipe.get("categories", [])
        if not categories:
            return "other"
        return str(categories[0]).lower()

    def _allergy_conflict(self, recipe: Dict[str, Any], user: UserProfile) -> bool:
        recipe_allergy = recipe.get("allergy", [])
        ingredients_text = " ".join(recipe.get("ingredients", []))
        allergy_map = {
            "Nuts": ["nuts", "nut", "almond", "walnut", "cashew", "peanut", "pistachio", "hazelnut"],
            "Lactose": ["lactose", "milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "feta", "parmesan", "halloumi", "mozzarella", "cream cheese"],
            "Gluten": ["gluten", "wheat", "barley", "rye", "bread", "pasta", "flour", "bun", "granola", "bagel", "wrap", "pita", "ciabatta"],
        }
        for allergy in user.allergies:
            if allergy == "None":
                continue
            if allergy.lower() in recipe_allergy:
                return True
            keywords = allergy_map.get(allergy, [allergy.lower()])
            if self._contains_any(ingredients_text, keywords):
                return True
        return False

    def _diet_match(self, recipe: Dict[str, Any], user: UserProfile) -> bool:
        recipe_diets = recipe.get("diet", [])
        protein = float(recipe.get("protein", 0) or 0)
        carbs = float(recipe.get("carbs", 999) or 999)
        pref = user.dietary_preference

        if pref == "High protein":
            return protein >= 18
        if pref == "Vegan":
            return "vegan" in recipe_diets
        if pref == "Keto":
            return "keto" in recipe_diets or carbs <= 20
        if pref == "Low carb":
            return carbs <= 35
        return True

    def _health_safe(self, recipe: Dict[str, Any], user: UserProfile) -> bool:
        conditions = set(user.health_conditions)
        carbs = float(recipe.get("carbs", 0) or 0)
        fats = float(recipe.get("fats", 0) or 0)
        ingredients_text = " ".join(recipe.get("ingredients", []))
        name_text = str(recipe.get("name", "")).lower()
        combined_text = f"{name_text} {ingredients_text}".lower()

        if "Diabetes" in conditions and carbs > 45:
            return False
        if "Heart Disease" in conditions:
            if fats > 22:
                return False
            if self._contains_any(combined_text, ["bacon", "sausage", "fried", "cream cheese", "halloumi"]):
                return False
        if "High Blood Pressure" in conditions:
            if fats > 20:
                return False
            if self._contains_any(combined_text, ["soy sauce", "miso", "pickles", "smoked", "capers", "bacon", "sausage"]):
                return False
        return True

    def _passes_user_daily_caps(self, recipe: Dict[str, Any], user: UserProfile) -> bool:
        if user.max_calories is not None and float(recipe.get("calories", 0) or 0) > user.max_calories:
            return False
        if user.max_protein is not None and float(recipe.get("protein", 0) or 0) > user.max_protein:
            return False
        if user.max_carbs is not None and float(recipe.get("carbs", 0) or 0) > user.max_carbs:
            return False
        if user.max_fats is not None and float(recipe.get("fats", 0) or 0) > user.max_fats:
            return False
        return True

    def _get_category_macro_profile(self, category: str) -> Dict[str, tuple]:
        profiles = {
            "meal": {
                "calories": (0.40, 0.75),
                "protein": (0.30, 0.85),
                "carbs": (0.20, 0.75),
                "fats": (0.15, 0.75),
            },
            "sandwich": {
                "calories": (0.18, 0.50),
                "protein": (0.15, 0.65),
                "carbs": (0.12, 0.55),
                "fats": (0.10, 0.45),
            },
            "salad": {
                "calories": (0.08, 0.35),
                "protein": (0.05, 0.45),
                "carbs": (0.05, 0.35),
                "fats": (0.05, 0.40),
            },
            "soup": {
                "calories": (0.06, 0.28),
                "protein": (0.03, 0.35),
                "carbs": (0.04, 0.32),
                "fats": (0.03, 0.25),
            },
            "snack": {
                "calories": (0.06, 0.25),
                "protein": (0.04, 0.40),
                "carbs": (0.04, 0.30),
                "fats": (0.03, 0.30),
            },
        }
        return profiles.get(category, profiles["meal"])

    def _passes_category_target_safety(self, recipe: Dict[str, Any], meal_target: Dict[str, float]) -> bool:
        category = self._category_name(recipe)
        profile = self._get_category_macro_profile(category)

        calories = float(recipe.get("calories", 0) or 0)
        protein = float(recipe.get("protein", 0) or 0)
        carbs = float(recipe.get("carbs", 0) or 0)
        fats = float(recipe.get("fats", 0) or 0)

        def within_range(value: float, target: float, ratio_range: tuple, extra_margin: float = 0.0) -> bool:
            lower = max(0.0, ratio_range[0] * target)
            upper = ratio_range[1] * target + extra_margin
            return lower <= value <= max(upper, lower)

        return (
            within_range(calories, meal_target["calories"], profile["calories"], extra_margin=40.0)
            and within_range(protein, meal_target["protein"], profile["protein"], extra_margin=8.0)
            and within_range(carbs, meal_target["carbs"], profile["carbs"], extra_margin=10.0)
            and within_range(fats, meal_target["fats"], profile["fats"], extra_margin=6.0)
        )

    def _is_too_similar(self, recipe_a: Dict[str, Any], recipe_b: Dict[str, Any]) -> bool:
        name_a = str(recipe_a.get("name", "")).lower()
        name_b = str(recipe_b.get("name", "")).lower()

        tokens_a = {token for token in re.findall(r"[a-z]+", name_a) if len(token) > 2}
        tokens_b = {token for token in re.findall(r"[a-z]+", name_b) if len(token) > 2}
        common_name_tokens = tokens_a & tokens_b

        ingredients_a = {str(item).strip().lower() for item in recipe_a.get("ingredients", [])}
        ingredients_b = {str(item).strip().lower() for item in recipe_b.get("ingredients", [])}
        ingredient_overlap = len(ingredients_a & ingredients_b)

        same_category = self._category_name(recipe_a) == self._category_name(recipe_b)

        return (
            (same_category and len(common_name_tokens) >= 2)
            or (same_category and ingredient_overlap >= 2)
            or (len(common_name_tokens) >= 3)
        )

    def _passes_meal_target_safety(self, recipe: Dict[str, Any], meal_target: Dict[str, float]) -> bool:
        return self._passes_category_target_safety(recipe, meal_target)

    def _profile_bonus(self, recipe: Dict[str, Any], meal_target: Dict[str, float], user: UserProfile) -> float:
        bonus = 0.0
        calories = float(recipe.get("calories", 0) or 0)
        protein = float(recipe.get("protein", 0) or 0)
        carbs = float(recipe.get("carbs", 0) or 0)
        fats = float(recipe.get("fats", 0) or 0)

        if self._diet_match(recipe, user):
            bonus += 0.08
        if self._health_safe(recipe, user):
            bonus += 0.06
        if recipe.get("discount"):
            bonus += 0.02

        if user.fitness_goal == "Lose weight":
            if calories <= meal_target["calories"] and fats <= meal_target["fats"] * 1.05:
                bonus += 0.06
        elif user.fitness_goal == "Gain weight":
            if calories >= meal_target["calories"] * 0.9:
                bonus += 0.06
        elif user.fitness_goal == "Build muscle":
            if protein >= meal_target["protein"] * 0.8:
                bonus += 0.07
        elif user.fitness_goal == "Improve health":
            if fats <= meal_target["fats"] and carbs <= meal_target["carbs"] * 1.05:
                bonus += 0.05
        elif user.fitness_goal == "Maintain weight":
            if abs(calories - meal_target["calories"]) <= meal_target["calories"] * 0.15:
                bonus += 0.05

        if user.activity_level in {"Moderate", "High"} and protein >= meal_target["protein"] * 0.75:
            bonus += 0.03
        return bonus

    def _score_recipe(self, recipe: Dict[str, Any], meal_target: Dict[str, float], user: UserProfile) -> float:
        calories = float(recipe.get("calories", 0) or 0)
        protein = float(recipe.get("protein", 0) or 0)
        carbs = float(recipe.get("carbs", 0) or 0)
        fats = float(recipe.get("fats", 0) or 0)

        calorie_score = self._relative_match(calories, meal_target["calories"])
        protein_score = self._relative_match(protein, meal_target["protein"])
        carbs_score = self._relative_match(carbs, meal_target["carbs"])
        fats_score = self._relative_match(fats, meal_target["fats"])

        base_score = (
            0.35 * calorie_score
            + 0.25 * protein_score
            + 0.20 * carbs_score
            + 0.10 * fats_score
        )
        preference_score = 0.10 if self._diet_match(recipe, user) else 0.0
        final_score = base_score + preference_score + self._profile_bonus(recipe, meal_target, user)
        return min(round(float(final_score), 4), 1.0)


    def _get_distribution_by_goal(self, user: UserProfile) -> Dict[str, float]:
        goal = user.fitness_goal

        if goal == "Lose weight":
            return {
                "meal": 0.30,
                "salad": 0.25,
                "soup": 0.20,
                "sandwich": 0.15,
                "snack": 0.10,
            }

        if goal == "Gain weight":
            return {
                "meal": 0.40,
                "sandwich": 0.25,
                "snack": 0.15,
                "salad": 0.10,
                "soup": 0.10,
            }

        if goal == "Build muscle":
            return {
                "meal": 0.40,
                "sandwich": 0.20,
                "salad": 0.15,
                "snack": 0.15,
                "soup": 0.10,
            }

        if goal == "Improve health":
            return {
                "meal": 0.30,
                "salad": 0.25,
                "soup": 0.20,
                "snack": 0.15,
                "sandwich": 0.10,
            }

        # Maintain weight and default fallback
        return {
            "meal": 0.35,
            "sandwich": 0.20,
            "salad": 0.20,
            "soup": 0.10,
            "snack": 0.15,
        }

    def _select_by_distribution(self, candidates: List[Dict[str, Any]], top_k: int, user: UserProfile) -> List[Dict[str, Any]]:
        if top_k <= 0 or not candidates:
            return []

        distribution = self._get_distribution_by_goal(user)
        category_order = ["meal", "sandwich", "salad", "soup", "snack"]

        grouped = {
            category: [c for c in candidates if self._category_name(c) == category]
            for category in category_order
        }

        targets = {category: int(top_k * distribution.get(category, 0.0)) for category in category_order}

        assigned = sum(targets.values())
        remaining_slots = top_k - assigned

        for category in category_order:
            if remaining_slots <= 0:
                break
            if len(grouped[category]) > targets[category]:
                targets[category] += 1
                remaining_slots -= 1

        recommended: List[Dict[str, Any]] = []
        seen_names = set()

        for category in category_order:
            needed = min(targets[category], len(grouped[category]))
            picked = 0

            for recipe in grouped[category]:
                recipe_name = recipe.get("name")
                if recipe_name in seen_names:
                    continue

                recommended.append(recipe)
                seen_names.add(recipe_name)
                picked += 1

                if picked >= needed:
                    break

        if len(recommended) < top_k:
            for recipe in candidates:
                recipe_name = recipe.get("name")
                if recipe_name in seen_names:
                    continue

                recommended.append(recipe)
                seen_names.add(recipe_name)

                if len(recommended) >= top_k:
                    break

        return recommended[:top_k]


    def recommend(self, user: UserProfile, daily_targets: Dict[str, float], top_k: int = 10) -> Dict[str, Any]:
        meal_target = {
            "calories": daily_targets["calories"] / user.meals_per_day,
            "protein": daily_targets["protein"] / user.meals_per_day,
            "carbs": daily_targets["carbs"] / user.meals_per_day,
            "fats": daily_targets["fats"] / user.meals_per_day,
        }

        candidates: List[Dict[str, Any]] = []
        for recipe in self.recipes:
            if self._allergy_conflict(recipe, user):
                continue
            if not self._diet_match(recipe, user):
                continue
            if not self._health_safe(recipe, user):
                continue
            if not self._passes_user_daily_caps(recipe, user):
                continue
            if not self._passes_meal_target_safety(recipe, meal_target):
                continue

            item = dict(recipe)
            item["score"] = self._score_recipe(recipe, meal_target, user)
            candidates.append(item)

        candidates.sort(key=lambda r: r["score"], reverse=True)
        recommendations = self._select_by_distribution(candidates, top_k=top_k, user=user)

        category_distribution = {
            "meal": sum(1 for r in recommendations if self._category_name(r) == "meal"),
            "sandwich": sum(1 for r in recommendations if self._category_name(r) == "sandwich"),
            "salad": sum(1 for r in recommendations if self._category_name(r) == "salad"),
            "soup": sum(1 for r in recommendations if self._category_name(r) == "soup"),
            "snack": sum(1 for r in recommendations if self._category_name(r) == "snack"),
       }

        return {
            "daily_targets": {k: round(v, 1) for k, v in daily_targets.items()},
            "meal_target": {k: round(v, 1) for k, v in meal_target.items()},
            "recommendations": recommendations,
            "available_recommendations": len(candidates),
            "category_distribution": category_distribution,
            "selected_model_inputs": {
                "activity_level": user.activity_level,
                "fitness_goal": user.fitness_goal,
                "dietary_preference": user.dietary_preference,
                "allergies": user.allergies,
                "health_conditions": user.health_conditions,
                "meals_per_day": user.meals_per_day,
            },
        }




def _apply_user_caps_to_daily_targets(daily_targets: Dict[str, float], user: UserProfile) -> Dict[str, float]:
    capped = dict(daily_targets)
    if user.max_calories is not None:
        capped["calories"] = min(capped["calories"], float(user.max_calories))
    if user.max_protein is not None:
        capped["protein"] = min(capped["protein"], float(user.max_protein))
    if user.max_carbs is not None:
        capped["carbs"] = min(capped["carbs"], float(user.max_carbs))
    if user.max_fats is not None:
        capped["fats"] = min(capped["fats"], float(user.max_fats))
    return {k: round(float(v), 1) for k, v in capped.items()}


class MealRecommendationSystem:
    def __init__(self, model_path: str, recipes_path: str) -> None:
        self.model = UserNutritionModel()
        self.model.load(model_path)
        self.recommender = RecipeRecommender(recipes_path)

    def recommend(self, user: UserProfile, top_k: int = 10) -> Dict[str, Any]:
        predicted_daily_targets = self.model.predict_daily_targets(user)
        final_daily_targets = _apply_user_caps_to_daily_targets(predicted_daily_targets, user)
        result = self.recommender.recommend(user, final_daily_targets, top_k=top_k)
        result["predicted_daily_targets"] = {k: round(float(v), 1) for k, v in predicted_daily_targets.items()}
        return result


def ensure_model_exists(model_path: str, nutrition_csv_path: str, cv: int = 5) -> None:
    if os.path.exists(model_path):
        return
    train_best_model(csv_path=nutrition_csv_path, model_output_path=model_path, cv=cv)
