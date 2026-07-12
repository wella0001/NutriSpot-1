import json
import os
import re
from typing import Any, Dict, List, Optional


def load_env(filepath=".env"):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()


load_env()

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_allergy(value: str) -> str:
    """Normalize an allergy value to standardized form."""
    if not value:
        return "None"
    normalized = str(value).strip().lower()
    mapping = {
        "lactose": "Lactose",
        "gluten": "Gluten",
        "nuts": "Nuts",
        "none": "None",
        "lactose intolerance": "Lactose",
        "egg": "None",
        "eggs": "None",
        "dairy": "Lactose",
        "milk": "Lactose",
    }
    return mapping.get(normalized, "None")


def normalize_health_condition(value: str) -> str:
    """Normalize a health condition value to standardized form."""
    if not value:
        return "None"
    normalized = str(value).strip().lower()
    mapping = {
        "high blood pressure": "High Blood Pressure",
        "hypertension": "High Blood Pressure",
        "heart disease": "Heart Disease",
        "heart": "Heart Disease",
        "cardiac": "Heart Disease",
        "diabetes": "Diabetes",
        "diabetic": "Diabetes",
        "none": "None",
    }
    return mapping.get(normalized, "None")


def normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def safe_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_text(x) for x in value]
    if value is None:
        return []
    return [normalize_text(value)]


def recipe_matches_diet(recipe: Dict[str, Any], diet_type: str) -> bool:
    diet_type = normalize_text(diet_type)
    recipe_diets = safe_list(recipe.get("diet", []))

    # Handle standardized diet types
    if not diet_type:
        return True
    if diet_type == "high protein":
        return float(recipe.get("protein", 0)) >= 18
    if diet_type == "low carb":
        return float(recipe.get("carbs", 999)) <= 35
    if diet_type == "keto":
        return "keto" in recipe_diets or float(recipe.get("carbs", 999)) <= 20
    if diet_type == "vegan":
        return "vegan" in recipe_diets
    # Default: matches any recipe
    return True


def recipe_has_allergy_conflict(recipe: Dict[str, Any], allergies: List[str]) -> bool:
    # Normalize all allergies to standardized values
    normalized_allergies = [normalize_text(normalize_allergy(a)) for a in allergies if normalize_text(normalize_allergy(a)) != "none"]
    if not normalized_allergies:
        return False

    recipe_allergy = safe_list(recipe.get("allergy", []))
    ingredients_text = " ".join(recipe.get("ingredients", [])).lower()

    allergy_keywords = {
        "nuts": ["nut", "almond", "walnut", "cashew", "peanut", "pecan", "pistachio", "hazelnut"],
        "lactose": ["milk", "cheese", "yogurt", "cream", "butter", "dairy", "whey", "feta", "parmesan", "greek yogurt"],
        "gluten": ["wheat", "bread", "pasta", "flour", "barley", "rye", "bun", "wrap", "granola", "croutons"],
    }

    for allergy in normalized_allergies:
        if allergy in recipe_allergy:
            return True
        for keyword in allergy_keywords.get(allergy, []):
            if keyword in ingredients_text:
                return True
    return False


def recipe_matches_conditions(recipe: Dict[str, Any], health_conditions: List[str]) -> bool:
    # Normalize all health conditions to standardized values
    normalized_conditions = [normalize_text(normalize_health_condition(c)) for c in health_conditions if normalize_text(normalize_health_condition(c)) != "none"]
    if not normalized_conditions:
        return True

    recipe_carbs = float(recipe.get("carbs", 0))
    recipe_fats = float(recipe.get("fats", 0))

    for condition in normalized_conditions:
        if condition == "diabetes" and recipe_carbs > 45:
            return False
        if condition == "heart disease" and recipe_fats > 30:
            return False
        if condition == "high blood pressure" and recipe_fats > 25:
            return False

    return True


class FoodChatbot:
    def __init__(self, foods_data: List[Dict[str, Any]], recommendation_engine=None):
        self.foods = foods_data
        self.engine = recommendation_engine
        self._build_search_index()
        self._build_intent_patterns()

    def _build_search_index(self):
        docs = []
        for food in self.foods:
            doc_parts = [
                food.get("name", ""),
                " ".join(food.get("ingredients", [])),
                " ".join(food.get("categories", [])),
                " ".join(food.get("diet", [])),
                " ".join(food.get("diseases", [])),
                " ".join(food.get("allergy", [])),
                str(food.get("meal_type", "")),
            ]
            docs.append(" ".join(doc_parts).lower())

        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform(docs)

    def _build_intent_patterns(self):
        self.intents = {
            "greeting": {
                "patterns": [r"\\b(hi|hello|hey)\\b", r"(مرحبا|اهلا|ازيك|ازاي|اه)"],
                "handler": self._handle_greeting,
            },
            "thanks": {
                "patterns": [r"(thanks|thank you|شكرا|تسلم)"],
                "handler": self._handle_thanks,
            },
            "help": {
                "patterns": [r"\\b(help|مساعدة)\\b"],
                "handler": self._handle_help,
            },
        }

    def _classify_intent(self, msg: str):
        msg_lower = msg.lower()
        for intent_name, intent_data in self.intents.items():
            for pattern in intent_data["patterns"]:
                match = re.search(pattern, msg_lower)
                if match:
                    return intent_name, match
        return "unknown", None

    def _handle_greeting(self, msg, match):
        return (
            "أهلاً بيك 👋\\n\\n"
            "أنا شات بوت التغذية الخاص بمشروعك.\\n"
            "بدخل على الداتا المحلية مباشرة من recipes.\\n\\n"
            "أمثلة:\\n"
            "- عايز أكل صحي\\n"
            "- i need healthy food for diabetes\\n"
            "- أعلى وجبة بروتين\\n"
            "- وجبات قليلة الكارب"
        )

    def _handle_thanks(self, msg, match):
        return "العفو 😊"

    def _handle_help(self, msg, match):
        return (
            "أقدر أساعدك في:\\n"
            "- البحث المباشر في recipes\\n"
            "- اقتراح وجبات صحية\\n"
            "- وجبات مناسبة للسكر أو القلب\\n"
            "- أعلى بروتين أو أقل سعرات\\n"
            "- فلترة حسب نوع الدايت والحساسية"
        )

    def _search_foods_by_text(self, query: str, top_n: int = 8):
        query_vec = self.vectorizer.transform([query.lower()])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        scored = [
            (self.foods[i], float(sims[i]))
            for i in range(len(self.foods))
            if sims[i] > 0.03
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def _dataset_first_candidates(self, user_msg: str, user_profile: Optional[Dict[str, Any]] = None, top_n: int = 8):
        matches = [food for food, _ in self._search_foods_by_text(user_msg, top_n=top_n)]

        if not matches:
            matches = self.foods[:top_n]

        if user_profile:
            filtered = []
            allergies = user_profile.get("allergies", [])
            diet_type = user_profile.get("diet_type", "balanced")
            health_conditions = user_profile.get("health_conditions", [])

            for food in matches:
                if recipe_has_allergy_conflict(food, allergies):
                    continue
                if not recipe_matches_diet(food, diet_type):
                    continue
                if not recipe_matches_conditions(food, health_conditions):
                    continue
                filtered.append(food)

            if filtered:
                matches = filtered

        return matches[:top_n]

    def _build_context(self, user_msg: str, user_profile: Optional[Dict[str, Any]] = None):
        matches = self._dataset_first_candidates(user_msg, user_profile=user_profile, top_n=8)

        foods_context = []
        for food in matches:
            foods_context.append({
                "name": food.get("name"),
                "meal_type": food.get("meal_type"),
                "calories": food.get("calories"),
                "protein": food.get("protein"),
                "carbs": food.get("carbs"),
                "fats": food.get("fats"),
                "diet": food.get("diet", []),
                "diseases": food.get("diseases", []),
                "allergy": food.get("allergy", []),
                "ingredients": food.get("ingredients", [])[:10],
            })

        return {
            "user_profile": user_profile or {},
            "matched_foods_from_dataset": foods_context,
            "user_question": user_msg,
        }

    def _local_fallback_answer(self, context: Dict[str, Any]) -> str:
        question = normalize_text(context["user_question"])
        foods = context.get("matched_foods_from_dataset", []) or []

        if not foods:
            return "مش لاقي نتائج مناسبة في الداتا الحالية."

        if "اعلى بروتين" in question or "أعلى بروتين" in question or "highest protein" in question:
            best = max(foods, key=lambda x: float(x.get("protein", 0)))
            return f"أعلى وجبة بروتين من الداتا هي **{best['name']}** وفيها **{best['protein']}g protein**."

        if "اقل سعرات" in question or "أقل سعرات" in question or "lowest calories" in question:
            best = min(foods, key=lambda x: float(x.get("calories", 0)))
            return f"أقل وجبة سعرات من الداتا هي **{best['name']}** وفيها **{best['calories']} kcal**."

        if "سكر" in question or "diabet" in question:
            diabetic = [f for f in foods if "diabetes" in [str(x).lower() for x in f.get("diseases", [])]]
            if diabetic:
                lines = [
                    f"- **{f['name']}** — {f['calories']} kcal, {f['protein']}g protein"
                    for f in diabetic[:4]
                ]
                return "دي وجبات من الداتا مناسبة لمرضى السكري:\\n\\n" + "\\n".join(lines)

        if "healthy" in question or "صحي" in question:
            lines = [
                f"- **{f['name']}** — {f['calories']} kcal, {f['protein']}g protein"
                for f in foods[:5]
            ]
            return "دي وجبات صحية من الداتا:\\n\\n" + "\\n".join(lines)

        lines = [
            f"- **{f['name']}** — {f['calories']} kcal, {f['protein']}g protein, {f['carbs']}g carbs"
            for f in foods[:5]
        ]
        return "دي أقرب وجبات من الداتا لسؤالك:\\n\\n" + "\\n".join(lines)

    def _call_openrouter_with_context(self, context: Dict[str, Any]) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY", "")

        if not api_key:
            return self._local_fallback_answer(context)

        system_prompt = """You are a friendly, knowledgeable nutrition assistant chatbot.
You have access to a local recipe dataset provided in the context.

RULES:
1. Use ONLY meals from the provided dataset — never invent meals or nutrition values.
2. Respond naturally and conversationally, like a helpful friend who knows about food.
3. Do NOT just list meals with their macros every time. Instead, have a real conversation:
   - Explain WHY you're suggesting a meal (e.g. 'This one is great because it's high in protein and low in carbs').
   - Ask follow-up questions if the user's request is vague.
   - Vary your response format — sometimes a short paragraph, sometimes a comparison, sometimes a tip.
4. Match the user's language: if they write in Arabic, reply in Arabic. If English, reply in English.
5. Be concise but warm. Don't repeat the same closing line every time.
6. If the user asks about health conditions, give general food suggestions from the dataset but never give medical advice.
7. Don't dump all nutritional info unless the user specifically asks for macros/details.
8. Respect allergies, diet type, and health conditions when user_profile is available.""".strip()

        user_prompt = f"""Here are some relevant meals from our database:
{json.dumps(context.get('matched_foods_from_dataset', []), ensure_ascii=False)}

User profile: {json.dumps(context.get('user_profile', {}), ensure_ascii=False)}

User question: {context.get('user_question', '')}""".strip()

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            pass

        return self._local_fallback_answer(context)

    def respond(self, user_msg: str, user_profile: Optional[Dict[str, Any]] = None, recommendations: Optional[List[Dict[str, Any]]] = None) -> str:
        msg = user_msg.strip()
        if not msg:
            return "اكتب سؤالك."

        intent, match = self._classify_intent(msg)
        if intent != "unknown":
            return self.intents[intent]["handler"](msg, match)

        context = self._build_context(user_msg=msg, user_profile=user_profile)
        return self._call_openrouter_with_context(context)
