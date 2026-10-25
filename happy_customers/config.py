"""Defaults and fixed choices for the customer happiness analysis."""

from pathlib import Path

SEED = 201
SCORING = "f1"
TEST_SIZE = 0.12
CV_FOLDS = 10
SCORING_CHOICES = ("f1", "f1_macro", "accuracy")

TARGET_COLUMN = "Y"
FEATURE_COLUMNS = ["X1", "X2", "X3", "X4", "X5", "X6"]
F1_TARGET = 0.73

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ACME-HappinessSurvey2020.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
RESULTS_PATH = PROJECT_ROOT / "reports" / "training_results.json"

PARAMETER_GRIDS = {
    "Logistic regression": {
        "model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300],
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 9, 11, 15, 19],
        "model__weights": ["uniform", "distance"],
    },
    "SVM": {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": [0.001, 0.01, 0.1, 1],
    },
    "Decision tree": {
        "model__max_depth": [1, 2, 3, 4, 5, 6],
        "model__min_samples_leaf": [1, 2, 4, 6, 8],
    },
    "Random forest": {
        "model__n_estimators": [50, 100, 150, 200],
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__min_samples_leaf": [1, 2, 4, 6],
    },
    "XGBoost": {
        "model__n_estimators": [25, 50, 100, 150],
        "model__max_depth": [1, 2, 3, 4],
        "model__learning_rate": [0.01, 0.03, 0.1, 0.3],
    },
}
