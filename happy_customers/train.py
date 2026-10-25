"""Train and evaluate models with the same workflow as the notebook."""

import argparse
import json
import pickle

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from happy_customers.config import (
    CV_FOLDS,
    F1_TARGET,
    FEATURE_COLUMNS,
    MODEL_PATH,
    PARAMETER_GRIDS,
    RAW_DATA_PATH,
    RESULTS_PATH,
    SCORING,
    SCORING_CHOICES,
    SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_data():
    """Read the fixed survey CSV and return a validated DataFrame.

    Repeated responses are retained. Raise ValueError for invalid survey data.
    """
    data = pd.read_csv(RAW_DATA_PATH)
    if data.empty:
        raise ValueError("The dataset is empty.")
    if set(data.columns) != {TARGET_COLUMN, *FEATURE_COLUMNS}:
        raise ValueError("Expected columns Y and X1 through X6.")
    if data.isna().any().any():
        raise ValueError("The dataset contains missing values.")
    if set(data[TARGET_COLUMN]) != {0, 1}:
        raise ValueError("Y must contain both happiness labels: 0 and 1.")
    if not data[FEATURE_COLUMNS].isin([1, 2, 3, 4, 5]).all().all():
        raise ValueError("Survey scores must be integers from 1 to 5.")
    return data


def split_data(data, seed=SEED, test_size=TEST_SIZE):
    """Split validated survey data; return X_train, X_test, y_train, y_test.

    Use seed to repeat the stratified split. test_size is a fraction from 0 to 1.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    return train_test_split(
        data[FEATURE_COLUMNS],
        data[TARGET_COLUMN],
        test_size=test_size,
        stratify=data[TARGET_COLUMN],
        random_state=seed,
    )


def build_pipelines(seed=SEED):
    """Return the six model pipelines, using seed for randomized models."""
    models = {
        "Logistic regression": LogisticRegression(random_state=seed),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(random_state=seed),
        "Decision tree": DecisionTreeClassifier(random_state=seed),
        "Random forest": RandomForestClassifier(random_state=seed),
        "XGBoost": XGBClassifier(random_state=seed),
    }
    pipelines = {}
    for name, model in models.items():
        scaler = (
            StandardScaler() if name in ["Logistic regression", "KNN", "SVM"] else "passthrough"
        )
        pipelines[name] = Pipeline(
            [("select", SelectKBest(score_func=f_classif)), ("scale", scaler), ("model", model)]
        )
    return pipelines


def train_models(X_train, y_train, seed=SEED, scoring=SCORING, cv_folds=CV_FOLDS):
    """Search the fixed grids using only training data and the chosen scoring.

    seed controls shuffled stratified folds and randomized models. cv_folds
    must be at least 2 and no greater than the smallest class count.
    Return the fitted winner, sorted CV results, and the mean baseline score.
    Print each model name as its search starts.
    """
    if scoring not in SCORING_CHOICES:
        raise ValueError(f"scoring must be one of {SCORING_CHOICES}.")
    if not 2 <= cv_folds <= y_train.value_counts().min():
        raise ValueError("cv_folds must be between 2 and the smallest training class count.")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    baseline_score = cross_val_score(
        DummyClassifier(strategy="most_frequent"), X_train, y_train, cv=cv, scoring=scoring
    ).mean()
    pipelines = build_pipelines(seed)
    score_column = f"cv_{scoring}"
    results = []
    for name, pipeline in pipelines.items():
        print(f"Searching {name}...", flush=True)
        search = GridSearchCV(
            pipeline,
            param_grid={"select__k": [1, 2, 3, 4, 5, 6], **PARAMETER_GRIDS[name]},
            scoring=scoring,
            refit=False,
            cv=cv,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        cv_results = search.cv_results_
        results.append(
            pd.DataFrame(
                {
                    "model": name,
                    "questions": cv_results["param_select__k"].astype(int),
                    score_column: cv_results["mean_test_score"],
                    f"{score_column}_std": cv_results["std_test_score"],
                    "params": cv_results["params"],
                }
            )
        )

    # Prefer fewer questions when mean CV scores tie exactly.
    results_df = (
        pd.concat(results, ignore_index=True)
        .sort_values([score_column, "questions"], ascending=[False, True])
        .reset_index(drop=True)
    )
    best = results_df.iloc[0]
    final_model = clone(pipelines[best["model"]]).set_params(**best["params"])
    final_model.fit(X_train, y_train)
    return final_model, results_df, float(baseline_score)


def evaluate_model(model, X_test, y_test):
    """Return test predictions and F1, macro F1, accuracy, and the F1 target check."""
    predictions = model.predict(X_test)
    test_f1 = f1_score(y_test, predictions, zero_division=0)
    scores = {
        "test_f1": float(test_f1),
        "test_f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "F1_target_met": bool(test_f1 >= F1_TARGET),
    }
    return predictions, scores


def run_training(seed=SEED, scoring=SCORING, test_size=TEST_SIZE, cv_folds=CV_FOLDS):
    """Run the analysis with four settings and return its report dictionary.

    Print the CV and test summaries. Save the pipeline and JSON report to the
    fixed project paths, replacing the previous run's generated files.
    """
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data, seed, test_size)
    model, results, baseline_score = train_models(X_train, y_train, seed, scoring, cv_folds)
    print(f"\nMajority-class mean CV {scoring}: {baseline_score:.3f}")
    print(results.drop_duplicates("model").drop(columns="params").round(3).to_string(index=False))

    best = results.iloc[0]
    selected_features = [
        column
        for column, keep in zip(FEATURE_COLUMNS, model.named_steps["select"].get_support())
        if keep
    ]
    _, test_scores = evaluate_model(model, X_test, y_test)
    report = {
        "settings": {
            "seed": seed,
            "scoring": scoring,
            "test_size": test_size,
            "cv_folds": cv_folds,
        },
        "training_responses": len(y_train),
        "test_responses": len(y_test),
        "model": best["model"],
        "parameters": best["params"],
        "selected_features": selected_features,
        "baseline_cv_score": baseline_score,
        "cv_score": float(best[f"cv_{scoring}"]),
        "cv_score_std": float(best[f"cv_{scoring}_std"]),
        **test_scores,
    }
    print(f"\nSelected model: {report['model']}")
    print(f"Selected parameters: {report['parameters']}")
    print(f"Keep: {', '.join(selected_features)}")
    print("\nFinal test:")
    print(pd.DataFrame([test_scores]).round(3).to_string(index=False))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved report: {RESULTS_PATH}")
    return report


def main():
    """Read the four command-line options and run the analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed (default: %(default)s)."
    )
    parser.add_argument(
        "--scoring",
        choices=SCORING_CHOICES,
        default=SCORING,
        help="Metric used for model selection (default: %(default)s).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Fraction reserved for the final test (default: %(default)s).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=CV_FOLDS,
        help="Number of stratified CV folds (default: %(default)s).",
    )
    run_training(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
