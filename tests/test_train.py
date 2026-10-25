import json
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from happy_customers import train


def test_split_is_repeatable_stratified_and_disjoint(survey):
    first = train.split_data(survey, seed=17, test_size=0.25)
    second = train.split_data(survey, seed=17, test_size=0.25)
    X_train, X_test, y_train, y_test = first

    for left, right in zip(first, second):
        assert left.equals(right)
    assert len(X_train) == 60
    assert len(X_test) == 20
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.index) | set(X_test.index) == set(survey.index)
    assert y_train.mean() == y_test.mean() == 0.5
    assert not X_test.index.equals(train.split_data(survey, seed=18, test_size=0.25)[1].index)


@pytest.mark.parametrize("test_size", [-0.1, 0, 1, 1.1])
def test_split_rejects_invalid_test_fraction(survey, test_size):
    with pytest.raises(ValueError, match="test_size"):
        train.split_data(survey, test_size=test_size)


@pytest.mark.parametrize("scoring", ["f1", "f1_macro", "accuracy"])
def test_chosen_metric_controls_selection_and_columns(survey, small_search, scoring):
    X_train, _, y_train, _ = train.split_data(survey, seed=17, test_size=0.25)
    model, results, baseline = train.train_models(
        X_train, y_train, seed=17, scoring=scoring, cv_folds=3
    )

    assert set(results.filter(like="cv_").columns) == {f"cv_{scoring}", f"cv_{scoring}_std"}
    assert results[f"cv_{scoring}"].is_monotonic_decreasing
    assert set(results["model"]) == set(train.PARAMETER_GRIDS)
    assert set(results["questions"]) == {1, 2, 3, 4, 5, 6}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
    scores = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv)
    assert results.iloc[0][f"cv_{scoring}"] == pytest.approx(scores.mean())
    assert results.iloc[0][f"cv_{scoring}_std"] == pytest.approx(scores.std())
    baseline_scores = cross_val_score(
        DummyClassifier(strategy="most_frequent"), X_train, y_train, scoring=scoring, cv=cv
    )
    assert baseline == pytest.approx(baseline_scores.mean())


def test_training_is_repeatable(survey, small_search):
    X_train, X_test, y_train, _ = train.split_data(survey)
    first_model, first_results, first_baseline = train.train_models(X_train, y_train, cv_folds=3)
    second_model, second_results, second_baseline = train.train_models(
        X_train, y_train, cv_folds=3
    )

    pd.testing.assert_frame_equal(first_results, second_results)
    np.testing.assert_array_equal(first_model.predict(X_test), second_model.predict(X_test))
    assert first_baseline == second_baseline


def test_tied_scores_prefer_fewer_questions(survey, monkeypatch):
    pipeline = Pipeline(
        [("select", SelectKBest(f_classif)), ("model", DummyClassifier(strategy="most_frequent"))]
    )
    monkeypatch.setattr(train, "build_pipelines", lambda seed: {"Dummy": pipeline})
    monkeypatch.setattr(train, "PARAMETER_GRIDS", {"Dummy": {}})
    X_train, _, y_train, _ = train.split_data(survey)

    model, results, _ = train.train_models(X_train, y_train, cv_folds=3)

    assert results["cv_f1"].nunique() == 1
    assert results.iloc[0]["questions"] == 1
    assert model.named_steps["select"].get_support().sum() == 1


@pytest.mark.parametrize("cv_folds", [1, 41])
def test_training_rejects_invalid_fold_count(survey, cv_folds):
    with pytest.raises(ValueError, match="cv_folds"):
        train.train_models(survey.drop(columns="Y"), survey["Y"], cv_folds=cv_folds)


def test_training_rejects_unknown_scoring(survey):
    with pytest.raises(ValueError, match="scoring"):
        train.train_models(survey.drop(columns="Y"), survey["Y"], scoring="unknown")


def test_final_metrics_and_target_use_correct_f1_averages():
    X_test = np.zeros((4, 1))
    y_test = [0, 1, 1, 1]
    model = DummyClassifier(strategy="constant", constant=1).fit(X_test, y_test)

    predictions, scores = train.evaluate_model(model, X_test, y_test)

    np.testing.assert_array_equal(predictions, [1, 1, 1, 1])
    assert scores["test_f1"] == pytest.approx(6 / 7)
    assert scores["test_f1_macro"] == pytest.approx(3 / 7)
    assert scores["test_accuracy"] == 0.75
    assert scores["F1_target_met"] is True


def test_run_saves_settings_and_a_working_model(
    survey, small_search, tmp_path, monkeypatch, capsys
):
    raw_path = tmp_path / "survey.csv"
    model_path = tmp_path / "models" / "model.pkl"
    results_path = tmp_path / "reports" / "training_results.json"
    survey.to_csv(raw_path, index=False)
    monkeypatch.setattr(train, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(train, "MODEL_PATH", model_path)
    monkeypatch.setattr(train, "RESULTS_PATH", results_path)
    settings = {"seed": 17, "scoring": "f1_macro", "test_size": 0.25, "cv_folds": 3}

    report = train.run_training(**settings)

    assert json.loads(results_path.read_text()) == report
    assert report["settings"] == settings
    assert report["training_responses"] == 60
    assert report["test_responses"] == 20
    with model_path.open("rb") as model_file:
        restored_model = pickle.load(model_file)
    _, X_test, _, y_test = train.split_data(survey, seed=17, test_size=0.25)
    _, restored_scores = train.evaluate_model(restored_model, X_test, y_test)
    for name, value in restored_scores.items():
        assert report[name] == value
    assert len(report["selected_features"]) == restored_model.named_steps["select"].k
    training_output, final_output = capsys.readouterr().out.split("Final test:")
    assert "cv_f1_macro" in training_output
    assert "accuracy" not in training_output
    assert "test_f1_macro" in final_output
    assert "test_accuracy" in final_output


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], {"seed": 201, "scoring": "f1", "test_size": 0.12, "cv_folds": 10}),
        (
            ["--seed", "17", "--scoring", "f1_macro", "--test-size", "0.25", "--cv-folds", "3"],
            {"seed": 17, "scoring": "f1_macro", "test_size": 0.25, "cv_folds": 3},
        ),
    ],
)
def test_cli_passes_the_four_settings(monkeypatch, arguments, expected):
    calls = []
    monkeypatch.setattr(sys, "argv", ["train", *arguments])
    monkeypatch.setattr(train, "run_training", lambda **settings: calls.append(settings))

    train.main()

    assert calls == [expected]


@pytest.mark.parametrize(
    "option", ["--models", "--question-counts", "--n-jobs", "--input", "--output"]
)
def test_cli_rejects_options_outside_the_agreed_scope(monkeypatch, option):
    monkeypatch.setattr(sys, "argv", ["train", option, "1"])

    with pytest.raises(SystemExit) as error:
        train.main()

    assert error.value.code == 2
