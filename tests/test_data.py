import numpy as np
import pandas as pd
import pytest

from happy_customers import train


def test_load_data_keeps_repeated_responses(survey, tmp_path, monkeypatch):
    data = pd.concat([survey, survey.iloc[[0]]], ignore_index=True)
    path = tmp_path / "survey.csv"
    data.to_csv(path, index=False)
    monkeypatch.setattr(train, "RAW_DATA_PATH", path)

    pd.testing.assert_frame_equal(train.load_data(), data)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("X1", np.nan, "missing values"),
        ("Y", 2, "happiness labels"),
        ("X1", 0, "Survey scores"),
        ("X1", 6, "Survey scores"),
        ("X1", 2.5, "Survey scores"),
    ],
)
def test_load_data_rejects_invalid_values(survey, tmp_path, monkeypatch, column, value, message):
    data = survey.astype(float)
    data.loc[0, column] = value
    path = tmp_path / "survey.csv"
    data.to_csv(path, index=False)
    monkeypatch.setattr(train, "RAW_DATA_PATH", path)

    with pytest.raises(ValueError, match=message):
        train.load_data()


@pytest.mark.parametrize("problem", ["empty", "missing_column", "extra_column", "one_class"])
def test_load_data_rejects_invalid_structure(survey, tmp_path, monkeypatch, problem):
    invalid_data = {
        "empty": survey.iloc[:0],
        "missing_column": survey.drop(columns="X6"),
        "extra_column": survey.assign(extra=1),
        "one_class": survey.assign(Y=1),
    }
    path = tmp_path / "survey.csv"
    invalid_data[problem].to_csv(path, index=False)
    monkeypatch.setattr(train, "RAW_DATA_PATH", path)

    with pytest.raises(ValueError):
        train.load_data()
