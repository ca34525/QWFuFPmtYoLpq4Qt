import numpy as np
import pandas as pd
import pytest

from happy_customers import train
from happy_customers.config import FEATURE_COLUMNS


@pytest.fixture
def survey():
    """Build a small, repeatable survey without reading the private CSV."""
    rng = np.random.default_rng(12)
    data = pd.DataFrame(rng.integers(1, 6, size=(80, 6)), columns=FEATURE_COLUMNS)
    data["Y"] = np.tile([0, 1], 40)
    data["X1"] = np.minimum(data["X1"] + data["Y"], 5)
    return data


@pytest.fixture
def small_search(monkeypatch):
    """Keep all six pipelines but shorten their grids for fast tests."""
    grids = {
        name: {parameter: values[:1] for parameter, values in grid.items()}
        for name, grid in train.PARAMETER_GRIDS.items()
    }
    grids["Random forest"]["model__n_estimators"] = [3]
    grids["XGBoost"]["model__n_estimators"] = [3]
    monkeypatch.setattr(train, "PARAMETER_GRIDS", grids)
