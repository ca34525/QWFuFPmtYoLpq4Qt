# Happy Customers

Predict customer happiness from six survey answers. Compare models and
identify questions that a shorter survey could omit. The project target is
an F1 score of at least 0.73 for happy customers (`Y=1`).

## Run the analysis

Open a terminal in the project folder. Install the environment:

```powershell
uv sync
```

Place the survey CSV at `data/raw/ACME-HappinessSurvey2020.csv`.
It must contain `Y` (0 or 1) and `X1` through `X6` (scores from 1 to 5).
The script reads this file and keeps repeated responses.

Run with the defaults:

```powershell
uv run python -m happy_customers.train
```

The command supports four options:

| Option | Default | Purpose |
|---|---|---|
| `--seed` | `201` | Repeat the split, CV folds, and randomized model fitting. |
| `--scoring` | `f1` | Select models using `f1`, `f1_macro`, or `accuracy`. |
| `--test-size` | `0.12` | Reserve this fraction of responses for the final test. |
| `--cv-folds` | `10` | Set the number of stratified cross-validation folds. |

For example, select models using macro F1:

```powershell
uv run python -m happy_customers.train --seed 201 --scoring f1_macro --test-size 0.12 --cv-folds 10
```

Use `--help` to list these options. The six models, question counts (1 through 6),
parameter grids, and file paths are fixed in the source code.

## How it works

1. Validate the survey and reserve the test responses.
2. Measure a majority-class baseline using cross-validation (CV).
3. Search the fixed model parameters and question counts using the same CV folds.
4. Choose the highest mean score. Prefer fewer questions when scores tie exactly.
5. Fit the winner on all training responses, then evaluate it on the test responses.

Question selection and scaling happen inside each training fold. The notebook
uses the same Python functions and adds exploration, plots, and explanations.
Use CV results to select models and settings; keep test results for the final report.

Training tables show only the chosen metric and its standard deviation across
folds. The final test shows three metrics:

- **F1:** F1 for happy customers (`Y=1`).
- **Macro F1:** The mean of the two classes' F1 scores, with equal weight for each class.
- **Accuracy:** The fraction of correct predictions.

The 0.73 target always applies to happy-customer F1, including when another
metric guides model selection. See [scikit-learn's F1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

Each command-line run saves two files, replacing the previous run's outputs:

- `models/model.pkl`: the fitted pipeline, including question selection and scaling.
- `reports/training_results.json`: the four settings, chosen model and parameters,
  selected questions, CV scores, and final test scores.

## Files to read

| File | Contents |
|---|---|
| `happy_customers/config.py` | Defaults, columns, paths, and fixed parameter grids. |
| `happy_customers/train.py` | Data checks, model training, evaluation, and the command-line entry point. |
| `notebooks/1.0-CTA-customer-happiness-analysis.ipynb` | Exploration and results using the shared functions. |
| `tests/` | Checks for data validation, reproducibility, scoring, and saved outputs. |

This split between notebooks and reusable code follows
[Cookiecutter Data Science's guidance](https://cookiecutter-data-science.drivendata.org/opinions/#notebooks-are-for-exploration-and-communication-source-files-are-for-repetition).

## Run the checks

```powershell
uv run python -m pytest -q
uv run ruff check
uv run ruff format --check
```

Tests use generated survey data and smaller parameter grids. They do not require
the private CSV or replace the saved project model.
