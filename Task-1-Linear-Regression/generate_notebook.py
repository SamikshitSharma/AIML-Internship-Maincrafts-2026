"""Generate the Task 1 notebook, plots, metrics, and saved model.

This script is intentionally reproducible: running it from the Task 1 directory
recreates the notebook and all required machine learning artifacts.
"""

from __future__ import annotations

import json
import pickle
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_dataframe() -> pd.DataFrame:
    """Load California Housing as a pandas DataFrame with target included."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    return df


def train_and_generate_artifacts() -> dict[str, float | int | str]:
    """Train the baseline model and save required visual artifacts."""
    sns.set_theme(style="whitegrid", context="notebook")
    df = load_dataframe()

    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    df.hist(figsize=(14, 10), bins=30, color="#3B82F6", edgecolor="white")
    plt.suptitle("California Housing Feature and Target Distributions", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(PLOTS_DIR / "histograms.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.45, color="#2563EB", edgecolors="none")
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="#DC2626", linewidth=2)
    plt.title("Actual vs Predicted Median House Values")
    plt.xlabel("Actual Median House Value")
    plt.ylabel("Predicted Median House Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.45, color="#059669", edgecolors="none")
    plt.axhline(y=0, color="#DC2626", linestyle="--", linewidth=2)
    plt.title("Residual Plot")
    plt.xlabel("Predicted Median House Value")
    plt.ylabel("Residuals")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residual_plot.png", dpi=180, bbox_inches="tight")
    plt.close()

    with (BASE_DIR / "house_model.pkl").open("wb") as model_file:
        pickle.dump(model, model_file)

    coefficients = pd.DataFrame(
        {"Feature": X.columns, "Coefficient": model.coef_}
    ).sort_values("Coefficient", ascending=False)
    coefficients.to_csv(BASE_DIR / "model_coefficients.csv", index=False)

    comparison = pd.DataFrame(
        {"Actual Values": y_test.iloc[:10].values, "Predicted Values": y_pred[:10]}
    )
    comparison.to_csv(BASE_DIR / "sample_predictions.csv", index=False)

    metrics = {
        "samples": int(df.shape[0]),
        "features": int(X.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "intercept": round(float(model.intercept_), 4),
        "strongest_positive_correlation": "MedInc",
        "notebook": "task1_ml_linear_regression.ipynb",
    }
    (BASE_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def markdown_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def build_notebook(metrics: dict[str, float | int | str]) -> None:
    """Create the required educational Jupyter Notebook."""
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    }

    cells = [
        markdown_cell(
            """
            # Task 1: Build and Evaluate a Linear Regression Model

            **Project:** House Price Predictor  
            **Internship:** Maincrafts Technology Artificial Intelligence & Machine Learning Internship  
            **Dataset:** California Housing dataset from scikit-learn
            """
        ),
        markdown_cell(
            """
            ## Project Objective

            The objective of this project is to build a beginner-friendly Linear Regression model that predicts median house values using the California Housing dataset. The notebook demonstrates the complete machine learning workflow: data loading, exploratory data analysis, data preparation, model training, evaluation, visualization, model saving, and final conclusions.
            """
        ),
        markdown_cell("## Import Libraries\n\nThe required Python libraries are imported below."),
        code_cell(
            """
            import pickle
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from sklearn.datasets import fetch_california_housing
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split

            sns.set_theme(style="whitegrid", context="notebook")
            PLOTS_DIR = Path("plots")
            PLOTS_DIR.mkdir(exist_ok=True)
            """
        ),
        markdown_cell(
            "## Dataset Loading\n\nThe California Housing dataset is loaded directly from scikit-learn using `fetch_california_housing(as_frame=True)`. The target variable is `MedHouseVal`, which represents median house value in units of $100,000."
        ),
        code_cell(
            """
            housing = fetch_california_housing(as_frame=True)
            df = housing.frame.copy()
            df.head()
            """
        ),
        markdown_cell("## Dataset Overview\n\nThe following cells display the dataset shape, first five rows, dataset information, summary statistics, and missing-value counts."),
        code_cell("print(f\"Dataset shape: {df.shape}\")\ndf.head()"),
        code_cell("df.info()"),
        code_cell("df.describe()"),
        code_cell("df.isnull().sum()"),
        markdown_cell(
            """
            ## Exploratory Data Analysis (EDA)

            EDA helps us understand data distributions, relationships between variables, and potential data quality issues before model training.
            """
        ),
        code_cell(
            """
            df.hist(figsize=(14, 10), bins=30, color="#3B82F6", edgecolor="white")
            plt.suptitle("California Housing Feature and Target Distributions", fontsize=16)
            plt.tight_layout(rect=(0, 0, 1, 0.97))
            plt.savefig(PLOTS_DIR / "histograms.png", dpi=180, bbox_inches="tight")
            plt.show()
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(10, 8))
            correlation_matrix = df.corr(numeric_only=True)
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ### Correlation Interpretation

            The strongest positive correlation with `MedHouseVal` is typically `MedInc`, meaning areas with higher median income tend to have higher median house values. Latitude and longitude also show meaningful relationships because housing values vary by geography. Correlation does not prove causation, but it helps identify useful predictive features.
            """
        ),
        markdown_cell("## Data Preparation\n\nThe feature matrix `X` contains all input columns. The target vector `y` contains `MedHouseVal`."),
        code_cell(
            """
            X = df.drop(columns="MedHouseVal")
            y = df["MedHouseVal"]

            print(f"Feature matrix shape: {X.shape}")
            print(f"Target vector shape: {y.shape}")
            """
        ),
        markdown_cell("## Train-Test Split\n\nThe dataset is split into 80% training data and 20% testing data using `random_state=42` for reproducibility."),
        code_cell(
            """
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42
            )

            print(f"Training samples: {X_train.shape[0]}")
            print(f"Testing samples: {X_test.shape[0]}")
            """
        ),
        markdown_cell("## Model Training\n\nA Linear Regression model is trained using the training dataset."),
        code_cell(
            """
            model = LinearRegression()
            model.fit(X_train, y_train)

            print(f"Intercept: {model.intercept_:.4f}")

            coefficients = pd.DataFrame({
                "Feature": X.columns,
                "Coefficient": model.coef_
            }).sort_values("Coefficient", ascending=False)

            coefficients
            """
        ),
        markdown_cell(
            """
            ### Coefficient Explanation

            A coefficient estimates how much the target value changes when a feature increases by one unit, assuming the other features stay constant. Positive coefficients increase the prediction, while negative coefficients decrease it. Because features use different units, coefficient size should be interpreted carefully.
            """
        ),
        markdown_cell("## Predictions\n\nThe trained model is used to predict house values for the test set."),
        code_cell(
            """
            y_pred = model.predict(X_test)

            comparison = pd.DataFrame({
                "Actual Values": y_test.iloc[:10].values,
                "Predicted Values": y_pred[:10]
            })

            comparison
            """
        ),
        markdown_cell("## Model Evaluation\n\nThe model is evaluated using MAE, RMSE, and R2 Score."),
        code_cell(
            """
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"R2 Score: {r2:.4f}")
            """
        ),
        markdown_cell(
            f"""
            ### Metric Interpretation

            - **MAE:** The model is off by about `{metrics['mae']}` in target units on average. Since the target is measured in $100,000 units, this is roughly `{float(metrics['mae']) * 100000:,.0f}` dollars.
            - **RMSE:** The RMSE is `{metrics['rmse']}`, which penalizes larger errors more heavily than MAE.
            - **R2 Score:** The R2 score is `{metrics['r2']}`, meaning the linear model explains a meaningful portion of the variation in house values, but there is room for improvement.
            """
        ),
        markdown_cell("## Visualization\n\nThe following plots compare actual vs predicted values and examine residuals."),
        code_cell(
            """
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.45, color="#2563EB", edgecolors="none")
            min_value = min(y_test.min(), y_pred.min())
            max_value = max(y_test.max(), y_pred.max())
            plt.plot([min_value, max_value], [min_value, max_value], color="#DC2626", linewidth=2)
            plt.title("Actual vs Predicted Median House Values")
            plt.xlabel("Actual Median House Value")
            plt.ylabel("Predicted Median House Value")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=180, bbox_inches="tight")
            plt.show()
            """
        ),
        code_cell(
            """
            residuals = y_test - y_pred

            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred, residuals, alpha=0.45, color="#059669", edgecolors="none")
            plt.axhline(y=0, color="#DC2626", linestyle="--", linewidth=2)
            plt.title("Residual Plot")
            plt.xlabel("Predicted Median House Value")
            plt.ylabel("Residuals")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "residual_plot.png", dpi=180, bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ### Residual Interpretation

            The residuals are centered around zero, which is expected for a fitted regression model. However, the spread is not perfectly random across all predicted values. This suggests that a simple linear model is a useful baseline, but non-linear models and additional feature engineering may improve performance.
            """
        ),
        markdown_cell("## Model Saving\n\nThe trained model is saved with pickle so it can be loaded later for inference."),
        code_cell(
            """
            with open("house_model.pkl", "wb") as file:
                pickle.dump(model, file)

            with open("house_model.pkl", "rb") as file:
                loaded_model = pickle.load(file)

            loaded_prediction = loaded_model.predict(X_test.iloc[[0]])
            print(f"Loaded model sample prediction: {loaded_prediction[0]:.4f}")
            """
        ),
        markdown_cell(
            """
            ## Conclusions

            The Linear Regression model successfully predicts California median house values using the provided eight numeric features. Median income is the most important positive indicator in the dataset, while geographic features also influence model behavior. The model is reproducible and suitable as a baseline regression solution.
            """
        ),
        markdown_cell(
            """
            ## Future Improvements

            - Compare against Ridge Regression and Lasso Regression.
            - Train tree-based models such as Random Forest Regressor and Gradient Boosting Regressor.
            - Use cross validation to obtain more reliable performance estimates.
            - Add feature engineering, especially for location-based patterns.
            - Tune hyperparameters for stronger predictive performance.
            """
        ),
    ]

    nb["cells"] = cells
    with (BASE_DIR / "task1_ml_linear_regression.ipynb").open("w", encoding="utf-8") as file:
        nbf.write(nb, file)


def main() -> None:
    metrics = train_and_generate_artifacts()
    build_notebook(metrics)
    print("Generated Task 1 notebook and artifacts.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
