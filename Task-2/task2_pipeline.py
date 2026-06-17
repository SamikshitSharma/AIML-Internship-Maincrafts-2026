
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    df = df.rename(columns={"MedHouseVal": "HousePrice"})
    X = df[FEATURE_NAMES]
    y = df["HousePrice"]
    return df, X, y


def evaluate_models() -> dict:
    PLOTS_DIR.mkdir(exist_ok=True)
    df, X, y = load_dataset()

    missing_values = {column: int(value) for column, value in df.isna().sum().items()}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=42),
    }

    rows = []
    fitted_models = {}
    predictions_by_model = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))
        rows.append({"Model": name, "RMSE": rmse, "R2 Score": r2})
        fitted_models[name] = model
        predictions_by_model[name] = predictions

    comparison = pd.DataFrame(rows).sort_values(
        by=["R2 Score", "RMSE"], ascending=[False, True]
    ).reset_index(drop=True)
    best_model_name = str(comparison.loc[0, "Model"])
    best_model = fitted_models[best_model_name]
    best_predictions = predictions_by_model[best_model_name]

    comparison.to_csv(BASE_DIR / "model_comparison.csv", index=False)

    sample_predictions = pd.DataFrame(
        {
            "Actual Values": y_test.reset_index(drop=True).iloc[:20].round(4),
            "Predicted Values": np.round(best_predictions[:20], 4),
        }
    )
    sample_predictions.to_csv(BASE_DIR / "sample_predictions.csv", index=False)

    model_bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "target_name": "HousePrice",
        "selection_rule": "Highest R2 Score, then lowest RMSE",
    }
    joblib.dump(model_bundle, BASE_DIR / "best_model.joblib")

    reloaded = joblib.load(BASE_DIR / "best_model.joblib")
    reloaded_predictions = reloaded["model"].predict(X_test[:5])

    metrics = {
        "project_title": "Feature Engineering, Model Optimization & Performance Comparison",
        "dataset": "California Housing dataset from sklearn.datasets.fetch_california_housing(as_frame=True)",
        "target_variable": "HousePrice",
        "features": FEATURE_NAMES,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values": missing_values,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "test_size_ratio": 0.20,
        "random_state": 42,
        "scaler": "StandardScaler",
        "models": comparison.to_dict(orient="records"),
        "best_model": best_model_name,
        "best_model_metrics": comparison.loc[0].to_dict(),
        "selection_rule": "Highest R2 Score; if tied, lowest RMSE",
        "reloaded_model_prediction_sample": [float(x) for x in reloaded_predictions],
    }
    (BASE_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_model_comparison(comparison)
    plot_actual_vs_predicted(y_test, best_predictions, best_model_name)
    plot_feature_importance(best_model, best_model_name)

    return metrics


def plot_model_comparison(comparison: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(comparison))
    width = 0.35

    ax1.bar(x - width / 2, comparison["RMSE"], width, color="#3366cc", label="RMSE")
    ax1.set_ylabel("RMSE (lower is better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison["Model"], rotation=15, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, comparison["R2 Score"], width, color="#109618", label="R2 Score")
    ax2.set_ylabel("R2 Score (higher is better)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper center")
    ax1.set_title("Model Comparison: RMSE and R2 Score")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def plot_actual_vs_predicted(y_test: pd.Series, predictions: np.ndarray, best_model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.scatter(y_test, predictions, alpha=0.35, s=18, color="#3366cc", edgecolors="none")
    lower = min(float(y_test.min()), float(predictions.min()))
    upper = max(float(y_test.max()), float(predictions.max()))
    ax.plot([lower, upper], [lower, upper], color="#cc3311", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual HousePrice")
    ax.set_ylabel("Predicted HousePrice")
    ax.set_title(f"Actual vs Predicted Values - {best_model_name}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)


def plot_feature_importance(model, model_name: str) -> None:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        label = "Feature Importance"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_))
        label = "Absolute Coefficient Magnitude"
    else:
        values = np.zeros(len(FEATURE_NAMES))
        label = "Importance Proxy"

    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh(np.asarray(FEATURE_NAMES)[order], values[order], color="#ff9900")
    ax.set_xlabel(label)
    ax.set_title(f"Feature Importance - {model_name}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    result = evaluate_models()
    print(json.dumps(result["best_model_metrics"], indent=2))
