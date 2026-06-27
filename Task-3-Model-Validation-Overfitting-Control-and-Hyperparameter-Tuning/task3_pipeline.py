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
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
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
RANDOM_STATE = 42


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy().rename(columns={"MedHouseVal": "HousePrice"})
    X = df[FEATURE_NAMES]
    y = df["HousePrice"]
    return df, X, y


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_task3() -> dict:
    PLOTS_DIR.mkdir(exist_ok=True)
    df, X, y = load_dataset()

    missing_values = {column: int(value) for column, value in df.isna().sum().items()}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=RANDOM_STATE
    )

    baseline_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Original Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
    }

    fitted_models = {}
    predictions_by_model = {}
    baseline_rows = []
    for name, model in baseline_models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        baseline_rows.append(
            {
                "Model": name,
                "RMSE": rmse(y_test, predictions),
                "R2 Score": float(r2_score(y_test, predictions)),
                "Model Type": "Baseline",
            }
        )
        fitted_models[name] = model
        predictions_by_model[name] = predictions

    original_tree = fitted_models["Original Decision Tree"]
    train_predictions = original_tree.predict(X_train)
    test_predictions = predictions_by_model["Original Decision Tree"]
    overfitting_analysis = {
        "model": "Original Decision Tree",
        "train_rmse": rmse(y_train, train_predictions),
        "test_rmse": rmse(y_test, test_predictions),
        "rmse_gap": rmse(y_test, test_predictions) - rmse(y_train, train_predictions),
        "interpretation": "The baseline Decision Tree has a lower train RMSE than test RMSE, indicating some overfitting. The controlled max_depth keeps the gap moderate compared with an unrestricted tree.",
    }

    cv_model = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        cv_model,
        X_scaled,
        y,
        cv=5,
        scoring="neg_root_mean_squared_error",
    )
    cv_rmse = -cv_scores
    cross_validation = pd.DataFrame(
        {
            "Fold": np.arange(1, len(cv_rmse) + 1),
            "RMSE": cv_rmse,
        }
    )
    cross_validation.loc[len(cross_validation)] = ["Mean", float(cv_rmse.mean())]
    cross_validation.loc[len(cross_validation)] = ["Std", float(cv_rmse.std())]
    cross_validation.to_csv(BASE_DIR / "cross_validation_results.csv", index=False)

    param_grid = {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10],
    }
    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    hyperparameter_results = pd.DataFrame(grid_search.cv_results_)
    hyperparameter_results = hyperparameter_results[
        [
            "param_max_depth",
            "param_min_samples_split",
            "mean_train_score",
            "std_train_score",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ].copy()
    hyperparameter_results["mean_train_rmse"] = -hyperparameter_results["mean_train_score"]
    hyperparameter_results["mean_cv_rmse"] = -hyperparameter_results["mean_test_score"]
    hyperparameter_results["std_cv_rmse"] = hyperparameter_results["std_test_score"]
    hyperparameter_results = hyperparameter_results.sort_values("rank_test_score").reset_index(drop=True)
    hyperparameter_results.to_csv(BASE_DIR / "hyperparameter_results.csv", index=False)

    tuned_tree = grid_search.best_estimator_
    tuned_tree.fit(X_train, y_train)
    tuned_predictions = tuned_tree.predict(X_test)
    tuned_row = {
        "Model": "Tuned Decision Tree",
        "RMSE": rmse(y_test, tuned_predictions),
        "R2 Score": float(r2_score(y_test, tuned_predictions)),
        "Model Type": "Optimized",
    }
    comparison = pd.DataFrame([*baseline_rows, tuned_row]).sort_values(
        by=["R2 Score", "RMSE"], ascending=[False, True]
    ).reset_index(drop=True)
    comparison.insert(0, "Rank", np.arange(1, len(comparison) + 1))
    comparison.to_csv(BASE_DIR / "model_comparison.csv", index=False)

    best_model_name = str(comparison.loc[0, "Model"])
    all_models = {**fitted_models, "Tuned Decision Tree": tuned_tree}
    all_predictions = {**predictions_by_model, "Tuned Decision Tree": tuned_predictions}
    best_model = all_models[best_model_name]
    best_predictions = all_predictions[best_model_name]

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
        "best_hyperparameters": grid_search.best_params_,
        "cv_scoring": "negative root mean squared error",
    }
    joblib.dump(model_bundle, BASE_DIR / "best_model.joblib")
    reloaded = joblib.load(BASE_DIR / "best_model.joblib")
    reloaded_predictions = reloaded["model"].predict(X_test[:5])

    metrics = {
        "project_title": "Model Validation, Overfitting Control & Hyperparameter Tuning",
        "dataset": "California Housing dataset from sklearn.datasets.fetch_california_housing(as_frame=True)",
        "target_variable": "HousePrice",
        "features": FEATURE_NAMES,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values": missing_values,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "test_size_ratio": 0.20,
        "random_state": RANDOM_STATE,
        "scaler": "StandardScaler",
        "baseline_models": baseline_rows,
        "overfitting_analysis": overfitting_analysis,
        "cross_validation": {
            "model": "Original Decision Tree",
            "cv": 5,
            "fold_rmse": [float(x) for x in cv_rmse],
            "mean_rmse": float(cv_rmse.mean()),
            "std_rmse": float(cv_rmse.std()),
        },
        "grid_search": {
            "model": "DecisionTreeRegressor",
            "param_grid": param_grid,
            "cv": 5,
            "scoring": "neg_root_mean_squared_error",
            "best_params": grid_search.best_params_,
            "best_cv_rmse": float(-grid_search.best_score_),
        },
        "models": comparison.to_dict(orient="records"),
        "best_model": best_model_name,
        "best_model_metrics": comparison.loc[0].to_dict(),
        "selection_rule": "Highest R2 Score; if tied, lowest RMSE",
        "reloaded_model_prediction_sample": [float(x) for x in reloaded_predictions],
    }
    (BASE_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_train_vs_test(overfitting_analysis)
    plot_cross_validation(cv_rmse)
    plot_grid_search(hyperparameter_results)
    plot_model_comparison(comparison)
    plot_feature_importance(best_model, best_model_name)

    return metrics


def plot_train_vs_test(overfitting_analysis: dict) -> None:
    labels = ["Train RMSE", "Test RMSE"]
    values = [overfitting_analysis["train_rmse"], overfitting_analysis["test_rmse"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(labels, values, color=["#3366cc", "#cc3311"], label="RMSE")
    ax.set_ylabel("RMSE")
    ax.set_title("Decision Tree Overfitting Check: Train vs Test RMSE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "train_vs_test_rmse.png", dpi=180)
    plt.close(fig)


def plot_cross_validation(cv_rmse: np.ndarray) -> None:
    folds = np.arange(1, len(cv_rmse) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(folds, cv_rmse, marker="o", linewidth=2, color="#3366cc", label="Fold RMSE")
    ax.axhline(cv_rmse.mean(), color="#109618", linestyle="--", label=f"Mean RMSE {cv_rmse.mean():.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE")
    ax.set_title("5-Fold Cross-Validation Scores")
    ax.set_xticks(folds)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cross_validation_scores.png", dpi=180)
    plt.close(fig)


def plot_grid_search(hyperparameter_results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for depth, group in hyperparameter_results.sort_values("param_min_samples_split").groupby("param_max_depth"):
        ax.plot(
            group["param_min_samples_split"].astype(int),
            group["mean_cv_rmse"],
            marker="o",
            linewidth=2,
            label=f"max_depth={depth}",
        )
    ax.set_xlabel("min_samples_split")
    ax.set_ylabel("Mean CV RMSE")
    ax.set_title("GridSearchCV Results for Decision Tree")
    ax.grid(alpha=0.25)
    ax.legend(title="Tree depth")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "grid_search_results.png", dpi=180)
    plt.close(fig)


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
    ax1.set_title("Model Comparison: Baseline vs Tuned Model")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_comparison.png", dpi=180)
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
    ax.barh(np.asarray(FEATURE_NAMES)[order], values[order], color="#ff9900", label=label)
    ax.set_xlabel(label)
    ax.set_title(f"Feature Importance - {model_name}")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    result = evaluate_task3()
    print(json.dumps(result["best_model_metrics"], indent=2))
