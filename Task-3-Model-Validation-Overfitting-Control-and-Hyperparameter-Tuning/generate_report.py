from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
REPORT_PATH = BASE_DIR / "report.pdf"


def wrap_line(text: str, width: int = 98) -> list[str]:
    words = text.split()
    lines = []
    current = []
    current_len = 0
    for word in words:
        next_len = current_len + len(word) + (1 if current else 0)
        if next_len > width and current:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = next_len
    if current:
        lines.append(" ".join(current))
    return lines


def add_wrapped(ax, y: float, text: str, *, size: float = 9.5, weight: str = "normal") -> float:
    for line in wrap_line(text):
        ax.text(0, y, line, fontsize=size, fontweight=weight, va="top")
        y -= 0.025
    return y


def add_text_page(pdf: PdfPages, metrics: dict) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.9])
    ax.axis("off")

    best = metrics["best_model_metrics"]
    cv = metrics["cross_validation"]
    grid = metrics["grid_search"]
    overfit = metrics["overfitting_analysis"]

    y = 0.98
    ax.text(0, y, "Model Validation, Overfitting Control & Hyperparameter Tuning", fontsize=15, fontweight="bold", va="top")
    y -= 0.045
    ax.text(0, y, "Maincrafts Technology AIML Internship - Task 3", fontsize=11, color="#333333", va="top")
    y -= 0.05

    sections = [
        ("Introduction", "This report summarizes a professional validation workflow for the California Housing prediction problem. The project extends the earlier model-comparison workflow with overfitting analysis, cross-validation, GridSearchCV tuning, optimized model evaluation, and reproducible model persistence."),
        ("Objective", "The objective is to detect overfitting, estimate model generalization with five-fold cross-validation, tune Decision Tree hyperparameters, and select the final regression model using RMSE and R2."),
        ("Methodology", f"The dataset contains {metrics['row_count']:,} rows, {len(metrics['features'])} features, and the target variable HousePrice. Features were scaled with {metrics['scaler']} and split into {metrics['train_size']:,} training rows and {metrics['test_size']:,} test rows using random_state={metrics['random_state']}. Baselines include Linear Regression, Ridge Regression, and the Task 2-style max_depth=5 Decision Tree."),
        ("Overfitting Analysis", f"The original Decision Tree achieved train RMSE {overfit['train_rmse']:.4f} and test RMSE {overfit['test_rmse']:.4f}, a gap of {overfit['rmse_gap']:.4f}. The lower training error indicates some overfitting, while the controlled max_depth keeps the gap from becoming extreme."),
        ("Cross Validation", f"Five-fold cross-validation produced a mean RMSE of {cv['mean_rmse']:.4f} with standard deviation {cv['std_rmse']:.4f}. This is more reliable than a single split because every fold becomes validation data once and the result reflects variability across multiple partitions."),
        ("GridSearchCV", f"GridSearchCV tuned max_depth values {grid['param_grid']['max_depth']} and min_samples_split values {grid['param_grid']['min_samples_split']} with cv={grid['cv']}. The best parameters were {grid['best_params']} with best CV RMSE {grid['best_cv_rmse']:.4f}."),
        ("Optimized Model", f"The best final model was {metrics['best_model']} with test RMSE {best['RMSE']:.4f} and R2 {best['R2 Score']:.4f}. It was selected by the rule: highest R2 Score, then lowest RMSE."),
        ("Conclusion", "The tuned Decision Tree balances predictive performance and generalization by controlling tree complexity with cross-validated hyperparameters. The workflow generates reusable artifacts, transparent metrics, and validation evidence suitable for portfolio review."),
    ]

    for title, body in sections:
        ax.text(0, y, title, fontsize=11, fontweight="bold", va="top")
        y -= 0.03
        y = add_wrapped(ax, y, body)
        y -= 0.014

    pdf.savefig(fig)
    plt.close(fig)


def add_results_page(pdf: PdfPages, metrics: dict) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.9])
    ax.axis("off")
    ax.text(0, 0.98, "Results Summary", fontsize=15, fontweight="bold", va="top")

    y = 0.92
    ax.text(0, y, "Model ranking", fontsize=11, fontweight="bold", va="top")
    y -= 0.035
    header = f"{'Rank':<6}{'Model':<28}{'RMSE':>10}{'R2':>10}"
    ax.text(0, y, header, fontsize=9.5, family="monospace", fontweight="bold", va="top")
    y -= 0.025
    for row in metrics["models"]:
        line = f"{int(row['Rank']):<6}{row['Model']:<28}{row['RMSE']:>10.4f}{row['R2 Score']:>10.4f}"
        ax.text(0, y, line, fontsize=9.5, family="monospace", va="top")
        y -= 0.024

    y -= 0.02
    ax.text(0, y, "Cross-validation fold RMSE", fontsize=11, fontweight="bold", va="top")
    y -= 0.035
    for index, value in enumerate(metrics["cross_validation"]["fold_rmse"], start=1):
        ax.text(0, y, f"Fold {index}: {value:.4f}", fontsize=9.8, va="top")
        y -= 0.024
    y -= 0.01
    ax.text(0, y, f"Mean: {metrics['cross_validation']['mean_rmse']:.4f}", fontsize=9.8, va="top")
    y -= 0.024
    ax.text(0, y, f"Std: {metrics['cross_validation']['std_rmse']:.4f}", fontsize=9.8, va="top")

    y -= 0.055
    justification = "Model justification: The final model was selected because it delivered the strongest held-out test performance after complexity controls were tuned through cross-validation. GridSearchCV improves confidence by avoiding manual hyperparameter choice and comparing candidate settings on multiple validation folds. The chosen depth and split threshold reduce variance compared with a deeper unrestricted tree while preserving non-linear predictive power."
    ax.text(0, y, "Model Justification", fontsize=11, fontweight="bold", va="top")
    y -= 0.032
    add_wrapped(ax, y, justification)

    pdf.savefig(fig)
    plt.close(fig)


def add_visual_page(pdf: PdfPages) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))
    fig.suptitle("Task 3 Visualizations", fontsize=15, fontweight="bold", y=0.985)
    image_files = [
        ("Train vs Test RMSE", PLOTS_DIR / "train_vs_test_rmse.png"),
        ("Cross-Validation Scores", PLOTS_DIR / "cross_validation_scores.png"),
        ("Grid Search Results", PLOTS_DIR / "grid_search_results.png"),
        ("Model Comparison", PLOTS_DIR / "model_comparison.png"),
        ("Feature Importance", PLOTS_DIR / "feature_importance.png"),
    ]
    flat_axes = axes.ravel()
    for ax, (title, path) in zip(flat_axes, image_files):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    flat_axes[-1].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    pdf.savefig(fig)
    plt.close(fig)


def generate_report() -> None:
    metrics = json.loads((BASE_DIR / "metrics.json").read_text(encoding="utf-8"))
    with PdfPages(REPORT_PATH) as pdf:
        add_text_page(pdf, metrics)
        add_results_page(pdf, metrics)
        add_visual_page(pdf)
    print(f"Generated {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()
