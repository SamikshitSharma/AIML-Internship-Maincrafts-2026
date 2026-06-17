
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


def add_text_page(pdf: PdfPages, metrics: dict) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.9])
    ax.axis("off")

    best = metrics["best_model_metrics"]
    models = metrics["models"]
    lines = [
        "Feature Engineering, Model Optimization & Performance Comparison",
        "Maincrafts Technology AIML Internship - Task 2",
        "",
        "Introduction: This report summarizes an enhanced house price prediction workflow using the California Housing dataset. The project demonstrates feature scaling, multiple regression models, model evaluation, comparison, visualization, and final model persistence.",
        "",
        "Objective: Compare Linear Regression, Ridge Regression, and Decision Tree Regressor models using RMSE and R2 Score, then select the best model using highest R2 Score and lowest RMSE as the tie-breaker.",
        "",
        f"Dataset: {metrics['row_count']:,} records, {len(metrics['features'])} input features, target variable HousePrice. Features used: {', '.join(metrics['features'])}.",
        "",
        f"Feature Engineering: The data was inspected for missing values, separated into features and target, and scaled using {metrics['scaler']} before the 80/20 train-test split.",
        "",
        "Models Evaluated:",
    ]
    for model in models:
        lines.append(f"- {model['Model']}: RMSE={model['RMSE']:.4f}, R2={model['R2 Score']:.4f}")
    lines.extend([
        "",
        f"Best Model: {metrics['best_model']} was selected with RMSE={best['RMSE']:.4f} and R2={best['R2 Score']:.4f}.",
        "",
        "Conclusion: The selected model achieved the strongest test-set explanatory performance under the required selection rule. The workflow is reproducible and generates saved metrics, comparison tables, sample predictions, plots, and a reusable model bundle.",
    ])

    y = 0.98
    for idx, line in enumerate(lines):
        if idx == 0:
            ax.text(0, y, line, fontsize=16, fontweight="bold", va="top")
            y -= 0.045
            continue
        if idx == 1:
            ax.text(0, y, line, fontsize=11, color="#333333", va="top")
            y -= 0.04
            continue
        if line == "":
            y -= 0.022
            continue
        wrapped = wrap_line(line)
        for part in wrapped:
            ax.text(0, y, part, fontsize=10.2, va="top")
            y -= 0.026
    pdf.savefig(fig)
    plt.close(fig)


def add_visual_page(pdf: PdfPages) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
    fig.suptitle("Task 2 Visualizations", fontsize=15, fontweight="bold", y=0.985)
    image_files = [
        ("Model Comparison", PLOTS_DIR / "model_comparison.png"),
        ("Actual vs Predicted", PLOTS_DIR / "actual_vs_predicted.png"),
        ("Feature Importance", PLOTS_DIR / "feature_importance.png"),
    ]
    for ax, (title, path) in zip(axes, image_files):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)


def generate_report() -> None:
    metrics = json.loads((BASE_DIR / "metrics.json").read_text(encoding="utf-8"))
    with PdfPages(REPORT_PATH) as pdf:
        add_text_page(pdf, metrics)
        add_visual_page(pdf)
    print(f"Generated {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()
