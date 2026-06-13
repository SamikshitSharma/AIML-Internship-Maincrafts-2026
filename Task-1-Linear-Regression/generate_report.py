"""Generate the required Task 1 PDF report."""

from __future__ import annotations

import json
from pathlib import Path

from fpdf import FPDF


BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"


class ReportPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 64, 175)
        self.cell(0, 8, "Maincrafts Technology AIML Internship - Task 1", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def section(pdf: FPDF, title: str, body: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 7, title)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5.4, body)
    pdf.ln(2)


def add_plot(pdf: FPDF, title: str, image_name: str, width: int = 86) -> None:
    path = PLOTS_DIR / image_name
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 5.5, title)
    pdf.set_x(pdf.l_margin)
    pdf.image(str(path), w=width)
    pdf.ln(2)


def main() -> None:
    metrics = json.loads((BASE_DIR / "metrics.json").read_text(encoding="utf-8"))

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 10, "Build and Evaluate a Linear Regression Model")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 7, "House Price Predictor using the California Housing Dataset")
    pdf.ln(3)

    section(
        pdf,
        "1. Introduction",
        "This report summarizes Task 1 of the Maincrafts Technology Artificial Intelligence and Machine Learning Internship. The project builds a baseline Linear Regression model to predict median house values using the California Housing dataset.",
    )
    section(
        pdf,
        "2. Objective",
        "The objective is to demonstrate the complete machine learning workflow: loading data, exploring patterns, preparing features, splitting the dataset, training a model, evaluating results, visualizing predictions, saving the model, and identifying future improvements.",
    )
    section(
        pdf,
        "3. Dataset Overview",
        f"The dataset contains {metrics['samples']} samples and {metrics['features']} input features. The target variable is MedHouseVal, representing median house value in units of $100,000. The feature set includes income, house age, room and bedroom averages, population, occupancy, latitude, and longitude.",
    )
    section(
        pdf,
        "4. Methodology",
        f"The data was split into {metrics['train_samples']} training samples and {metrics['test_samples']} testing samples using an 80/20 split with random_state=42. A scikit-learn LinearRegression model was trained on the feature matrix and evaluated on unseen test data.",
    )

    pdf.add_page()
    section(
        pdf,
        "5. Exploratory Data Analysis",
        "Histograms were used to inspect the distribution of each numerical variable. A correlation heatmap was used to study relationships between input features and the target. Median income showed the strongest positive relationship with median house value, while geographic variables also contributed meaningful signal.",
    )
    add_plot(pdf, "Feature and Target Distributions", "histograms.png", width=145)
    add_plot(pdf, "Correlation Heatmap", "correlation_heatmap.png", width=92)

    pdf.add_page()
    section(
        pdf,
        "6. Model Training Process",
        f"The Linear Regression model learned an intercept of {metrics['intercept']}. Coefficients were mapped to feature names to understand how each feature contributes to the prediction while holding other variables constant.",
    )
    section(
        pdf,
        "7. Evaluation Results",
        f"Mean Absolute Error (MAE): {metrics['mae']}\nRoot Mean Squared Error (RMSE): {metrics['rmse']}\nR2 Score: {metrics['r2']}\n\nMAE shows the average absolute prediction error. RMSE penalizes larger errors more strongly. R2 indicates how much variation in median house values is explained by the model.",
    )
    add_plot(pdf, "Actual vs Predicted Values", "actual_vs_predicted.png", width=92)
    add_plot(pdf, "Residual Plot", "residual_plot.png", width=92)

    pdf.add_page()
    section(
        pdf,
        "8. Key Findings",
        "The model provides a clear and reproducible baseline for house value prediction. Median income is a strong predictor of median house value. The actual vs predicted plot shows the model captures the general trend, while residual patterns indicate that a purely linear model does not capture all underlying complexity.",
    )
    section(
        pdf,
        "9. Limitations",
        "Linear Regression assumes a linear relationship between input features and the target. Housing prices are influenced by non-linear, geographic, and socioeconomic factors, so a baseline linear model may underperform compared with more flexible algorithms.",
    )
    section(
        pdf,
        "10. Future Improvements",
        "Recommended improvements include Ridge Regression, Lasso Regression, Random Forest Regressor, Gradient Boosting Regressor, cross validation, feature engineering, and location-based feature interactions.",
    )
    section(
        pdf,
        "11. Conclusion",
        "Task 1 was completed successfully. The project demonstrates a complete machine learning workflow and produces all required deliverables: notebook, PDF report, saved model, requirements file, reproducible generator scripts, and evaluation plots.",
    )

    pdf.output(BASE_DIR / "report.pdf")
    print("Generated report.pdf")


if __name__ == "__main__":
    main()
