# Task 1: Linear Regression House Price Predictor

## Project Overview

This project completes Task 1 of the Maincrafts Technology Artificial Intelligence & Machine Learning Internship. It builds and evaluates a Linear Regression model for predicting median house values using the California Housing dataset from scikit-learn.

The project demonstrates the complete beginner-friendly machine learning workflow: data loading, exploratory data analysis, data preparation, train-test splitting, model training, prediction, evaluation, visualization, reporting, and model persistence.

## Dataset Description

The dataset is loaded with:

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
```

Dataset characteristics:

- Total samples: 20,640
- Input features: 8
- Target variable: `MedHouseVal`
- Problem type: Regression
- Target unit: median house value in units of $100,000

Feature columns:

- `MedInc`: median income in block group
- `HouseAge`: median house age in block group
- `AveRooms`: average rooms per household
- `AveBedrms`: average bedrooms per household
- `Population`: block group population
- `AveOccup`: average household occupancy
- `Latitude`: block group latitude
- `Longitude`: block group longitude

## Technologies Used

- Python 3.11
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- pickle

## Project Structure

```text
Task-1-Linear-Regression/
├── task1_ml_linear_regression.ipynb
├── report.pdf
├── requirements.txt
├── README.md
├── house_model.pkl
├── generate_report.py
├── generate_notebook.py
└── plots/
    ├── histograms.png
    ├── correlation_heatmap.png
    ├── actual_vs_predicted.png
    └── residual_plot.png
```

## Installation Instructions

From this task directory, install the required dependencies:

```powershell
pip install -r requirements.txt
```

## How to Run

Generate the notebook, plots, saved model, and metrics:

```powershell
python generate_notebook.py
```

Generate the PDF report:

```powershell
python generate_report.py
```

Open and run the notebook:

```powershell
jupyter notebook task1_ml_linear_regression.ipynb
```

## Model Evaluation Summary

The notebook evaluates the Linear Regression model using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R2) Score

The final metrics are printed in the notebook and summarized in `report.pdf`.

## Key Findings

The strongest positive relationship with median house value is typically median income (`MedInc`). The linear model provides a useful baseline, but some residual patterns remain because housing prices are influenced by non-linear geographic, economic, and neighborhood-level factors.

## Future Improvements

- Apply Ridge Regression to reduce coefficient instability.
- Apply Lasso Regression for feature selection.
- Compare with Random Forest Regressor.
- Compare with Gradient Boosting Regressor.
- Use cross validation for more reliable evaluation.
- Engineer additional location and interaction features.
- Tune model hyperparameters.

## Deliverables

- `task1_ml_linear_regression.ipynb`: complete notebook with explanations, code, plots, and conclusions.
- `report.pdf`: professional 2-4 page report.
- `house_model.pkl`: saved Linear Regression model.
- `plots/`: generated EDA and model evaluation visualizations.
- `generate_notebook.py`: reproducible notebook and artifact generator.
- `generate_report.py`: reproducible PDF report generator.
