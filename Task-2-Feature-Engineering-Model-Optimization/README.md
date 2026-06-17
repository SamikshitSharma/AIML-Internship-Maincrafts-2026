# Feature Engineering, Model Optimization & Performance Comparison

## Project Overview

This project completes Maincrafts Technology AIML Internship Task 2. It builds an enhanced house price prediction workflow using the California Housing dataset and compares multiple regression models using a consistent evaluation process.

The workflow includes feature inspection, feature scaling, train-test validation, model training, metric comparison, visualization, best-model selection, and model persistence for reuse.

## Objective

Compare multiple regression algorithms for house price prediction and select the best-performing model using:

- Highest R2 Score
- Lowest RMSE as the tie-breaker

## Dataset Information

- Dataset: California Housing dataset from `sklearn.datasets.fetch_california_housing(as_frame=True)`
- Rows: 20,640
- Features: 8
- Target variable: `HousePrice`
- Missing values: none detected across the modeling dataset

### Features Used

- `MedInc`
- `HouseAge`
- `AveRooms`
- `AveBedrms`
- `Population`
- `AveOccup`
- `Latitude`
- `Longitude`

## Models Evaluated

- Linear Regression
- Ridge Regression with `alpha=1.0`
- Decision Tree Regressor with `max_depth=5` and `random_state=42`

## Evaluation Metrics

- RMSE: Root Mean Squared Error. Lower values indicate smaller prediction error.
- R2 Score: Coefficient of determination. Higher values indicate stronger explanatory performance.

## Results Table

| Rank | Model | RMSE | R2 Score |
|---:|---|---:|---:|
| 1 | Decision Tree Regressor | 0.7242 | 0.5997 |
| 2 | Ridge Regression | 0.7456 | 0.5758 |
| 3 | Linear Regression | 0.7456 | 0.5758 |

## Best Model

The selected best model is **Decision Tree Regressor**.

- RMSE: `0.7242`
- R2 Score: `0.5997`
- Selection rule: Highest R2 Score; if tied, lowest RMSE

The Decision Tree Regressor performs best in this required comparison because it can capture non-linear relationships in the California Housing data while the linear models are limited to linear patterns. Linear and Ridge Regression remain more interpretable, but the Decision Tree gives the strongest test-set R2 score in this workflow.

## Visualizations

The project generates the following plots:

- `plots/model_comparison.png`
- `plots/actual_vs_predicted.png`
- `plots/feature_importance.png`

## Project Files

- `AI_ML_Task2_Model_Comparison.ipynb`: executable notebook with explanations and outputs
- `task2_pipeline.py`: reproducible pipeline used to generate metrics, plots, model, and predictions
- `generate_notebook.py`: notebook generator
- `generate_report.py`: PDF report generator
- `report.pdf`: 1-2 page professional report
- `best_model.joblib`: saved best model bundle with scaler and metadata
- `metrics.json`: structured evaluation results
- `model_comparison.csv`: ranked model comparison table
- `sample_predictions.csv`: 20 actual-vs-predicted examples
- `requirements.txt`: Python dependencies

## Installation

From the repository root or this Task 2 folder, install dependencies with:

```bash
pip install -r requirements.txt
```

## Reproducibility Instructions

Run the following commands from `Task-2`:

```bash
python task2_pipeline.py
python generate_notebook.py
python -m nbconvert --to notebook --execute AI_ML_Task2_Model_Comparison.ipynb --output AI_ML_Task2_Model_Comparison.ipynb --ExecutePreprocessor.timeout=300
python generate_report.py
```

## Validation Summary

Validated on Python 3.11 with scikit-learn 1.5.2. The notebook executes top-to-bottom, required model artifacts are generated, the saved model bundle reloads successfully, and all expected plots, CSVs, JSON metrics, notebook, and PDF report are present.

## Future Improvements

- Add cross-validation for more robust performance estimates
- Tune Decision Tree hyperparameters with grid search
- Test ensemble models such as Random Forest or Gradient Boosting
- Add residual plots and error analysis by geography
- Use a scikit-learn Pipeline for production-style preprocessing and modeling
