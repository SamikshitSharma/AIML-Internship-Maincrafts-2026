# Model Validation, Overfitting Control & Hyperparameter Tuning

**Maincrafts Technology - Artificial Intelligence & Machine Learning Internship**  
**Task 3**

This project enhances the California Housing prediction workflow with professional model validation, overfitting analysis, cross-validation, GridSearchCV hyperparameter tuning, optimized model comparison, and reusable model persistence.

## Objective

- Detect overfitting by comparing train and test RMSE.
- Use 5-fold cross-validation to estimate generalization performance.
- Tune Decision Tree hyperparameters with GridSearchCV.
- Compare baseline and tuned models using RMSE and R2 Score.
- Save the best model and generate reproducible portfolio artifacts.

## Concepts Covered

- Train-test validation
- Overfitting and generalization gap analysis
- 5-fold cross-validation
- Negative RMSE scoring with `cross_val_score`
- Hyperparameter tuning with `GridSearchCV`
- Regression metrics: RMSE and R2 Score
- Model comparison and selection
- Model persistence with Joblib

## Dataset

The project uses the California Housing dataset from scikit-learn:

```python
fetch_california_housing(as_frame=True)
```

The original target column `MedHouseVal` is renamed to `HousePrice`.

## Workflow

1. Load and inspect the California Housing dataset.
2. Select the eight official predictor features.
3. Apply `StandardScaler`, matching the Task 2 preprocessing approach.
4. Split data into 80% training and 20% testing sets with `random_state=42`.
5. Train Linear Regression, Ridge Regression, and baseline Decision Tree models.
6. Analyze Decision Tree train vs test RMSE for overfitting.
7. Run 5-fold cross-validation with negative RMSE scoring.
8. Tune Decision Tree `max_depth` and `min_samples_split` with GridSearchCV.
9. Compare all models and select the best by highest R2 Score, then lowest RMSE.
10. Save the best model bundle and generate reports, CSV outputs, and plots.

## Cross Validation

The baseline Decision Tree was evaluated with 5-fold cross-validation.

| Fold | RMSE |
| ---: | ---: |
| 1 | 0.8509 |
| 2 | 0.7457 |
| 3 | 0.7558 |
| 4 | 0.8935 |
| 5 | 0.8244 |
| Mean | 0.8141 |
| Std | 0.0563 |

Cross-validation is more reliable than a single train-test split because every fold is used for validation once, making the estimate less dependent on one random partition.

## Hyperparameter Tuning

GridSearchCV tuned the following Decision Tree parameters:

```python
max_depth = [3, 5, 7, 10]
min_samples_split = [2, 5, 10]
cv = 5
```

Best hyperparameters:

```text
max_depth = 10
min_samples_split = 10
```

Best cross-validated RMSE: `0.6366`

## Results

| Rank | Model | RMSE | R2 Score | Type |
| ---: | --- | ---: | ---: | --- |
| 1 | Tuned Decision Tree | 0.6454 | 0.6821 | Optimized |
| 2 | Original Decision Tree | 0.7242 | 0.5997 | Baseline |
| 3 | Ridge Regression | 0.7456 | 0.5758 | Baseline |
| 4 | Linear Regression | 0.7456 | 0.5758 | Baseline |

## Final Model

The selected final model is the **Tuned Decision Tree**. It was selected because it achieved the strongest held-out test performance after cross-validated tuning and reduced prediction error compared with the original Decision Tree baseline.

## Repository Structure

```text
Task-3-Model-Validation-Overfitting-Control-and-Hyperparameter-Tuning/
├── AI_ML_Task3_Model_Validation_Tuning.ipynb
├── report.pdf
├── README.md
├── requirements.txt
├── best_model.joblib
├── metrics.json
├── cross_validation_results.csv
├── hyperparameter_results.csv
├── model_comparison.csv
├── sample_predictions.csv
├── generate_notebook.py
├── generate_report.py
├── task3_pipeline.py
└── plots/
    ├── train_vs_test_rmse.png
    ├── cross_validation_scores.png
    ├── grid_search_results.png
    ├── model_comparison.png
    └── feature_importance.png
```

## Installation

From this directory:

```bash
pip install -r requirements.txt
```

## Execution Instructions

Run the full pipeline:

```bash
python task3_pipeline.py
```

Regenerate the notebook:

```bash
python generate_notebook.py
```

Regenerate the PDF report after pipeline outputs exist:

```bash
python generate_report.py
```

## Future Improvements

- Use a scikit-learn `Pipeline` to fit scaling only inside each training fold.
- Add nested cross-validation for more conservative model-selection estimates.
- Compare ensemble models such as Random Forest and Gradient Boosting.
- Add residual plots and prediction interval analysis.

