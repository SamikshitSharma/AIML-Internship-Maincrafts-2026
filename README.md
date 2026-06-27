# AIML Internship Maincrafts 2026

A professional machine learning portfolio repository for the 4-week Artificial Intelligence & Machine Learning Internship at Maincrafts Technology.

## Internship Overview

This repository contains all internship tasks, experiments, notebooks, reports, models, and supporting assets completed during the Maincrafts Technology AIML Internship. It is structured to support clean development, reproducible analysis, technical reporting, and long-term portfolio presentation.

## About Maincrafts Technology AIML Internship

The internship focuses on applying core artificial intelligence and machine learning concepts through practical, task-based projects. Each task strengthens the complete machine learning workflow: problem understanding, data loading, exploratory analysis, preprocessing, model training, evaluation, documentation, reporting, and GitHub-based portfolio maintenance.

## Repository Objectives

- Maintain a professional record of all internship work.
- Organize each task in a dedicated, reproducible project folder.
- Demonstrate practical machine learning skills using Python and common data science libraries.
- Provide readable notebooks, reports, saved artifacts, metrics, and documentation suitable for GitHub, LinkedIn, resumes, and interviews.
- Build a scalable structure that can grow across the full internship duration.

## Technologies Used

- Python
- Jupyter Notebook
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Git and GitHub

## Internship Progress

| Task | Title | Status |
| --- | --- | --- |
| Task 1 | Linear Regression House Price Predictor | Completed ✅ |
| Task 2 | Feature Engineering, Model Optimization & Performance Comparison | Completed ✅ |
| Task 3 | Model Validation, Overfitting Control & Hyperparameter Tuning | Completed ✅ |
| Task 4 | Pending Assignment | Pending ⏳ |

## Task 2 Summary

Task 2 builds an enhanced California Housing prediction workflow with feature scaling, multiple regression models, model comparison, best-model selection, visualizations, a saved model artifact, and a professional PDF report.

### Models Evaluated

- Linear Regression
- Ridge Regression
- Decision Tree Regressor

### Results

| Model | RMSE | R2 Score |
| --- | ---: | ---: |
| Decision Tree Regressor | 0.7242 | 0.5997 |
| Ridge Regression | 0.7456 | 0.5758 |
| Linear Regression | 0.7456 | 0.5758 |

### Selected Best Model

Decision Tree Regressor

## Task 3 Summary

Task 3 adds a professional validation and tuning workflow to the California Housing prediction project. It focuses on identifying model generalization issues, improving Decision Tree performance through cross-validation and GridSearchCV, and selecting the final model using objective evaluation metrics.

### Objective

- Detect overfitting
- Perform cross-validation
- Apply GridSearchCV
- Optimize model performance
- Select the final model scientifically

### Concepts Covered

- Cross Validation
- GridSearchCV
- Hyperparameter Tuning
- Overfitting Detection
- Model Optimization

### Deliverables

- Jupyter Notebook
- PDF Report
- Optimized Model
- Cross Validation Results
- Hyperparameter Results
- Visualizations

### Validation and Tuning Results

| Model | RMSE | R2 Score |
| --- | ---: | ---: |
| Tuned Decision Tree | 0.6454 | 0.6821 |
| Original Decision Tree | 0.7242 | 0.5997 |
| Ridge Regression | 0.7456 | 0.5758 |
| Linear Regression | 0.7456 | 0.5758 |

### Selected Best Model

Tuned Decision Tree with `max_depth=10` and `min_samples_split=10`.

## Repository Structure

```text
AIML-Internship-Maincrafts-2026/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── assets/
│   └── README.md
├── Task-1-Linear-Regression/
│   ├── task1_ml_linear_regression.ipynb
│   ├── report.pdf
│   ├── README.md
│   ├── requirements.txt
│   ├── house_model.pkl
│   ├── metrics.json
│   ├── model_coefficients.csv
│   ├── sample_predictions.csv
│   ├── generate_notebook.py
│   ├── generate_report.py
│   └── plots/
├── Task-2-Feature-Engineering-Model-Optimization/
│   ├── AI_ML_Task2_Model_Comparison.ipynb
│   ├── report.pdf
│   ├── README.md
│   ├── requirements.txt
│   ├── best_model.joblib
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── sample_predictions.csv
│   ├── generate_notebook.py
│   ├── generate_report.py
│   ├── task2_pipeline.py
│   └── plots/
├── Task-3/
│   ├── AI_ML_Task3_Model_Validation_Tuning.ipynb
│   ├── report.pdf
│   ├── README.md
│   ├── requirements.txt
│   ├── best_model.joblib
│   ├── metrics.json
│   ├── cross_validation_results.csv
│   ├── hyperparameter_results.csv
│   ├── model_comparison.csv
│   ├── sample_predictions.csv
│   ├── generate_notebook.py
│   ├── generate_report.py
│   ├── task3_pipeline.py
│   └── plots/
└── Task-4/
    └── README.md
```

## Task Tracking

| Task | Topic | Status | Key Deliverables |
| --- | --- | --- | --- |
| Task 1 | Linear Regression House Price Predictor | Completed ✅ | Notebook, report, README, trained model, metrics, predictions, plots |
| Task 2 | Feature Engineering, Model Optimization & Performance Comparison | Completed ✅ | Notebook, report, saved model, metrics, comparison CSV, sample predictions, plots |
| Task 3 | Model Validation, Overfitting Control & Hyperparameter Tuning | Completed ✅ | Notebook, report, optimized model, cross-validation results, hyperparameter results, visualizations |
| Task 4 | Pending Assignment | Pending ⏳ | To be added |

## Future Scope

Planned improvements across the internship include:

- Add well-documented notebooks for each remaining task.
- Include model evaluation reports and visualizations for each project.
- Save reusable models and prediction scripts where appropriate.
- Add project-specific datasets or dataset access instructions.
- Expand documentation with setup steps, results, and learning outcomes.
- Maintain clean commits for each internship milestone.

## Contact / Author

**Author:** Milug  
**Program:** Maincrafts Technology Artificial Intelligence & Machine Learning Internship  
**Repository Purpose:** Internship submission and machine learning portfolio

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
