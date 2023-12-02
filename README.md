# Student Performance Prediction Projects

## Overview
This repository contains two projects aimed at predicting student performance using a Random Forest Regressor. Both projects use `pandas`, `matplotlib`, `sklearn`, and `shap` for data handling, visualization, model training, and explanation. The main difference lies in the hyperparameter tuning approach:
- `main_grid` uses GridSearchCV for hyperparameter tuning.
- `main_optuna` employs Optuna for hyperparameter optimization.

## Dependencies
- `pandas`
- `matplotlib`
- `sklearn`
- `shap`
- `optuna` (for main_optuna)

## Common Steps in Both Projects
- Loading and cleaning data from 'Student_Performance.csv'.
- Creating new features and dropping low variance columns.
- Standardizing the data using `StandardScaler`.
- Splitting the data into training and test sets.
- Employing RandomForestRegressor for the prediction model.
- SHAP analysis to understand the impact of the features.

## Hyperparameter Tuning
### main_grid
- Uses GridSearchCV for finding the best hyperparameters.
- Saves the SHAP summary plot as 'Student_Performance_Grid.pdf'.

### main_optuna
- Uses Optuna for optimizing hyperparameters.
- Saves the SHAP summary plot as 'Student_Performance_Optuna.pdf'.

## Usage
1. Install the required dependencies.
2. Load your dataset (ensure it matches the expected format).
3. Run the respective script for hyperparameter tuning and model evaluation.
4. The SHAP plots will be saved as PDF files.

## License
[MIT](https://choosealicense.com/licenses/mit/)
