import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load and clean the data
df = pd.read_csv('Student_Performance.csv')
df['Extracurricular Activities'] = df['Extracurricular Activities'].map(
    {'Yes': 1, 'No': 0})
df.fillna(df.mean(), inplace=True)

# Create new parameter
df['Study Sleep Interaction'] = df['Hours Studied'] ** df['Sleep Hours']

# Check for low variance features
variance = df.var()
low_variance_columns = variance[variance < 0.1].index
df.drop(low_variance_columns, axis=1, inplace=True)

# Standardization of the data
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('Performance Index', axis=1))
y = df['Performance Index']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Narrowed hyperparameters based on the previous tests
param_grid_refined = {
    'n_estimators': [190],
    'max_depth': [15],
    'min_samples_split': [2],
    'max_features': ['sqrt']
}

# Initialize the Random Forest Regressor
rf_regressor_refined = RandomForestRegressor(random_state=42)

# Hyperparameter optimization with refined GridSearchCV, add different values
# for hyperparameters and modify n_jobs to your environment
grid_search_refined = GridSearchCV(estimator=rf_regressor_refined,
                                   param_grid=param_grid_refined, cv=2,
                                   n_jobs=4, verbose=2)
grid_search_refined.fit(X_train, y_train)

# Best parameters and best score after refined tuning
print('Refined best parameters from Grid Search:')
print(grid_search_refined.best_params_)
best_rf_regressor_refined = grid_search_refined.best_estimator_

# Re-fit the best model
best_rf_regressor_refined.fit(X_train, y_train)
refined_predictions = best_rf_regressor_refined.predict(X_test)
print('Refined Random Forest Regressor Performance:')
print('MSE:', mean_squared_error(y_test, refined_predictions))
print('R-squared:', r2_score(y_test, refined_predictions))

# Sample a smaller subset of data for SHAP value calculation
sample_size = 1000  # Adjust this number based on your system's capability
X_sample = shap.utils.sample(X_train, sample_size)

# Create a simpler model for SHAP analysis
simplified_model = RandomForestRegressor(n_estimators=50, max_depth=5,
                                         random_state=42)
simplified_model.fit(X_train, y_train)

# Create the SHAP explainer and calculate SHAP values on the smaller sample
explainer = shap.TreeExplainer(simplified_model)
shap_values = explainer.shap_values(X_sample)

# Create a SHAP summary plot
shap.summary_plot(shap_values, X_sample,
                  feature_names=df.drop('Performance Index', axis=1).columns,
                  show=False)

# Save the plot to a file
plt.savefig('Student_Performance.pdf', bbox_inches='tight')  # Saves a PDF file
plt.close()
