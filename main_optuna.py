import shap
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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


# Define Optuna objective function
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 300)
    max_depth = trial.suggest_int('n_estimators', 10, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    max_features = trial.suggest_categorical('max_features',
                                             ['sqrt', 'log2'])

    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  max_features=max_features, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)

    return r2


# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # Change the number of trials

# Print the best parameters
print('Best hyperparameters:', study.best_params)

# Train the final model with the best hyperparameters
best_params = study.best_params
final_model = RandomForestRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)
final_predictions = final_model.predict(X_test)

# Evaluate the final model
final_mse = mean_squared_error(y_test, final_predictions)
final_r2 = r2_score(y_test, final_predictions)
print('Final Model MSE:', final_mse)
print('Final Model R-squared:', final_r2)

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
plt.savefig('Student_Performance_Optuna.pdf',
            bbox_inches='tight')  # Saves a PDF file
plt.close()
