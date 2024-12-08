import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Import joblib for saving the model


# Load the dataset
ds = pd.read_csv('../../Library/Application Support/JetBrains/PyCharmCE2024.2/scratches/credit_risk_dataset_resampled.csv')

# Define the features and target variable
selected_features = [
    'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'person_home_ownership_RENT', 'cb_person_cred_hist_length_binned',
    'person_age_binned', 'person_income_binned', 'person_emp_length_binned'
]

# Add all features related to 'person_home_ownership'
home_ownership_columns = [col for col in ds.columns if 'person_home_ownership' in col]
selected_features.extend(home_ownership_columns)

# Drop duplicates
selected_features = list(set(selected_features))

# Define X (features) and y (target variable)
X = ds[selected_features]
y = ds['loan_status']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class weights based on the distribution of classes
# class_weights = y_train.value_counts(normalize=True).to_dict()
# weight_pos_class = class_weights[0] / class_weights[1]

# Define the Optuna optimization function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-5, 1e-1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        # 'scale_pos_weight': weight_pos_class
    }

    # Initialize the model with suggested hyperparameters
    model = XGBClassifier(**params, random_state=42)

    # Perform cross-validation to evaluate the model
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    # Return the mean cross-validation score as the optimization objective (maximize accuracy)
    return cv_scores.mean()

# Create an Optuna study
study = optuna.create_study(direction='maximize')

# Optimize the objective function
study.optimize(objective, n_trials=60)

# Print the best trial found by Optuna
print("Best hyperparameters:", study.best_params)
print("Best cross-validation accuracy: {:.2f}".format(study.best_value))


# Use the best parameters to train the final model
best_params = study.best_params
xgb_best = XGBClassifier(**best_params, random_state=42)
xgb_best.fit(X_train, y_train)

# Predict and evaluate the model on the test dataset
y_pred = xgb_best.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test dataset with optimized parameters: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the best model to a file using joblib
joblib.dump(xgb_best, 'xgb_best_model.joblib')
print("Model has been saved to 'xgb_best_model.joblib'")
