import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
ds = pd.read_csv('credit_risk_dataset_resampled.csv')

# Print the second row of the dataset to inspect its values
print(ds.iloc[1, :])

# Define the target variable 'loan_status' and feature variables
X = ds.drop('loan_status', axis=1)  # Drop the target variable
y = ds['loan_status']  # The target variable

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest model
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)

# Get the feature importances
feature_importances = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=['importance'])

# Sort the features by importance
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Display the feature importances
print(feature_importances)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.index,
            y=feature_importances['importance'],
            hue=feature_importances.index,
            legend=False
            )
plt.xticks(rotation=90)
plt.title('Feature Importance - Random Forest')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Perform feature selection with SelectFromModel
sfm = SelectFromModel(rf, threshold=0.05, prefit=True)  # Select features with importance > 0.05
X_selected = sfm.transform(X_train)  # Transform training data to selected features

# Get the names of the selected features
selected_features = X.columns[sfm.get_support()]

# Print the selected features and their importance
for f in range(len(selected_features)):
    print(
        "%2d) %-*s %f"
        % (f + 1, 30, selected_features[f], feature_importances.loc[selected_features[f], "importance"])
    )
