import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

from helpers import apply_quantile_binning, boxcox_transformation

# Load the dataset from a CSV file into a pandas DataFrame
ds_raw = pd.read_csv('../../Library/Application Support/JetBrains/PyCharmCE2024.2/scratches/credit_risk_dataset.csv')

# Display basic information about the dataset: data types, non-null values, etc.
ds_raw.info()

# Display the second row to verify the data content
print(ds_raw.iloc[1, :])

# Separate the target variable (loan_status) from the feature set
target = ds_raw['loan_status']
X = ds_raw.drop('loan_status', axis=1)

# Separate numerical and categorical columns
num_columns = X.select_dtypes(include=['number']).columns.tolist()  # Numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()  # Categorical columns

# Set seaborn plot theme for consistent styling
sns.set_theme(style="whitegrid")

# Plot distributions of categorical features
plt.figure(figsize=(15, 10))  # Set the figure size for the categorical plots

# Calculate the number of rows and columns for the grid layout of plots
n_cols = 2  # Two columns in the grid
n_rows = (len(categorical_columns) // n_cols) + (
    1 if len(categorical_columns) % n_cols != 0 else 0)  # Rows dynamically calculated

# Plot a countplot for each categorical feature
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.countplot(x=ds_raw[col], hue=ds_raw[col], palette='Set2', legend=False)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')

# Adjust the layout to prevent overlapping of the plots
plt.tight_layout()
plt.show()

# Map loan grades to numerical values (e.g., A -> 7, B -> 6, etc.)
grade_values = ds_raw['loan_grade'].unique().tolist()  # Get the unique values of loan_grade
grade_dict = dict(zip(sorted(grade_values), range(7, 0, -1)))  # Map grades to numbers (A -> 7, B -> 6, ...)

# Apply the mapping to the 'loan_grade' column
X['loan_grade'] = X['loan_grade'].map(grade_dict)
# Remove 'loan_grade' from the list of categorical columns
categorical_columns.remove('loan_grade')

# Create dummy variables (one-hot encoding) for the remaining categorical columns
X_dummies = pd.get_dummies(X[categorical_columns],
                           drop_first=True)  # Avoid multicollinearity by dropping the first category

# Drop the original categorical columns as they have been encoded
X = X.drop(categorical_columns, axis=1)

# Concatenate the dummy variables back to the dataset
X = pd.concat([X, X_dummies], axis=1)

# Plot distributions for numerical features
plt.figure(figsize=(15, 10))  # Set the figure size for the distribution plots

# Calculate the number of rows needed for the plot grid layout
n_cols = 3  # Number of columns in the plot grid
n_rows = (len(num_columns) // n_cols) + (1 if len(num_columns) % n_cols != 0 else 0)  # Calculate rows dynamically

# Loop through numerical columns and create a histogram with Kernel Density Estimation (KDE) for each
for i, col in enumerate(num_columns, 1):
    plt.subplot(n_rows, n_cols, i)  # Place the plot in a dynamic grid layout
    sns.histplot(X[col], kde=True, bins=30, color='skyblue', stat='density')  # Plot histogram with KDE
    plt.title(f'Distribution of {col}')  # Set title for each plot
    plt.xlabel(col)  # Set x-axis label
    plt.ylabel('Density')  # Set y-axis label to "Density" to represent normalized frequency

# Adjust the layout to prevent overlapping of plots
plt.tight_layout()
plt.show()

# Create box plots to visualize the distribution of numerical features by target variable (loan_status)
plt.figure(figsize=(12, 8))  # Set the figure size for boxplots

# Loop through each numerical column and create a boxplot for loan_status
for i, col in enumerate(num_columns, 1):
    plt.subplot(n_rows, n_cols, i)  # Position the plot in the grid layout
    sns.boxplot(x=target, y=X[col])  # Boxplot of the numerical feature against the target variable
    plt.title(f"Box plot: {col} vs Target")  # Set the title for the boxplot

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()

# Display the columns of the updated dataset (X)
print(X.columns.tolist())

# Apply quantile binning to categorize 'cb_person_cred_hist_length' into 4 categories
X = apply_quantile_binning(X, 'cb_person_cred_hist_length')

# Filter the 'person_age' column to remove any records where age is above 100 years
X = X[X['person_age'] <= 100]

# Apply quantile binning for 'person_age', 'person_income', and 'person_emp_length'
X = apply_quantile_binning(X, 'person_age')
X = apply_quantile_binning(X, 'person_income')
X = apply_quantile_binning(X, 'person_emp_length')

# Display the updated dataset to verify changes
print(X.iloc[1, :])  # Print the second row of the updated dataset
print(X.columns.tolist())  # Print the column names of the updated dataset

# Recalculate numerical columns after binning
num_columns = X.select_dtypes(['number']).columns.tolist()

# Create a figure with two subplots side by side to compare distributions
plt.figure(figsize=(18, 8))  # Adjust the figure size to fit both plots side by side

# Plot the original distribution of 'loan_amnt' before Box-Cox transformation (First subplot)
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
sns.histplot(ds_raw['loan_amnt'], kde=True, bins=30, color='skyblue', stat='density')  # Original distribution
plt.title('Original Distribution of Loan Amount')
plt.xlabel('Loan Amount')
plt.ylabel('Density')

# Apply Box-Cox transformation to 'loan_amnt' for normalization
X = boxcox_transformation(X, 'loan_amnt')

# Plot the transformed distribution of 'loan_amnt' (Second subplot)
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
sns.histplot(X['loan_amnt'], kde=True, bins=30, color='orange', stat='density')  # Transformed distribution
plt.title('Transformed Distribution of Loan Amount')
plt.xlabel('Loan Amount')
plt.ylabel('Density')

# Adjust layout to prevent overlap between subplots
plt.tight_layout()

# Display the plots
plt.show()

# Display the value counts for the 'loan_grade' column after transformation (optional)
print(X['loan_grade'].value_counts())  # Show how many records fall into each loan grade category

X = X.dropna()
target = target[X.index]

# Apply SMOTE for oversampling
smote = SMOTE(sampling_strategy='auto', random_state=42)  # Initialize SMOTE
X_resampled, y_resampled = smote.fit_resample(X, target)  # Resample features and target

# Plot the class distribution after oversampling
plt.figure(figsize=(8, 6))
sns.countplot(x=y_resampled)
plt.title('Class Distribution in Loan Status (After Oversampling)')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

# Recombine the resampled features and target
X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
y_resampled_df = pd.DataFrame(y_resampled, columns=['loan_status'])
# Concatenate the resampled features and target into one DataFrame
resampled_df = pd.concat([X_resampled_df, y_resampled_df], axis=1)

corr_matrix = resampled_df.corr()['loan_status'].drop('loan_status')

plt.figure(figsize=(20, 6))
sns.barplot(x=corr_matrix.index, y=corr_matrix.values, hue=corr_matrix.index, palette="coolwarm", legend=False)
plt.title("Correlation Between Features and Loan Status (After Oversampling)")
plt.xlabel("Features")
plt.ylabel("Correlation with Loan Status")
plt.xticks(rotation=45)
plt.show()


# Save the resampled dataset to a CSV file
resampled_df.to_csv('credit_risk_dataset_resampled.csv', index=False)
