from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
ds = pd.read_csv('credit_risk_dataset_resampled.csv')

# Select columns related to 'person_home_ownership'
home_ownership_columns = [col for col in ds.columns if 'person_home_ownership' in col]

# Extract the data for PCA
home_ownership_data = ds[home_ownership_columns]

# Standardize the data
scaler = StandardScaler()
home_ownership_scaled = scaler.fit_transform(home_ownership_data)

# Apply PCA to reduce to one component
pca = PCA(n_components=1)
home_ownership_pca = pca.fit_transform(home_ownership_scaled)

# Add the PCA component back to the dataset
ds['person_home_ownership_pca'] = home_ownership_pca

# Drop the original 'person_home_ownership' columns
# ds = ds.drop(columns=home_ownership_columns)

# Inspect the dataset with the new PCA feature
print(ds[['person_home_ownership_pca']].head())

ds.to_csv('pca_data.csv', index=False)