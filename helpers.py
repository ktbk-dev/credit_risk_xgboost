import kaggle
import pandas as pd
from typing import List, Optional
from scipy import stats
import numpy as np


def get_kaggle_ds(dataset, path='.', unzip=True):
    kaggle.api.dataset_download_files(dataset=dataset, path=path, unzip=unzip)
    return f'{dataset.split('/')[-1]}.csv'.replace('-', '_')


def apply_quantile_binning(data: pd.DataFrame, column_name: str,
                           labels: Optional[List[int]] = [1, 2, 3, 4]) -> pd.DataFrame:
    """
    Apply quantile-based binning to the given column of the DataFrame and drop the original column.

    This function exactly mimics the process from the provided code, including calculating the quantiles,
    defining bin edges, applying the binning, and then dropping the original column.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the column to be binned.
    - column_name (str): The name of the column to be binned.
    - num_bins (int): Number of bins (default is 4, for quartiles).
    - labels (Optional[List[int]]): Optional list of integer labels for the bins (e.g., [1, 2, 3, 4]).

    Returns:
    - pd.DataFrame: The DataFrame with the binned column and the original column removed.
    """
    # Calculate the quartiles (25%, 50%, 75%)
    Q1 = data[column_name].quantile(0.25)
    Q2 = data[column_name].quantile(0.5)  # Median
    Q3 = data[column_name].quantile(0.75)

    # Calculate the minimum and maximum values
    min_value = data[column_name].min()
    max_value = data[column_name].max()

    # Define the bin edges (min, Q1, Q2, Q3, max)
    bins = [min_value, Q1, Q2, Q3, max_value]

    # Apply binning to the column using pd.cut
    data[f'{column_name}_binned'] = pd.cut(data[column_name], bins=bins, labels=labels, include_lowest=True)

    # Drop the original column after binning
    data = data.drop(column_name, axis=1)

    return data

    # Apply binning to the column based on quantiles
    data[f'{column_name}_binned'] = pd.cut(data[column_name], bins=quantiles, labels=labels, include_lowest=True)

    # Drop the original column after binning (optional)
    data = data.drop(column_name, axis=1, inplace=True)

    return data


def boxcox_transformation(df: pd.DataFrame, column_name: str, lambda_val: Optional[float] = None):
    """
    Applies the Box-Cox transformation to a specified column in a pandas DataFrame.

    Parameters:
    - df: pandas DataFrame, the DataFrame containing the data.
    - column_name: str, the name of the column to transform.
    - lambda_val: float or None, the Box-Cox transformation parameter (lambda).
                  If None, the function will find the optimal lambda.

    Returns:
    - transformed_df: pandas DataFrame with the transformed column.
    """

    # Ensure the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Extract the specified column and check for positive values
    data = df[column_name].values
    if np.any(data <= 0):
        raise ValueError(f"Data in column '{column_name}' must be positive for Box-Cox transformation.")

    # Apply Box-Cox transformation
    if lambda_val is None:
        # Find the optimal lambda using scipy's boxcox function
        transformed_data, lambda_optimal = stats.boxcox(data)
    else:
        # Apply the Box-Cox transformation with the specified lambda
        transformed_data = (data ** lambda_val - 1) / lambda_val if lambda_val != 0 else np.log(data)

    # Create a new DataFrame with the transformed column
    transformed_df = df.copy()
    transformed_df[column_name] = transformed_data

    return transformed_df
