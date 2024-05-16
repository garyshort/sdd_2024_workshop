# Author: gary@darach.ai
# Date: 2024-05-16
# Description:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes


def main():
    # Load the diabetes dataset
    diabetes = load_diabetes()

    # Create a DataFrame for easier data manipulation
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['Progression'] = diabetes.target

    # Display the first few rows of the DataFrame
    print("First few rows of the dataset:")
    print(df.head())

    # Summary statistics of the dataset
    print("\nSummary Statistics:")
    print(df.describe())

    # Check for missing values in the dataset
    print("\nMissing values in each column:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()
