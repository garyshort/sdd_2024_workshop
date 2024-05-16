# Author: gary@darach.ai
# Date: 2024-05-16
# Description:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    california = fetch_california_housing()

    # Convert dataset to a DataFrame
    data = pd.DataFrame(california.data, columns=california.feature_names)
    data['MedHouseVal'] = california.target

    # Select features and target
    X = data[['MedInc']]  # MedInc: median income in block
    # MedHouseVal: Median house value for California districts
    y = data['MedHouseVal']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X_train, y_train)

    # Predict the values for testing data
    y_pred = model.predict(X_test)

    # Calculate the Root Mean Squared Error and the coefficient of determination (R^2)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Root Mean Squared Error:", rmse)
    print("R^2 Score:", r2)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='black', label='Actual data')
    plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Fitted line')
    plt.xlabel('Median Income')
    plt.ylabel('Median House Value (in $100,000s)')
    plt.title('Linear Regression on California Housing Dataset')
    plt.legend()
    plt.savefig('lesson2.png')


if __name__ == "__main__":
    main()
