# Author: gary@darach.ai
# Date: 2024-05-16
# Description:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def main():
    # Load the dataset
    diabetes = load_diabetes()

    # Create a DataFrame for easier data manipulation
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['Disease_progression'] = diabetes.target

    # Extract all independent variables
    X = df.drop(columns='Disease_progression')
    y = df['Disease_progression']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Generate polynomial features for all independent variables
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predict and calculate the mean squared error
    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Plotting actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue',
                alpha=0.5, label='Predicted values')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(),
             y_test.max()], color='red', lw=2, label='Ideal fit')
    plt.title('Actual vs Predicted Disease Progression')
    plt.xlabel('Actual Disease Progression')
    plt.ylabel('Predicted Disease Progression')
    plt.legend()
    plt.grid(True)
    plt.savefig('diabetes_polynomial_regression.png')

    print("RMSE:", rmse)


if __name__ == "__main__":
    main()
