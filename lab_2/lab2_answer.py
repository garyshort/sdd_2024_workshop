# Author: gary@darach.ai
# Date: 2024-05-16
# Description:

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main():
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Select only one feature - BMI (which is the third feature in the dataset)
    # np.newaxis increases the dimension to use in linear regression
    X = diabetes.data[:, np.newaxis, 2]

    # Define the target variable
    y = diabetes.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model using the training sets
    model.fit(X_train, y_train)

    # Predict using the test set
    y_pred = model.predict(X_test)

    # Calculate the root mean square error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Plotting
    plt.scatter(X_test, y_test, color='black', label='Actual data')
    plt.plot(X_test, y_pred, color='blue', linewidth=3,
             label='Linear regression line')
    plt.xlabel('Body Mass Index')
    plt.ylabel('Diabetes Progression')
    plt.title('Regression Line and Actual Data Points')
    plt.legend()
    plt.savefig('lab2_answer.png')

    print("Coefficient:", model.coef_[0])
    print("Intercept:", model.intercept_)
    print("Root Mean Square Error: {:.2f}".format(rmse))


if __name__ == "__main__":
    main()
