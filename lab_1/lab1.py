# Author: gary@darach.ai
# Date: 2024-05-10
# Description: Lesson 1 - Create a mean model for prediction and state if it is
#              a good fit based on the standard deviation of the dataset.

import numpy as np


def main():
    # Generate a dataset of random numbers
    data = np.random.normal(loc=50, scale=10, size=1000)

    # Calculate the mean of the dataset
    mean = np.mean(data)

    # Calculate the standard deviation of the dataset
    std_dev = np.std(data)

    # Determine if the mean is a good model for prediction
    if std_dev < mean / 2:
        print(
            f"Mean ({mean:.2f}) is a good model for prediction. Standard Deviation is low ({std_dev:.2f}).")
    else:
        print(
            f"Mean ({mean:.2f}) is not a good model for prediction. Standard Deviation is high ({std_dev:.2f}).")


if __name__ == "__main__":
    main()
