import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import LabelEncoder


def load_data(file_name):
    """
    Loads the data from a student dataset file_name and converts it to a training set (x_train, y_train).
    The input x_train includes the features ["sex", "age", "Pstatus", "Mjob", "Fjob", "higher", "activities"],
    the output y_train contains the final grade G3.

    Parameters:
        file_name (string): path to a student dataset

    Returns:
        x_train (ndarray): Shape(m, 7), m - number of training examples (students) Input to the model
        y_train (ndarray): Shape(m,) Output of the model
    """
    # importing the dataset
    data = pd.read_csv(file_name)

    # Editing the raw dataset to get x_train and y_train
    data = data[["school", "sex", "age", "Mjob", "Fjob", "higher", "activities", "G1", "G2", "G3"]]

    #Eliminating rows with NaN values
    data = data.dropna()

    # Turning categorical features into numbers
    # Dummy matrices + Label Encoding
    non_num = data.select_dtypes(include="object")
    encoder = LabelEncoder()
    for column in non_num.columns:
        if len(non_num[column].unique()) == 2:
            data[column] = encoder.fit_transform(data[column])

        else:
            non_num[column] = non_num[column].apply(lambda x: column[0].lower() + "_" + x)
            dummies = pd.get_dummies(non_num[column])
            dummies = dummies.drop([dummies.columns[-1]], axis=1)
            data = pd.concat([data, dummies], axis=1)
            data = data.drop([column], axis=1)

    # Extracting x_train and y_train from the table
    x_train = data.drop(["G3"], axis=1)
    y_train = data["G3"]

    return x_train, y_train


def pd_to_np(x, y):
    """

    Converts a Dataframe to a Numpy array.

    Parameters:
        x (pandas dataframe): Training set as DataFrame
        y (pandas dataframe): Output set as DataFrame

    Returns:
        x (ndarray): Training set as Numpy array
        y (ndarray): Output set as Numpy array
    """

    x = x.to_numpy()
    y = y.to_numpy()

    x = x.astype('float64')
    y = y.astype('float64')

    return x, y