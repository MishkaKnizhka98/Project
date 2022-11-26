import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


def load_data(file_name):
    """
    Loads the data from a student dataset file_name and extracts the columns 
    ["school", "sex", "age", "Mjob", "Fjob", "higher", "activities", "G1", "G2", "G3"]. 
    The columns ["school", "sex", "age", "Mjob", "Fjob", "higher", "activities", "G1", "G2"] are assigned to the input x of the model.
    The column "G3" (final grade) is assigned to the output y of the model.
    

    Parameters:
        file_name (string): Path to a student dataset

    Returns:
        x (pandas dataframe): Shape(m,n) (m - number of examples (rows), n - number of features (columns)) Input to the model
        y (pandas dataframe): Shape(m,) Output of the model
    """
    # importing the dataset
    data = pd.read_csv(file_name)

    # Editing the raw dataset to get x and y
    data = data[["school", "sex", "age", "Mjob", "Fjob", "higher", "activities", "G1", "G2", "G3"]]
    
    #Eliminating rows with NaN values
    data = data.dropna()
    
    x = data.drop(["G3"], axis=1)
    y = data["G3"]
    
    return x, y




def dummy_matrices(data):
    """
    Turns categoraical features into dummy matrices.
    
    Parameters:
        data (pandas dataframe): Data with categorical features
        
    Returns:
        data (pandas dataframe): Data with dummy matrices
    """

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

    return data


def pd_to_np(x):
    """

    Converts a Dataframe to a Numpy array.

    Parameters:
        x (pandas dataframe): Training set as DataFrame

    Returns:
        x (ndarray): Training set as Numpy array
    """

    x = x.to_numpy()
    x = x.astype('float64')

    return x





def normalize(x):
    """

    Performs feature scaling in the range [0,1] by division of each feature by its maximum value.

    Parameters:
        x (ndarray): Training set (features of students)

    Returns:
        x (ndarray): Training set exposed to feature scaling (input to the model)
    """

    x = x.astype('float64')
    for column in range(x.shape[1]):
        x[:, column] = x[:, column] / x[:, column].max()

    return x


def compute_model(x, y):
    """

    Computes a linear regression model using the LinearRegression class.

    Parameters:
        x (ndarray): Shape (m,n) Input for the model
        y (ndarray): Shape (m,) Output for the model

    Returns:
        w (ndarray): Shape (n,) Fitted parameters of the model (coefficients)
        b (scalar): Fitted parameter of the model (intercept)
    """
    model = LinearRegression()
    model.fit(x, y)

    w = model.coef_
    b = model.intercept_

    return w, b