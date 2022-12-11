# This module contains all functions that are employed in the linear regression analysis in main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #To indicate binary categorical features as 0 and 1
from sklearn.linear_model import LinearRegression #To fit a linear model


def load_data(file_name):
    """
    Loads the data from a student dataset file_name and extracts the columns 
    ["school", "sex", "age", "Mjob", "Fjob", "higher", "activities", "G1", "G2", "G3"]. 
    The columns ["school", "sex", "age", "Mjob", "Fjob", "higher", "activities", "G1", "G2"] are assigned to the input x of the model.
    The column "G3" (final grade) is assigned to the output y of the model.

    Parameters:
        ----------
        file_name : string
                Path to a student dataset

    Returns:
        ----------
        x : pandas dataframe
                Shape (m,n) (m - number of examples (rows), n - number of features (columns))
                Input to the model
        y : pandas dataframe
                Shape (m,) Output of the model
    """

    # Importing the dataset from a file
    data = pd.read_csv(file_name)

    # Extracting necessary columns from the dataset
    data = data[["school", "sex", "age", "Mjob", "Fjob", "higher", "activities", "G1", "G2", "G3"]]
    
    # Eliminating rows with NaN values
    data = data.dropna()
    
    x = data.drop(["G3"], axis=1)
    y = data["G3"]
    
    return x, y





def dummy_matrices(data):
    """
    Turns categorical features of a training dataset into dummy matrices. In particular, each column
    which has several values of a categorical feature, is split into a set of columns. Each column in the set
    is assigned to a certain value of the feature and contains indicators 0 and 1. When the student's feature
    is equal to the value of a specific column, its indicator will be 1, otherwise, it will be 0.
    For example, the feature "Mjob" has five values ["at_home", "health", "other", "services", "teacher"].
    Then after splitting the feature into dummy matrices it will be replaced by five columns with headings written as
    "m_ + value name" (the value name is altered in order to avoid confusion with "Fjob" values).
    If a student has the "Mjob" feature as "health", the indicator in the "m_health" will be 1, whereas
    the columns "m_at_home", "m_other", "m_services", "m_teacher" will have 0 in the student's row.

    If a feature has binary categorical value, its values will be indicated as 0 and 1. Such process is called a label encoding.
    For example, the feature "activities" has two values: "yes" and "no". Therefore, after applying label encoding
    to this feature, "yes" will be indicated with 1 and "no" will be indicated with 0. In order to perform label encoding,
    the LabelEncoder class is used.

    
    Parameters:
        ----------
        data : pandas dataframe
                Shape (m,n) Data with categorical features
        
    Returns:
        ----------
        data : pandas dataframe
                Shape (m,k) (m - number of examples (rows), k - number of features with dummy matrices (columns), k>=n)
                Data with dummy matrices
    """

    #Creating a copy of dataframe in order to avoid unpredictable errors coming up due to chained assignments
    data = data.copy()

    # Turning categorical features into numbers
    non_num = data.select_dtypes(include="object")
    encoder = LabelEncoder()
    for column in non_num.columns:
        #Label Encoding
        if len(non_num.loc[:, column].unique()) == 2:
            data.loc[:, column] = encoder.fit_transform(data.loc[:, column])

        else:
            #Dummy matrices
            non_num.loc[:,column] = non_num.loc[:, column].apply(lambda x: column[0].lower() + "_" + x)
            dummies = pd.get_dummies(non_num.loc[:, column])
            data = pd.concat([data, dummies], axis=1)
            data = data.drop([column], axis=1)

    return data
    




def compute_model(x, y):
    """
    Computes a linear regression model using the LinearRegression class.

    Parameters:
        ----------
        x : pandas dataframe
                Shape (m,k) Input to the model
        y : pandas dataframe
                Shape (m,) Output of the model

    Returns:
        ----------
        w : numpy ndarray
                Shape (k,) Parameters of the fitted model (slope)
        b : scalar
                Parameter of the fitted model (intercept)
        r_sq : scalar
                Coefficient of determination R^2
    """

    model = LinearRegression()
    model.fit(x, y)

    w = model.coef_
    b = model.intercept_
    r_sq = model.score(x,y)

    return w, b, r_sq





def new_student(x):
    """
    Creates a new student. The function gets the features that belong to students in the input x
    and assigns them to a new student. For each feature a user types into a number within a specific range
    (15-25 for "age" and 0-20 for "G1" and "G2") or an allowable value (for a categorical feature).
    The function returns a 1-row DataFrame with the features of the new student.
    
    Parameters:
        ----------
        x : pandas dataframe
                Shape (m,n) Input to the model. Defines the features for a new student
        
    Returns:
        ----------
        new : pandas dataframe
                Shape (1,n) A new example of a student
    """

    #Dictionary to contain new student's features
    new_data = {}
    
    for column in x.columns:
        inp = ""
        # Typing categorical features
        if x[column].dtype == "object":
            inp = input("Please write the " + column + " of a student (" + str(x[column].unique()) + "): ")
            while inp not in x[column].unique():
                print("Invalid feature! Please try again!")
                inp = input("Please write the " + column + " of a student (" + str(x[column].unique()) + "): ")

        # Typing "age"
        elif column == "age":
            while True:
                try:
                    inp = int(input("Please write the " + column + " of a student (from " + str(15) + " to " + str(25) + "): "))
                    while inp < 15 or inp > 25:
                        print("Invalid feature! Please try again!")
                        inp = int(input("Please write the " + column + " of a student (from " + str(15) + " to " + str(25) + "): "))
                except ValueError: 
                    print("Value Error for age! Please try again!")
                    continue
                
                if inp >= 15 or inp <= 25:
                    break

        #Typing "G1" and "G2"
        elif column == "G1" or column == "G2":
            while True:
                try:
                    inp = int(input("Please write the " + column + " of a student (from " + str(0) + " to " + str(20) + "): "))
                    while inp < 0 or inp > 20:
                        print("Invalid grade! Please try again!")
                        inp = int(input("Please write the " + column + " of a student (from " + str(0) + " to " + str(20) + "): "))
                except ValueError:
                    print("Value Error for grade! Please try again!")
                    continue
                    
                if inp >= 0 or inp <= 20:
                    break

        new_data[column] = inp
    
    new = pd.DataFrame(new_data, index=[0])

    return new





def dummy_matrix_of_new_student(new_student, x):
    """
    Turns categorical features of a new student into dummy variables. The function creates a new array
    x_new by appending the new_student to the input x. x_new is then exposed to dummy_matrices()
    which turns categorical features into dummy matrices. The function returns the last element of x_new, which is
    the new student's row with categorical features replaced by dummy matrices.
    
    Parameters:
        ----------
        new_student : pandas dataframe
                Shape (1,n) A new student array with categorical features
        x : pandas dataframe
                Shape (m,n) Input to the model
        
    Returns:
     ----------
        new_student_dummy : pandas dataframe
                Shape (1,k) The new student array with dummy matrices
    """
    
    x_new = x.append(new_student)
    new_dummy = dummy_matrices(x_new)
    new_student_dummy = new_dummy.iloc[-1]
    
    return new_student_dummy





def predict(example, w, b):
    """
    Predicts the target y (final grade "G3") of a new student (indicated as "example") using a linear model
    with trained parameters w and b. If the new student's final grade is higher than 20, the function will print out the
    notification. The np.dot() method is invoked to calculate the dot product of arrays x and w.
    
    Parameters:
        ----------
        example : pandas dataframe
                Shape (1,k) An example of a student with categorical features replaced by dummy matrices
        w : numpy ndarray
                Shape (k,) Parameters of the trained model (slope)
        b : scalar
                Parameter of the trained model (intercept)
        
    Returns:
        ----------
        y_pred : scalar
                Predicted target
    """

    y_pred = np.dot(w, example) + b

    if y_pred > 20:
        print("The new student's final grade G3 is higher than 20!")

    return y_pred





def plot(x, y, w, b):
    """
    Creates plots with dependence of the output y and its predicted values on numerical values of input x.
    In case of a dataset with features ["school", "sex", "age", "Mjob", "Fjob", "activities", "higher", "G1", "G2"]
    and target "G3", the function will scatter given and predicted values of "G3" against "age", "G1" and "G2".
    In order to predict "G3" for each student, trained parameters w and b are inserted into a linear function,
    where np.dot() method is invoked to obtain the dot product of arrays x and w.
    Parameters:
        ----------
        x : pandas dataframe
                Shape (m,n) Input to the model
        y : pandas dataframe
                Shape (m,) Output of the model
        w : numpy.ndarray
                Shape (n,) Parameters of the trained model (slope)
        b : scalar
                Parameter of the trained model (intercept)
        
    Returns:
        ----------
        None
    """

    #Here we predict G3 dor each student in training set x
    m = x.shape[0]
    predicted = np.zeros(m)
    x_dummy = dummy_matrices(x)
    for i in range(m):
        predicted[i] = np.dot(w, x_dummy.iloc[i]) + b

    #Selecting numeric features and plotting them
    num_column = x.select_dtypes(include="int64")
    i= 1   
    plt.figure(figsize=(20, 7))
    plt.subplots_adjust(top=0.8)
    for column in num_column.columns:
        plt.subplot(1,len(num_column.columns), i)
        plt.scatter(x[column], predicted, c="b")
        plt.scatter(x[column], y, c="r", marker="x")
        plt.title("Dependence of G3 vs " + column)
        plt.ylabel("G3")
        plt.xlabel(column)
        i += 1
    plt.suptitle("Data values (red crosses) vs predicted values (blue dots)", fontsize = 20, y = 1)
    plt.show()
        
         