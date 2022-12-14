# This module contains tests for each function in my_functions.py
import pandas as pd
import os
import pytest
import my_functions as mf
import math
from unittest import mock


@pytest.fixture
def dummy_data():
    x, y = mf.load_data("test_data/test_data.csv")
    x_dummy = mf.dummy_matrices(x)
    return x_dummy


def test_data_load_correctly():
    """
    This function tests that load_data() uploads and splits data into x and y arrays correctly.
    x consists of columns ["school", "sex", "age", "Mjob", "Fjob", "higher", "activities", "G1", "G2"].
    y consists of the column ["G3"].

    GIVEN: a file test_data.csv is being opened with read_csv() method
    WHEN: the data is split into arrays x and y
    THEN: the result consists of the arrays x and y
    """

    test_data = {"school": ["Hogwarts", "Nevermore"],
                 "sex": ["M", "F"],
                 "age": [15, 16],
                 "Mjob": ["teacher", "services"],
                 "Fjob": ["health", "services"],
                 "higher": ["yes", "yes"],
                 "activities": ["yes", "yes"],
                 "G1": [17, 18],
                 "G2": [14, 15],
                 "G3": [19, 19]}

    test_data = pd.DataFrame(test_data)
    test_data_csv = test_data.to_csv("test_data.csv")

    x, y = mf.load_data("test_data.csv")
    loaded_data = pd.concat([x, y], axis=1)

    assert loaded_data.equals(test_data)

    file_name = "test_data.csv"
    if os.path.isfile(file_name):
        os.remove(file_name)



def test_data_misses_column():
    """
    This function tests the limit case of using load_data() when the parameter is None.
    In this case TypeError is raised.
    """
    test_data = {"school": ["Hogwarts", "Nevermore"],
                 "sex": ["M", "F"],
                 "age": [15, 16],
                 "Mjob": ["teacher", "services"],
                 "Fjob": ["health", "services"],
                 "higher": ["yes", "yes"],
                 "activities": ["yes", "yes"],
                 "G1": [17, 18],
                 #"G2": [1,2],
                 "G3": [19, 19]}

    test_data = pd.DataFrame(test_data)
    test_data_csv = test_data.to_csv("test_data.csv")

    with pytest.raises(KeyError):
        x, y = mf.load_data("test_data.csv")

    file_name = "test_data.csv"
    if os.path.isfile(file_name):
        os.remove(file_name)



def test_dummy_matrices_performed_correctly():
    """
    This function tests that input with categorical features
    can be correctly decomposed into dummy matrices consisting of indicator variables
    that take on the values 0 or 1. For example, if there are three schools
    (located in Moscow, S.-Petersburg and Zelenograd), the feature "school" will be split into three columns.
    For binary categorical features one value is indicated by 1, whereas 0 is assigned to the second value.

    GIVEN: the data is split into arrays x and y, x is the input with categorical features:
           ["school" -> 3 values, "Mjob" -> 5 values, "Fjob" -> 4 values, "sex", "higher", "activities" -> binary features]
    WHEN: each column with categorical features is decomposed into columns with indicator value,
          each column with binary categorical features is replaced by a column with indicators
    THEN: the number of columns increases from 9 to 18, training set becomes numeric
    """

    x = {"school": ["Hogwarts", "Hogwarts", "Nevermore"],
                 "sex": ["M", "F", "F"],
                 "age": [15, 14, 16],
                 "Mjob": ["teacher", "at_home", "services"],
                 "Fjob": ["health", "other", "services"],
                 "higher": ["yes", "no", "yes"],
                 "activities": ["no", "yes", "yes"],
                 "G1": [17, 18, 19],
                 "G2": [14, 15, 16],
                 }
    x = pd.DataFrame(x)
    x_dummy = mf.dummy_matrices(x)

    x_dummy_test = {"school": [0, 0, 1],
                         "sex": [1, 0, 0],
                         "age": [15, 14, 16],
                         "higher": [1, 0, 1],
                         "activities": [0, 1, 1],
                         "G1": [17, 18, 19],
                         "G2": [14, 15, 16],
                         "m_at_home": [0, 1, 0],
                         "m_services": [0, 0, 1],
                         "m_teacher": [1, 0, 0],
                         "f_health": [1, 0, 0],
                         "f_other": [0, 1, 0],
                         "f_services": [0, 0, 1]
                 }
    x_dummy_test = pd.DataFrame(x_dummy_test)
    x_dummy = x_dummy.astype("int64")
    assert x_dummy_test.equals(x_dummy)



def test_data_with_dummy_matrices_has_no_categorical_features():
    """
    This function tests that after applying dummy_matrices() to data it will not contain
    categorical features. For this purpose the function is given a parameter dummy_data from pytest-fixture
    containing dataset with dummy matrices. In this case the number of columns with categorical features is 0.
    """

    x = {"school": ["Hogwarts", "Hogwarts", "Nevermore"],
                 "sex": ["M", "F", "F"],
                 "age": [15, 14, 16],
                 "Mjob": ["teacher", "at_home", "services"],
                 "Fjob": ["health", "other", "services"],
                 "higher": ["yes", "no", "yes"],
                 "activities": ["no", "yes", "yes"],
                 "G1": [17, 18, 19],
                 "G2": [14, 15, 16],
                 }
    x = pd.DataFrame(x)
    x_dummy = mf.dummy_matrices(x)

    assert x_dummy.select_dtypes(include="object").shape[1] == 0


def test_dummy_matrices_do_not_change_data_with_numeric_values():
    """
    This function tests a limit case when dummy_matrices() does not alter data with numeric features.
    """

    x_num = pd.DataFrame([1, 2, 3], columns = ["Numeric values"])
    assert mf.dummy_matrices(x_num).equals(x_num)


def test_compute_model():
    """
    This function tests that compute_model() fits a linear regression model correctly.
    For this case a simple dataset with zero variation is used.
    Slope w, intercept b and coefficient of determination r_sq are calculated.

    GIVEN: a model is given simple input x and output y from a linear function: y(x) = 3x + 2
    WHEN: a class LinearRegression is used to fit the model
    THEN: w and b are expected to be equal to 3 and 2, respectively, r_sq is expected to be 1
    """

    x = [1, 2, 3]
    y = [5, 8, 11]

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    w, b, r_sq = mf.compute_model(x, y)

    assert math.isclose(w, 3) and math.isclose(b, 2) and r_sq == 1


def test_compute_model_with_single_point():
    """
    This function tests that a linear regression model is not well-defined for single samples
    and will return a NaN value for r_sq, if the number of samples is less than two.
    """

    x = [10]
    y = [10]

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    w, b, r_sq = mf. compute_model(x, y)

    assert math.isnan(r_sq) and math.isclose(w, 0) and math.isclose(b, 10)


def test_new_student():
    """
    This function tests that new_student() creates a 1-row table with a new student's features.

    GIVEN: input x is formed from the dataset test_data.csv
    WHEN: a user inputs features to a new student Alex. In order to make the test pass,
    Alex's school should be indicated as "Moscow"
    THEN: a variable alex is assigned to a 1-row pandas dataframe. In order to compare elements of a DataFrame,
    the method .any() is used
    """

    x, y = load_data("test_data/test_data.csv")
    alex = new_student(x)
    assert (alex["school"] == "Moscow").any()


def test_two_new_students():
    """
    This function tests that two new students created with new_student() and given different features,
    will be different from each other.

    GIVEN: input x is formed from the dataset test_data.csv
    WHEN: a user inputs features to new students Alex and Lorenzo. In order to make the test pass,
    at least one Alex's feature should differ from identical Lorenzo's feature.
    THEN: variables alex and lorenzo are assigned to 1-row pandas dataframes and then compared
    """

    x, y = load_data("test_data/test_data.csv")

    alex = new_student(x)
    print("\n")
    lorenzo = new_student(x)

    assert lorenzo.equals(alex) == False


def test_dummy_matrix_of_new_student():
    """
    This function tests that dummy_matrix_of_new_student() correctly decomposes a new student's features
    into dummy matrices and indicates binary features with 0 and 1.

    GIVEN: input x is formed from the dataset test_data.csv and a new student Alex is created
    with new_student(). In order to make the test pass, Alex's school should be indicated as "Moscow".
    The variable alex is assigned to a 1-row pandas dataframe with Alex's features
    WHEN: dummy_matrix_for_new_student() is given alex and x as parameters
    THEN: categorical features are decomposed into dummy matrices, binary features are replaced with
    indicators 0 and 1. If Alex's school is indicated as "Moscow", the feature value s_Moscow is equal to 1
    """

    x, y = load_data("test_data/test_data.csv")
    alex = new_student(x)

    alex = dummy_matrix_of_new_student(alex, x)
    assert (alex["s_Moscow"] == 1).any()


def test_dummy_matrix_for_new_student_has_no_categorical_features():
    """
    This function tests that after applying dummy_matrix_for_new_student() to a new student's features
    they will not contain categorical features.
    """

    x, y = load_data("test_data/test_data.csv")
    alex = new_student(x)
    alex = dummy_matrix_of_new_student(alex, x)
    assert alex.dtype == "int64"


def test_predict():
    """
    This function tests that predict() correctly predicts the output y using a linear regression model.

    GIVEN: simple input x and output y arrays are given from a linear function: y(x) = 3x + 2
    WHEN: compute_model() fits the model and returns trained parameters w = 3 and b = 2.
          y_pred is assigned to a predicted output for new input x_pred
    THEN: if x_pred = 4, then y_pred should be 14
    """

    x = [1, 2, 3]
    y = [5, 8, 11]

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    w, b, r_sq = compute_model(x, y)

    y_pred = predict(4, w, b)

    assert math.isclose(y_pred, 14)


def test_that_predicted_output_higher_than_twenty_notified():
    """
    This function tests that if a predicted output "G3" is higher than 20 (maximum G3 value),
    then a user will get a notification about it.

    GIVEN: simple input x and output y arrays are given from a linear function: y(x) = 3x + 2
    WHEN: compute_model() fits the model and returns trained parameters w = 3 and b = 2.
          y_pred is assigned to a predicted output for new input x_pred
    THEN: if x_pred = 10, then y_pred should be 32 and the alert
          "The new student's final grade G3 is higher than 20!" is raised
    """

    x = [1, 2, 3]
    y = [5, 8, 11]

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    w, b, r_sq = compute_model(x, y)

    y_pred = predict(10, w, b)

    assert math.isclose(y_pred, 32)


@mock.patch("my_functions.plt.show")
def test_plot(mock_plot):
    """
    This function tests that plot() works correctly by asserting that the method plt.show() is invoked.
    The approach is to mock plt.show() inside my_functions.py module and to assert that it got called.
    Since plt.show() is invoked at the end of plot(), if something breaks leading up to plt.show(),
    this mock will not get called and the test will fail.

    GIVEN: The @mock.patch("my_functions.plt.show") decorator "patches" the plt.show() method
           imported inside my_functions.py and injects it as a mock object (mock_plot)
           to the test_plot() as a parameter
    WHEN:  x and y arrays form the training set from test_data.csv. x array is exposed to dummy_matrices()
           and returns x_dummy with categorical features replaced by dummy matrices.
           Inserting x_dummy and y to compute_model(), we train the model and obtain parameters w, b and
           the coefficient of determination r_sq
    THEN:  x, y, w and b are put as parameters to plot(), which generates plots with dependencies of
           the target y on numerical features
    """

    x, y = load_data("test_data/test_data.csv")
    x_dummy = dummy_matrices(x)
    w, b, r_sq = compute_model(x_dummy, y)
    plot(x, y, w, b)
    assert mock_plot.called
