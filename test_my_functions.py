# This module contains tests for each function in my_functions.py
import pandas as pd
import os
import pytest
import my_functions as mf
import math
from unittest import mock
import builtins


def test_data_load_correctly():
    """
    This function tests that mf.load_data() uploads and splits data into x and y arrays correctly.
    x consists of columns ["school", "sex", "age", "Mjob", "Fjob", "higher", "activities", "G1", "G2"].
    y consists of the column ["G3"].

    GIVEN:  A pandas dataframe test_data is created and saved as "test_data.csv" in the project directory.

    WHEN:   mf.load_data() splits the data from "test_data.csv" into arrays x and y.

    THEN:   x and y are joined together as loaded_data. loaded_data is then compared with the initial dataframe test_data.
            After comparison is done, "test_data.csv" is removed from the directory.
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
    This function tests that KeyError is raised, if mf.load_data() loads a dataset with an absent feature. In this test
    the feature "G2" is absent.

    GIVEN:  A pandas dataframe test_data is created and saved as "test_data.csv" in the project directory. Column "G2"
            is absent (marked as a comment).

    WHEN:   mf.load_data() splits the data from "test_data.csv" into arrays x and y.

    THEN:   Since x lacks column "G2", KeyError should be raised.
            After exception is raised, "test_data.csv" is removed from the directory.

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
    This function tests that input with categorical features can be correctly decomposed into dummy matrices consisting
    of indicator variables that take on the values 0 or 1. For example, if there are three variants of a mother's job
    ("Mjob" -> "teacher", "at_home" or "services"), the feature "Mjob" will be split into three columns.
    For binary categorical features one value is indicated by 1, whereas 0 is assigned to the second value.

    GIVEN:  x is a pandas dataframe created as the input with categorical features:
            ["Mjob" -> 3 values, "Fjob" -> 3 values, "school", "sex", "higher", "activities" -> binary features].

    WHEN:   x is exposed to mf.dummy_matrices(). Each column with categorical features is decomposed into columns
            with indicator value, each column with binary categorical features is replaced by a column with indicators.
            The resulting dataset is assigned to x_dummy.

    THEN:   x_dummy_test is created as the input x with dummy matrices and then compared to x_dummy.
            Since in x_dummy_test all features have the type "int64", in x_dummy some features have the type "uint8",
            x_dummy was converted to "int64" with .astype().
    """

    x = {"school": ["Hogwarts", "Hogwarts", "Nevermore"],
                 "sex": ["M", "F", "F"],
                 "age": [15, 14, 16],
                 "Mjob": ["teacher", "at_home", "services"],
                 "Fjob": ["health", "other", "services"],
                 "higher": ["yes", "no", "yes"],
                 "activities": ["no", "yes", "yes"],
                 "G1": [17, 18, 19],
                 "G2": [14, 15, 16]}
    x = pd.DataFrame(x)
    x_dummy = mf.dummy_matrices(x)
    x_dummy = x_dummy.astype("int64")

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
                         "f_services": [0, 0, 1]}

    x_dummy_test = pd.DataFrame(x_dummy_test)

    assert x_dummy_test.equals(x_dummy)





def test_data_with_dummy_matrices_has_no_categorical_features():
    """
    This function tests that after applying mf.dummy_matrices() to data it will not contain
    categorical features.

    GIVEN:  x is a pandas dataframe created as the input with categorical features:
            ["Mjob" -> 3 values, "Fjob" -> 3 values, "school", "sex", "higher", "activities" -> binary features].

    WHEN:   x is exposed to mf.dummy_matrices(). Each column with categorical features is decomposed into columns
            with indicator value, each column with binary categorical features is replaced by a column with indicators.
            The resulting dataset is assigned to x_dummy.

    THEN:   Categorical features with dtypes == "object" are assigned to non_num using .select_dtypes().
            Since x_dummy has no categorical features, the number of columns in non_num is 0.
    """

    x = {"school": ["Hogwarts", "Hogwarts", "Nevermore"],
                 "sex": ["M", "F", "F"],
                 "age": [15, 14, 16],
                 "Mjob": ["teacher", "at_home", "services"],
                 "Fjob": ["health", "other", "services"],
                 "higher": ["yes", "no", "yes"],
                 "activities": ["no", "yes", "yes"],
                 "G1": [17, 18, 19],
                 "G2": [14, 15, 16]}
    x = pd.DataFrame(x)
    x_dummy = mf.dummy_matrices(x)

    non_num = x_dummy.select_dtypes(include="object") #columns with dtypes == "object" are assigned to non_num

    assert non_num.shape[1] == 0





def test_dummy_matrices_do_not_change_data_with_numeric_values():
    """
    This function tests a limit case when mf.dummy_matrices() does not alter data with numeric features.

    GIVEN:  x_num is a pandas dataframe with one column "Numeric values" as a feature.

    WHEN:   x_num is exposed to mf.dummy_matrices(). The resulting dataframe is indicated as x_num_dummy.

    THEN:   x_num_dummy is compared with x_num. Since mf.dummy_matrices() does not change numeric features,
            the arrays should be equal.
    """

    x_num = pd.DataFrame([1, 2, 3], columns = ["Numeric values"])
    x_num_dummy = mf.dummy_matrices(x_num)
    assert x_num_dummy.equals(x_num)





def test_compute_model():
    """
    This function tests that mf.compute_model() fits a linear regression model correctly.
    For this case a simple dataset with zero variation is used.
    Slope w, intercept b and coefficient of determination r_sq are calculated.

    GIVEN:  A model is given simple input x and output y from a linear function: y(x) = 3x + 2.

    WHEN:   The class LinearRegression is used to fit the model.

    THEN:   w and b are expected to be equal to 3 and 2, respectively, r_sq is expected to be 1.
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

    GIVEN:  x and y form a single point (10, 10).

    WHEN:   mf.compute_model() is given x and y as parameters.

    THEN:   Parameters w and b are equal to 0 and 10, respectively. r_sq is equal to NaN value.
    """

    x = [10]
    y = [10]

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    w, b, r_sq = mf. compute_model(x, y)

    assert math.isnan(r_sq) and math.isclose(w, 0) and math.isclose(b, 10)






def test_new_student(monkeypatch):
    """
    This function tests that mf.new_student() creates a 1-row table with a new student's features. For this purpose
    the monkeypatch fixture is given as a parameter. A monkeypatch object can alter an attribute in a dictionary,
    and then restore its original value at the end of the test. In this case the built-in input() function
    (located inside mf.new_student()) is a value of "builtins" dictionary, so we can alter it. After input() is done,
    the result is compared with the pre-built 1-row dataframe.

    GIVEN:  new_example is a 1-row pandas dataframe.

    WHEN:   inputs is the array that contains all feature values to be passed to mf.new_student() with monkeypatch.setattr()
            as the input() data. mf.new_student() returns the 1-row result to result (variable).
            new_example is given to mf.new_student() as a parameter to define the features (columns) of result.

    THEN:   Feature values in inputs are the same as in the new_example dataframe. Therefore, result and new_example
            should be equal.
    """

    new_example = {
        "school": "Nevermore",
        "sex": "M",
        "age": 16,
        "Mjob": "teacher",
        "Fjob": "health",
        "higher": "yes",
        "activities": "yes",
        "G1": 17,
        "G2": 14}
    new_example = pd.DataFrame(new_example, index=[0])

    inputs = iter(["Nevermore", "M", 16, "teacher", "health", "yes", "yes", 17, 14])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    result = mf.new_student(new_example)

    assert result.equals(new_example)





def test_two_new_students(monkeypatch):
    """
    This function tests that mf.new_student() given different feature values, will create two different new students.
    For this purpose monkeypatch fixture is set as a parameter to mock a user's input.

    GIVEN:  test_data is a pandas dataframe which defines features for mf.new_student() to fill in.

    WHEN:   monkeypatch.setattr() passes feature values from inputs_1 to student_1 and from inputs_2 to student_2
            through the input() function in mf.new_student(). Feature values for student_1 and student_2 are different.

    THEN:   student_1 and student_2 are compared. Since they have different feature values, student_1 is not equal to
            student_2, thus, student_1.equals(student_2) should be False.
    """

    test_data = {"school": ["Hogwarts", "Nevermore"],
                 "sex": ["M", "F"],
                 "age": [15, 16],
                 "Mjob": ["teacher", "services"],
                 "Fjob": ["health", "services"],
                 "higher": ["yes", "yes"],
                 "activities": ["yes", "yes"],
                 "G1": [17, 18],
                 "G2": [1, 2]}
    test_data = pd.DataFrame(test_data)

    inputs_1 = iter(["Nevermore", "M", 16, "teacher", "health", "yes", "yes", 17, 14])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs_1))
    student_1 = mf.new_student(test_data)

    inputs_2 = iter(["Nevermore", "F", 16, "services", "health", "yes", "yes", 17, 14])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs_2))
    student_2 = mf.new_student(test_data)

    assert student_1.equals(student_2) == False





def test_dummy_matrix_of_new_student():
    """
    This function tests that mf.dummy_matrix_of_new_student() correctly decomposes a new student's features
    into dummy matrices and indicates binary features with 0 and 1.

    GIVEN:  x is used as an input set in mf.dummy_matrix_of_new_student(). new_student is a 1-row dataframe with a
            new student's features. dummy_test is a 1-row dataset which is equal to new_student dataset with
            categorical features replaced with dummy matrices.

    WHEN:   new_student and x are given to mf.dummy_matrix_of_new_student(). The result is indicated as new_student_dummy.

    THEN:   new_student_dummy is compared with dummy_test. Since dummy_test replicates the result of mf.dummy_matrix_of_new_student(),
            their equality should be True.
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

    new_student = {
        "school": "Nevermore",
        "sex": "F",
        "age": 16,
        "Mjob": "teacher",
        "Fjob": "health",
        "higher": "yes",
        "activities": "yes",
        "G1": 17,
        "G2": 14}
    new_student = pd.DataFrame(new_student, index=[0])

    new_student_dummy = mf.dummy_matrix_of_new_student(new_student, x)

    dummy_test = {"school": 1,
                         "sex": 0,
                         "age": 16,
                         "higher": 1,
                         "activities": 1,
                         "G1": 17,
                         "G2": 14,
                         "m_at_home": 0,
                         "m_services": 0,
                         "m_teacher": 1,
                         "f_health": 1,
                         "f_other": 0,
                         "f_services": 0}
    dummy_test = pd.Series(dummy_test)

    assert new_student_dummy.equals(dummy_test)





def test_dummy_matrix_for_new_student_has_no_categorical_features():
    """
    This function tests that after applying mf.dummy_matrix_for_new_student() to a new student's features
    they will not contain categorical features.

    GIVEN:  x is created as an input set in mf.dummy_matrix_of_new_student(). new_student is a 1-row dataframe with a
            new student's features.

    WHEN:   x and new_student are given to mf.dummy_matrix_of_new_student(). The result is assigned to new_student_dummy.
            In order to extract categorical features (dtypes == "object"), the Series new_student_dummy
            is converted to DataFrame with .to_frame(). Categorical columns are extracted with .select_dtypes() and
            assigned to non_num.

    THEN: Since new_student_dummy has no categorical features, the number of columns in non_num is equal to 0.

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

    new_student = {
        "school": "Nevermore",
        "sex": "F",
        "age": 16,
        "Mjob": "teacher",
        "Fjob": "health",
        "higher": "yes",
        "activities": "yes",
        "G1": 17,
        "G2": 14}
    new_student = pd.DataFrame(new_student, index=[0])

    new_student_dummy = mf.dummy_matrix_of_new_student(new_student, x)
    new_student_dummy = new_student_dummy.to_frame()  # Converting to DataFrame
    non_num = new_student_dummy.select_dtypes(include="object")

    assert non_num.shape[1] == 0





def test_predict():
    """
    This function tests that mf.predict() correctly predicts the output y using a linear regression model.

    GIVEN:  Simple input x and output y arrays are given from a linear function: y(x) = 3x + 2

    WHEN:   mf.compute_model() fits the model and returns trained parameters w = 3 and b = 2.
            y_pred is assigned to a predicted output for new input x_pred

    THEN:   If x_pred = 4, then y_pred should be 14
    """

    x = [1, 2, 3]
    y = [5, 8, 11]

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    w, b, r_sq = mf.compute_model(x, y)

    y_pred = mf.predict(4, w, b)

    assert math.isclose(y_pred, 14)





def test_that_predicted_output_higher_than_twenty_notified():
    """
    This function tests that if a predicted output "G3" is higher than 20 (maximum G3 value),
    then a user will get a warning and the predicted value will be trimmed to 20.

    GIVEN:  Simple input x and output y arrays are given from a linear function: y(x) = 3x + 2

    WHEN:   mf.compute_model() fits the model and returns trained parameters w = 3 and b = 2.
            y_pred is assigned to a predicted output for new input x_pred

    THEN:   If x_pred = 10, the warning "The new student's final grade G3 is higher than 20!" is raised and y_pred is
            trimmed to 20.
    """

    with pytest.warns(UserWarning):
        x = [1, 2, 3]
        y = [5, 8, 11]

        x = pd.DataFrame(x)
        y = pd.DataFrame(y)

        w, b, r_sq = mf.compute_model(x, y)

        y_pred = mf.predict(10, w, b)

    assert math.isclose(y_pred, 20)





@mock.patch("my_functions.plt.show")
def test_plot(mock_plot):
    """
    This function tests that mf.plot() works correctly by asserting that the method plt.show() is invoked.
    The approach is to mock plt.show() inside my_functions.py module and to assert that it gets called.
    Since plt.show() is invoked at the end of mf.plot(), if something breaks leading up to plt.show(),
    this mock will not get called and the test will fail.

    GIVEN:  The @mock.patch("my_functions.plt.show") decorator "patches" the plt.show() method
            imported inside my_functions.py and injects it as a mock object (mock_plot)
            to the test_plot() as a parameter.

    WHEN:   x and y arrays form the training set. x array is exposed to mf.dummy_matrices() and returns x_dummy
            with categorical features replaced by dummy matrices. Inserting x_dummy and y to mf.compute_model(),
            we train the model and obtain parameters w, b and the coefficient of determination r_sq.

    THEN:   x, y, w and b are put as parameters to mf.plot(), which generates plots with dependencies of
            the target y on numerical features.
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

    y = pd.DataFrame([17, 18, 20], columns = ["G3"])

    w, b, r_sq = mf.compute_model(x_dummy, y)
    mf.plot(x, y, w, b)
    assert mock_plot.called
