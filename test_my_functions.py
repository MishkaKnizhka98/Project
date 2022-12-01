import pytest
from my_functions import *
import math

@pytest.fixture
def data():
    x, y = load_data("test_data/test_data.csv")
    return x, y

@pytest.fixture
def dummy_data():
    x, y = load_data("test_data/test_data.csv")
    x_dummy = dummy_matrices(x)
    return x_dummy


def test_data_load_correctly():
    """
    This function tests that load_data() uploads and splits data in x and y arrays correctly.
    x consists of columns ["school", "sex", "age", "Mjob", "Fjob", "higher", "activities", "G1", "G2"].
    y consists of the column ["G3"].

    GIVEN: a file test_load_data.csv is being opened with read_csv() method
    WHEN: the data is splitted into arrays x and y
    THEN: the result consists of the arrays x and y
    """
    x, y = load_data("test_data/test_data.csv")
    assert y[3] == 15
    assert x["Fjob"].iloc[3] == "teacher"


def test_data_load_limit_case():
    """
    This function tests the limit case of using load_data() when the parameter is None.
    In this case TypeError is raised.

    """
    with pytest.raises(TypeError):
        x, y = load_data()




def test_dummy_matrices_performed_correctly():
    """
    This function tests that training data with categorical features
    can be decomposed into one or more indicator variables that take on the values 0 or 1.
    For example, if there are three schools (located in Moscow, S.-Petersburg and Zelenograd),
    the feature "school" will be splitted into three columns.
    For binary categorical features one value is indicated by 1, whereas 0 is assigned to the second value.

    GIVEN: the data is splitted into arrays x and y, x is the training set with categorical features:
           ["school" -> 3 values, "Mjob" -> 5 values, "Fjob" -> 4 values, "sex", "higher", "activities" -> binary features]
    WHEN: each column with categorical features is decomposed into columns with indicator value,
          each column with binary categorical features is replaced by a column with indicators
    THEN: the number of columns increases from 9 to 18, training set becomes numeric

    """
    x, y = load_data("test_data/test_data.csv")
    x_dummy = dummy_matrices(x)
    assert x_dummy.shape[1] == 18
    assert x_dummy["s_Moscow"].iloc[1] == 1



def test_data_with_dummy_matrices_has_no_categorical_features(dummy_data):
    """
    This function tests that after applying dummy_matrices() to data it will not contain
    categorical features.
    """
    assert dummy_data.select_dtypes(include = "object").shape[1] == 0

def test_dummy_matrices_do_not_change_data_with_numeric_values():
    """
    This function tests a limit case when dummy_matrices() does not alter data with numeric features.
    """
    x_num, y_num = load_data("test_data/test_data_numeric_features.csv")
    assert dummy_matrices(x_num).equals(x_num)





def test_compute_model():
    """
    This function tests that compute_model() builds linear regression correctly.
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

    w, b, r_sq = compute_model(x, y)

    assert math.isclose(w, 3) and math.isclose(b, 2) and r_sq == 1



def test_compute_model_with_single_point():
    """
    This function tests that a linear regression model is not well-defined for single samples
    and will return a NaN value for r_sq,if the number of samples is less than two.
    """
    x = [10]
    y = [10]

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    w, b, r_sq = compute_model(x, y)

    assert math.isnan(r_sq)






