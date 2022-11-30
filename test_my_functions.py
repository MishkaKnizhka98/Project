import pandas as pd
import pytest
from my_functions import *


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






