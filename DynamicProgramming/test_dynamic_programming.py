"""Unit testing file for the Dynamic Programming lab"""


import dynamic_programming
import numpy as np
import pytest

def test_eat_cake():
    """
    Create a unit test to test your function from problem 6, your eat_cake function.
    You can do this by making sure the matrices produced for the example match up
    with what is in the pdf for the lab.
    """
    A_exp = np.array([[0, 0, 0, 0],
                      [0.5, 0.5, 0.5, 0.5],
                      [0.95, 0.95, 0.95, 0.707],
                      [1.355, 1.355, 1.157, 0.866],
                      [1.7195, 1.562, 1.343, 1]])
    P_exp = np.array([[0, 0, 0, 0],
                      [.25, .25, .25, .25],
                      [.25, .25, .25, .5],
                      [.25, .25, .5, .75],
                      [.25, .5, .5, 1]])
    
    # See if the matrices were close enough
    A, P = dynamic_programming.eat_cake(3, 4, 0.9)
    assert np.linalg.norm(A - A_exp) < 0.01, "Incorrect matrix A"
    assert np.linalg.norm(P - P_exp) < 0.01, "Incorrect matrix P"



def test_calc_stopping():
    # Finds the expected value and the stopping index for N = 4.
    expected_val, index = dynamic_programming.calc_stopping(4)

    # Checks to make sure the stopping values are correct.
    assert abs(expected_val - 0.4583) <= 1e-3, "Incorrect expected value"
    assert index == 1, "Incorrect index"
