"""Unit testing file for CVXPY Intro lab"""


import cvxpy_intro
import pytest
import numpy as np

def test_prob5():
    """
    Write at least one unit test for problem 5.
    """
    # Set up the matrix and vector
    A = np.array([[1, 2, 1, 1], 
                  [0, 3, -2, -1]])
    b = np.array([7, 4])

    optimizer, val = cvxpy_intro.prob5(A, b)  # Run the function

    # Checks for the correct vector and minimum value
    assert np.linalg.norm(optimizer - np.array([0., 1, 0, 0])) <= 1e-3, "Returned the wrong minimizer"
    assert abs(val - 5.099) <= 1e-3, "Returned the wrong mimimum"

def test_l1Min():
    # Sets up the matrix and vector
    A = np.array([[1, 2, 1, 1],
                 [0, 3, -2, -1]])
    b = np.array([7,4])

    # Runs the l1Min function on the matrix and vector
    x, ans = cvxpy_intro.l1Min(A, b)
    
    # Checks for the correct vector and minimum value
    assert np.linalg.norm(x - np.array([0.0, 2.571, 1.857, 0.0])) <= 1e-3, "Returned the wrong minimizer"
    assert abs(ans - 4.429) <= 1e-3, "Returned the wrong mimimum"