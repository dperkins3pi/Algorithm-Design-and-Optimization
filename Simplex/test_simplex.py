"""Unit testing file for the Simplex lab"""

import simplex
import pytest
import numpy as np
from scipy import linalg as la

def test_simplex():
    """
    Write at least one unit test for problem 5, the simplex solver.
    """
    # Initialize data form the example
    c = np.array([-3, -2])
    A = np.array([[1, -1],
                  [3, 1], 
                  [4, 3]])
    b = np.array([2, 5, 7])

    # Get the solutions
    solver = simplex.SimplexSolver(c, A, b)
    minimial_val, dependent_vars, independent_vars = solver.solve()
    dependent_indices = np.array(list(dependent_vars.keys()))
    dependent_vals = np.array(list(dependent_vars.values()))
    independent_indices = np.array(list(independent_vars.keys()))
    independent_vals = np.array(list(independent_vars.values()))

    print("deep", dependent_indices)
    assert abs(minimial_val + 5.2) < 0.0001, f"Incorrect minimal value"
    assert la.norm(dependent_indices - np.array([0, 1, 2])) < 0.0001, "Incorrect dependent variable indices"
    assert la.norm(dependent_vals - np.array([1.6, .2, .6])) < 0.0001, "Incorrect dependent variable values"
    assert la.norm(independent_indices - np.array([3, 4])) < 0.0001, "Incorrect independent variable indices"
    assert la.norm(independent_vals - np.array([0, 0])) < 0.0001, "Incorrect independent variable values"



def test_simplex_example():
    # Sets up the values for the simplex problem.
    c = np.array([-3, -2])
    b = np.array([2, 5, 7])
    A = np.array([[1, -1], [3, 1], [4, 3]])

    # Runs the simplex solver.
    solver = simplex.SimplexSolver(c, A, b)
    sol = solver.solve()

    # Checks if it returned the correct value
    assert sol[0] == -5.2, "Incorrect result from the simplex method"