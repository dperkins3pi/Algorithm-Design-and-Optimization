"""Unit testing file for Gradient Descent Methods lab"""

import gradient_methods
import pytest
import numpy as np
from scipy import optimize as opt
from scipy import linalg as la

def test_nonlinear_conjugate_gradient():
    """
    Write at least one unit test for problem 3, the nonlinear conjugate gradient function.
    """
    # Rosenbrock function
    x0 = np.array([10, 10])
    solution = opt.fmin_cg(opt.rosen, x0, opt.rosen_der)
    min, converged, k = gradient_methods.nonlinear_conjugate_gradient(opt.rosen, opt.rosen_der, x0, maxiter=10000)
    if(converged):
        assert la.norm(min - solution) < 0.001  # See if solutions are similar

def test_conjugate_gradient():
    # Tests different sized matrices to see if they converge
    for n in range(1,5):
        A = np.random.random((n,n))
        b = np.random.random(n)
        Q = A.T @ A
        x, conv, k = gradient_methods.conjugate_gradient(Q, b, np.random.random(n))
        if conv:
            assert np.allclose(Q @ x, b), "Incorrect vector found"