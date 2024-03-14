"""Unit testing file for Interior Point (Linear) Lab"""


import interior_point_linear
import pytest
import numpy as np

def test_interiorPoint():
    """
    Write at least one unit test for your interiorPoint function.
    """
    j, k = 7, 5
    A, b, c, x = interior_point_linear.randomLP(j, k)
    point, value = interior_point_linear.interiorPoint(A, b, c)
    assert np.allclose(x, point[:k]), "Converged to the wrong value"

    j, k = 9, 7
    A, b, c, x = interior_point_linear.randomLP(j, k)
    point, value = interior_point_linear.interiorPoint(A, b, c)
    assert np.allclose(x, point[:k]), "Converged to the wrong value"