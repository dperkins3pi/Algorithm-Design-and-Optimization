"""Unit testing file for Gaussian Quadrature"""
import gaussian_quadrature
import numpy as np
import pytest


def test_init():
    # Test correct storage of attributes
    gauss1 = gaussian_quadrature.GaussianQuadrature(20, 'legendre')
    gauss2 = gaussian_quadrature.GaussianQuadrature(5, 'chebyshev')

    assert (gauss1.ptype == 'legendre'), "Polytype not stored"
    assert (gauss2.ptype == 'chebyshev'), "Polytype not stored"

    # Test the respective weight functions
    input_value = 0.25
    assert (gauss1.w_inv(input_value) == 1)
    assert (gauss2.w_inv(input_value) == np.sqrt(1-input_value**2))

    # Test giving invalid polytype
    with pytest.raises(ValueError) as error:
        gauss3 = gaussian_quadrature.GaussianQuadrature(1, 'Legendre')

    with pytest.raises(ValueError) as error:
        gauss4 = gaussian_quadrature.GaussianQuadrature(20, 'invalid')


def test_point_weights():
    # Write unit tests for testing the point_weights function
    
    # Create objects of class
      # Create objects of class
    gauss1 = gaussian_quadrature.GaussianQuadrature(5, 'legendre')
    gauss2 = gaussian_quadrature.GaussianQuadrature(7, 'legendre')

    # See if x's matxh
    x = np.zeros(5)
    x[0] = -(1/3)*np.sqrt(5+2*np.sqrt(10/7))
    x[1] = -(1/3)*np.sqrt(5-2*np.sqrt(10/7))
    x[3] = (1/3)*np.sqrt(5-2*np.sqrt(10/7))
    x[4] = (1/3)*np.sqrt(5+2*np.sqrt(10/7))

    # See if x's match
    w = np.zeros(5)
    w[0] = (322 - 13*np.sqrt(70)) / 900
    w[1] = (322 + 13*np.sqrt(70)) / 900
    w[2] = (128) / 225
    w[3] = (322 + 13*np.sqrt(70)) / 900
    w[4] = (322 - 13*np.sqrt(70)) / 900

    print(x)
    xi, wi = gauss1.points_weights(5)
    assert abs(np.sum(x - xi)) < 0.001
    assert abs(np.sum(w - wi)) < 0.001

    # Write unit tests for testing the point_weights function

    # See if the integral approximation is within certain bount
    p = lambda x: x**2
    integral = np.sum(p(gauss1.points)*gauss1.weights)
    assert (abs(integral - (2/3)) < 0.01 )

    integral = np.sum(p(gauss2.points)*gauss2.weights)
    assert (abs(integral - (2/3)) < 0.01 )

   
    # Function 2
    gauss1 = gaussian_quadrature.GaussianQuadrature(6, 'legendre')
    gauss2 = gaussian_quadrature.GaussianQuadrature(4, 'legendre')

    # See if the integral approximation is within certain bount
    p = lambda x: np.sin(x)**2 / (x - 2)
    integral = np.sum(p(gauss1.points)*gauss1.weights)
    assert (abs(integral - (-0.320437)) < 0.01 )

    integral = np.sum(p(gauss2.points)*gauss2.weights)
    assert (abs(integral - (-0.320437)) < 0.01 )