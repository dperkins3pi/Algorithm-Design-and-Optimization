# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
Daniel Perkins
MATH 323
1/14/24
"""

import numpy as np
from scipy.linalg import eig

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        # Raise value error if invalid polytype
        if polytype != "legendre" and polytype != "chebyshev":
            raise ValueError("Polytype must be Legendre or Chebyshev")
        
        # Store attributes
        self.n = n
        self.ptype = polytype

        # Calculate inverse weight function
        if polytype == "legendre":
            w_inverse = lambda x: 1
        elif polytype == "chebyshev":
            w_inverse = lambda x: np.sqrt(1 - x**2)

        # Store inverse weight function
        self.w_inv = w_inverse

        # Find the points and weights
        points, weights = self.points_weights(n)
        self.points = points
        self.weights = weights

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        # Find each B_k
        if self.ptype == "legendre":
            B = np.array([k**2 / (4*k**2 - 1) for k in range(1, n)])
        elif self.ptype == "chebyshev":
            B = np.ones(n - 1) * (1/4)
            B[0] = (1/2)

        # Create Jacobi matrix matrix (note that each a_k=0)
        J = np.zeros((n, n))
        for k in range(1, n):
            J[k - 1, k] = np.sqrt(B[k - 1])
            J[k, k - 1] = np.sqrt(B[k - 1])
        print(J)

        # Find the eigenvalues and eigenvectors of J
        eigenvalues, eigenvectors = eig(J)
        x = eigenvalues

        # Calculate the weights
        if self.ptype == "legendre":
            mu = 2
        elif self.ptype == "chebyshev":
            mu = np.pi
        weights = mu * eigenvectors[0]**2

        return x, weights

    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        raise NotImplementedError("Problem 4 Incomplete")

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        raise NotImplementedError("Problem 6 Incomplete")


# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    raise NotImplementedError("Problem 5 Incomplete")


if __name__ == "__main__":
    # prob 1
    quad = GaussianQuadrature(10, "legendre")
    # w = quad.w_inverse
    # print(w(0.01))

    # prob 2
    x, w = quad.points_weights(5)
    print(x)
    print(w)
