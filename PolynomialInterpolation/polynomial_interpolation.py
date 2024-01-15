# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
Daniel Perkins
MATH 322
1/9/24
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import BarycentricInterpolator
from scipy import linalg as la
from numpy.fft import fft
from numpy.polynomial.chebyshev import poly2cheb

# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    # print("x", xint)
    # print("y", yint)
    # print("points", points)

    # Compute the denominator of each Lj
    L_denom = np.zeros_like(xint).astype(float)  # Initialize array for denominators
    for j in range(len(L_denom)):
        denom = 1   # Start at one
        for k in range(len(L_denom)):
            if j != k:
                denom *= (xint[j] - xint[k])   # apply equal 14.1
        L_denom[j] = denom
    # print("denom", L_denom)

    # Evaluate Lj at all points in domain
    L = np.ones((len(xint), len(points)))
    for j in range(len(L_denom)):
        solutions = np.ones_like(points)   # initialize at zero
        for k in range(len(L_denom)):
            if j != k: solutions *= (points - xint[k])  # apply  14.1
        L[j] = solutions / L_denom[j]   # divide by denominator
    # print(L)

    # Evaluate interpolating polynomial at each point in domain
    p = yint @ L

    return p

    
# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        # Compute Barycentric weights
        n = len(xint)   # Number of interpolating points
        w = np.ones(n)     # Array for storing weights
        C = (np.max(xint) - np.min(xint)) / 4   # Capacity

        # Randomize order of product
        shuffle = np.random.permutation(n-1)
        for j in range(n):
            temp = (xint[j] - np.delete(xint, j)) / C
            temp = temp[shuffle]
            w[j] /= np.product(temp)

        # Store attributes
        self.x = xint
        self.y = yint
        self.w = w
        self.n = n

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        computation = np.zeros_like(points)
        i = 0  # for indexing
        for point in points:
            if point in self.x: 
                computation[i] = self.y[np.where(self.x == point)] # if already in y
            else:
                num = (self.w * self.y) / (point - self.x) # numerator as array of all values
                den = (self.w) / (point - self.x)  # denominator as array of all values
                computation[i] = np.sum(num) / np.sum(den)  # evaluate the point
            i += 1
        return computation

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        # Extend points
        xint = np.concatenate((self.x, xint))
        yint = np.concatenate((self.y, yint))
        
        # self.__init__(xint, yint)  # Coult just call original function
        # Compute Barycentric weights
        n = len(xint)   # Number of interpolating points
        w = np.ones(n)     # Array for storing weights
        C = (np.max(xint) - np.min(xint)) / 4   # Capacity

        # Randomize order of product
        shuffle = np.random.permutation(n-1)
        for j in range(n):
            temp = (xint[j] - np.delete(xint, j)) / C
            temp = temp[shuffle]
            w[j] /= np.product(temp)

        # Store attributes
        self.x = xint
        self.y = yint
        self.w = w
        self.n = n
        


# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    domain = np.linspace(-1,1,400, endpoint=True)
    f = lambda x: 1/(1+25*x**2)  # Function to be interpolated

    ks = np.arange(2, 9)
    ns = 2**ks
    errors_b = []
    errors_c = []
    for i in ks:  # for n=2^2, 2^3, ..., 2^8
        n = 2**i
        pts = np.linspace(-1, 1, n, endpoint=True)  # interpolate n equally spaced points
        poly = BarycentricInterpolator(pts, f(pts))   # Use class to find interpolating polynomial

        y = f(domain)
        y_approx = poly(domain)
        try:   # Sometimes it divides by 0, so we default those to large numbers
            errors_b.append(la.norm(y - y_approx, ord=np.inf))
        except: 
            errors_b.append(10**10)

        # Same thing but now with Chebyshev extremal points
        j = np.arange(0, n+1)
        cheb = np.cos(j * np.pi / n)
        poly = BarycentricInterpolator(cheb, f(cheb))   # Use class to find interpolating polynomial
        y_approx = poly(domain)
        errors_c.append(la.norm(y - y_approx, ord=np.inf))
    
        # Plot the Barycentric polynomials
        # plt.subplot(4,2, i-1)
        # plt.plot(domain, f(domain), label="f")
        # plt.plot(domain, poly(domain), label="Approximation")
        # plt.title(f"n = 2^{i}")
        # plt.legend(loc="upper center")
    
    # Log log plots of error
    plt.subplot(121)
    plt.title("Equally Spaced Points")
    plt.ylabel("error")
    plt.xlabel("n")
    plt.loglog(ns, errors_b)
    plt.subplot(122)
    plt.title("Chebyshec Extremal Points")
    plt.ylabel("error")
    plt.xlabel("n")
    plt.loglog(ns, errors_c)

    plt.tight_layout()
    plt.show()


# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    # Compute Chebyshev extremal points
    j = np.arange(0, 2*n)
    cheb = np.cos(j * np.pi / n)

    # Compute the yks
    y = np.ones(2*n) * 2
    y[0] = 1
    y[n] = 1

    # Find the aks
    a = y * np.real(fft(f(cheb)))

    # Scale it
    a *= (1/(2*n))

    return a[:n]   # Return only the first n


# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    # Load data
    data = np.load("airdata.npy")
    m = len(data)

    # Get interpolating points
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a, b = 0, 366 - 1/24

    # Find interpolation
    domain = np.linspace(0, b, m)
    points = fx(a, b, n)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)
    poly = Barycentric(domain[temp2], data[temp2])

    # Plot
    plt.subplot(121)
    plt.title("Original Data")
    plt.xlabel("time")
    plt.ylabel("PM_2.5")
    plt.plot(domain, data)
    plt.subplot(122)
    plt.title("Approximation")
    plt.xlabel("time")
    plt.ylabel("PM_2.5")
    plt.plot(domain, poly(domain))
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    # Prob 1
    # x = np.array([0, 1, 2, 3, 4])
    # x = (x - 2) / 2
    # y = 1 / (1 + 25 * x**2)
    # points = np.linspace(-1, 1, 4)
    # lagrange(x, y, points)

    # Prob 2
    # x = np.linspace(-1, 1, 200)
    # f = 1 / (1 + 25 * x**2)
    # # n=5
    # plt.subplot(1, 2, 1)
    # plt.plot(x, f, label="Original")
    # xs = np.linspace(-1, 1, 5, endpoint=True)
    # ys = 1 / (1 + 25 * xs**2)
    # plt.plot(x, lagrange(xs, ys, x), label="Interpolation")
    # plt.legend()
    # # n = 11
    # plt.subplot(1, 2, 2)
    # plt.plot(x, f, label="Original")
    # xs = np.linspace(-1, 1, 11, endpoint=True)
    # ys = 1 / (1 + 25 * xs**2)
    # plt.plot(x, lagrange(xs, ys, x), label="Interpolation")
    # plt.legend()
    # plt.show()

    # Prob 3
    # x = np.linspace(-1, 1, 200)
    # f = 1 / (1 + 25 * x**2)
    # # n=5
    # plt.subplot(1, 2, 1)
    # plt.plot(x, f, label="Original")
    # xs = np.linspace(-1, 1, 5, endpoint=True)
    # ys = 1 / (1 + 25 * xs**2)
    # Barycentric_array = Barycentric(xs, ys)
    # plt.plot(x, Barycentric_array(x), label="Interpolation")
    # plt.legend()
    # # n = 11
    # plt.subplot(1, 2, 2)
    # plt.plot(x, f, label="Original")
    # xs = np.linspace(-1, 1, 11, endpoint=True)
    # ys = 1 / (1 + 25 * xs**2)
    # Barycentric_array = Barycentric(xs, ys)
    # plt.plot(x, Barycentric_array(x), label="Interpolation")
    # plt.legend()
    # plt.show()

    # Prob 4
    # x = np.linspace(-1, 1, 200)
    # f = 1 / (1 + 25 * x**2)
    # # n=5
    # plt.subplot(1, 2, 1)
    # plt.plot(x, f, label="Original")
    # xs = np.linspace(-1, 1, 5, endpoint=True)
    # ys = 1 / (1 + 25 * xs**2)
    # Barycentric_array = Barycentric(xs, ys)
    # plt.plot(x, Barycentric_array(x), label="Interpolation")
    # plt.legend()
    # # n = 11
    # plt.subplot(1, 2, 2)
    # plt.plot(x, f, label="Original")
    # xs = np.array([-.9, -.6321, -.11, .7, 0.88])
    # ys = 1 / (1 + 25 * xs**2)
    # Barycentric_array.add_weights(xs, ys)
    # plt.plot(x, Barycentric_array(x), label="Interpolation")
    # plt.legend()
    # plt.show()

    # prob5
    # prob5()

    # prob6
    # f = lambda x: -3 + 2*x**2 - x**3 + x**4
    # pcoeffs = [-3, 0, 2, -1, 1]
    # ccoeffs = poly2cheb(pcoeffs)
    # print(np.allclose(ccoeffs, chebyshev_coeffs(f, 5)))

    # prob7
    # prob7(50)

    import time
    T = np.random.randint(10,30,50)
    start = time.time()
    product = 1
    for t in T:
        product *= t
    end = time.time()
    print(end - start)

    start = time.time()
    np.product(T)
    end = time.time()
    print("y", end - start)

    # start = time.time()
    # np.mult(T)
    # end = time.time()
    # print(end - start)

    start = time.time()
    np.linalg.prod_array(T)
    end = time.time()
    print(end - start)
