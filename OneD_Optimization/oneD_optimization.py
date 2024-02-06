# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
Daniel Perkins
MATH 323
1/20/24
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy.optimize import line_search

# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=100):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    x0 = (a + b) / 2   # Set initial minimizer to midpoint
    varphi = (1+np.sqrt(5))/2   # Golden ratio
    converged = False

    for i in range(1, maxiter+1):
        # Get enpoints of smaller interval
        c = (b - a) / varphi
        a_tilde = b - c
        b_tilde = a + c

        # Shrink the interval
        if f(a_tilde) < f(b_tilde): b = b_tilde
        else: a = a_tilde
        x1 = (a + b) / 2  # Set new minimizer as midpoint
        
        if abs(x0 - x1) < tol: 
            converged = True
            break  # Stop iterating if approximation doesn't change enough
        x0 = x1
    
    return x1, converged, i


# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=100):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converged = False  # Initialize to False

    for i in range(1, maxiter+1):
        x1 = x0 - df(x0) / d2f(x0)
        if abs(x1 - x0) < tol:   # x is not changing enough
            converged = True
            break
        x0 = x1  # Update terms
    return x1, converged, i


# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=100):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converged = False  # Initialize to False

    for i in range(1, maxiter+1):
        x2 = (x0 * df(x1) - x1 * df(x0)) / (df(x1) - df(x0))  # Formula 16.3
        if np.abs(x2 - x1) < tol:   # x is not changing enough
            converged = True
            break
        # Update last terms
        x0 = x1
        x1 = x2
    
    return x2, converged, i


# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    Dfp = Df(x).T @ p # Compute these two values only once
    fx = f(x)
    # Algoirthm 2
    while (f(x + alpha * p) > fx + c * alpha * Dfp):
        alpha = rho * alpha
    return alpha


if __name__=="__main__":

    # Prob 1
    # f = lambda x: np.exp(x) - 4*x
    # a = 0
    # b = 3
    # minimizer_guess = golden_section(f, a, b)[0]
    # minimizer = opt.golden(f, brack=(0,3), tol=.001)
    # print(minimizer_guess, minimizer)

    # xs = np.linspace(0, 3, 500)
    # plt.plot(xs, f(xs), label="f(x)")
    # plt.scatter(minimizer_guess, f(minimizer_guess), c="red", label="Minimizer")
    # plt.legend()
    # plt.show()

    
    # Prob 2
    # df = lambda x: 2*x + 5*np.cos(5*x)
    # d2f = lambda x: 2 - 25*np.sin(5*x)
    # x0 = 0
    # minimizer_guess = newton1d(df, d2f, x0)[0]
    # minimizer = opt.newton(df, x0=x0, fprime=d2f, tol=1e-10, maxiter=500)
    # print(minimizer_guess, minimizer)
    # print(newton1d(df, d2f, x0))

    # xs = np.linspace(-3, 3, 500)
    # plt.plot(xs, df(xs), label="f'(x)")
    # plt.scatter(minimizer_guess, df(minimizer_guess), c="red", label="Minimizer")
    # plt.legend()
    # plt.show()


    # Prob 3
    # f = lambda x: x**2 + np.sin(x) + np.sin(10*x)
    # df = lambda x: 2*x + np.cos(x) + 10*np.cos(10*x)
    # x0 = 0
    # x1 = -1
    # minimizer_guess = secant1d(df, x0, x1)[0]
    # minimizer = opt.newton(df, x0=x0, tol=1e-10, maxiter=500)
    # print(minimizer_guess, minimizer)
    # print(secant1d(df, x0, x1))

    # xs = np.linspace(-3, 3, 500)
    # plt.plot(xs, f(xs), label="f(x)")
    # plt.scatter(minimizer_guess, f(minimizer_guess), c="red", label="Minimizer (My Function)")
    # plt.scatter(minimizer, f(minimizer), c="orange", label="Minimizer (Scipy)")
    # plt.legend()
    # plt.show()


    # Prob 4
    f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
    Df = lambda x: np.array([2*x[0], 2*x[1], 2*x[2]])
    x = np.array([150., .03, 40.])
    p = np.array([-.5, -100., -4.5])
    guess = backtracking(f, Df, x, p)
    print(guess)

    alpha = line_search(f, Df, x, p)[0]
    print(alpha)

    pass