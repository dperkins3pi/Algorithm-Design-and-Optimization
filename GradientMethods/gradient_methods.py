# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
Daniel Perkins
MATH 323
1/30/24
"""

import numpy as np
from scipy import optimize as opt
from scipy import linalg as la
from scipy import optimize as opt
from matplotlib import pyplot as plt

# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    x = x0
    converged = False
    for k in range(1, maxiter+1):
        a_x = lambda a: f(x - a*Df(x).T)  # Define function of alpha
        step_size = opt.minimize_scalar(a_x)  # Return minimizer
        x = x - step_size.x * Df(x).T   # 18.2
        if la.norm(Df(x), np.inf) < tol:  # Already close enough
            converged = True
            break 

    return x, converged, k


# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converged = False
    n = len(Q)  # number of basis vectors
    r = Q @ x0 - b   # Direction of steepest descent
    d = -r
    k = 0   # To count the number of iterations
    x = x0
    while(la.norm(r) >= tol and k < n):  # Algorithm 1
        a = np.dot(r, r) / np.dot(d, Q @ d)  # Use dot product to find r^Tr
        x = x + a * d
        rk = r + a * (Q @ d)
        b = np.dot(rk, rk) / np.dot(r, r)
        d = -rk + b * d
        r = rk   # Set new value for r
        k = k + 1
    if la.norm(r) < tol: converged = True
    return x, converged, k


# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converged = False
    r = -df(x0) 
    d = r
    g = lambda a: f(x0 + a*d) 
    a = opt.minimize_scalar(g).x
    x = x0 + a*d
    k = 1   # To count the number of iterations
    while(la.norm(r) >= tol and k < maxiter):  # Algorithm 2
        rk = -df(x)
        b = np.dot(rk, rk) / np.dot(r, r)
        d = rk + b*d

        g = lambda a: f(x + a*d) 
        a = opt.minimize_scalar(g).x

        x = x + a*d
        r = rk
        k = k + 1
    if la.norm(r) < tol: converged = True
    return x, converged, k


# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    # Load in the data
    data = np.loadtxt(filename)
    b = np.copy(data[:, 0])
    data[:, 0] = np.ones_like(b)
    A = data

    # Use problem 2 to solve A^TAx=A^Tb
    solution = conjugate_gradient(A.T @ A, A.T @ b, x0)[0]
    return solution


# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        def neg_log_like(B):
            summand = np.log(1 + np.exp(-(B[0] + B[1]*x))) + (1 - y)*(B[0] + B[1]*x)  # (18.4)
            return np.sum(summand)
        minimizer = opt.fmin_cg(neg_log_like, guess)  # Minimize the function

        # Store minimizers
        self.B0 = minimizer[0]
        self.B1 = minimizer[1]

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        return 1 / (1 + np.exp(-(self.B0 + self.B1*x)))   # probability that x is assigned label y=1


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    # Get the data
    data = np.load(filename)
    x = data[:, 0]
    y = data[:, 1]

    guess = np.array([20, -1])  # initial guess

    # Fit the data
    log_reg = LogisticRegression1D()
    log_reg.fit(x, y, guess)

    # Get the graph
    xs = np.linspace(30, 100, endpoint=True)
    prediction = log_reg.predict(xs)
    
    # Plot it
    plt.title("Probability of O-Ring Damage")
    plt.ylabel('O-Ring Damage')
    plt.xlabel("Temperature")
    plt.plot(xs, prediction, c="orange")
    plt.scatter(x, y)
    plt.legend()
    plt.show()

    return log_reg.predict(31)  # predicted probability of damage


if __name__=="__main__":
    # prob 1
    # easy function
    f = lambda x: x[0]**4 + x[1]**4 + x[2]**4
    Df = lambda x: np.array([4*x[0]**4, 4*x[1]**3, 4*x[2]**3])
    x0 = np.array([1, 1, 2])
    # print(steepest_descent(f, Df, x0, tol=1e-5, maxiter=100)[0])
    # # Rosenbrock function
    # f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    # Df = lambda x: np.array([-2*(1-x[0])+100*2*(x[1]-x[0]**2)*-2*x[0], 100*2*(x[1]-x[0]**2)])
    # x0 = np.array([1, 8])
    # print(steepest_descent(f, Df, x0, tol=1e-5, maxiter=10000))


    # prob2
    # Q = np.array([[2, 0], [0, 4]])
    # b = np.array([1, 8])
    # x0 = np.random.random(2)
    # print(np.allclose(conjugate_gradient(Q, b, x0, tol=1e-4)[0], np.array([1/2, 2.])))

    # for i in range(10):
    #     n = 4
    #     A = np.random.random((n, n))
    #     Q = A.T @ A
    #     b, x0 = np.random.random((2, n))
    #     x = la.solve(Q, b)
    #     x_approx, converged, k = conjugate_gradient(Q, b, x0, tol=1e-4)
    #     if(converged):
    #         print(np.allclose(Q @ x, b))
    #     else:
    #         print("The algorithm did not converge. Guess:" + x_approx, "actual: " + x)


    # Prob 3
    # f = lambda x: x[0]**4 + x[1]**4 + x[2]**4
    # Df = lambda x: np.array([4*x[0]**4, 4*x[1]**3, 4*x[2]**3])
    # x0 = np.array([1., 1., 2.])

    # x0 = np.array([10, 10])
    # solution = opt.fmin_cg(opt.rosen, x0, opt.rosen_der)
    # print("Solution:", solution)
    # print()

    # # # Rosenbrock function
    # x0 = np.array([10, 10])
    # min = nonlinear_conjugate_gradient(opt.rosen, opt.rosen_der, x0, maxiter=10000)
    # print("Approximation", min)
    # print("opt solution evaluated:", f(solution), "my solution evaluated:", f(min[0]))

    # Prob 4
    # print(prob4())


    # Prob 5 and 6
    # print(prob6())

    pass