# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
Danny Perkins
MATH 322
3/10/24
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(j,k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j,k))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k,:] @ x
    b[k:] = A[k:,:] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    def F(x, l, mu):  # Equation for F
        top = A.T @ l - c
        middle = A @ x - b
        bottom = mu * x   # Don't actually need to create M as a matrix
        return np.concatenate((top, middle, bottom))
    
    def DF(x, l, mu):   # Equation for DF
        top = np.block([np.zeros((len(A.T),len(A.T))), A.T, np.eye(len(A.T))])
        middle = np.block([A, np.zeros((len(A),len(A))), np.zeros_like(A)])
        bottom = np.block([np.diag(mu), np.zeros((len(A.T),len(A))), np.diag(x)])
        return np.vstack((top, middle, bottom))
    
    def search_direction(x, l, mu):
        sigma = 1/10
        DF1 = DF(x, l, mu)   # Get the derivative

        F1 = F(x, l, mu)
        v = np.dot(x, mu) / len(x)  # Duality measure
        right = np.concatenate((np.zeros(len(x)), np.zeros(len(l)), sigma * v * np.ones(len(mu))))  # right side of 23.2
        
        # Solve the equations efficiently
        lu, piv = la.lu_factor(DF1)
        direction  = (la.lu_solve((lu, piv),  -F1 + right))
        return direction, v
    
    def step_size(x, mu, direction): # compute step size
        n = len(mu)
        alpha_max = min(-mu / direction[-n:])  # delta mu is last n elements
        delta_max = min(-x / direction[:n])  # delta x is first n elements
        alpha = min(1, 0.95*alpha_max)
        delta = min(1, 0.95*delta_max)
        return alpha, delta
    
    x, lam, mu = starting_point(A, b, c)

    for i in range(niter):  # run the algorithm
        n = len(x)
        direction, v = search_direction(x, lam, mu)  # get search direction and duality measure
        alpha, delta = step_size(x, mu, direction)   # get step size
        x = x + delta * direction[:n]   # compute next element in sequence
        lam = lam + alpha * direction[n:-n]
        mu = mu + alpha * direction[-n:]
        print(v)
        if v < tol: break   # already converged

    return x, c.T @ x



def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    raise NotImplementedError("Problem 5 Incomplete")



if __name__ == "__main__":
    j, k = 7, 5
    A, b, c, x = randomLP(j, k)
    print("randomLP", x)
    point, value = interiorPoint(A, b, -c)
    print("my way", point[:k])
    print("val", value)
    
    # HELP!!!!! IS MY DF CORRECT???????
    # IS IT NOT SUPPOSED TO BE INVERTIBLE