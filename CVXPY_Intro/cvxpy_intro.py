# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
Daniel Perkins
MATH 323
2/27/24
"""

import cvxpy as cp
import numpy as np


def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(3, nonneg=True) # Declare x
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)  # Declare objective function

    # Constraints
    A1 = np.array([1, 2, 0])
    A2 = np.array([0, 1, -4])
    A3 = np.array([2, 10, 3])
    P = np.eye(3)
    constraints = [A1 @ x <= 3, A2 @ x <= 1, A3 @ x >= 12, P @ x >= 0]

    # Assemle the problem and solve ir
    problem = cp.Problem(objective, constraints)
    optimal_val = problem.solve()
    optimizer = x.value
    return optimizer, optimal_val


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(len(A[0]), nonneg=True) # Declare x
    c = cp.norm(x, 1)
    objective = cp.Minimize(c)  # Declare objective function

    # Constraints
    constraints = [A @ x == b]

    # Assemle the problem and solve ir
    problem = cp.Problem(objective, constraints)
    optimal_val = problem.solve()
    optimizer = x.value
    return optimizer, optimal_val
    


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """

    p = cp.Variable(6, nonneg=True) # Declare x
    c = np.array([4, 7, 6, 8, 8, 9])
    objective = cp.Minimize(c.T @ p)  # Declare objective function

    # Constraints
    available = np.array([7, 2, 4])   # Number of pianos avaiable at each supply center
    Aout = np.array([[1, 1, 0, 0, 0, 0],  # Represents the pianos leaving
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 1, 1]])
    Ain = np.array([[1, 0, 1, 0, 1, 0],   # Represents the incoming pianos
                    [0, 1, 0, 1, 0, 1]])
    demand = np.array([5, 8])   # Number of pianos needed in each center
    constraints = [Aout @ p == available, Ain @ p == demand]

    # Assemble the problem and solve ir
    problem = cp.Problem(objective, constraints)
    optimal_val = problem.solve()
    optimizer = p.value
    return optimizer, optimal_val


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    Q = np.array([[3/2, 1, 1/2],    # Matrix Q (after solving for it)
                  [1, 2, 1],
                  [1/2, 1, 3/2]])
    r = np.array([3, 0, 1])   # Last two components
    x = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, Q) + r.T @ x))  # Define the problem
    optimal_val = prob.solve()
    optimizer = x.value
    return optimizer, optimal_val


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    n = len(A[0])
    x = cp.Variable(n, nonneg=True) # Declare x
    c = cp.norm(A @ x - b, 2)
    objective = cp.Minimize(c)  # Declare objective function

    # Constraints
    constraints = [cp.sum(x) == 1]   # Use sum function as affine

    # Assemble the problem and solve ir
    problem = cp.Problem(objective, constraints)
    optimal_val = problem.solve()
    optimizer = x.value
    return optimizer, optimal_val


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    data = np.load("food.npy", allow_pickle=True)   # Read the data
    p = data[:,0]   # Prices
    s = data[:,1]   # Servings
    data[:, 5:] *= -1  # So that we minimize rather than maximizing
    for i in range(2, 8):   # Multiple each thing by the serving size
        data[:, i] *= s    
    A = data[:, 2:].T # The rest of the matrix

    n = len(data)   # Number of foods
    x = cp.Variable(n, nonneg=True) # Declare x
    objective = cp.Minimize(p.T @ x)  # Declare objective function for the price

    # Constraints
    b = np.array([2000, 65, 50, -1000, -25, -46])
    constraints = [A @ x <= b]   # Use sum function as affine

    # Assemble the problem and solve ir
    problem = cp.Problem(objective, constraints)
    optimal_val = problem.solve()
    optimizer = x.value   # The most optimal things to eat are potatoes (the most), milk and cheese
    return optimizer, optimal_val


if __name__=="__main__":
    # Prob 1
    # optimizer, val = prob1()
    # print("Optimizer:", optimizer)
    # print("Optimal Value:", val)

    # Prob 2
    A = np.array([[1, 2, 1, 1], [0, 3, -2, -1]])
    b = np.array([7, 4])
    optimizer, val = l1Min(A, b)
    print("Optimizer:", optimizer)
    print("Optimal Value:", val)

    # Prob 3
    # print(prob3())

    # Prob 4
    # print(prob4())

    # Prob 5
    # A = np.array([[1, 2, 1, 1], [0, 3, -2, -1]])
    # b = np.array([7, 4])
    # optimizer, val = prob5(A, b)
    # print("Optimizer:", optimizer)
    # print("Optimal Value:", val)

    # Prob 6
    # optimizer, val = prob6()
    # print("Optimizer:", optimizer)
    # print("Optimal Value:", val)

    pass