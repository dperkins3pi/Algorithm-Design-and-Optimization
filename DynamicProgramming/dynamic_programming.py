# dynamic_programming.py
"""Volume 2: Dynamic Programming.
Daniel Perkins
MATH 323
3/17/24
"""

import numpy as np
from matplotlib import pyplot as plt


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    v = np.zeros(N)  # Initialize it to be all zeros
    for i in range(N - 1, 0, -1):   # Iterate backwards starting at 2nd-to last element
        v[i-1] = max((i/(i+1))*v[i] + 1/N, v[i])  # (24.1, with indices shifted)

    return max(v), np.argmax(v) + 1   # Add one since index starts at 0


# Problem 2
def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    # Initialize vairiables
    Ns = np.arange(3, M+1)
    optimal_percentages = []
    max_vals = []

    for N in Ns:
        max_val, t0 = calc_stopping(N)  # Call prob 1
        optimal_percentages.append(t0 / N)  # Gather data
        max_vals.append(max_val)

    # Plot it
    plt.plot(Ns, optimal_percentages, label="Optimal Stopping Percentage")
    plt.plot(Ns, max_vals, label="Maximum Probability")
    plt.legend()
    plt.xlabel("M")
    plt.show()

    return max_val


# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    C = np.zeros((N+1, N+1))   # Start with all zeros
    for i in range(N+1): 
        for j in range(N+1):  # Only update lower-left triangle
            if i > j: C[i,j] = (i - j) / N   # calculate w
    C = u(C)   # apply utility function
    return C


# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    A = np.zeros((N+1, T+1))
    w = np.arange(N + 1)/N

    CV = np.zeros((N+1, N+1))  # Initialize it to be 0s
    P = np.zeros((N+1, T+1))   # To determine the optimal policy

    A[:,T] = u(w)   # Compute matrix at t=T
    P[:,T] = w  # Compute last column
    for t in range(T, 0, -1):   # For each t (starting at the end)
        for i in range(N+1):   # Rows
            for j in range(N+1):   # Columns
                pieces = w[i] - w[j]   # wi-wj
                # If less than 0, keep it at 0
                if pieces >= 0: CV[i,j] = u(pieces) + B*A[j, t]
        A[:,t-1] = np.max(CV, axis=1)  # Update the corresponding column in A

        # Compute the optimal policy
        for i in range(N+1):
            j = np.argmax(CV[i])
            P[i, t-1] = w[i] - w[j]

    return A, P


# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T+1,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    A, P = eat_cake(T, N, B, u)   # Get policy matrix
    c = np.zeros(T+1)

    i = 0
    row = N   # Start at row N
    for i in range(T + 1):
        action = P[row, i]
        row -= round(action * N)   # Convert to integer and move up that amount
        c[i] = action
    return c


if __name__=="__main__":
    # Prob 1
    # print(calc_stopping(4))

    # Prob 2
    # print(graph_stopping_times(1000))
    # Is my graph correct??????

    # Prob 3
    # print(get_consumption(4))

    # Prob 4-6
    # A, P = eat_cake(3, 4, 0.9)
    # print(A)
    # print(P)

    # Prob 7
    print(find_policy(3, 4, 0.9))