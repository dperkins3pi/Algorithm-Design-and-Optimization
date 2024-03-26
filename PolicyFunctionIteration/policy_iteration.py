# policy_iteration.py
"""Volume 2: Policy Function Iteration.
Daniel Perkins
MATH 323
3/21/24
"""

import numpy as np
import gymnasium as gym

# Intialize P for test example
#Left =0
#Down = 1
#Right = 2
#Up= 3

P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0,0,0,False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0,0,0,False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0,0,0,False)]
P[1][3] = [(0,0,0,False)]
P[2][0] = [(0, 2, -1, False)]
P[2][1] = [(0, 2, -1, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 1, True)]



# Problem 1
def value_iteration(P, nS ,nA, beta=1.0, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """
    v = np.zeros(nS)
    vk = np.copy(v)
    for n in range(1, maxiter + 1):  # Loop until hit max
        for i in range(nS):   # For each state
            actions = np.zeros(nA)   # Part in {} for equation 25.5
            for j in range(nA):   # Calculate each part
                p, s, u, is_terminal = P[i][j][0]
                actions[j] += p * (u + beta * v[s])
            vk[i] = max(actions)   # equation 25.5
        if(np.linalg.norm(v - vk) < tol): break  # Already converged
        v = np.copy(vk)
    return v, n


# Problem 2
def extract_policy(P, nS, nA, v, beta=1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """
    policy = np.zeros_like(v)
    for i in range(nS):   # For each state
        actions = np.zeros(nA)   # Part in {} for equation 25.6
        for j in range(nA):   # Calculate each part
            p, s, u, is_terminal = P[i][j][0]
            actions[j] += p * (u + beta * v[s])
        policy[i] = np.argmax(actions)   # equation 25.6
    return policy


# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8):
    """Computes the value function for a policy using policy evalution.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.

    Returns:
        v (ndarray): The discrete values for the true value function.
    """
    v = np.zeros(nS)  # Initialize
    vk = np.zeros(nS)
    while True:  # Loop until hit max
        v = np.copy(vk)
        for i in range(nS):
            a = policy[i]   # Get action from optimal action
            p, s, u, is_terminal = P[i][a][0]
            vk[i] = p * (u + beta * v[s])  # (25.7)
        if (np.linalg.norm(v - vk) < tol): break   # Convergence
    return vk


# Problem 4
def policy_iteration(P, nS, nA, beta=1.0, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
    	v (ndarray): The discrete values for the true value function
        policy (ndarray): which direction to move in each square.
        n (int): number of iterations
    """
    pi = np.ones(nS)   # Initialize to all ones
    for k in range(maxiter):
        v = compute_policy_v(P, nS, nA, pi)   # Evaluate polict
        pik = extract_policy(P, nS, nA, v)    # Improve polict
        if (np.linalg.norm(pik - pi)) < tol: break   # Convergence
        pi = pik
    return v, pi, k+1


# Problem 5 and 6
def frozen_lake(basic_case=True, M=1000, render=False):
    """ Finds the optimal policy to solve the FrozenLake problem.

    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns.
    M (int): The number of times to run the simulation using problem 6.
    render (boolean): Whether to draw the environment.

    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The mean expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value function for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The mean expected value for following the policy iteration optimal policy.
    """
    # Initialize environemnt
    if basic_case: 
        if render: env = gym.make("FrozenLake-v1", desc=None, map_name='4x4', is_slippery=True, render_mode='human')
        else: env = gym.make("FrozenLake-v1", desc=None, map_name='4x4', is_slippery=True)
    else: 
        if render: env = gym.make("FrozenLake-v1", desc=None, map_name='8x8', is_slippery=True, render_mode='human')
        else: env = gym.make("FrozenLake-v1", desc=None, map_name='8x8', is_slippery=True)
    
    # Get data from gymnasium
    observation, info = env.reset()
    nS = env.observation_space.n
    nA = env.action_space.n
    P = env.P
    env.close()

    # ARE THESE THE RIGHT THINGS????????
    vi_policy, _ = value_iteration(P, nS, nA)
    vi_total_rewards = 0
    pi_value_func = extract_policy(P, nS, nA, vi_policy)
    _, pi_policy, _ = policy_iteration(P, nS, nA)
    pi_total_rewards = 0
    return vi_policy, vi_total_rewards, pi_value_func, pi_policy, pi_total_rewards



# Problem 6
def run_simulation(env, policy, beta=1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
    env (gym environment): The gym environment.
    policy (ndarray): The policy used to simulate.
    beta (float): The discount factor.

    Returns:
    total reward (float): Value of the total reward recieved under policy.
    """
    raise NotImplementedError("Problem 6 Incomplete")


if __name__=="__main__":
    # Prob 1
    # print(value_iteration(P, 4, 4))

    # Prob 2
    # v, n = value_iteration(P, 4, 4)
    # print(extract_policy(P, 4, 4, v))

    # Prob 3
    # v, n = value_iteration(P, 4, 4)
    # policy = extract_policy(P, 4, 4, v)
    # print(compute_policy_v(P, 4, 4, policy))

    # Prob 4
    # V, pi, k = policy_iteration(P, 4, 4)
    # print(V)
    # print(pi)
    # print(k)

    # Prob 5
    vi_policy, vi_total_rewards, pi_value_func, pi_policy, pi_total_rewards = frozen_lake(M=100, render=False)
    print("vi_policy\n", vi_policy)
    print("vi_total_rewards\n", vi_total_rewards)
    print("pi_value_func\n", pi_value_func)
    print("pi_policy\n", pi_policy)
    print("pi_total_rewards\n", pi_total_rewards)