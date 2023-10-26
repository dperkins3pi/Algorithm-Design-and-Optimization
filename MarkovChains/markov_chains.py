# markov_chains.py
"""Volume 2: Markov Chains.
Daniel Perkins
MATH 321
10/22/23
"""

import numpy as np
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        (fill this out)
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        # Set attributes of tree
        self.A = A
        if states is None:
            self.labels = np.array([n for n in range(len(A))])   # Default labels
        else:
            self.labels = np.array(states)
        self.label_dict = dict()

        # Set the dictionary up
        i = 0           # for iteration
        for column in A.T:
            if abs(sum(column) - 1) > 0.0000001:     # Verify that the matrix is column stochastic
                print(i, column)
                print(sum(column))
                raise ValueError("The Matrix is not column stochastic")
            self.label_dict[self.labels[i]] = i    # Set each label to map to column number
            i += 1
            

    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        distribution = self.A.T[self.label_dict[state]]  # Get column of the state
        draw = np.random.multinomial(1, distribution)   # Take a draw
        return self.labels[np.argmax(draw)]     # Return the state it transitioned to


    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        labels = [start]  # intialize list
        state = start
        for i in range(N - 1):   # transition n-1 times
            state = self.transition(state)
            labels.append(state)
        return labels

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        labels = [start]  # initialize list
        state = start
        while state != stop:   # until the state is stop
            state = self.transition(state)
            labels.append(state)   # add to list
        return labels

    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        x = np.random.random(len(self.A))     # random vector
        x = x / np.linalg.norm(x, ord=1)      # Normalize so that it adds to one
        x_old = np.zeros_like(x)    # to allow for while loop
        k = 0                 # find number of iters
        while la.norm(x_old - x, ord=1) > tol:
            if k > maxiter:
                raise ValueError("A^k does not converge")
            x_old = x
            x = self.A @ x    # Multiplty x_k+1=Ax_k
            k += 1    # Keep track of iterations
        return x

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        # Read the file
        with open(filename, "r") as file:
            data = file.readlines()
        data = [line.split() for line in data]

        # Get the state labels
        set_of_words = set()   # Use to store each word without repitition
        for sentence in data:
            for word in sentence:
                set_of_words.add(word)
        words = list(set_of_words)   # Make it a set to preserve order
        words.insert(0, "$tart")
        words.append("$top")

        # Create Matrix of zeroes of appropriate size
        A = np.zeros((len(words),len(words)))
        # self.A = MarkovChain(A, set_of_words)

        # Create training set
        for sentence in range(len(data)):
            # Prepend start and end
            data[sentence].insert(0, "$tart")
            data[sentence].append("$top")
            prev_word = data[sentence][0] # Start with first word
            for word in data[sentence][1:]:  # For each word (exludint the first since nothing was before it)
                A[words.index(word), words.index(prev_word), ] += 1   # Add 1 to entry from previous word to current word
                prev_word = word        # Update previous word
        A[len(A) - 1, len(A) - 1] = 1   # Map End to itself

        # Normalize the columns
        norm = np.linalg.norm(A, ord=1, axis=0)
        A = A / norm

        # Set all attributes
        chain = MarkovChain(A, words)
        self.A = chain.A  # Save training set as Markov Chain
        self.labels = chain.labels
        self.label_dict = chain.label_dict


    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        sentence = self.path("$tart", "$top")
        sentence = " ".join(sentence[1:-1])
        return sentence

if __name__=="__main__":
    # A = np.array([[0.5, 0.3, 0.1, 0], [0.3, 0.3, 0.3, 0.3], [0.2, 0.3, 0.4, 0.5], [0, 0.1, 0.2, 0.2]])
    # # A = np.array([[.7, .6], [.3, .4]])
    # my_chain = MarkovChain(A, ["hot", "mild", "cold", "freezing"])

    # # Prob 1
    # # print(my_chain.A)
    # # print(my_chain.labels)
    # # print(my_chain.label_dict)

    # # Prob 2
    # # print(my_chain.transition("cold"))

    # # Prob 3
    # # print(my_chain.walk("mild", 5))
    # # print(my_chain.path("hot", "freezing"))

    # # Prob 4 Testing
    # # x = my_chain.steady_state()
    # # print(my_chain.A @ x)
    # # print(np.linalg.matrix_power(my_chain.A, 15))

    # # Prob 4, validate results for prob 3
    # B = np.array([[.7, .6], [.3, .4]])
    # my_chain = MarkovChain(B, ["hot", "cold"])
    # steady_state = my_chain.steady_state()
    # print("steady state", steady_state)
    # large_walk = my_chain.walk("hot", 100)
    # print("num of hot days in long walk of 100: ", sum(np.core.defchararray.count(large_walk, "hot")))

    # Prob 5 and 6
    # chain = SentenceGenerator("yoda.txt")
    # for _ in range(5):
    #     print(chain.babble())
    #     print()
    print()