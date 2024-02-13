"""Volume 2: Simplex

Daniel Perkins
2/13/24
MATH 323
"""

import numpy as np

# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        x = np.zeros_like(c)
        if(np.sum(A @ x <= b) != len(b)):   # If at least one of the components does not return true
            raise ValueError("The given system is infeasible at the origin")
        # Initialize the dictionary
        self.dictionary = self._generatedictionary(c, A, b)


    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        # Get size of b
        m = len(b)

        # Set up the dictionary
        I = np.eye(m)
        A = -np.block([A, I])
        c = np.hstack((c, np.zeros(m)))
        top = np.r_[np.zeros(1), c]   # Combine 0 and c horizonatally
        bottom = np.c_[b, A]        # Combine b and A horizontlaly
        return np.block([[top], [bottom]])   # Concoatenate top and bottom


    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        for col in range(1, len(self.dictionary[0])):  # for each column except the first
            if(self.dictionary[0][col] < 0): return col  # Find first column with negative entry
        return None   # if no column has negative entry, return none

    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        ratios = []   # list of all the entries
        for row in range(1, len(self.dictionary)):   # for each row (excluding the first)
            if self.dictionary[row, index] < 0:   # If the element is negative
                if(self.dictionary[row, index] == 0): ratios.append(np.inf)  # avoid dividing by 0
                else: ratios.append(-self.dictionary[row, 0] / self.dictionary[row, index])
            else:
                ratios.append(np.inf)    # If nonnegative, append infinity (so it is not chosen)
        
        if min(ratios) == np.inf: return None  # If there is no solution, return None
        
        # Find smallest ration (or first smallest if there are multiple)
        row = np.argmin(ratios) + 1  # add 1 since we started at row index 1
        return row
        


    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        col = self._pivot_col()
        if col is None: raise ValueError("The problem is unbounded")  # Raise value error if all all columns are nonnegative
        row = self._pivot_row(col)
        if row is None: raise ValueError("The problem is unbounded")  # Raise value error if all all rows are nonnegative

        # Make the pivot
        self.dictionary[row] /= -self.dictionary[row, col]   # Divide pivot row by negative value of pivot column
        for i in range(len(self.dictionary)):
            if i == row: continue   # No more row operations performed on pivot row
            self.dictionary[i] = self.dictionary[i] + self.dictionary[i, col] * self.dictionary[row]  # row operation to zero out

    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        # See if all values are nonnegative on top row (excluding the first entry)
        if((np.sum(self.dictionary[0, 1:] >= 0)) == len(self.dictionary[0, 1:])): done = True
        else: done = False
        while not done:
            self.pivot()  # make the pivot
            # Check again
            if((np.sum(self.dictionary[0, 1:] >= 0)) == len(self.dictionary[0, 1:])): done = True

        minimal_val = self.dictionary[0,0]  # get min value

        basic_var = dict()  # gets dependent variables
        nonbasic_var = dict()   # gets independent variables
        for i in range(1, len(self.dictionary[0])):
            if(self.dictionary[0, i] != 0): nonbasic_var[i - 1] = 0  # independent variable
            else: 
                if(i + 1 < len(self.dictionary)):   # if index not out of bounds
                    basic_var[i - 1] = self.dictionary[i + 1, 0]
                else:   # index out of bounds, so loop back
                    basic_var[i - 1] = self.dictionary[i - len(self.dictionary[0]), 0]

        return minimal_val, basic_var, nonbasic_var

# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    # Load the data
    data = np.load(filename)
    A = data["A"]
    p = data["p"]
    m = data["m"]
    d = data["d"]

    # """Class for solving the standard linear optimization problem

    #                     minimize        c^Tx
    #                     subject to      Ax <= b
    #                                      x >= 0
    # via the Simplex algorithm.
    # """
    # # Problem 1
    # def __init__(self, c, A, b):

    print("A", A)
    print("p", p)
    print("m", m)
    print("d", d)

    b = np.concatenate((m, d))
    new_A = np.block([[A], [np.eye(len(d))]])

    print()
    print()
    print("New A")
    print(new_A)
    print("B", b)
    print("C", -p)
    # IS THIS THE WAY TO SET IT UP??????????????????????????/



    # solver = SimplexSolver(-p, A, m)
    # solver.solve()



if __name__=="__main__":

    # c = np.array([-3, -2])
    # A = np.array([[1, -1],
    #               [3, 1], 
    #               [4, 3]])
    # b = np.array([2, 5, 7])
    # solver = SimplexSolver(c, A, b)

    # Prob 1 and 2
    # print(solver.dictionary)

    # Prob 3
    # col = solver._pivot_col()
    # print("pivot column:", col)
    # print("pivot row:", solver._pivot_row(col))

    # Prob 4
    # print(solver.dictionary)
    # solver.pivot()
    # print(solver.dictionary)

    # Prob 5
    # print(solver.solve())

    # Prob 6
    prob6()