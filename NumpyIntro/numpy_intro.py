# numpy_intro.py
"""Python Essentials: Intro to NumPy.
Daniel Perkins
MATH 321
09/04/23
"""


import numpy as np


def prob1():
    """ Define the matrices A and B as arrays. Return the matrix product AB. """
    A = np.array([[3, -1, 4], [1, 5, -9]])
    B = np.array([[2, 6, -5, 3],[5, -8, 9, 7],[9, -3, -2, -3]])
    return A @ B


def prob2():
    """ Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A. """
    A = np.array([[3, 1, 4], [1, 5, 9], [-5, 3, 1]])
    return -(A @ A @ A) + 9*(A @ A) - 15*A


def prob3():
    """ Define the matrices A and B as arrays using the functions presented in
    this section of the manual (not np.array()). Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    A = np.ones((7, 7), dtype=np.int64) #7x7 matrix of all ones
    A = np.triu(A) #upper-right triangle portion
    B = 5 * np.ones((7, 7), dtype=np.int64) #7x7 matrix of all 5's
    B = B - 6*np.tril(np.ones((7, 7),dtype=np.int64)) #Make bottom right triangle -1's
    return A @ B @ A


def prob4(A):
    """ Make a copy of 'A' and use fancy indexing to set all negative entries of
    the copy to 0. Return the resulting array.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    A_copy = np.copy(A) #copy the matrix so that it doesn't change the initial one
    mask = A_copy < 0  #boolean mask for all negative numbers
    A_copy[mask] = 0  #set negative numbers to 0
    return A_copy


def prob5():
    """ Define the matrices A, B, and C as arrays. Use NumPy's stacking functions
    to create and return the block matrix:
                                | 0 A^T I |
                                | A  0  0 |
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.arange(6).reshape((3,2)).T
    B = np.tril(np.full((3,3),3))
    C = -2 * np.eye(3)
    I = np.eye(3)
    top_row = np.hstack((np.zeros((A.T.shape[0],A.shape[1])), A.T, I))
    middle_row = np.hstack((A, np.zeros((A.shape[0],A.T.shape[1])), np.zeros((A.shape[0],I.shape[1]))))
    bottom_row = np.hstack((B, np.zeros((B.shape[0],A.T.shape[1])), C))
    full_matrix = np.vstack((top_row, middle_row, bottom_row))
    return full_matrix.astype(np.int64)


def prob6(A):
    """ Divide each row of 'A' by the row sum and return the resulting array.
    Use array broadcasting and the axis argument instead of a loop.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    row_sums = A.sum(axis=1)  #find sum of each row
    row_sums = row_sums.reshape((-1,1))  #make the matrix vertical
    return A / row_sums  #array broadcasting


def prob7():
    """ Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid. Use slicing, as specified in the manual.
    """
    grid = np.load("grid.npy")
    horizontal_array = grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:] #each (i,j) entry is the product the four numbers to the right
    vertical_array = grid[:-3, :] * grid[1:-2,:] * grid[2:-1,:] * grid[3:,:]
    diagonal_down_array = grid[:-3,:-3] * grid[1:-2,1:-2] * grid[2:-1,2:-1] * grid[3:,3:] #same as the example of highlighted numbers
    diagonal_up_array = grid[3:,:-3] * grid[2:-1,1:-2] * grid[1:-2,2:-1] * grid[:-3,3:]#similar to last one, but the diagonal is in the other direction
    list_of_maxes = [np.max(horizontal_array),np.max(vertical_array),np.max(diagonal_down_array),np.max(diagonal_up_array)] #max value of each of the above arrays
    return np.max(list_of_maxes) #max of the maxes


if __name__=="__main__":
    #print(prob1())
    #print(prob2())
    #print(prob3())
    #print(prob4(np.array([-3,-1,3])))
    #print(prob5())
    #print(prob6(np.array([[1,1,0],[0,1,0],[1,1,1]])))
    #print(prob7())
    print()