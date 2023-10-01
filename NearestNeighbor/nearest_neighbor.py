# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
Daniel Perkins
MATH 321
9/26/23
"""

import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree
from scipy import stats
from matplotlib import pyplot as plt

# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    norms = []   #list of distances
    for point in X:
        norms.append(la.norm(point - z))   #add distance of each point
    return X[np.argmin(norms)]   #returns the point with smallest distance




# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        if type(x) is not np.ndarray:  #raise an error if not an array
            raise TypeError("The input was not a numpy array")
        self.value = x
        self.left = None
        self.right = None
        self.pivot = None

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        if self.root is None:   #tree is empty
            new_node = KDTNode(data)
            new_node.pivot = 0
            self.root = new_node
            self.k = len(data)   #set new k value since tree was empty
            return
        elif self.k != len(data):   #data not correct size
            raise ValueError(f"Data to be inserted is not in R^{self.k}")
    
        # Recursive function to step through each node
        def _step(current, pivot):
            if current is None:
                new_node.pivot = pivot                     #Set pivot value based on level in tree
                return
            elif data[pivot] == current.value[pivot]:
                raise ValueError(f"Node with {data[pivot]} on pivot {pivot} is already in the tree")
            elif data[pivot] < current.value[pivot]:         # Go to left subtree
                _step(current.left, (pivot + 1) % self.k)  # We call pivot % self.k to find pivot in equivelence class Z_k
                if current.left is None:                   # Add node to left when empty
                    current.left = new_node
            elif data[pivot] > current.value[pivot]:      # Go to right subtree
                _step(current.right, (pivot + 1) % self.k) 
                if current.right is None:                  # Add node to right when empty
                    current.right = new_node

        # Create new node and find its parent
        new_node = KDTNode(data)
        _step(self.root, 0)


    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        #recursive function
        def kd_search(current, nearest, d):
            if current is None:         #Base case: dead end
                return nearest, d
            
            # get pivot and vector at the current node
            x = current.value
            i = current.pivot

            #Check if current is closer to z than nearest
            if la.norm(x - z) < d:
                nearest = current
                d = la.norm(x - z)

            if z[i] < x[i]:  #Search to the left
                nearest, d = kd_search(current.left, nearest, d)
                if z[i] + d >= x[i]:   #Search to the right if needed
                    nearest, d = kd_search(current.right, nearest, d)
            else:   #Search to the right of the tree
                nearest, d = kd_search(current.right, nearest, d)
                if z[i] - d <= x[i]:    #Search to the left if needed
                    nearest, d = kd_search(current.left, nearest, d)
            return nearest, d
        
        #call the recursive function and return
        node, d = kd_search(self.root, self.root, la.norm(self.root.value - z))
        return node.value, d


    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.

    n_neighbors - the number of neighbots to include in the vote (k in k-nearest neighbor)
    """
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.tree = KDTree(X)  # make a tree from the data to improve nearest search efficiency
        self.y = y

    def predict(self, z):
        """
        Querys the tree for n_neighbor elements of X closes to z
        Returns the most common label of those neighbors, for machine learning prediction
        """
        # find list of the smallest n_neighbors Euclidean distances
        distances, indices = self.tree.query(z, k=self.n_neighbors)
        y = [self.y[index] for index in indices]   #convert indices to the elements in the labels
        return stats.mode(y)[0]   #return the most common label




# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    # Extract the data and load into test and training sets
    data = np.load("mnist_subset.npz")
    X_train = data["X_train"].astype(np.float64)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float64)
    y_test = data["y_test"]
    
    KNeighbor = KNeighborsClassifier(n_neighbors)
    KNeighbor.fit(X_train, y_train)
    print("prediction", KNeighbor.predict(X_test))
    print("y-test", y_test)


if __name__ == "__main__":
    #problems 1-2
    # z = np.array([1, 1])
    # X = np.array([[2, 2],[0, 0]])
    # print(exhaustive_search(X, z))
    # print(type(z) is )
    # my_node = KDTNode(np.array([2,4]))
    # # print(my_node.value)

    # #problems 3-4
    # my_tree = KDT()
    # my_tree.insert(np.array([5, 5]))
    # my_tree.insert(np.array([3, 2]))
    # my_tree.insert(np.array([8, 4]))
    # my_tree.insert(np.array([2, 6]))
    # # print(my_tree.query(np.array([8, 0])))
    # # print(exhaustive_search(np.array([[5, 5], [3, 2], [8, 4], [2, 6]]), np.array([8, 0])))

    # KNeighbor = KNeighborsClassifier(2)
    # KNeighbor.fit(np.array([[-981, 1], [27, 2], [4, 4], [4, 6]]), np.array([11, 42, 553, 5533]))
    # print(KNeighbor.predict(np.array([4, 5])))

    # data = np.random.random((100,5))
    # target = np.random.random(5)
    # tree = KNeighborsClassifier(3)
    # tree.fit(data, target)
    # .fit(np.array([[-981, 1], [27, 2], [4, 4], [4, 6]]), np.array([11, 42, 553, 5533]))
    # print(KNeighbor.predict(np.array([0, 0])))

    prob6(4)
    