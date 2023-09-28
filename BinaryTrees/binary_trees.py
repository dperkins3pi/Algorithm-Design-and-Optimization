# binary_trees.py
"""Volume 2: Binary Trees.
Daniel Perkins
MATH 321
09/19/23
"""

# These imports are used in BST.draw().
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
import numpy as np
import random
import time
from matplotlib import pyplot as plt

class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        #recursive function to traverse the truee
        def _step(current):
            if current is None:  #not in tree
                raise ValueError(str(data) + " is not in the list")
            elif current.value == data:  #found the node
                return current
            else:   #move to next node
                return _step(current.next)
            
        #start the recursion
        return _step(self.head)


class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        # Recursive function call
        def _step(current):
            """Recursively step through the tree until
            If the data is already in the tree, raise a ValueError
            """
            new_Node = BSTNode(data)            # Create the new node
            if self.root is None:               # Base case 1: empty tree, assign root
                self.root = new_Node
            elif current is None:               # Base case 2: Found spot for the new node
                current = new_Node
            elif data == current.value:         # Base case 3: data already in the tree
                raise ValueError(f"{str(data)} is already in the Tree")
            elif data < current.value:          # Recursively search left.
                _step(current.left)
                if current.left is None:        # if it is the correct spot, connect in tree
                    current.left = new_Node
                    new_Node.prev = current
            else:                              # Recursively search left. 
                _step(current.right)
                if current.right is None:      # if it is the correct spot, connect in tree
                    current.right = new_Node
                    new_Node.prev = current
            
        #call the recursive function
        _step(self.root)

    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        print(f"removing {data}")
        # Recursive function call
        if self.root is None:                       #Base case 1: Tree is empty
            raise ValueError("The tree is empty")
        
        def _step(current):
            """Recursively step through the tree until
            If the data is not in the tree, raise a ValueError
            """
            if current is None:                     #Node not in tree
                raise ValueError(f"There is no node in the tree containing {data}")
            elif current.value == data:              #Value is found

                # Leaf node
                if current.left is None and current.right is None:
                    if current is self.root:                  # The leaf node is the root
                        self.root = None
                    else:
                        if current.prev.left is current:      # If the node is a left child
                            current.prev.left = None
                        elif current.prev.right is current:   # If the node is a right child
                            current.prev.right = None
                        current.prev = None

                # Node with only right child
                elif current.right is not None and current.left is None:
                    if current is self.root:               # The node is the root
                        current.right.prev = None
                        self.root = current.right
                        current.right = None
                    else:
                        if current.prev.left is current:      # If the node is a left child
                            current.prev.left = current.right
                        elif current.prev.right is current:   # If the node is a right child
                            current.prev.right = current.right
                        current.right.prev = current.prev
                        current.prev = None
                        current.right = None
                
                # Node with only left child
                elif current.right is None and current.left is not None:
                    if current is self.root:               # The node is the root
                        current.left.prev = None
                        self.root = current.left
                        current.left = None
                    else:
                        if current.prev.left is current:      # If the node is a left child
                            current.prev.left = current.left
                        elif current.prev.right is current:   # If the node is a right child
                            current.prev.right = current.left
                        current.left.prev = current.prev
                        current.prev = None
                        current.left = None

                # Node with two children
                if current.right is not None and current.left is not None:
                    # Find the predecessor
                    predecessor = current.left
                    while predecessor.right is not None:
                        predecessor = predecessor.right
                    # Store the predecessor value, remove it, set current to predecessor
                    predecessor_value = predecessor.value
                    self.remove(predecessor_value)
                    current.value = predecessor_value

            #step through tree to find the data
            elif data > current.value:        # Search in right half of the tree
                 _step(current.right)
            elif data < current.value:        # Search in left half of the tree
                 _step(current.left)
            
        _step(self.root)        

        

    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    with open("english.txt", 'r') as my_file:
        data = my_file.read().splitlines()  #read file into list

    #initilize data for the graphs
    size = np.arange(3, 11)**2
    build_time_linked_list = []
    build_time_BST = []
    build_time_AVL = []
    search_time_linked_list = []
    search_time_BST = []
    search_time_AVL = []

    for k in size:
        subset = random.sample(data, k)  #create random subset
        to_find = random.sample(subset, 5)  #subset to find items
        # initialize lists

        #Building time for linked list
        start = time.time()
        my_linked_list = SinglyLinkedList()
        for word in subset:
            my_linked_list.append(word)
        end = time.time()
        build_time_linked_list.append(end - start)

        #Searching time for linked list
        start = time.time()
        for x in to_find:
            my_linked_list.iterative_find(x)
        end = time.time()
        search_time_linked_list.append(end - start)

        #Building time for BST
        start = time.time()
        my_tree = BST()
        for word in subset:
            my_tree.insert(word)
        end = time.time()
        build_time_BST.append(end - start)

        #Searching time for BST
        start = time.time()
        for x in to_find:
            my_tree.find(x)
        end = time.time()
        search_time_BST.append(end - start)

        #Building time for AVL
        start = time.time()
        avl_tree = AVL()
        for word in subset:
            avl_tree.insert(word)
        end = time.time()
        build_time_AVL.append(end - start)

        #Searching time for AVL
        start = time.time()
        for x in to_find:
            avl_tree.find(x)
        end = time.time()
        search_time_AVL.append(end - start)

    print(search_time_linked_list)
    print(search_time_BST)
    print(search_time_AVL)

    fig, axes = plt.subplots(1, 2) #arrange plots into 1x2 grid
    fig.tight_layout(pad = 4) #space out the axes
    plt.suptitle("Operation Time of Different Data Structures")

    #plot 1
    axes[0].set_title("Building The Structure")
    axes[0].loglog(size, build_time_linked_list, label="Linked List")
    axes[0].loglog(size, build_time_BST, label="BST")
    axes[0].loglog(size, build_time_AVL, label="AVL")
    axes[0].legend()
    axes[0].set_xlabel("Size of the Data Structure")
    axes[0].set_ylabel("Time taken (seconds)")

    #plot 2
    axes[1].set_title("Finding An Element")
    axes[1].loglog(size, search_time_linked_list, label="Linked List")
    axes[1].loglog(size, search_time_BST, label="BST")
    axes[1].loglog(size, search_time_AVL, label="AVL")
    axes[1].legend()
    axes[1].set_xlabel("Size of the Data Structure")
    axes[1].set_ylabel("Time taken (seconds)")

    plt.show()

if __name__=="__main__":
    # problem 1
    # my_linked_list = SinglyLinkedList()
    # my_linked_list.append(5)
    # my_linked_list.append(6)
    # print(my_linked_list.iterative_find(8))
    # print(my_linked_list.recursive_find(8))

    # # problem 2
    # my_Tree = BST()
    # my_Tree.insert(5)
    # my_Tree.insert(8)
    
    # #problem 3
    # print(my_Tree)
    # for x in [5]:
    #     my_Tree.remove(x)
    # print(my_Tree)

    #prob4()

    print()

    