"""Unit testing file for binary_trees.py"""

import binary_trees
import pytest

@pytest.fixture #pytest fixture to help in the construction of a tree
def build_the_tree():
    tree_1 = binary_trees.BST()
    for i in [4,3,8,1,2,9,23,6,5,7]:
        tree_1.insert(i)
    return tree_1
    

def test_insert(build_the_tree):
    """Unit test problem 2, creating an insert method for your BST class"""
    
    tree1 = build_the_tree
    assert tree1.root.value == 4, "root inserted correctly"

    parent = tree1.root
    assert parent.right.value == 8, "right child of root inserted incorrectly"
    assert parent.left.value == 3, "left child of root inserted incorrectly"

    val1 = parent.left
    assert val1.left.value == 1, "left child with no right child inserted incorrectly"
    assert val1.right is None, "right child non-empty"

    val2 = parent.right.left
    assert val2.left.value == 5, "left child of node in middle of tree is correct"
    assert val2.right.value == 7, "right child of node in middle of tree is correct"

    val3 = parent.right.right.right
    assert val3.left is None, "base level of tree should have no left child"
    assert val3.right is None, "base level of tree should have no right child"

    with pytest.raises(ValueError) as excinfo:
        tree1.insert(1), "Did not raise a value error for trying to insert a duplicate node"
    
    #check if insertion of one node is correct
    tree2 = binary_trees.BST()
    tree2.insert(1)
    val4 = tree2.root
    
    assert val4.left is None
    assert val4.right is None
    assert val4.prev is None


def test_remove():
    """Unit Test for Problem 3: creating a remove method for your BST class"""
    

    my_Tree = binary_trees.BST()
    for x in [6, 4, 8, 1, 5, 7, 10, 3, 9]:
        my_Tree.insert(x)
    for x in [7, 10, 1, 4, 3]:
        my_Tree.remove(x) 
    assert my_Tree.root.value == 6, "[7, 10, 1, 4, 3] removed incorrectly"
    assert my_Tree.root.left.value == 5, "[7, 10, 1, 4, 3] removed incorrectly"
    assert my_Tree.root.right.value == 8, "[7, 10, 1, 4, 3] removed incorrectly"
    assert my_Tree.root.right.right.value == 9, "[7, 10, 1, 4, 3] removed incorrectly"

    my_Tree = binary_trees.BST()
    for x in [5, 3, 6, 1, 4, 7, 8]:
        my_Tree.insert(x)
    for x in [8, 6, 3, 5]:
        my_Tree.remove(x)
    assert my_Tree.root.value == 4, "[8, 6, 3, 5] removed incorrectly"
    assert my_Tree.root.left.value == 1, "[8, 6, 3, 5] removed incorrectly"
    assert my_Tree.root.right.value == 7, "[8, 6, 3, 5] removed incorrectly"

    my_Tree = binary_trees.BST()
    my_Tree.insert(5)
    my_Tree.remove(5)
    assert my_Tree.root is None, "[5] removed incorrectly"

    with pytest.raises(ValueError) as excinfo:
        my_Tree = binary_trees.BST()
        my_Tree.insert(5)
        my_Tree.remove(3)
    assert excinfo.value.args[0] == "There is no node in the tree containing 3"

    with pytest.raises(ValueError) as excinfo:
        my_Tree = binary_trees.BST()
        my_Tree.remove(3)
    assert excinfo.value.args[0] == "The tree is empty"


            #     >>> print(12)                       | >>> print(t3)
            # [6]                                 | [5]
            # [4, 8]                              | [3, 6]
            # [1, 5, 7, 10]                       | [1, 4, 7]
            # [3, 9]                              | [8]
            # >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            # ...     t1.remove(x)                | ...     t3.remove(x)
            # ...                                 | ...
            # >>> print(t1)                       | >>> print(t3)
            # [6]                                 | [4]
            # [5, 8]                              | [1, 7]
            # [9]                                 |
            #                                     | >>> print(t4)
            # >>> print(t2)                       | [5]
            # [2]                                 | >>> t4.remove(1)
            # [1, 3]                              | ValueError: <message>
            # >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            # ...     t2.remove(x)                | >>> print(t4)
            # ...                                 | []
            # >>> print(t2)                       | >>> t4.remove(5)
            # []                                  | ValueError: <message>






