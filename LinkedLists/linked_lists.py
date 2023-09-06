# linked_lists.py
"""Volume 2: Linked Lists.
Daniel Perkins
MATH 321
09/07/23
"""


# Problem 1
class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute.
                
        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        if type(data) not in [int, float, str]:
            raise TypeError("Data in Node must be an int, float, or str")
        self.value = data


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
        length (int): the length of the list
    """
    def __init__(self):
        """Initialize the head, tail, and length attributes by setting
        them to None or 0, since the list is empty initially.
        """
        self.head = None
        self.tail = None
        self.length = 0        # Current size of list

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
        self.length += 1    #the list is 1 longer

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """
        #Set initial index
        index = self.head

        if index is None: #do not operate if list is empty
            raise ValueError(f"'{data}' is not in the list")
        
        while index is not None: #iterate through list until value does not exist
            if index.value == data: #the item is found
                return index #return the object, not the value
            index = index.next #go to next item

        #if statement does not return, the node is not in the list
        raise ValueError(f"'{data}' is not in the list")

        


    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        if i < 0 or i >= self.length:  #check if index is in range
            raise IndexError(f"index '{i}' out of range")
        
        #start with first node
        the_node = self.head
        for index in range(i):
            the_node = the_node.next  #update node depending on index number
        
        return the_node

        

    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        return self.length

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        if self.length == 0: #the list is empty
            return "[]"
        
        the_list = "["  #first part of string
        index = self.head   
        while index is not None:  #iterate through linked list
            the_list += repr(index.value)  #add object to string, in quotes if it is a string
            index = index.next  #move to next index
            if index is None: #no need to add comma
                break  
            the_list += ", " #separate the objects
        the_list += "]" #finish the list

        return the_list

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        target = self.find(data) #find node and return ValueError if not in list
        if target == self.head: #if the target is the first node
            self.head = target.next #move head of list
        else:
            target.prev.next = target.next  #connect previous node to following node
        if target == self.tail: #if the target is at the end of the list
            self.tail = target.prev #move tail of list
        else:
            target.next.prev = target.prev #connect next node to previous node
        self.length -= 1

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """
        if index < 0 or index > self.length: #index out of range
            raise IndexError(f"The index, {index} is out of range of the list of length {self.length}")
        elif index == self.length: #just add data to end of list
            self.append(data)
        else:
            new_node = LinkedListNode(data) #create the new node
            if index == 0: #add to head
                new_node.next = self.head #set link to next value
                self.head = new_node #change head
                self.length += 1 #adjust list length
            else:
                target = self.head #start from head
                for i in range(index): #iterate through list
                    target = target.next #move to next spot
                new_node.prev = target.prev #new node reference left to index before
                new_node.next = target #new node reference right to what used to be there
                target.prev.next = new_node #links previous node to the new one
                target.prev = new_node #object previously there references left to new_node
                self.length += 1 #adjust list length


            
            


# Problem 6: Deque class.
class Deque(LinkedList):
    """
    Double-ended queue. Only data can be accessed/added to end or begining

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
        length (int): the length of the list
    """
    def __init__(self):
        """
        Initialize the head, tail, length attributes by setting
        them to None or 0, since the list is empty initially.
        """
        LinkedList.__init__(self)

    def pop(self):
        """
        Removes and returns the last item from the Deque
        """
        if self.length == 0:
            raise ValueError("The list is empty")
        elif self.length == 1: #only node in list, so just remove the list
            the_object = self.tail #store object before removing it
            self.head = None
            self.tail = None
            self.length = 0
            return the_object
        else:
            the_object = self.tail #store object before removing it
            self.tail.prev.next = None #remove connection from last object to it
            self.tail = self.tail.prev #move tail of deque
            self.length -= 1 #deque is smaller
            return the_object
        
    def popleft(self):
        """
        Removes and returns the first item from the Deque
        """
        if self.length == 0:
            raise ValueError("The list is empty")
        elif self.length == 1: #only node in list, su just remove the listr
            self.head = None
            self.tail = None
            self.length = 0
            return the_object
        else:
            the_object = self.head #store object before removing it
            self.head.next.prev = None #remove connection from first object to it
            self.head = self.head.next #move head
            self.length -= 1 #deque is smaller
            return the_object
        
    def appendleft(self, data):
        """Append a new node to the left containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            self.head.prev = new_node               # head --> new_node
            new_node.next = self.head               # head <-- new_node
            # Now the first node in the list is new_node, so reassign the head
            self.head = new_node
        self.length += 1    #the list is 1 longer

    def remove(*args, **kwargs):
        """Ovewrite O(n) function in list"""
        raise NotImplementedError("Use pop() or popleft() for removal")
    
    def insert(*args, **kwargs):
        """Ovewrite O(n) function in list"""
        raise NotImplementedError("Use append() or appendleft() for insertion")
    

# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    #initialize the Deque
    contents = Deque()

    #read from file
    with open(infile, 'r') as my_file:
        line = True
        while line:  #until file is done
            line = str(my_file.readline()) #get line
            contents.append(line) #append the line

    #write into the outfile
    print(contents)
    with open(outfile, 'w') as my_file:
        #write twice to get rid of error of printing two lines on same line
        my_file.write(contents.pop().value)
        my_file.write(contents.pop().value)
        my_file.write("\n")
        while len(contents) > 0:
            my_file.write(contents.pop().value)

        


if __name__ == "__main__":
    # l = Deque()
    # for x in ['a', 'b', 'c', 'd']:
    #     l.append(x)
    # print(l, len(l))
    # l.remove(2,"2")
    prob7("english.txt", "output_file.txt")