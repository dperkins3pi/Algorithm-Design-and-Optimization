# breadth_first_search.py
"""Volume 2: Breadth-First Search.
Daniel Perkins
MATH 321
10/5/23
"""

import collections
import networkx as nx
from matplotlib import pyplot as plt

# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        # if the key is not in there
        if n not in self.d.keys():
            # Create a dictionary with the node and its value as the empty set
            node = {n: set()}
            self.d.update(node)




    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        # add the nodes if they are not present
        self.add_node(u)
        self.add_node(v)

        #add the nodes to the adjacency dictionary of each element
        self.d[u].update({v})
        self.d[v].update({u})

    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        # Raise error if not in graph
        if n not in self.d.keys():
            raise KeyError(f"Node {n} is not in the graph")
        else:
            edges = self.d[n]   #edges to remove
            self.d.pop(n)   #remove the value
            # remove the edges adjacent to the node
            for edge in edges:
                self.d[edge].discard(n)

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        # Raise error if no edge in the graph
        try: 
            if u not in self.d[v]:
                raise KeyError(f"the edge {u}=>{v} is not in the graph")
            if v not in self.d[u]:
                raise KeyError(f"the edge {u}<={v} is not in the graph")
        except:
            raise KeyError(f"the edge {u}<=>{v} is not in the graph")
        
        # Remove the edge from each adjacency dictionary
        self.d[u].remove(v)
        self.d[v].remove(u)


    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        # initailize the data structures
        V = []
        Q = collections.deque()  # Create Q as a deck
        M = set()
        Q.append(source)
        M.add(source)

        if source not in self.d.keys():
            raise KeyError("The source node is not in the graph")

        # until every node is traversed
        while len(Q) > 0:  #until Q is empty
            current_node = Q.popleft()  # Pop the element that was added first
            V.append(current_node)
            neighbors = self.d[current_node]  # Set of elements that current node points to
            for neighbor in neighbors:
                if neighbor not in M:
                    Q.append(neighbor)
                    M.add(neighbor)
        return V
        

    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        # initailize the data structures
        Q = collections.deque()  # Create Q as a deck
        M = set()
        Q.append(source)
        M.add(source)
        back_traverse = {}   # To backtrack and find the path used to get the target
        target_found = False

        if source not in self.d.keys():
            raise KeyError("The source node is not in the graph")

        # To create the back_traverse from target to source
        while not target_found:  #until target is found
            current_node = Q.popleft()  # Pop the source element
            neighbors = self.d[current_node]  # Set of elements that current node points to
            for neighbor in neighbors:
                if neighbor not in M:
                    Q.append(neighbor)
                    M.add(neighbor)
                    back_traverse[neighbor] = current_node  # map the visited node to visiting
                if neighbor == target:
                    target_found = True
            if len(Q) == 0:             # Backup if statement if not found
                raise KeyError("The source node is not in the graph")
        
        # Traverse through back_traverse to find list of nodes visited in the parth
        current_node = target
        path = [current_node]
        while current_node != source:  # Until source is found
            current_node = back_traverse[current_node]
            path.append(current_node)
        path.reverse()   # Reverse order of list, so that it starts at the source
        return path


# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        # Initialize attributes
        self.movie_titles = set()
        self.actor_names = set()
        self.graph = nx.Graph()

        # Read the file
        with open(filename, "r") as file:
            data = file.read()
            data = data.splitlines()
            data = [line.split("/") for line in data]   # Split the strings in the list

        # Upload the data into the class atributes
        self.movie_titles.update(set([item[0] for item in data]))
        actors = [item[1:] for item in data]
        
        i = 0  # used to increment access each movie by index
        for movie in actors:
            self.actor_names.update(set(movie)) # Add the actors to the set by each movie
            for actor in movie:
                self.graph.add_edge(data[i][0], actor)  # Add the edge in the graph of each actor
            i += 1


    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        # Find shortest path and divide by two to ignore the movies in the path
        path = nx.shortest_path(self.graph, source, target)
        # To find length, subtract 1 from list and divide by two to ignore movies
        length = int((len(path) - 1) / 2)
        return path, length

    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        # Find shortest path lengths
        path_lengths = nx.shortest_path_length(self.graph, target)
        # Divide each path length by two to ignore the movies in the paths
        for key in path_lengths:
            path_lengths[key] = path_lengths[key] / 2

        # Plot the histogram
        plt.hist(path_lengths.values(), bins=[i-.5 for i in range(8)])
        plt.title(f"The {target} number's of other actos")
        plt.xlabel(f"The {target} number")
        plt.ylabel("Number of Actors")
        plt.show()

if __name__ == "__main__":
    # prob 1
    # my_Graph = Graph()
    # print(my_Graph)
    # my_Graph.add_edge(1,8)
    # my_Graph.add_edge(6,8)
    # my_Graph.add_edge(1,3)
    # my_Graph.add_edge(8,3)
    # my_Graph.add_edge(1,9)
    # print(my_Graph)
    # my_Graph.remove_node(1)
    # print(my_Graph)
    # my_Graph.remove_edge(3, 8)
    # print(my_Graph)

    #prob 2
    # my_Graph = Graph()
    # my_Graph.add_edge("A","B")
    # my_Graph.add_edge("A","D")
    # my_Graph.add_edge("B","D")
    # my_Graph.add_edge("C","D")
    # print(my_Graph)
    # print("Traversal: ", my_Graph.traverse("A"))   
    # # Ask TA: Order not always the same 
    # #(not always possible since no path from B to C)

    # #prob 3
    # print("Shortest path: ", my_Graph.shortest_path("A", "C"))   

    # prob 4
    # my_Graph = MovieGraph()
    # print(len(my_Graph.movie_titles))
    # print(len(my_Graph.actor_names))
    # print(my_Graph.graph.edges("Toby Jones"))

    # prob 5
    # my_Graph = MovieGraph()
    # print(my_Graph.path_to_actor("Kevin Bacon", "Jennifer Lawrence"))
    # print(my_Graph.path_to_actor("Kevin Bacon", "Christopher Robin"))
    # print(my_Graph.path_to_actor("Kevin Bacon", "Tim Robbins"))
    # print(my_Graph.path_to_actor("Kevin Bacon", "Zach Grenier"))
    # print(my_Graph.path_to_actor("Kevin Bacon", "Chris Pratt"))
    # print(my_Graph.path_to_actor("Kevin Bacon", "Adam Sandler"))

    # prov 6
    my_Graph = MovieGraph("movie_data_small.txt")
    my_Graph.average_number("Kevin Bacon")