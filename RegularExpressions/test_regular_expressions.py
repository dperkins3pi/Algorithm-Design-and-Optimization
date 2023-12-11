"""Unit testing file for Regular Expressions Lab"""


import regular_expressions
import pytest

def test_prob4():
    """
    Write at least one unit test for problem 4,
    testing to make sure your function correctly finds valid Python identifiers.
    """
    # Sets up the pattern
    pattern = regular_expressions.prob4()
    
    # Sets of strings that should work and shouldn't
    good_strings = ["Mouse", "_num =  2.3", "arg_ = 'hey'", "__x__", "var24"
                    "max=total", "string= ''", "num_guesses", "cheese = 2"]
    bad_strings = ["3rats", "_num = 2.3.2", "arg_ = 'one'two", "sq(x)", " x", "max=2total", 
                   "is_4(value==4)", "pattern = r'^one|two fish$'", "one =  'one', two = ' two  ' "]
    
    # Tests all of the good strings
    for s in good_strings:
        assert pattern.search(s), f"'{s}' doesn't match."
        
    # Tests all of the bad strings
    for s in bad_strings:
        assert not pattern.search(s), f"'{s}' matches when it shouldn't."

def test_prob3():
    # Sets up the pattern
    pattern = regular_expressions.prob3()
    
    # Sets of strings that should work and shouldn't
    good_strings = ["Book store", "Book supplier",
                    "Mattress store", "Mattress supplier",
                    "Grocery store", "Grocery supplier"]
    bad_strings = ["Booky store", "Book Book", "Book Grocery store",
                   "This is a Book store", "Grocery stores rock",
                   "Book storesupplier", " Book store", "Bookstore"]
    
    # Tests all of the good strings
    for s in good_strings:
        assert pattern.search(s), f"'{s}' doesn't match."
        
    # Tests all of the bad strings
    for s in bad_strings:
        assert not pattern.search(s), f"'{s}' matches when it shouldn't."