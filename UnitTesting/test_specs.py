# test_specs.py
"""Python Essentials: Unit Testing.
Daniel Perkins
MATH 321
09/04/23
"""

import specs
import pytest
import random


def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    assert specs.smallest_factor(1) == 1, "Base Case of 1"  #No error
    assert specs.smallest_factor(2) == 2, "Test on next lowers number" #No error
    assert specs.smallest_factor(4) == 2, "Square number, 4" #ERRORRR!!!! -Solved
    assert specs.smallest_factor(16) == 2,  "Other square number (a square number squared)" #No Error
    assert specs.smallest_factor(25) == 5, "Third square number (a prime number squared)"  #ERROR!!!!! - Solved
    assert specs.smallest_factor(125) == 5, "Cubic number, 125" #No error
    assert specs.smallest_factor(53) == 53, "Large prime number" #No error
    assert specs.smallest_factor(637) == 7, "Large number, large prime factor" #No error
    #The errors occured when n was a square of a prime number

# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    assert specs.month_length("January") == 31, "Month of January should have 31 days"
    assert specs.month_length("February", leap_year=False) == 28, "Month of February should have 28 days outside of Leap Year"
    assert specs.month_length("February", leap_year=True) == 29, "Month of February should have 29 days on Leap year"
    assert specs.month_length("March") == 31, "Month of March should have 31 days"
    assert specs.month_length("April") == 30, "Month of April should have 30 days"
    assert specs.month_length("May") == 31, "Month of May should have 31 days"
    assert specs.month_length("June") == 30, "Month of June should have 30 days"
    assert specs.month_length("July") == 31, "Month of July should have 31 days"
    assert specs.month_length("August") == 31, "Month of August should have 31 days"
    assert specs.month_length("September") == 30, "Month of September should have 30 days"
    assert specs.month_length("October") == 31, "Month of October should have 31 days"
    assert specs.month_length("November") == 30, "Month of November should have 30 days"
    assert specs.month_length("December") == 31, "Month of December should have 31 days"
    assert specs.month_length("John") == None, "John is not a month"
    #Note, I didn't need to do that many tests for coverage, just one month with 31 days, one with 30, 29, 28, and None


# Problem 3: write a unit test for specs.operate().
def test_operate():
    with pytest.raises(TypeError) as excinfo: #must raise an error is oper is not a string
        specs.operate(1, 1, 1)
    assert excinfo.value.args[0] == "oper must be a string"

    assert specs.operate(1, 1, '+') == 2, "1+1=2"
    assert specs.operate(1, 1, '-') == 0, "1-1=0"
    assert specs.operate(1, 1, '*') == 1, "1*1=1"

    assert specs.operate(2, 1, "/") == 2, "2/1=2"
    assert specs.operate(1, 2, "/") == 0.5, "1/2=0.5 (check for float)"
    with pytest.raises(ZeroDivisionError) as excinfo: #must raise an error when dividing by 0
        specs.operate(1, 0, "/")
    assert excinfo.value.args[0] == "division by zero is undefined"

    with pytest.raises(ValueError) as excinfo: #must raise an error if oper is another value
        specs.operate(3, 7, "g")
    assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"



# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3

def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7

    with pytest.raises(ZeroDivisionError) as excinfo: #must raise error if the denominator is 0
        frac = specs.Fraction(1,0)
    assert excinfo.value.args[0] == "denominator cannot be zero"

    with pytest.raises(TypeError) as excinfo: #must raise error if the arguments are not integers
        frac = specs.Fraction(0.5, 1)
    assert excinfo.value.args[0] == "numerator and denominator must be integers"
    with pytest.raises(TypeError) as excinfo:
        frac = specs.Fraction(1, 0.5)
    assert excinfo.value.args[0] == "numerator and denominator must be integers"


def test_fraction_str(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    assert str(specs.Fraction(4, 1)) == "4" #new case, if denominator is 1

def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)
    assert 2.5 == specs.Fraction(5, 2)  #new case comparing it to a float

def test_fraction_add(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3 + frac_1_2 == specs.Fraction(5, 6)
    assert frac_1_3 + frac_1_3 == specs.Fraction(2, 3)
    assert frac_1_3 + frac_n2_3 == specs.Fraction(-1, 3)
    assert frac_1_2 + frac_n2_3 == specs.Fraction(-1, 6)

def test_fraction_sub(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3 - frac_1_2 == specs.Fraction(-1, 6)
    assert frac_1_3 - frac_1_3 == 0
    assert frac_1_3 - frac_n2_3 == 1
    assert frac_1_2 - frac_n2_3 == specs.Fraction(7, 6)

def test_fraction_mul(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3 * frac_1_2 == specs.Fraction(1, 6)
    assert frac_1_3 * frac_1_3 == specs.Fraction(1, 9)
    assert frac_1_3 * frac_n2_3 == specs.Fraction(-2, 9)
    assert frac_1_2 * frac_n2_3 == specs.Fraction(-2, 6)

def test_fraction_truediv(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3 / frac_1_2 == specs.Fraction(2, 3)
    assert frac_1_3 / frac_1_3 == 1
    assert frac_1_3 / frac_n2_3 == specs.Fraction(-1, 2)
    assert frac_1_2 / frac_n2_3 == specs.Fraction(-3, 4)

    with pytest.raises(ZeroDivisionError) as excinfo: #must raise error if it divides by 0
        frac = specs.Fraction(1, 6) / specs.Fraction(0,1)
    assert excinfo.value.args[0] == "cannot divide by zero"

# Problem 5: Write test cases for Set.
def test_count_sets():
    hand_one = ["1111", "2122", "0200", "1211", "2222", "2122", "2022", "2222"] #will raise a value error because only 8 cards
    with pytest.raises(ValueError) as excinfo:
        num_of_sets = specs.count_sets(hand_one)
    assert excinfo.value.args[0] == "there are not exactly 12 cards"
    
    hand_two = ["0000", "1111", "2222", "0100", "1111", "2122", "0200", "1211", "2222", "2122", "2022", "2222"] #will raise a value error becuase not all cards are unique
    with pytest.raises(ValueError) as excinfo:
        num_of_sets = specs.count_sets(hand_two)
    assert excinfo.value.args[0] == "the cards are not all unique"

    hand_three = ["000", "0001", "0002", "0010", "0011", "0012", "0020", "0021", "0022", "0100", "0101", "0102"] #will raise a value error becuase not all cards are exactly 4 digits
    with pytest.raises(ValueError) as excinfo:
        num_of_sets = specs.count_sets(hand_three)
    assert excinfo.value.args[0] == "one or more cards does not have exactly 4 digits"

    hand_four = ["0000", "0001", "0002", "0010", "0011", "0012", "0030", "0021", "0022", "0100", "0101", "0102"] #will raise a value error becuase not all digits are 0,1,2
    with pytest.raises(ValueError) as excinfo:
        num_of_sets = specs.count_sets(hand_four)
    assert excinfo.value.args[0] == "one or more cards has a character other than 0, 1, or 2"
    
    hand_five = ["0000", "0001", "0002", "0010", "0011", "0012", "0020", "0021", "0022", "0100", "0101", "0102"] #4 matches
    assert specs.count_sets(hand_five) == 4
    hand_six = ["0000", "0001", "0002", "0010", "0011", "0012", "0020", "0021", "0022", "0100", "0101", "1102"] #3 matches
    assert specs.count_sets(hand_six) == 3
    

def test_is_set():
    assert specs.is_set("1022", "1122", "1020") == False #second digits are 0, 1, 0
    assert specs.is_set("0010", "0020", "1010") == False #3rd digits are 1, 2, 1
    assert specs.is_set("1022", "1122", "1222") == True #1st, 3rd, 4th digits are the same, 2nd is 0, 1, 2
    assert specs.is_set("0000", "1111", "2222") == True #all different
    assert specs.is_set("0120", "0210", "0000") == True #1st 2nd are all different, 1st and 4th are the same 
