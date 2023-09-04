# python_intro.py
"""Python Essentials: Introduction to Python.
Daniel Perkins
MATH 321
09/04/23
"""

# Problem 1 (write code below)
if __name__ == "__main__":
    print("Hello, world!")

# Problem 2
def sphere_volume(r):
    """ Return the volume of the sphere of radius 'r'.
    Use 3.14159 for pi in your computation.
    """
    return (4/3) * 3.14159 * r**3


# Problem 3
def isolate(a, b, c, d, e):
    """ Print the arguments separated by spaces, but print 5 spaces on either
    side of b.
    """
    print(a, "   ", b, "   ", c, d, e)


# Problem 4
def first_half(my_string):
    """ Return the first half of the string 'my_string'. Exclude the
    middle character if there are an odd number of characters.

    Examples:
        >>> first_half("python")
        'pyt'
        >>> first_half("ipython")
        'ipy'
    """
    return my_string[:len(my_string) // 2]

def backward(my_string):
    """ Return the reverse of the string 'my_string'.

    Examples:
        >>> backward("python")
        'nohtyp'
        >>> backward("ipython")
        'nohtypi'
    """
    return my_string[::-1]


# Problem 5
def list_ops():
    """ Define a list with the entries "bear", "ant", "cat", and "dog".
    Perform the following operations on the list:
        - Append "eagle".
        - Replace the entry at index 2 with "fox".
        - Remove (or pop) the entry at index 1.
        - Sort the list in reverse alphabetical order.
        - Replace "eagle" with "hawk".
        - Add the string "hunter" to the last entry in the list.
    Return the resulting list.

    Examples:
        >>> list_ops()
        ['fox', 'hawk', 'dog', 'bearhunter']
    """
    my_list = ["bear", "ant", "cat", "dog"]
    my_list.append("eagle")
    my_list[2] = "fox"
    my_list.pop(1)
    my_list.sort(reverse=True)
    my_list[my_list.index("eagle")] = "hawk"
    my_list[len(my_list)-1] += "hunter"
    return my_list


# Problem 6
def pig_latin(word):
    """ Translate the string 'word' into Pig Latin, and return the new word.

    Examples:
        >>> pig_latin("apple")
        'applehay'
        >>> pig_latin("banana")
        'ananabay'
    """
    if word[0] in ['a', 'e', 'i', 'o', 'u']:
        word += "hay"
    else:
        first_letter = word[0]
        word = word[1:]
        word += first_letter + "ay"
    return word


# Problem 7
def palindrome():
    """ Find and retun the largest panindromic number made from the product
    of two 3-digit numbers.
    """
    max = 0
    for x in range(999,0, -1):  #go through all 3 digit numbers, in reverse
        for y in range(999, 0, -1):
            num = str(x * y)  
            reversed = num[::-1]   
            if num == reversed:   #checks if the numbe ris a palindrome
                value = x * y
                if value > max:    #replaces the max value if the palindrome is greater than all previous ones
                    max = value
                break
    return max

# Problem 8
def alt_harmonic(n):
    """ Return the partial sum of the first n terms of the alternating
    harmonic series, which approximates ln(2).
    """
    sum = 0
    for i in range(1, n + 1):  #sum from n=1 to n
       a_i = (-1)**(i+1) / i
       sum += a_i
    return sum