# regular_expressions.py
"""Volume 3: Regular Expressions.
Daniel Perkins
MATH 322
12/7/23
"""

import re

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile("python")  # Match python

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    # Put \ in front of every metacharacter
    return re.compile(r"\^\{@\}\(\?\)\[%\]\{\.\}\(\*\)\[_\]\{&\}\$")

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    # | acts as an or
    return re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$")

# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    # [a-zA-Z_]\w*\s*  matches a python identifier
    # \d*(\.)?\d* matches any number
    # \s*'[^']*' matches any string with single quotes
    pattern = r"^[a-zA-Z_]\w*\s*(=\s*\d*(\.)?\d*|=\s*'[^']*'|=\s*[a-zA-Z_]\w*\s*)?$"
    return re.compile(pattern)

# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """

    lines = code.split("\n")  # Split up into lines which are easier to work with
    new_lines = []   # For new lines that will be modified
    for line in lines:   # For each line
        pattern1 = r"\s*(if|elif|for|while|with|def|class)(.*)"   # Patterns with expressions at end
        pattern2 = r"\s*(else|try|except|finally)"      # Patterns with no expression
        block1 = re.compile(pattern1)   # Check if it is a block
        block2 = re.compile(pattern2)   # Check if it is a block
        if block1.search(line): 
            new_line = block1.sub(line + ":", line)  # If block, add colon
        if block2.search(line):
            new_line = block2.sub(line + ":", line)
        elif not block1.search(line):
            new_line = line     # Id not, don't change anything
        new_lines.append(new_line)
    return "\n".join(new_lines)   # Convert list to string

# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """

    with open(filename, "r") as file:
        lines = [line.rstrip() for line in file]  # Read lines and remove \n

    people = {}  # Dictionary of each person and their information

    # Set up regexp for each data information
    name_pattern = re.compile(r"([a-zA-z]+ )([A-Z]\. )?([a-zA-z]+)")
    date_pattern = re.compile(r"([01]?)([0-9])/([0-3]?)([0-9])/(\d\d)?(\d\d)")
    email_pattern = re.compile(r"\S*@\S*")
    phone_pattern = re.compile(r"\d*-?\(?(\d\d\d)\)?-?(\d\d\d-\d\d\d\d)")

    for line in lines:
        names = name_pattern.findall(line)[0]  # Returns tuple that matched name
        name = names[0] + names[2]   # Gets name in right format
        people[name] = {"birthday" : None, "email": None, "phone": None}

        dates = date_pattern.findall(line)
        if dates:  # if the date exists
            dates = list(dates[0])  # Make it mutable
            if dates[0] == "": dates[0] = '0'   # If 0 not included
            if dates[2] == "": dates[2] = '0'   # If 0 not included
            if dates[4] == "": dates[4] = '20'   # If 0 not included
            date = dates[0] + dates[1] + "/" + dates[2] + dates[3] + "/" + dates[4] + dates[5]
            people[name]["birthday"] = date

        emails = email_pattern.findall(line)
        if emails: # if the email exits
            people[name]["email"] = emails[0]

        phones = phone_pattern.findall(line)
        if phones:  # if the phone number exists
            phones = phones[0]
            phone = "(" + phones[0] + ")" + phones[1]
            people[name]["phone"] = phone
        
    return people


if __name__=="__main__":
    # Prob 1
    # print(bool(prob1().match("Ipython")))
    # print(bool(prob1().search("Ipython")))

    # Prob 2
    # print(bool(prob2().search("^{@}(?)[%]{.}(*)[_]{&}$")))

    # Prob 3
    # good_strings = ["Book store", "Book supplier",
    #                 "Mattress store", "Mattress supplier",
    #                 "Grocery store", "Grocery supplier"]
    # bad_strings = ["Booky store", "Book Book", "Book Grocery store",
    #                "This is a Book store", "Grocery stores rock",
    #                "Book storesupplier", " Book store", "Bookstore"]
    # for x in good_strings:
    #     print(bool(prob3().search(x)))
    # for x in bad_strings:
    #     print(bool(prob3().search(x)))

    # Prob 4
    # good_strings = ["Mouse", "_num =  2.3", "arg_ = 'hey'", "__x__", "var24"
    #                 "max=total", "string= ''", "num_guesses", "cheese = 2"]
    # bad_strings = ["3rats", "_num = 2.3.2", "arg_ = 'one'two", "sq(x)", " x", "max=2total", 
    #                "is_4(value==4)", "pattern = r'^one|two fish$'", "one =  'one', two = ' two  ' "]
    # for x in good_strings:
    #     print(bool(prob4().search(x)), x)
    # print()
    # for x in bad_strings:
    #     print(bool(prob4().search(x)), x)

    # prob 5
    # code = """k, i, p = 999, 1, 0
    # while k > i
    #     i *= 2
    #     p += 1
    #     if k != 999
    #         print("k should not have changed")
    #     else
    #         pass
    # print(p)"""
    # print(prob5(code))

    # prob 6
    # print(prob6())

    print()