# shell2.py
"""Volume 3: Unix Shell 2.
Daniel Perkins
MATH 321
11/2/23
"""

import os
from glob import glob
import subprocess

# Problem 3
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.

    Returns:
        matched_files (list): list of the filenames that matched the file
               pattern AND the target string.
    """
    file_pattern = "**/" + file_pattern   # To allow for matching more than one directory
    files = glob(file_pattern, recursive=True)   # Find all files that match the pattern
    matches = []
    for the_file in files:          # For each file that matched the pattern
        with open(the_file, "r") as file: # Open the file Checks to see if the file contains the target string
            data = file.read()
            if target_string in data:      # If target string is in the file
                matches.append(the_file)
    return matches


# Problem 4
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    raise NotImplementedError("Problem 4 Incomplete")

if __name__ == "__main__":
    # Prob 3
    # print(grep("format", "*.py"))

    # Prob 4
    print()