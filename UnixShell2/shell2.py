# shell2.py
"""Volume 3: Unix Shell 2.
Daniel Perkins
MATH 321
11/2/23
"""

import os
from glob import glob
import subprocess
from queue import PriorityQueue

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
    the_files = set()           # Set to store tuples of file size and file path
    list_of_files = []
    all_files = glob("**/*", recursive=True)   # find all files in the directory
    main = os.getcwd()   # current directory
    
    for file in all_files:
        if os.path.isdir(file):   # If it is a directory, go into it
            os.chdir(file)
            file_info = subprocess.check_output(["ls", "-s"]).decode()   # Find  size of each file
            file_info = file_info.split("\n")[1:-1]  # Remove total part and empty string, and make it a list
            for info in file_info:
                info = info.strip()   # Strip whitespace
                info = info.split(" ")      # Separate file size from name of file 
                if not os.path.isdir(info[1]):    # If not directory
                    the_files.add((int(info[0]), file + "/" + info[1]))   # and file size and path to set of files
            os.chdir(main)   # Go back to original directory
    
    for _ in range(n):   # For the n largest files
        list_of_files.append(max(the_files)[1])   # Add largest file to list
        the_files.remove(max(the_files))    # Remove the largest file from the set

    smallest = min(the_files)[1]           # Find smallest file
    args = [f"cat {smallest} | wc -l > smallest.txt"]   # Writ enumber of lines ot smallest file
    subprocess.Popen(args, shell=True)

    return list_of_files                # Return list of n largest files

if __name__ == "__main__":
    # Prob 3
    # print(grep("format", "*.py"))

    # Prob 4
    print(largest_files(5))