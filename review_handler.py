"""
This is a file that contains the method to help you to pre-process the string in the review data.
"""

#1. Import regular expression module
import re

def remove_unwanted_strings(review):
    for index, data in enumerate(review):
        # Anything within the <> will be removed 
        # ? to tell it dont be greedy so it won't capture everything from the 
        # first < to the last > in the document
        review[index] = re.sub('<.*?>', ' ', data) 
        review[index] = re.sub('[^a-zA-Z]',' ',data).lower().split()
    return review