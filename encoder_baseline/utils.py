"""
Module: data_structure_utils

This module provides utility functions for manipulating and analyzing data structures 
such as lists and dictionaries.

Functions:
    - tupling(a: list or tuple): Converts all nested lists or tuples within the input into tuples.
    - type_depth(thing): Recursively analyzes the structure of lists and dictionaries, returning a
      dictionary that represents the type structure of the input.

Example Usage:
    >>> from data_structure_utils import tupling, type_depth
    >>> result = tupling([1, [2, 3], (4, 5)])
    >>> print(result)
    (1, (2, 3), (4, 5))
    
    >>> structure = type_depth({"a": [1, 2, {"b": 3}], "c": 4})
    >>> print(structure)
    {'a': [<class 'int'>, <class 'int'>, {'b': <class 'int'>}], 'c': <class 'int'>}

Author:
    Baptiste Brunet

Date:
    2024-10-22  
"""

def tupling(a:list or tuple):
    """
    Converts all nested lists or tuples within the input into tuples.
    
    Args:
        a (list or tuple): The input nested list or tuple to be converted.
    
    Returns:
        tuple: The input structure with all nested lists and tuples converted to tuples.
    """
    if isinstance(a, (list, tuple)):
        return tuple(tupling(b) for b in a)
    return a

def type_depth(thing):
    """
    Recursively analyzes the structure of lists and dictionaries, returning a
    dictionary that represents the type structure of the input.
    
    Args:
        thing: The input object which can be a list, dictionary, or any other type.
    
    Returns:
        dict or type: A dictionary representing the structure of lists and dictionaries,
                      or the type of the object if it is not a list or dictionary.
    """
    if isinstance(thing, dict):
        res = {}
        for i, j in thing.items():
            temp = type_depth(j)
            if (i, temp) not in res.items():
                res[i] = temp
    elif isinstance(thing, list):
        res = []
        for i in thing:
            temp = type_depth(i)
            if not temp in res:
                res.append(temp)
    else:
        return type(thing)
    return res
