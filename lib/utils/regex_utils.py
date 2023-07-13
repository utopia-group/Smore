from typing import List, Iterable

from lib.interpreter.span import MatchSpan
import numpy as np


def repeat_matching_helper(string_size: int, arg_matrix: np.array, lower_limit: int, upper_limit: int) -> np.array:
    """
    Helper method for the repeat matcher. Given the spans of the desired element (given in matrix representation),
    find all spans such that the desired element repeats between lower_limit and upper_limit times.
    """
    lower_limit = int(max(0, lower_limit))
    upper_limit = int(min(string_size, upper_limit))

    if lower_limit > upper_limit:
        return np.zeros(shape=(string_size + 1, string_size + 1))

    if lower_limit == 0:
        res_matrix = np.identity(string_size + 1)
    else:
        res_matrix = np.zeros(shape=(string_size + 1, string_size + 1))

    curr_matrix = np.identity(string_size + 1)  # Current span matrix
    for i in range(1, upper_limit + 1):
        curr_matrix = np.clip(curr_matrix @ arg_matrix, 0, 1)
        if i >= lower_limit:
            res_matrix += curr_matrix
    return res_matrix


def repeat_token_mode_matching_helper(string_size: int, arg_matrix: np.array, space_matrix: np.array,
                                      lower_limit: int, upper_limit: int) -> np.array:
    """
    Helper method for the repeat matcher in token mode. Given the spans of the desired element (given in
    matrix representation) as well as the span of spaces, find all spans such that the desired element repeats
    between lower_limit and upper_limit times.
    """
    lower_limit = int(max(0, lower_limit))
    upper_limit = int(min(string_size, upper_limit))

    if lower_limit > upper_limit:
        return np.zeros(shape=(string_size + 1, string_size + 1))

    if lower_limit == 0:
        res_matrix = np.identity(string_size + 1)
    else:
        res_matrix = np.zeros(shape=(string_size + 1, string_size + 1))

    curr_matrix = arg_matrix  # Current span matrix, starts off at the state after 1 repetition
    # To go from i repetitions to i+1 repetitions, we first have the spaces, and then the repeated element
    multiplier_matrix = (space_matrix + np.identity(string_size + 1)) @ arg_matrix

    for i in range(1, upper_limit):
        if i >= lower_limit:
            res_matrix += curr_matrix
        curr_matrix = np.clip(curr_matrix @ multiplier_matrix, 0, 1)

    res_matrix += curr_matrix  # One more to count for upper_limit

    return res_matrix


def create_matrix_representation(string_size: int, spans: Iterable[MatchSpan]) -> np.array:
    """
    Creates a matrix representation of a list of spans
    """
    arr = np.zeros(shape=(string_size + 1, string_size + 1))
    for span in spans:
        arr[span.start][span.end] = 1
    return arr


def create_span_representation(string_size: int, arr: np.array) -> List[MatchSpan]:
    """
    Creates a list of spans from the matrix representation of spans
    """
    res = []
    for i in range(string_size + 1):
        for j in range(i, string_size + 1):
            if arr[i][j] > 0:
                res.append(MatchSpan(i, j))
    return res
