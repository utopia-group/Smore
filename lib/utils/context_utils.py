from typing import Tuple, List


def context_tokenize_similarity_matching(s: str) -> List[Tuple[str, int, int]]:
    """
    find out all tokens in the context and their corresponding indices
    """

    context_tokenize = []
    curr_token_starting_span = -1
    char_i: str
    for i, char_i in enumerate(s):
        if char_i.isspace() or char_i == '/' or char_i == '_' or char_i == ',':
            if curr_token_starting_span == -1:
                continue
            else:
                curr_token_ending_span = i
                context_tokenize.append(
                    (s[curr_token_starting_span:curr_token_ending_span], curr_token_starting_span, curr_token_ending_span))
                curr_token_starting_span = -1
        else:
            if curr_token_starting_span == -1:
                curr_token_starting_span = i
            else:
                if i == len(s) - 1:
                    curr_token_ending_span = i + 1
                    context_tokenize.append(
                        (s[curr_token_starting_span:curr_token_ending_span], curr_token_starting_span, curr_token_ending_span))
            continue

    return context_tokenize
