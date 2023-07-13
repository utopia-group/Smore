import re
from typing import Tuple, List, Optional, Iterator

from lib.config import pd_print
from lib.utils.exceptions import NoPositiveMatchException


def locate_error(exception: Exception) -> Optional[Tuple[List[str], int, str, str, List[int]]]:
    """
    given an exception, find out which part of the regex is wrong
    """

    if isinstance(exception, NoPositiveMatchException):
        # we need to generate a new over-approximation (with no lazy matching)
        s_ctx = exception.context
        s = s_ctx.s
        over_approx = exception.sketch_object.get_overapprox(
            dict([(ty, [s_ctx.get_substr_regex(span) for span in str_spans]) for ty, str_spans in s_ctx.type_to_str_spans.items()]), lazy=False)
        # we first need to split the regex into components
        regex_components = decompose_regex(over_approx)
        pd_print("regex_components: {}".format(regex_components))

        # these are all information we need to track with optional mode
        optional_mode = False
        optional_rc_i = []
        indexes_without_optional = []

        # these should not be affected by optional mode
        possible_error_rc_i = []
        remaining_str = []

        indexes = [0]
        last_rc = 0

        i = 0
        while i < len(regex_components):
            rc = regex_components[i]

            # check if the current rc is a .*, if it is, we need to run .* and what follows together
            if ('.*' in rc or '(.)*' in rc) and i + 1 < len(regex_components):
                # we need to run the .* and what follows together
                rc = rc + regex_components[i + 1]
                i += 1

            # print('1:', i, rc)
            # print('2:', indexes)

            if len(indexes) == 0:
                break

            indexes2 = []
            indexes2_with_optional = []

            for idx in indexes:
                # print('idx:', idx)
                s1 = s[idx:]
                # print('s1:', s1)
                idx2 = consume_matches(rc, s1, 'forward', exception.sketch_object.no_type)
                # print('idx2:', idx2)

                if len(idx2) == 0:
                    if not optional_mode:
                        possible_error_rc_i.append(i)
                        remaining_str.append(s1)

                    # let's assume that the regex component is optional
                    indexes2_with_optional.append(idx)
                else:
                    indexes2.extend([(i2 + idx) for i2 in idx2])

            last_rc = i
            if len(indexes2) == 0:
                if len(indexes2_with_optional) == len(indexes):
                    # this means we *have* to consume the optional regex component (not matter what match we do)
                    optional_rc_i.append(i)

                    # this criterion overfits benchmark #4
                    if len(optional_rc_i) > 1:
                        break
                    else:
                        indexes_without_optional = indexes
                        indexes = indexes2_with_optional
                        optional_mode = True
                else:
                    break

            else:
                indexes = indexes2
            i += 1

        pd_print('indexes: {}'.format(indexes))
        pd_print('optional_rc_i: {}'.format(optional_rc_i))
        pd_print('indexes_without_optional: {}'.format(indexes_without_optional))

        # if the string is fully consumed and the optional_rc is not empty, then we conclude the right way to fix it is by adding optional
        if 0 < len(optional_rc_i) <= 1 and any(i == len(s) for i in indexes):
            return regex_components, -1, '', 'forward', optional_rc_i
        elif len(optional_rc_i) == 2 and any(i == len(s) for i in indexes) and regex_components[optional_rc_i[1]] == '[ ]':
            return regex_components, -1, '', 'forward', optional_rc_i
        else:
            if len(optional_rc_i) > 0:
                indexes = indexes_without_optional

        # check if the string is fully consumed
        if last_rc == len(regex_components) - 1 and not any(i == len(s) for i in indexes):
            # the string is not fully consumed
            possible_error_rc_i.append(len(regex_components))
            remaining_str.append(s[max(indexes):])

        pd_print('possible_error_rc_i: {}'.format(possible_error_rc_i))
        pd_print('remaining_str: {}'.format(remaining_str))

        if len(possible_error_rc_i) == 0:
            # should not reach here
            raise RuntimeError('locate_error: should not reach here for exception input {} {}'.format(exception.regex,
                                                                                                      exception.context))
        else:
            # figure out the one with the largest index
            error_i = max(possible_error_rc_i)
            # htype, hid = get_most_recent_hole(regex_components, error_i)
            if error_i != 0:
                return regex_components, error_i, remaining_str[possible_error_rc_i.index(error_i)], 'forward', []

        # we do backward mode (because the forward mode failed)

        # these are all information we need to track with optional mode
        optional_mode = False
        optional_rc_i = []
        indexes_without_optional = []

        indexes = [len(s)]
        possible_error_rc_i = []
        remaining_str = []

        last_rc = 0
        i = len(regex_components) - 1

        while i >= 0:

            rc = regex_components[i]
            # check if the current rc is a .*, if it is, we need to run what before and .* together
            if ('.*' in rc or '(.)*' in rc) and i - 1 >= 0:
                # we need to run the .* and what follows together
                rc = rc + regex_components[i - 1]
                i -= 1

            # print('1:', i, rc)
            # print('2:', indexes)

            if len(indexes) == 0:
                break

            indexes2 = []
            indexes2_with_optional = []

            for idx in indexes:
                # print('idx:', idx)
                s1 = s[:idx]
                # print('s1:', s1)
                idx2 = consume_matches(rc, s1, 'backward', exception.sketch_object.no_type)
                # print('idx2:', idx2)

                if len(idx2) == 0:
                    if not optional_mode:
                        possible_error_rc_i.append(i)
                        remaining_str.append(s1)

                    # let's assume that the regex component is optional
                    indexes2_with_optional.append(idx)
                else:
                    indexes2.extend(idx2)

            last_rc = i
            if len(indexes2) == 0:
                if len(indexes2_with_optional) == len(indexes):
                    # this means we *have* to consume the optional regex component (not matter what match we do)
                    optional_rc_i.append(i)

                    # this criterion overfits benchmark #4
                    if len(optional_rc_i) > 1:
                        break
                    else:
                        indexes_without_optional = indexes
                        indexes = indexes2_with_optional
                        optional_mode = True
                else:
                    break

            else:
                indexes = indexes2
            i -= 1

        pd_print('indexes: {}'.format(indexes))

        # if the string is fully consumed and the optional_rc is not empty, then we conclude the right way to fix it is by adding optional
        if 0 < len(optional_rc_i) <= 1 and any(i == 0 for i in indexes):
            return regex_components, -1, '', 'backward', optional_rc_i
        elif len(optional_rc_i) == 2 and any(i == 0 for i in indexes) and regex_components[optional_rc_i[0]] == '[ ]':
            return regex_components, -1, '', 'backward', optional_rc_i
        else:
            if len(optional_rc_i) > 0:
                indexes = indexes_without_optional

        # check if the string is fully consumed
        if last_rc == 0 and not any(i == 0 for i in indexes):
            # the string is not fully consumed
            possible_error_rc_i.append(0)
            remaining_str.append(s[min(indexes):])

        pd_print('possible_error_rc_i: {}'.format(possible_error_rc_i))
        pd_print('remaining_str: {}'.format(remaining_str))
        pd_print('optional_rc_i: {}'.format(optional_rc_i))

        if len(possible_error_rc_i) == 0:
            # should not reach here
            raise RuntimeError('locate_error: should not reach here for exception input {} {}'.format(exception.regex,
                                                                                                      exception.context))
        else:
            # figure out the one with the largest index
            error_i = min(possible_error_rc_i)
            if error_i != (len(regex_components) - 1):
                return regex_components, error_i, remaining_str[possible_error_rc_i.index(error_i)], 'backward', []

    else:
        raise NotImplementedError

    return None


def decompose_regex(regex: str) -> List[str]:
    """
    given a regex, decompose it to a list of sub-regexes that can be concatenated
    """
    r = []
    paren_count = 0  # count parenthesis
    buffer = ''
    in_cc = False

    idx = 0
    while idx < len(regex):
        if regex[idx] == '\\' and idx + 1 < len(regex):
            # In this branch, we've encountered a backslash
            # We have to add this character and the following char
            # to the buffer. The reason why we process both chars
            # at once instead of just processing the next character in
            # the next loop is because the following character could be
            # a special character, like a paren, which could mess things up
            buffer += regex[idx:idx + 2]
            idx += 1
        elif regex[idx] == '(' and not in_cc:
            if paren_count == 0:
                if buffer != '':
                    r.append(buffer)
                    buffer = ''
            buffer += regex[idx]
            paren_count += 1
        elif regex[idx] == ')' and not in_cc:
            paren_count -= 1
            buffer += regex[idx]
            if paren_count == 0:
                if buffer != '':
                    r.append(buffer)
                    buffer = ''
        elif regex[idx] == '[':
            in_cc = True
            buffer += regex[idx]
        elif regex[idx] == ']':
            in_cc = False
            buffer += regex[idx]
        else:
            buffer += regex[idx]
        idx += 1

    if buffer != '':
        r.append(buffer)

    return r


def decompose_sketch(sketch: str) -> List[str]:
    """
    given a sketch, decompose it to a list of sub-sketches that can be concatenated
    """
    r = []
    paren_count = 0  # count parenthesis
    buffer = ''
    in_hole = False
    in_cc = False

    for idx in range(0, len(sketch)):
        if sketch[idx] == '(' and not in_cc:
            if paren_count == 0:
                if buffer != '':
                    r.append(buffer)
                    buffer = ''
            buffer += sketch[idx]
            paren_count += 1
        elif sketch[idx] == ')' and not in_cc:
            paren_count -= 1
            buffer += sketch[idx]
            if paren_count == 0:
                if buffer != '':
                    r.append(buffer)
                    buffer = ''
        elif sketch[idx] == '{':
            if paren_count == 0 and sketch[idx + 1] == '?':
                if buffer != '':
                    r.append(buffer)
                    buffer = ''
                in_hole = True
                buffer += sketch[idx]
            elif paren_count > 0 and sketch[idx + 1] == '?':
                in_hole = True
                buffer += sketch[idx]
        elif sketch[idx] == '}':
            buffer += sketch[idx]
            if in_hole and paren_count == 0:
                r.append(buffer)
                buffer = ''
            in_hole = False
        elif sketch[idx] == '[':
            in_cc = True
            buffer += sketch[idx]
        elif sketch[idx] == ']':
            in_cc = False
            buffer += sketch[idx]
        else:
            buffer += sketch[idx]

    if buffer != '':
        r.append(buffer)

    return r


def get_error_component_type(regexes: List[str], idx: int) -> Tuple[str, int] | str:
    if idx == -1:
        return 'optional'
    elif idx == len(regexes):
        return 'append'
    else:
        curr_regex = regexes[idx]
        # print(curr_regex)
        if curr_regex.startswith('(?P'):
            s_idx = curr_regex.index('<')
            e_idx = curr_regex.index('>')
            hole_content = curr_regex[(s_idx + 1):e_idx]
            # print(hole_content)
            # split into hole name and hole id
            match = re.match(r"([A-Za-z]+)([0-9]+)", hole_content, re.I)
            if match:
                items = match.groups()
                return items[0], int(items[1])
        else:
            return 'concrete'


def check_date_repair(hole_type: str):
    return hole_type.lower() in ['year', 'month', 'day']


def check_time_repair(hole_type: str):
    return hole_type.lower() in ['hour', 'minute', 'second']


def check_float_repair(hole_type: str):
    raise NotImplementedError


def check_general_semantic_repair(hole_type: str):
    """
    These criteria are incomplete at this point
    """
    return hole_type.lower() not in ['string', 'integer', 'float', 'number', 'year', 'month', 'day', 'hour', 'minute',
                                     'second', 'date', 'time']


def _split_at_bar(s: str) -> List[str]:
    """
    Splits the string s at the "|" character.
    If the "|" character is preceded by a literal backslash ("\\"), no splitting occurs
    """
    # This uses lookbehind.
    return re.split(r'(?<!\\)\|', s)


def consume_matches(regex: str, s: str, mode: str, no_type=False) -> List[int]:
    """
    in this function so far we can only do non-greedy match for the or operator
    """

    if regex.endswith('>.*)') and not no_type:
        return [0]

    rx = re.compile(regex)

    if mode == 'forward':
        matches = [i for i in range(1, len(s) + 1) if re.fullmatch(rx, s[:i]) is not None]
        if len(matches) == 0 and regex.endswith('?)'):
            return [0]
        return matches

    assert mode == 'backward'
    matches = [i for i in range(len(s)) if re.fullmatch(rx, s[i:]) is not None]
    if len(matches) == 0 and regex.startswith('?)'):
        return [0]
    return matches


def generalize_error_regex(decomposed_regex: List[str], error_component_idx: int, mode: str) -> Tuple[str, str]:
    group_name = 'repair{}'.format(error_component_idx)
    if mode == 'forward':
        new_regex = decomposed_regex[:error_component_idx]
        new_regex.append('(?P<{}>(.*))'.format(group_name))
    else:
        assert mode == 'backward'
        new_regex = decomposed_regex[(error_component_idx + 1):]
        new_regex.insert(0, '(?P<{}>(.*))'.format(group_name))

    return ''.join(new_regex), group_name
