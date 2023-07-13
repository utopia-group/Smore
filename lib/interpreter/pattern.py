"""
Pattern Functions: This is basically the program syntax
"""
import re
from math import inf
from typing import List, Callable, Set, Optional, Tuple

from lib.interpreter.context import StrContext
from lib.interpreter.span import MatchSpan
from lib.interpreter.str_function import StrFunc
from lib.lang.constants import CaseTag, CC_REGEX_MAP, CONST_REGEX_MAP, DECOMPOSE_SPECIAL_TAGS
from lib.nlp.nlp import date_template_match3
from lib.utils.matcher_utils import find_date, check_regions, check_countries, num_text_to_num
from lib.utils.regex_utils import repeat_matching_helper, repeat_token_mode_matching_helper, \
    create_matrix_representation, create_span_representation

import numpy as np

"""
First we have all the abstract classes
"""


class Pattern:
    def __init__(self, token_mode: bool = False, pure_regex: bool = False):
        self.token_mode = token_mode
        self.pure_regex: bool = pure_regex

        # for Pattern  <-> Program conversion
        self.f_name: str = ""
        self.args = []

    def duplicate(self) -> 'Pattern':
        raise NotImplementedError

    def __repr__(self):
        pass

    def to_py_regex(self, *args):
        pass

    def update_token_case_mode(self, token_mode: bool, case_tag: str):
        return self


class SemPattern(Pattern):
    def __init__(self):
        super().__init__()

        self.args: List[Tag | PredicateFormula | StrFunc | float | str] = []

    def duplicate(self) -> 'SemPattern':
        raise NotImplementedError

    def __repr__(self):
        pass


class RegexPattern(Pattern):
    def __init__(self, token_mode: bool, case_tag: CaseTag, pure_regex: bool):
        super().__init__(token_mode, pure_regex)
        self.case_tag: CaseTag = case_tag

        self.args: List[RegexPattern | SemPattern | int] = []

    def duplicate(self) -> 'RegexPattern':
        raise NotImplementedError

    def update_token_case_mode(self, token_mode: bool, case_tag: CaseTag):

        if isinstance(self, CC) or isinstance(self, Const):
            # we do not update token mode for all base case (not necessary)
            self.token_mode = False
        else:
            self.token_mode = token_mode
        self.case_tag = case_tag

        return self

    def update_pure_regex_mode(self, args: List[Pattern]):
        self.pure_regex = all([arg.pure_regex for arg in args])

    def exec(self, *args) -> List[MatchSpan]:
        raise NotImplementedError

    def to_py_regex(self, *args) -> str:
        raise NotImplementedError


class Predicate:
    def __init__(self):
        self.f_name: str = ''

    def exec(self, s: StrContext) -> bool:
        raise NotImplementedError


class PredicateAtom(Predicate):
    def __init__(self, op: str, args):
        super().__init__()
        self.f_name: str = op
        self.args = args

    def exec(self, s: StrContext) -> bool:
        raise NotImplementedError

    def __repr__(self):
        if len(self.args) == 0:
            return self.f_name
        else:
            args = [str(e) for e in self.args]
            return "{}({})".format(self.f_name, ','.join(args))


class BooleanAtom(PredicateAtom):
    def __init__(self, str_val: str):
        super().__init__(op='BooleanAtom', args=[str_val])
        self.val = str_val == 'TRUE'

    def exec(self, s: StrContext) -> bool:
        return self.val

    def __repr__(self):
        return str(self.val)


class PredicateFormula(Predicate):
    """
    A **flatten-out** list of patterns, with connections and negations
    """

    def __init__(self, components: List[Predicate], op: str):
        super().__init__()
        assert op == 'and' or op == 'or' or op == 'not'
        assert len(components) == 1 if op == 'not' else len(components) == 2
        self.components: List = components
        self.op: str = op

    def __repr__(self):
        if self.op == 'not':
            return '~' + repr(self.components[0])
        return '(' + repr(self.components[0]) + ' ' + self.op + ' ' + repr(self.components[1]) + ')'

    def exec(self, s: StrContext) -> bool:
        if self.op == 'or':
            return self.components[0].exec(s) or self.components[1].exec(s)
        elif self.op == 'and':
            return self.components[0].exec(s) and self.components[1].exec(s)
        return not self.components[0].exec(s)  # not operation


class Tag:
    def __init__(self, tag: str):
        self.tag: str = tag
        self.f_name: str = ''

    def __repr__(self):
        return self.tag


class EntTag(Tag):
    """
    Entity Tags that are not parsable
    """

    def __init__(self, tag: str):
        super().__init__(tag)
        self.tag: str = tag
        self.f_name: str = 'EntTag'


class OntTag(Tag):
    """
    Entity Tags that are not parsable
    """

    def __init__(self, tag: str):
        super().__init__(tag)
        self.tag: str = tag
        self.f_name: str = 'OntTag'


class TypeTag(Tag):
    """
    Parsable Tags
    """

    def __init__(self, tag: str):
        super().__init__(tag)
        self.f_name: str = 'TypeTag'


class QueryStr(Tag):
    """
    String for the query, make this a Tag just for consistency
    """

    def __init__(self, tag: str):
        super().__init__(tag)
        self.f_name: str = 'QueryStr'


class ConstStr(Tag):
    """
    String for the query, make this a Tag just for consistency
    """

    def __init__(self, tag: str):
        super().__init__(tag)
        self.f_name: str = 'ConstStr'


class Similar(SemPattern):
    def __init__(self, keyword: ConstStr, mode: str, threshold: Optional[float] = None):
        super().__init__()
        self.keyword: ConstStr = keyword
        self.mode: str = mode
        self.threshold = threshold

        self.f_name = 'Similar'
        if threshold is None:
            self.args = [keyword, mode]
        else:
            self.args = [keyword, threshold]

    def __repr__(self):
        if self.threshold is not None:
            return 'Similar({}, {})'.format(self.keyword, self.threshold)
        else:
            return 'Similar({}, {})'.format(self.keyword, self.mode)


class MatchQuery(SemPattern):
    """
    Query GPT3
    """

    def __init__(self, query: QueryStr, func: StrFunc, in_context: Optional[Tuple[List[str], List[str]]] = None):
        super().__init__()
        self.query: QueryStr = query
        self.func: StrFunc = func

        # need to store some in-context examples
        self.in_context_examples: Tuple[List[str], List[str]] = in_context if in_context is not None else ([], [])

        self.f_name = 'MatchQuery'
        self.args = [self.query, self.func, self.in_context_examples]

    def duplicate(self) -> 'MatchQuery':
        new_query = MatchQuery(self.query, self.func, self.in_context_examples)
        return new_query

    def __repr__(self):
        # TODO: the sketch parser not able to handle this yet because of the in—context-examples
        return 'MatchQuery({},{},{})'.format(repr(self.query), repr(self.func), repr(self.in_context_examples))


class MatchEntity(SemPattern):
    """
    match entity
    """

    def __init__(self, ent: EntTag | OntTag):
        super().__init__()
        self.ent: EntTag = ent

        self.f_name = 'MatchEntity'
        self.args = [self.ent]

    def duplicate(self) -> 'MatchEntity':
        new_ent = MatchEntity(self.ent)
        return new_ent

    def __repr__(self):
        return '{{{}}}'.format(repr(self.ent))


class MatchType(SemPattern):
    """
    match parsable object
    """

    def __init__(self, tag: TypeTag, formula: Optional[Predicate] = None):
        super().__init__()
        self.tag: TypeTag = tag
        self.formula: Optional[Predicate] = formula

        self.f_name = 'MatchType'
        self.args = [self.tag, self.formula]

    def duplicate(self) -> 'MatchType':
        new_type = MatchType(self.tag, self.formula)
        return new_type

    def __repr__(self):
        return '{{{} -> {}}}'.format(repr(self.tag), repr(self.formula))


class NumPredicate(PredicateAtom):
    """
    num predicate pattern of the form n1 <= s <= n2, where s is the var
    op can be 'ge', 'geq', 'eq', 'le', 'leq'
    """

    def __init__(self, low, low_op: str, high, high_op: str):
        super().__init__(op='NumMatch', args=[low, low_op, high, high_op])
        self.low = low
        self.low_op = low_op
        self.high = high
        self.high_op = high_op

    def to_func(self) -> Callable:
        """
        output a function that we can evaluate directly
        """
        func_str = "lambda a: (lambda s: s {} {})(a) and (lambda s: s {} {})(a)".format(self.low_op, self.low, self.high_op, self.high)
        return eval(func_str)

    def exec(self, s: StrContext) -> bool:
        # print("executing NumMatch")
        # print("s is ", s.s)
        # print("num_text_to_num(s.s) is ", num_text_to_num(s.s))
        # print(self.low_op, self.low, self.high_op, self.high)
        if num_text_to_num(s.s) is None:
            return False
        return self.to_func()(num_text_to_num(s.s.strip()))

    def __repr__(self):
        return "NumMatch(x, {}, {}, {}, {})".format(self.low_op, self.low, self.high_op, self.high)


class DatePredicate(PredicateAtom):
    def __init__(self, op: str, *args):
        if len(args) == 0:
            super().__init__(op, [])
        else:
            super().__init__(op, args[0])

    def exec(self, s: StrContext):
        # parse date

        s = s.s

        # we need a separate function to parse date in some circumstances
        if re.search(date_template_match3, s) is not None:
            year = s[0:2]
            month = s[2:4]
            day = s[4:6]
            s = year + '-' + month + '-' + day

        match self.f_name:
            case "isYear":
                try:
                    s_dt_list = find_date(s, {'REQUIRE_PARTS': ['year']})
                except Exception:
                    print("Something wrong when parse dates!")
                    return False

                for s1, dt in s_dt_list:
                    if s1 == s:
                        year = dt.year
                        return self.args[0] <= year <= self.args[1]
            case "isMonth":
                try:
                    s_dt_list = find_date(s, {'REQUIRE_PARTS': ['month']})
                except Exception:
                    print("Something wrong when parse dates!")
                    return False

                for s1, dt in s_dt_list:
                    if s1 == s:
                        month = dt.month
                        return self.args[0] <= month <= self.args[1]
            case "isDate":
                try:
                    s_dt_list = find_date(s, {'REQUIRE_PARTS': ['day']})
                except Exception:
                    print("Something wrong when parse dates!")
                    return False

                for s1, dt in s_dt_list:
                    if s1 == s:
                        date = dt.day
                        return self.args[0] <= date <= self.args[1]
            case default:
                raise ValueError('op {} not in date predicates'.format(self.f_name))

        return False


class TimePredicate(PredicateAtom):
    def __init__(self, op: str, *args):
        if len(args) == 0:
            super().__init__(op, [])
        else:
            super().__init__(op, args[0])

    def exec(self, s: StrContext):
        # parse date

        s = s.s
        try:
            s_parsed = find_date(s)

        except Exception:
            print("Something wrong when parse dates!")
            return False

        s_dt = None
        for sp in s_parsed:
            if sp[0] == s:
                s_dt = sp[1]
        if s_dt is None:
            return False

        if s_dt.hour == 0 and s_dt.minute == 0:
            return False

        match self.f_name:
            case "btwHour":
                hour = s_dt.hour
                return self.args[0] <= hour <= self.args[1]
            case "btwMin":
                minute = s_dt.minute
                return self.args[0] <= minute <= self.args[1]
            case "btwSec":
                second = s_dt.second
                return self.args[0] <= second <= self.args[1]
            case "isMorning":
                hour = s_dt.hour
                return 6 <= hour <= 12
            case "isAfternoon":
                hour = s_dt.hour
                return 12 <= hour <= 18
            case "isEvening":
                hour = s_dt.hour
                return 18 <= hour <= 24 or 0 <= hour <= 6
            case default:
                raise ValueError('op {} not in time predicates'.format(self.f_name))


class PlacePredicate(PredicateAtom):
    def __init__(self, op: str, *args):
        if isinstance(args[0], str):
            super().__init__(op, [args[0]])
        else:
            super().__init__(op, args[0])

    def exec(self, s: StrContext) -> bool:
        # print(s.str_span_to_predicate_context)

        match self.f_name:
            case "inRegion":
                key = 'region'
            case "inCountry":
                key = 'country'
            case "inState":
                key = 'state'
            case _:
                raise ValueError('op {} not in place predicate'.format(self.f_name))

        # print("s.str_span_to_predicate_context:", s.str_span_to_predicate_context)
        # print(self.args)
        # print(key)

        for span_ctx in s.str_span_to_predicate_context.values():
            if key in span_ctx:
                # print('span_ctx:', span_ctx)
                # print('span_ctx[key]:', span_ctx[key])
                if any(s.lower() in [s.lower() for s in self.args] for s in span_ctx[key]):
                    return True
                else:
                    continue
        return False


class OptionalR(RegexPattern):
    def __init__(self, arg1: RegexPattern | SemPattern, token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE, pure_regex: bool = False):
        super().__init__(token_mode, case_tag, pure_regex)
        self.arg1: RegexPattern | SemPattern = arg1.update_token_case_mode(token_mode, case_tag)
        # self.update_pure_regex_mode([self.arg1])

        self.f_name = 'OptionalR'
        self.args = [self.arg1]

    def duplicate(self) -> 'OptionalR':
        return OptionalR(self.arg1.duplicate(), self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        # Optional is just simply add [0,0], [1,1], ...,

        assert len(args) == 2

        res_span = []
        arg1_spans: Set[MatchSpan] = args[0]
        empty_spans: Set[MatchSpan] = args[1]

        res_span.extend(arg1_spans)
        res_span.extend(empty_spans)

        return res_span

    def to_py_regex(self, *args) -> str:
        arg1_regex = self.arg1.to_py_regex(*args)
        if arg1_regex == '(.)*' or arg1_regex == '.*':
            return '({}?)'.format(arg1_regex)
        elif arg1_regex == '((.)*)':
            return '((.)*?)'
        else:
            return '(({})?)'.format(self.arg1.to_py_regex(*args))

    def __repr__(self):
        return '(({})?)'.format(repr(self.arg1))


class StarR(RegexPattern):
    def __init__(self, arg1: RegexPattern | SemPattern, token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE, pure_regex: bool = False):
        super().__init__(token_mode, case_tag, pure_regex)
        self.arg1: RegexPattern | SemPattern = arg1.update_token_case_mode(token_mode, case_tag)
        # self.update_pure_regex_mode([self.arg1])

        self.f_name = 'StarR'
        self.args = [self.arg1]

    def duplicate(self) -> 'StarR':
        return StarR(self.arg1.duplicate(), self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        assert len(args) == 3

        string_size: int = args[0]
        arg_matrix: np.array = create_matrix_representation(string_size, args[1])
        if self.token_mode:
            space_matrix: np.array = create_matrix_representation(string_size, args[2])
            return create_span_representation(string_size,
                                              repeat_token_mode_matching_helper(string_size, arg_matrix, space_matrix,
                                                                                0, string_size))
        return create_span_representation(string_size,
                                          repeat_matching_helper(string_size, arg_matrix, 0, string_size))

    def to_py_regex(self, *args) -> str:
        return '(({})*)'.format(self.arg1.to_py_regex(*args))

    def __repr__(self):
        return '(({})*)'.format(repr(self.arg1))


class Plus(RegexPattern):
    def __init__(self, arg: RegexPattern | SemPattern, token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE, pure_regex: bool = False):
        super().__init__(token_mode, case_tag, pure_regex)
        self.arg: RegexPattern | SemPattern = arg.update_token_case_mode(token_mode, case_tag)
        # self.update_pure_regex_mode([self.arg])

        self.f_name = 'Plus'
        self.args = [self.arg]

    def duplicate(self) -> 'Plus':
        return Plus(self.arg.duplicate(), self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        assert len(args) == 3

        string_size: int = args[0]
        arg_matrix: np.array = create_matrix_representation(string_size, args[1])
        if self.token_mode:
            space_matrix: np.array = create_matrix_representation(string_size, args[2])
            return create_span_representation(string_size,
                                              repeat_token_mode_matching_helper(string_size, arg_matrix, space_matrix,
                                                                                1, string_size))
        return create_span_representation(string_size,
                                          repeat_matching_helper(string_size, arg_matrix, 1, string_size))

    def to_py_regex(self, *args) -> str:
        return '(({})+)'.format(self.arg.to_py_regex(*args))

    def __repr__(self):
        return '(({})+)'.format(repr(self.arg))


class NotR(RegexPattern):
    def __init__(self, arg: RegexPattern | SemPattern, token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE, pure_regex: bool = False):
        super().__init__(token_mode, case_tag, pure_regex)
        self.arg: RegexPattern | SemPattern = arg.update_token_case_mode(token_mode, case_tag)

        self.f_name = 'NotR'
        self.args = [self.arg]

    def duplicate(self) -> 'NotR':
        return NotR(self.arg.duplicate(), self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        arg_spans: Set[MatchSpan] = args[0]
        universe_spans: Set[MatchSpan] = args[1]
        complement_set = universe_spans.difference(arg_spans)
        return list(complement_set)

    def to_py_regex(self, *args) -> str:
        # this operation not support in python regex
        raise NotImplementedError

    def __repr__(self):
        return '(~({}))'.format(repr(self.arg))


class Contain(RegexPattern):
    def __init__(self, arg: RegexPattern | SemPattern, token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE, pure_regex: bool = False):
        super().__init__(token_mode, case_tag, pure_regex)
        self.arg: RegexPattern | SemPattern = arg.update_token_case_mode(token_mode, case_tag)
        # self.update_pure_regex_mode([self.arg])

        self.f_name = 'Contain'
        self.args = [self.arg]

    def duplicate(self) -> 'Contain':
        return Contain(self.arg.duplicate(), self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        """
        Given a str s, given a match span [n1,n2], generate [i, j] where 0 <= i <= n1, n2 <= j <= len(s)
        """
        assert len(args) == 2

        string_size: int = args[0]
        arg_matrix: np.array = create_matrix_representation(string_size, args[1])
        for i in range(string_size - 1, -1, -1):
            arg_matrix[i] += arg_matrix[i + 1]
        arg_matrix = np.transpose(arg_matrix)
        for i in range(1, string_size + 1):
            arg_matrix[i] += arg_matrix[i - 1]
        arg_matrix = np.transpose(arg_matrix)
        return create_span_representation(string_size, arg_matrix)

    def to_py_regex(self, *args) -> str:
        return '.*({}).*'.format(self.arg.to_py_regex(*args))

    def __repr__(self):
        return 'Contain({})'.format(repr(self.arg))


class Startwith(RegexPattern):
    def __init__(self, arg: RegexPattern | SemPattern, token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE, pure_regex: bool = False):
        super().__init__(token_mode, case_tag, pure_regex)
        self.arg: RegexPattern | SemPattern = arg.update_token_case_mode(token_mode, case_tag)
        # self.update_pure_regex_mode([self.arg])

        self.f_name = 'Startwith'
        self.args = [self.arg]

    def duplicate(self) -> 'Startwith':
        return Startwith(self.arg.duplicate(), self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        """
        Given a str s, given a match span [n1,n2], generate [n1, j] where n2 <= j <= len(s)
        """
        assert len(args) == 2

        string_size: int = args[0]
        arg_matrix: np.array = create_matrix_representation(string_size, args[1])
        arg_matrix = np.transpose(arg_matrix)
        for i in range(1, string_size + 1):
            arg_matrix[i] += arg_matrix[i - 1]
        arg_matrix = np.transpose(arg_matrix)
        return create_span_representation(string_size, arg_matrix)

    def to_py_regex(self, *args) -> str:
        return '({}).*'.format(self.arg.to_py_regex(*args))

    def __repr__(self):
        return 'SW({})'.format(repr(self.arg))


class Endwith(RegexPattern):
    def __init__(self, arg: RegexPattern | SemPattern, token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE, pure_regex: bool = False):
        super().__init__(token_mode, case_tag, pure_regex)
        self.arg: RegexPattern | SemPattern = arg.update_token_case_mode(token_mode, case_tag)
        # self.update_pure_regex_mode([self.arg])

        self.f_name = 'Endwith'
        self.args = [self.arg]

    def duplicate(self) -> 'Endwith':
        return Endwith(self.arg.duplicate(), self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        """
        Given a str s, given a match span [n1,n2], generate [i, n2] where 0 <= i <= n1
        """
        assert len(args) == 2

        string_size: int = args[0]
        arg_matrix: np.array = create_matrix_representation(string_size, args[1])
        for i in range(string_size - 1, -1, -1):
            arg_matrix[i] += arg_matrix[i + 1]
        return create_span_representation(string_size, arg_matrix)

    def to_py_regex(self, *args) -> str:
        return '.*({})'.format(self.arg.to_py_regex(*args))

    def __repr__(self):
        return 'EW({})'.format(repr(self.arg))


class Concat(RegexPattern):
    def __init__(self, arg1: RegexPattern | SemPattern, arg2: RegexPattern | SemPattern, token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE,
                 pure_regex: bool = False):
        super().__init__(token_mode, case_tag, pure_regex)
        self.arg1: RegexPattern | SemPattern = arg1.update_token_case_mode(token_mode, case_tag)
        self.arg2: RegexPattern | SemPattern = arg2.update_token_case_mode(token_mode, case_tag)
        # self.update_pure_regex_mode([self.arg1, self.arg2])

        # for pattern <-> program communication
        self.args = [self.arg1, self.arg2]
        self.f_name = 'Concat'

    def duplicate(self) -> 'Concat':
        return Concat(self.arg1.duplicate(), self.arg2.duplicate(), self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        assert len(args) == 4

        string_size: int = args[0]
        arg1_matrix: np.array = create_matrix_representation(string_size, args[1])
        arg2_matrix: np.array = create_matrix_representation(string_size, args[2])
        res_matrix: np.array = arg1_matrix @ arg2_matrix  # Matrix multiplication is equivalent to concatenation

        if self.token_mode:
            space_matrix: np.array = create_matrix_representation(string_size, args[3])
            res_matrix += arg1_matrix @ space_matrix @ arg2_matrix

        return create_span_representation(string_size, res_matrix)

    def to_py_regex(self, *args):

        return '{}{}'.format(self.arg1.to_py_regex(*args), self.arg2.to_py_regex(*args))

    def __repr__(self):
        return '{}{}'.format(repr(self.arg1), repr(self.arg2))


class Repeat(RegexPattern):
    def __init__(self, arg1: RegexPattern | SemPattern, arg2: int, token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE,
                 pure_regex: bool = False):
        super().__init__(token_mode, case_tag, pure_regex)
        self.arg1: RegexPattern | SemPattern = arg1.update_token_case_mode(token_mode, case_tag)
        self.k: int = arg2
        # self.update_pure_regex_mode([self.arg1])

        self.f_name = 'Repeat'
        self.args = [self.arg1, self.k]

    def duplicate(self) -> 'Repeat':
        return Repeat(self.arg1.duplicate(), self.k, self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        assert len(args) == 3

        string_size: int = args[0]
        arg_matrix: np.array = create_matrix_representation(string_size, args[1])
        if self.token_mode:
            space_matrix: np.array = create_matrix_representation(string_size, args[2])
            return create_span_representation(string_size,
                                              repeat_token_mode_matching_helper(string_size, arg_matrix, space_matrix,
                                                                                self.k, self.k))
        return create_span_representation(string_size,
                                          repeat_matching_helper(string_size, arg_matrix, self.k, self.k))

    def to_py_regex(self, *args) -> str:
        return '(({}){{{}}})'.format(self.arg1.to_py_regex(*args), str(self.k))

    def __repr__(self):
        return '(({}){{{}}})'.format(repr(self.arg1), str(self.k))


class RepeatRange(RegexPattern):
    def __init__(self, arg1: RegexPattern | SemPattern, arg2: int | float, arg3: int | float, token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE,
                 pure_regex: bool = False):
        super().__init__(token_mode, case_tag, pure_regex)
        self.arg1: RegexPattern | SemPattern = arg1.update_token_case_mode(token_mode, case_tag)
        self.k1: int = arg2
        self.k2: int = arg3
        # self.update_pure_regex_mode([self.arg1])

        self.f_name = 'RepeatRange'
        self.args = [self.arg1, self.k1, self.k2]

    def duplicate(self) -> 'RepeatRange':
        return RepeatRange(self.arg1.duplicate(), self.k1, self.k2, self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        assert len(args) == 3

        string_size: int = args[0]
        arg_matrix: np.array = create_matrix_representation(string_size, args[1])
        if self.token_mode:
            space_matrix: np.array = create_matrix_representation(string_size, args[2])
            return create_span_representation(string_size,
                                              repeat_token_mode_matching_helper(string_size, arg_matrix, space_matrix,
                                                                                self.k1, self.k2))
        return create_span_representation(string_size,
                                          repeat_matching_helper(string_size, arg_matrix, self.k1, self.k2))

    def to_py_regex(self, *args) -> str:
        if self.k2 == inf:
            return '(({}){{{},}})'.format(self.arg1.to_py_regex(*args), str(self.k1))
        else:
            return '(({}){{{},{}}})'.format(self.arg1.to_py_regex(*args), str(self.k1), str(self.k2))

    def __repr__(self):
        if self.k2 == inf:
            return '(({}){{{},}})'.format(repr(self.arg1), str(self.k1))
        else:
            return '(({}){{{},{}}})'.format(repr(self.arg1), str(self.k1), str(self.k2))


class Or(RegexPattern):
    def __init__(self, args: List[RegexPattern | SemPattern], token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE, pure_regex: bool = False):
        super().__init__(token_mode, case_tag, pure_regex)
        self.args: List[RegexPattern | SemPattern] = [arg.update_token_case_mode(token_mode, case_tag) for arg in args]
        self.f_name = 'OrR'

    def duplicate(self) -> 'Or':
        return Or([arg.duplicate() for arg in self.args], self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        # I specified args of Or as an n-ary one
        res_span = []
        for arg in args:
            res_span.extend(arg)

        return res_span

    def to_py_regex(self, *args):
        return '({})'.format('|'.join(['({})'.format(arg.to_py_regex(*args)) for arg in self.args]))

    def __repr__(self):
        return '(' + ' || '.join([repr(arg) for arg in self.args]) + ')'


class And(RegexPattern):
    def __init__(self, args: List[RegexPattern | SemPattern], token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE, pure_regex: bool = False):
        super().__init__(token_mode, case_tag, pure_regex)
        self.args: List[RegexPattern | SemPattern] = [arg.update_token_case_mode(token_mode, case_tag) for arg in args]
        self.f_name = 'AndR'

    def duplicate(self) -> 'And':
        return And([arg.duplicate() for arg in self.args], self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        result_matching_span = set()
        for i, match_span in enumerate(args):
            if i == 0:
                result_matching_span.update(match_span)
            else:
                result_matching_span = result_matching_span.intersection(match_span)
        return list(result_matching_span)

    def to_py_regex(self, *args) -> str:
        # NOTE: And is not supported in python regex
        raise NotImplementedError

    def __repr__(self):
        return '(' + ' & '.join([repr(arg) for arg in self.args]) + ')'


class CC(RegexPattern):
    """
    Character class
    """

    def __init__(self, cc: str, token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE, pure_regex: bool = True):
        super().__init__(token_mode, case_tag, pure_regex)
        self.cc: str = cc.lower()
        self.f_name = 'CC'

    def duplicate(self) -> 'CC':
        return CC(self.cc, self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        raise NotImplementedError

    def to_py_regex(self, *args):
        if self.cc in CC_REGEX_MAP:
            return CC_REGEX_MAP[self.cc]
        elif self.cc in CONST_REGEX_MAP:
            return CONST_REGEX_MAP[self.cc]
        elif self.cc == 'empty':
            return '()'
        else:
            raise NotImplementedError(self.cc)

    def __repr__(self):
        if self.cc in CC_REGEX_MAP:
            return CC_REGEX_MAP[self.cc]
        elif self.cc in CONST_REGEX_MAP:
            return CONST_REGEX_MAP[self.cc]
        elif self.cc == 'empty':
            return '()'
        else:
            raise NotImplementedError(self.cc)


class Const(RegexPattern):
    """
    Const string value
    """

    def __init__(self, value: str, token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE, pure_regex: bool = True):
        super().__init__(token_mode, case_tag, pure_regex)
        self.value: str = value
        self.f_name = 'Const'

    def duplicate(self) -> 'Const':
        return Const(self.value, self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        raise NotImplementedError

    def to_py_regex(self, *args):
        if self.value in CONST_REGEX_MAP:
            return '{}'.format(CONST_REGEX_MAP[self.value])
        else:
            # if the value contains letter, we generate three different versions: lower, upper, and original
            if re.search(r'\([\w ]+/[\w ]+', self.value):
                # split by '/'
                split_value = self.value.split('/')
                return '({}|{}|{}|{}|{}|{}|{}|{})'.format(split_value[0].lower(), split_value[0].upper(), split_value[0], split_value[0].strip(), split_value[1].lower(), split_value[1].upper(), split_value[1], split_value[1].strip())

            if any([c.isalpha() for c in self.value]):
                return '({}|{}|{}|{})'.format(self.value.lower(), self.value.upper(), self.value, self.value.strip())
            else:
                return '({})'.format(self.value)

    def __repr__(self):
        if self.value in CONST_REGEX_MAP:
            return '{}'.format(CONST_REGEX_MAP[self.value])
        else:
            return '({})'.format(self.value)


class Null(RegexPattern):
    """
    well, this is null
    """
    def __init__(self, token_mode: bool, case_tag: CaseTag, pure_regex: bool):
        super().__init__(token_mode, case_tag, pure_regex)
        self.f_name = 'Null'

    def duplicate(self) -> 'Null':
        return Null(self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        # return nothing
        return []

    def to_py_regex(self, *args) -> str:
        raise NotImplementedError

    def __repr__(self):
        return 'null'


class Hole(RegexPattern):
    """
    Special pattern for Sketch. Does not actually do anything
    """
    def __init__(self, _type: str, hole_id: int, token_mode: bool, case_tag: CaseTag, pure_regex: bool):
        super().__init__(token_mode, case_tag, pure_regex)
        self.type: str = _type
        self.hole_id: int = hole_id

    def duplicate(self) -> 'Hole':
        return Hole(self.type, self.hole_id, self.token_mode, self.case_tag, self.pure_regex)

    def exec(self, *args) -> List[MatchSpan]:
        raise NotImplementedError

    def to_py_regex(self, *args) -> str:

        # print('arg0:', args[0])
        # print('arg1:', args[1])
        # print('type:', self.type)
        # print('hid:', self.hole_id)

        if self.type in ['integer', 'float', 'year', 'date', 'month', 'time', 'hour', 'second', 'minute', 'day']:
            assert self.hole_id in args[1]
            if len(args[1][self.hole_id]) > 0:
                regex = '({})'.format('|'.join(args[1][self.hole_id]))
            else:
                regex = '.*'
        elif self.type in ['string', 'charseq']:
            regex = '.*'
        elif self.hole_id in args[1] and len(args[1][self.hole_id]) > 0:
            regex = '({})'.format('|'.join(args[1][self.hole_id]))
        else:
            regex = '.*'

        group_name = self.type if len(self.type.split()) == 1 else ''.join(self.type.split())

        if self.hole_id in args[0]:
            return '(?P<{}{}>{}?)'.format(group_name, self.hole_id, regex)

        if regex == '.*':
            if -1 in args[0]:
                return '(?P<{}{}>{})'.format(group_name, self.hole_id, regex)
            else:
                return '(?P<{}{}>{}?)'.format(group_name, self.hole_id, regex)

        return '(?P<{}{}>{})'.format(group_name, self.hole_id, regex)

    def __repr__(self):
        return '{{??: {}}}'.format(self.type)
