from typing import List, Optional, Set

from lib.interpreter.context import StrContext, Context
from lib.interpreter.pattern import Pattern, Tag, Predicate, MatchType, Similar, MatchQuery
from lib.interpreter.span import MatchSpan

"""
Regex-pattern matchers
"""


class MatchContext(Context):
    """
    intermediate results for match functions
    """

    def __init__(self, s: StrContext, pattern: Pattern | Predicate | Tag, success: bool):
        super().__init__()
        self.success: bool = success  # success means whether pattern can match the entire s string
        self.s: StrContext = s
        self.pattern: Pattern | Predicate | Tag = pattern
        self.match_spans: Set[MatchSpan] = set()

    def check_success(self) -> bool:
        self.success = False

        for span in self.match_spans:
            # print("check success: span:{} s.get_end_index: {}".format(span, self.s.get_end_index()))
            if span.__eq__((0, self.s.get_end_index())):
                self.success = True
                break

        return self.success

    def __repr__(self):
        return 'MatchContext({}, {}, {})'.format(self.s, self.pattern, self.match_spans)


class RegexMatchContext(MatchContext):
    """
    subclasses of these should all just be regex operators
    """

    def __init__(self, s: StrContext, pattern: Pattern, success: bool = False):
        super().__init__(s, pattern, success)


"""
Num-pattern matcher
"""


class NumMatchContext(MatchContext):
    """
    number semantic matching results
    """

    def __init__(self, s: StrContext, pattern: Predicate, success: bool = False):
        super().__init__(s, pattern, success)


"""
date-time-place-pattern matcher
"""


class DTPMatchContext(MatchContext):
    """
    number semantic matching results
    """

    def __init__(self, s: StrContext, pattern: Predicate, success: bool = False):
        super().__init__(s, pattern, success)


"""
SemRegex-pattern matcher
"""


class TagMatchContext(MatchContext):
    """
    semantic tag context matching results
    """

    def __init__(self, s: StrContext, pattern: Tag, success: bool = False):
        super().__init__(s, pattern, success)


class SimilarMatchContext(MatchContext):
    """
    semantic tag context matching results
    """

    def __init__(self, s: StrContext, pattern: Similar, success: bool = False):
        super().__init__(s, pattern, success)


class QueryMatchContext(MatchContext):
    """
    query match result
    """

    def __init__(self, s: StrContext, pattern: MatchQuery, success: bool = False, tag_values: Optional[List[str]] = None):
        super().__init__(s, pattern, success)
        self.tag_values: List[str] = tag_values


class CustomSemMatchContext(TagMatchContext):
    """
    customized semantic tag context matching results
    currently we only support tag_values as a list of strings
    """
    def __init__(self, s: StrContext, pattern: Tag, tag_values: List[str], mode: str = "CUSTOM"):
        super().__init__(s, pattern)
        self.tag_values: List[str] = tag_values
        self.mode = mode


class PredMatchContext(MatchContext):
    """
    This just correspond to a predicate evaluation of a given string
    a predicate context contains the evaluation results of each of the atom
    """

    def __init__(self, s: StrContext, pattern: Predicate, components: Optional[List[MatchContext]] = None, success: bool = False):
        super().__init__(s, pattern, success)
        self.components: List[MatchContext] = components if components is not None else []


class ObjectMatchContext(MatchContext):
    """
    This class:
        correspond to the matching results for a pr symbol of the grammar, consists of the semantic part and the syntactic part
        each semantic match correspond to multiple syntactic match
        there can be multiple semantic match
        The overall phrase matching is success if there exists at least 1 semantic match such that one of its corresponding syntactic match successes
    """

    def __init__(self, s: StrContext, pattern: MatchType, success: bool = False):
        super().__init__(s, pattern, success)
