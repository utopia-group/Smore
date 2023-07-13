from typing import Tuple


class Span:
    """
    match span object, represent a [start, end] pair
    """

    def __init__(self, *args):
        if len(args) == 1:
            arg = args[0]
            assert isinstance(arg, Tuple)
            self.start = arg[0]
            self.end = arg[1]
        elif len(args) == 2:
            arg1 = args[0]
            arg2 = args[1]
            assert isinstance(arg1, int) and isinstance(arg2, int)
            self.start = arg1
            self.end = arg2
        else:
            raise NotImplementedError

    def __hash__(self):
        return hash((self.start, self.end))

    def __eq__(self, other):
        if isinstance(other, MatchSpan):
            return self.start == other.start and self.end == other.end
        elif isinstance(other, Tuple) and len(other) == 2 and isinstance(other[0], int) and isinstance(other[1], int):
            return self.start == other[0] and self.end == other[1]
        else:
            raise TypeError

    def __repr__(self):
         return '(' + str(self.start) + ', ' + str(self.end) + ')'
        # return '({}, {})'.format(self.start, self.end)


class MatchSpan(Span):

    def __init__(self, *args):
        super().__init__(*args)

    def update_span(self, start_idx: int) -> 'MatchSpan':
        self.start += start_idx
        self.end += start_idx

        return self
