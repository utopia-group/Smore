from lib.interpreter.context import StrContext


class NoPositiveMatchException(Exception):
    def __init__(self, regex: str, context: StrContext, sketch_object, message='Match should be at least successful for positive strings'):
        self.regex = regex
        self.context = context
        self.sketch_object = sketch_object
        super().__init__(message)


class SketchParsingException(Exception):
    pass


class ProgramParsingException(Exception):
    pass


class SketchInfeasibleException(Exception):
    pass


class TimeOutException(Exception):
    def __init__(self, message, errors=None):
        super(TimeOutException, self).__init__(message)
        self.errors = errors


class NoMoreSampleBudgetException(Exception):
    pass
