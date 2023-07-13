class StrFunc:
    def __init__(self):
        self.f_name: str = ''
        self.args = []

    def to_regex(self, value: str) -> str:
        raise NotImplementedError

    def __repr__(self):
        pass


class Var(StrFunc):
    def __init__(self):
        super().__init__()

    def to_regex(self, value: str):
        return value

    def __repr__(self):
        return 'x'


class Str(StrFunc):
    def __init__(self, arg: str, optional: bool = False):
        super(Str, self).__init__()
        self.arg: str = arg
        self.optional: bool = optional

        if self.optional:
            self.f_name = 'Str?'
            self.args = [self.arg]
        else:
            self.f_name = 'Str'
            self.args = [self.arg]

    def to_regex(self, value: str):
        return "({})".format(self.arg) if not self.optional else "({})?".format(self.arg)

    def __repr__(self):
        return self.arg if not self.optional else "{}?".format(self.arg)


class SubStr(StrFunc):
    def __init__(self, arg: StrFunc, idx1: int, idx2: int):
        super().__init__()
        self.idx1: int = idx1
        self.idx2: int = idx2
        self.arg: StrFunc = arg

        self.f_name = 'SubStr'
        self.args = [self.arg, self.idx1, self.idx2]

    def to_regex(self, value: str):
        ret_value = self.arg.to_regex(value)

        assert ')?' not in ret_value

        return ret_value[self.idx1: self.idx2]

    def __repr__(self):
        return 'substr({}, {}, {})'.format(repr(self.arg), self.idx1, self.idx2)


class Capitalize(StrFunc):
    def __init__(self, arg: StrFunc):
        super().__init__()
        self.arg: StrFunc = arg

        self.f_name = 'Capitalize'
        self.args = [arg]

    def to_regex(self, value: str) -> str:
        ret_value = self.arg.to_regex(value)

        return ret_value.capitalize()

    def __repr__(self):
        return 'capitalize({})'.format(repr(self.arg))


class LowerCase(StrFunc):
    def __init__(self, arg: StrFunc):
        super().__init__()
        self.arg: StrFunc = arg

        self.f_name = 'LowerCase'
        self.args = [arg]

    def to_regex(self, value: str) -> str:
        ret_value = self.arg.to_regex(value)

        return ret_value.lower()

    def __repr__(self):
        return 'lowercase({})'.format(repr(self.arg))


class Append(StrFunc):
    def __init__(self, arg1: StrFunc, arg2: StrFunc):
        super().__init__()
        self.arg1: StrFunc = arg1
        self.arg2: StrFunc = arg2

        self.f_name = 'Append'
        self.args = [self.arg1, self.arg2]

    def to_regex(self, value: str) -> str:
        ret_value1 = self.arg1.to_regex(value)
        ret_value2 = self.arg2.to_regex(value)

        return '{}{}'.format(ret_value1, ret_value2)

    def __repr__(self):
        return 'append({},{})'.format(repr(self.arg1), repr(self.arg2))


class SplitConcat(StrFunc):
    def __init__(self, arg: StrFunc, split_delim: str = ' ', merge_delim: str = ''):
        super().__init__()
        self.arg: StrFunc = arg
        self.split_delim: str = split_delim
        self.merge_delim: str = merge_delim

        self.f_name = 'SplitConcat'
        self.args = [self.arg, self.split_delim, self.merge_delim]

    def to_regex(self, value: str) -> str:
        concat_elements = []

        for token in value.split(self.split_delim):
            concat_elements.append(self.arg.to_regex(token))
        return self.merge_delim.join(concat_elements)

    def __repr__(self):
        return 'splitConcat({},{},{})'.format(repr(self.arg), self.split_delim, self.merge_delim)

