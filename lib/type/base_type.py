"""
create all the base types here
"""
from typing import Optional as OptionalT
from typing import Tuple, List


class Type:
    def __init__(self):
        pass


class BaseType(Type):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, BaseType):
            return self.name == other.name
        else:
            return False


class GPT3Type(BaseType):
    """
    Base type that is the supertype of any GPT3 generated class
    """
    def __init__(self, name: str = 'GPT3Type'):
        super().__init__(name)


class Any(BaseType):
    """
    This is the base type reserved for any production that I don't have typing rules for
    """
    def __init__(self, name: str = 'Any'):
        super().__init__(name)


class Optional(Type):
    def __init__(self, base: BaseType):
        super().__init__()
        self.base = base

    def __repr__(self):
        return 'Optional({})'.format(repr(self.base))


class Union(Type):
    def __init__(self, init_types: OptionalT[List[Type]] = None):
        super().__init__()
        self.init_types: List[Type] = init_types if init_types is not None else []

    def add_type(self, t):
        if isinstance(t, Type):
            self.init_types.append(t)
        elif isinstance(t, List):
            self.init_types.extend(t)
        else:
            raise NotImplementedError

    def simplify(self):
        """
        Simplify the current type
        """
        raise NotImplementedError


def constructor(self, name: str):
    self.name = name


def _repr(self):
    return self.name


def construct_base_type(name: str, superclasses: Tuple[type]) -> Tuple[type, BaseType]:
    new_class = type(name if ' ' not in name else name.replace(' ', ''),
                     superclasses,
                     {
                         # constructor
                         '__init__': constructor,

                         # data member
                         name: name,

                         # method function
                         "__repr__": lambda self: self.name
                     })
    return new_class, new_class(name)
