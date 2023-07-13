from typing import List, Type, Optional, Callable, Any

from lib.interpreter.pattern import Pattern, Predicate, StrFunc, Tag
from lib.lang.symbol import NonterminalSymbol, Symbol

"""
basic production class
since we implement the dsl separate, this class also connects the grammar with execution (i.e. the Pattern class)
"""


class Production:
    """
    General production class
    """

    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, arg_syms: List[Symbol] | List[List[Symbol]],
                 constructor: Optional[Type[Pattern] | Type[Predicate] | Type[StrFunc] | Callable[[Any], Predicate | Pattern | StrFunc]]):
        self.function_name: str = function_name

        self.ret_sym: NonterminalSymbol = ret_sym
        self.arg_syms: List[Symbol] = arg_syms

        # take in a pattern constructor (or a callable function that returns a pattern
        self.constructor: Optional[Type[Pattern] | Callable[[Any], Predicate | Pattern] | StrFunc] = constructor

    def __repr__(self):
        return '{}({}) -> {}'.format(self.function_name, ','.join([repr(s) for s in self.arg_syms]), self.ret_sym)


class IdProduction(Production):
    """
    Production of the pattern ret_sym := arg_sym
    # TODO: the constructor is really overloaded
    """

    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, arg_syms: List[Symbol],
                 constructor: Optional[Type[Pattern | Tag | StrFunc] | StrFunc | Callable[[Any], Predicate | Pattern | StrFunc]]):
        super().__init__(function_name, ret_sym, arg_syms, constructor)
