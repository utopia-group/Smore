from typing import Any, List, Optional

from lib.interpreter.pattern import Pattern, Const, Hole
from lib.lang.constants import CaseTag
from lib.program.node import TerminalNode, NonterminalNode, ContextNode, HoleNode


def to_pattern_helper_terminal(curr_node: Optional[TerminalNode]) -> Pattern | Any:

    if curr_node is None or curr_node.sym is None:
        return Const('')

    symbol_with_id_constructor: List[str] = ['CONST1', 'CC', 'TYPE', 'ENT', 'ONT', 'QUERY']

    if isinstance(curr_node, ContextNode):
        return curr_node.value

    if curr_node.sym.name in symbol_with_id_constructor:
        return curr_node.id_prod.constructor(curr_node.name)
    elif curr_node.sym.name == 'XVAR':
        return curr_node.id_prod.constructor()
    elif isinstance(curr_node.name, int):
        return curr_node.name
    elif isinstance(curr_node.name, str):
        return curr_node.name
    elif isinstance(curr_node.name, float):
        return curr_node.name
    else:
        raise NotImplementedError('Node {} is not supported'.format(curr_node))


def to_pattern_helper_nonterminal(curr_node: NonterminalNode, args_pattern) -> Pattern:
    pattern_constructor = curr_node.prod.constructor

    if curr_node.name == 'AndR' or curr_node.name == 'OrR':
        # these constructs take a list as the argument
        return pattern_constructor(args_pattern)

    elif curr_node.name == 'MatchType':
        if args_pattern[1] == 'TRUE' or args_pattern[1] == 'FALSE':
            args_pattern[1] = None
        return pattern_constructor(*args_pattern)

    elif curr_node.sym.name == 'p' or curr_node.sym.name == 'p1':
        # this is the predicate branch
        if pattern_constructor is None:
            return args_pattern[0]
        else:
            return pattern_constructor(*args_pattern)

    else:
        # these constructs takes sequence of elements as arguments
        if pattern_constructor is None:
            return args_pattern[0]
        else:
            return pattern_constructor(*args_pattern)


def to_pattern_helper_hole(curr_node: HoleNode) -> Hole:
    return Hole(curr_node.type, curr_node.hole_id, False, CaseTag.NONE, False)