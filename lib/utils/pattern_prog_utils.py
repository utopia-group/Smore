from typing import Optional, List

from lib.interpreter.pattern import Pattern, Tag, Predicate, RegexPattern, Const, CC, SemPattern, PredicateFormula, PredicateAtom, Hole
from lib.interpreter.str_function import StrFunc, Var, Str
from lib.lang.grammar import CFG
from lib.lang.symbol import TerminalSymbol
from lib.program.node import Node
from lib.program.program import Program


def pattern_to_prog(grammar: CFG, _id: int, pattern: Pattern) -> Program:
    # given a pattern, instantiate this to a program

    prog = Program(_id)

    def construct_node(p1: Pattern | Tag | StrFunc | Predicate, root: bool) -> Optional[Node]:

        if p1 is None:
            return prog.add_none_node()

        # print("current_pattern: ", p1)
        if isinstance(p1, tuple):
            node = prog.add_terminal_node('context', grammar.get_terminal_sym('IN_CONTEXT'), None)
            return node
        pattern_name = p1.f_name if not (isinstance(p1, int) or isinstance(p1, float) or isinstance(p1, str)) else ''

        if isinstance(p1, RegexPattern):

            if isinstance(p1, Const):
                node = prog.add_terminal_node(value=p1.value, prod=grammar.get_id_production('ToConst'), symbol=grammar.get_terminal_sym('CONST1'))
                return node
            elif isinstance(p1, CC):
                node = prog.add_terminal_node(value=p1.cc, prod=grammar.get_id_production('ToCC'), symbol=grammar.get_terminal_sym('CC'))
                return node
            elif isinstance(p1, Hole):
                node = prog.add_hole(_type=p1.type, hole_id=p1.hole_id)
                return node
            else:
                # create a new node
                parent_node = prog.add_nonterminal_node(name=pattern_name, symbol=grammar.get_nonterminal_sym('r'), prod=grammar.name_to_prod[pattern_name])
                if root:
                    prog.set_start_node(parent_node)
                children_node = [construct_node(cpt, False) for cpt in p1.args]
                prog.set_children(parent_node, children_node, no_depth=True)

                return parent_node

        elif isinstance(p1, SemPattern):
            parent_node = prog.add_nonterminal_node(name=pattern_name, symbol=grammar.get_nonterminal_sym('s'), prod=grammar.name_to_prod[pattern_name])
            if root:
                prog.set_start_node(parent_node)
            # first we need to find out what are the arguments
            children_pattern = p1.args
            # children_pattern = [value for key, value in vars(p1).items() if 'arg' in key]
            children_node = [construct_node(cpt, False) for cpt in children_pattern]
            # print("children_node: ", children_node)
            prog.set_children(parent_node, [c for c in children_node if c is not None], no_depth=True)

            return parent_node

        elif isinstance(p1, PredicateFormula):
            if p1.op == 'not':
                parent_node = prog.add_nonterminal_node(name='Not',
                                                        symbol=grammar.get_nonterminal_sym('p'),
                                                        prod=grammar.name_to_prod['Not'])
                children_node = [construct_node(p1.components[0], False)]
                prog.set_children(parent_node, children_node, no_depth=True)
                return parent_node

            else:
                # make this a binary-construct for simplicity
                # TODO: improve this later
                name = 'Or' if p1.op == 'or' else 'And'
                parent_node = prog.add_nonterminal_node(name=name,
                                                        symbol=grammar.get_nonterminal_sym('p'),
                                                        prod=grammar.name_to_prod[name])
                children_node = [construct_node(p1.components[0], False),
                                 construct_node(p1.components[1], False)]
                prog.set_children(parent_node, children_node, no_depth=True)

                return parent_node

        elif isinstance(p1, PredicateAtom):
            sub_parent_node = prog.add_nonterminal_node(name=pattern_name, symbol=grammar.get_nonterminal_sym('p'), prod=grammar.name_to_prod[pattern_name])
            children_node = [construct_node(cpt, False) for cpt in p1.args]
            prog.set_children(sub_parent_node, children_node, no_depth=True)

            parent_node = prog.add_nonterminal_node(name='ToAtom', symbol=grammar.get_nonterminal_sym('p'), prod=grammar.name_to_prod['ToAtom'])
            prog.set_children(parent_node, [sub_parent_node], no_depth=True)
            return parent_node

        elif isinstance(p1, StrFunc):
            if isinstance(p1, Var):
                node = prog.add_terminal_node(value='x', prod=grammar.get_id_production('ToVar'), symbol=grammar.get_terminal_sym('XVAR'))
                return node
            elif isinstance(p1, Str):
                if p1.optional:
                    node = prog.add_terminal_node(value=p1.arg, prod=grammar.get_id_production('Str?'), symbol=grammar.get_terminal_sym('CONST'))
                else:
                    node = prog.add_terminal_node(value=p1.arg, prod=grammar.get_id_production('Str'), symbol=grammar.get_terminal_sym('CONST'))

                return node
            else:
                parent_node = prog.add_nonterminal_node(name=pattern_name, symbol=grammar.get_nonterminal_sym('f'), prod=grammar.name_to_prod[pattern_name])
                children_node = [construct_node(cpt, False) for cpt in p1.args]
                prog.set_children(parent_node, children_node, no_depth=True)

                return parent_node

        elif isinstance(p1, int):
            node = prog.add_terminal_node(value=p1, prod=None, symbol=grammar.get_terminal_sym('N'))
            return node
        elif isinstance(p1, float):
            node = prog.add_terminal_node(value=p1, prod=None, symbol=grammar.get_terminal_sym('SIM'))
            return node
        elif isinstance(p1, str):
            op_sym = grammar.get_terminal_sym('OP')
            delim_sym = grammar.get_terminal_sym('DELIM')
            region_sym = grammar.get_terminal_sym('REGION')
            sim_sym = grammar.get_terminal_sym('SIM')
            bool_sym = grammar.get_terminal_sym('BOOL')
            if p1 in op_sym.values:
                node = prog.add_terminal_node(value=p1, prod=None, symbol=op_sym)
            elif p1 in delim_sym.values:
                node = prog.add_terminal_node(value=p1, prod=None, symbol=delim_sym)
            elif p1 in region_sym.values:
                node = prog.add_terminal_node(value=p1, prod=None, symbol=region_sym)
            elif p1 in sim_sym.values:
                node = prog.add_terminal_node(value=p1, prod=None, symbol=sim_sym)
            elif p1 in bool_sym.values:
                node = prog.add_terminal_node(value=p1, prod=None, symbol=bool_sym)
            else:
                raise NotImplementedError
            return node
        elif isinstance(p1, Tag):
            prod_arg_sym = grammar.name_to_prod[pattern_name].arg_syms[0]
            assert isinstance(prod_arg_sym, TerminalSymbol)
            node = prog.add_terminal_node(value=p1.tag, prod=grammar.get_id_production(pattern_name), symbol=prod_arg_sym)
            return node
        else:
            raise NotImplementedError("pattern {} is not supported".format(p1))

    construct_node(pattern, True)
    return prog


def get_all_semantic_types(pattern: Pattern) -> List[str]:
    all_types = []

    def get_type(p1: Pattern | Tag | StrFunc | Predicate):
        if isinstance(p1, RegexPattern):
            for arg in p1.args:
                get_type(arg)
        elif isinstance(p1, SemPattern):
            for arg in p1.args:
                if isinstance(arg, Tag):
                    all_types.append(arg.tag.lower())

    get_type(pattern)
    return all_types
