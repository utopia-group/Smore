import itertools
from typing import Optional, Dict, List, Tuple, Any

from lib.interpreter.pattern import Pattern, Predicate, StarR, CC, Null, MatchType, TypeTag, Or, MatchEntity, MatchQuery, EntTag, QueryStr, OptionalR
from lib.interpreter.str_function import StrFunc, Var
from lib.lang.constants import CaseTag
from lib.lang.production import Production, IdProduction
from lib.lang.symbol import NonterminalSymbol, TerminalSymbol, Symbol
from lib.program.node import NonterminalNode, Node, VariableNode, TerminalNode, NoneNode, HoleNode, ContextNode
from lib.type import type_system
from lib.type.base_type import Type, BaseType
from lib.type.type_enum import parsable_types, entity_types
from lib.utils.program_utils import to_pattern_helper_nonterminal, to_pattern_helper_terminal, to_pattern_helper_hole

"""
Program is a tree data structure that tracks both children and parent relationship between nodes 
"""


class Program:
    def __init__(self, _id: int, start_nid: Optional[int] = None):
        # classic fields
        self.id: int = _id
        self.depth: int = -1
        self.start_node_id: Optional[int] = None
        self.nodes: Dict[int, Node] = {}
        self.to_children: Dict[int, List[int]] = {}
        self.to_parent: Dict[int, int] = {}
        self.node_to_depth: Dict[int, int] = {}

        # for synthesizer
        self.var_nodes: Dict[int, str] = {}
        self.hole_id_to_hole_nid: Dict[int, int] = {}
        self.node_id_counter = itertools.count(start=1) if start_nid is None else itertools.count(start=start_nid)
        self.contain_leaf: bool = False

        # for typed synthesizer
        self.nid_to_goal_type: Dict[int, Type] = {}

    def duplicate(self, new_id: int) -> 'Program':
        prog = Program(new_id)
        prog.depth = self.depth
        prog.start_node_id = self.start_node_id
        prog.nodes = self.nodes.copy()
        prog.to_children = self.to_children.copy()
        prog.to_parent = self.to_parent.copy()
        prog.node_to_depth = self.node_to_depth.copy()

        prog.var_nodes = self.var_nodes.copy()
        prog.hole_id_to_hole_nid = self.hole_id_to_hole_nid.copy()
        prog.node_id_counter = itertools.tee(self.node_id_counter)[1]
        prog.contain_leaf = self.contain_leaf

        prog.nid_to_goal_type = self.nid_to_goal_type.copy()

        return prog

    def update(self, prog: 'Program'):
        """
        update the current program structure with the information from the new program
        """
        self.nodes.update(prog.nodes)
        self.to_children.update(prog.to_children)
        self.to_parent.update(prog.to_parent)
        self.node_to_depth.update(prog.node_to_depth)

    def increment_nid(self, base: int):
        start_node_id: Optional[int] = self.start_node_id + base
        nodes: Dict[int, Node] = dict([(nid + base, node) for nid, node in self.nodes.items()])
        # need to duplicate the nodes as well
        nodes2: Dict[int, Node] = {}
        for _id, node in nodes.items():
            nodes2[_id] = node.duplicate(_id)
        self.nodes = nodes2
        to_children: Dict[int, List[int]] = dict([(nid + base, [cid + base for cid in cnids]) for nid, cnids in self.to_children.items()])
        to_parent: Dict[int, int] = dict([(cid + base, pid + base) for cid, pid in self.to_parent.items()])
        node_to_depth: Dict[int, int] = {}

        self.start_node_id = start_node_id
        self.to_children = to_children
        self.to_parent = to_parent
        self.node_to_depth = node_to_depth

        # reset
        self.var_nodes = {}
        self.hole_id_to_hole_nid: Dict[int, int] = {}

    def is_concrete(self) -> bool:
        return len(self.var_nodes) == 0

    def set_start_node(self, node: Node):
        self.start_node_id = node.id
        self.depth = 1
        self.node_to_depth[node.id] = self.depth

    def get_depth(self) -> int:
        return self.depth

    def add_hole(self, _type: str, hole_id: int) -> HoleNode:
        new_node = HoleNode(next(self.node_id_counter), _type=_type, hole_id=hole_id)
        self.nodes[new_node.id] = new_node
        self.hole_id_to_hole_nid[hole_id] = new_node.id

        return new_node

    def add_variable_node(self, symbol: Symbol) -> VariableNode:
        new_node = VariableNode(next(self.node_id_counter), name='?', sym=symbol)
        # print("adding a variable node with id: {}".format(new_node.id))
        self.nodes[new_node.id] = new_node
        self.var_nodes[new_node.id] = ''
        return new_node

    def add_nonterminal_node(self, name: str, symbol: NonterminalSymbol, prod: Production, sub_id: Optional[int] = None) -> NonterminalNode:
        node_id = next(self.node_id_counter) if sub_id is None else sub_id
        new_node = NonterminalNode(node_id, name=name, sym=symbol, prod=prod)
        self.nodes[node_id] = new_node
        return new_node

    def add_terminal_node(self, value: str | int | float, symbol: TerminalSymbol, prod: Optional[IdProduction], sub_id: Optional[int] = None) -> TerminalNode:
        node_id = next(self.node_id_counter) if sub_id is None else sub_id
        if symbol.name == 'IN_CONTEXT':
            new_node = ContextNode(node_id, name=value, value=([], []), sym=symbol)
        else:
            new_node = TerminalNode(node_id, name=value, prod=prod, sym=symbol)
        self.nodes[node_id] = new_node
        if symbol.name in ['BOOL', 'ENT', 'ONT', 'CC', 'CONST1', 'QUERY', 'CONSTSTR']:
            self.contain_leaf = True
        return new_node

    def add_node(self, value: str | int | float, symbol: Symbol, prod: Optional[Production], sub_id: Optional[int]) -> TerminalNode | NonterminalNode:
        if isinstance(symbol, TerminalSymbol):
            assert prod is None or isinstance(prod, IdProduction)
            return self.add_terminal_node(value, symbol, prod, sub_id)
        elif isinstance(symbol, NonterminalSymbol):
            return self.add_nonterminal_node(value, symbol, prod, sub_id)
        else:
            raise ValueError

    def add_none_node(self) -> NoneNode:
        new_node = NoneNode(next(self.node_id_counter))
        self.nodes[new_node.id] = new_node

        return new_node

    def add_empty_node(self) -> TerminalNode:
        new_node = TerminalNode(next(self.node_id_counter), '', None, None)
        self.nodes[new_node.id] = new_node
        return new_node

    def instantiate_var_node(self, var_node: VariableNode, name: str, symbol: Symbol, prod: Optional[Production] = None) -> Node:
        new_node = self.add_node(value=name, symbol=symbol, prod=prod, sub_id=var_node.id)
        # remove var_node
        # print("removing a variable node with id: {}".format(new_node.id))
        del self.var_nodes[var_node.id]
        return new_node

    def instantiate_hole_node(self, hole_id: int, hole_type: BaseType, symbol: Symbol):
        # get hole node
        hole_node = self.get_node(self.hole_id_to_hole_nid[hole_id])
        variable_node = VariableNode(hole_node.id, name='?', sym=symbol)
        self.nodes[hole_node.id] = variable_node
        self.var_nodes[hole_node.id] = ''
        self.nid_to_goal_type[hole_node.id] = hole_type
        self.node_to_depth[hole_node.id] = 1

    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def get_children(self, parent: Node) -> List[Node]:
        return [self.get_node(cid) for cid in self.to_children[parent.id]]

    def set_children(self, parent: Node, children: List[Node], no_depth: bool = False):
        self.to_children[parent.id] = [c.id for c in children]
        for c in children:
            self.to_parent[c.id] = parent.id

        # set depth
        if not no_depth:
            child_depth = self.node_to_depth[parent.id] + 1
            for c in children:
                self.node_to_depth[c.id] = child_depth
            # program depth is the max of its current depth and the child_depth
            self.depth = max(self.depth, child_depth)
        # print("set children depth res: {}".format(self.node_to_depth))

    def get_ancestor_helper(self, nid: int, acc_list: List[NonterminalNode]):
        # the returned list also include the original node itself (if it is also a non-terminal node)
        if nid == self.start_node_id:
            pass
        else:
            curr_node = self.get_node(nid)
            if isinstance(curr_node, NonterminalNode):
                acc_list.append(curr_node)

            self.get_ancestor_helper(self.to_parent[nid], acc_list)

    def select_var_node(self) -> VariableNode:
        # Pick the variable node with the lowest depth
        lowest_depth = len(self.nodes) + 2
        lowest_depth_var_node_id = -1
        for var_node_id in self.var_nodes:
            if self.node_to_depth[var_node_id] < lowest_depth:
                lowest_depth_var_node_id = var_node_id
                lowest_depth = self.node_to_depth[var_node_id]

        selected_var_node = self.nodes[lowest_depth_var_node_id]
        assert isinstance(selected_var_node, VariableNode)
        return selected_var_node

    def to_under_approx_helper(self, curr_node: Node):
        """
        helper method to generate the under-approximation
        the reason is that null is not supported in pure_regex mode, but we can symbolically evaluate it actually
        intuition: when is null actually become something useful? (assuming we dont have not)
        [[Star(null)]] -> EmptyStr
        [[Optional(null]] -> EmptyStr
        [[Or(r1, ..., rn, null]] -> Or(r1, ..., rn)
        For the rest, we can just propagate null (as #)
        """
        if curr_node.name == 'Not':
            raise ValueError("Not node not allow in the partial program so far")

        if isinstance(curr_node, NoneNode):
            return None
        elif isinstance(curr_node, TerminalNode):
            return to_pattern_helper_terminal(curr_node)
        elif isinstance(curr_node, VariableNode):
            return '#'
        else:
            assert isinstance(curr_node, NonterminalNode)

            args_pattern = [self.to_under_approx_helper(arg) for arg in self.get_children(curr_node)]

            if '#' in args_pattern:
                # that means we found a null somewhere in the children
                if curr_node.name == 'OrR' or curr_node.name == 'StarR' or curr_node.name == 'Optional':
                    if curr_node.name == 'StarR' or curr_node.name == 'Optional':
                        # in this case, we can directly evaluate to empty
                        return CC(cc='empty')
                    else:
                        # in this case, we can directly evaluate to the rest of the arguments
                        new_args_pattern = [p for p in args_pattern if not p == '#']
                        pattern_constructor = curr_node.prod.constructor
                        return pattern_constructor(new_args_pattern)
                else:
                    # for the rest we just return '#'
                    return '#'
            else:
                # treat as a normal program
                return to_pattern_helper_nonterminal(curr_node, args_pattern)

    def to_over_approx_helper(self, curr_node: Node, use_type_info: bool):

        def get_over_approx_based_on_type(bt: BaseType):
            all_entity_types = {**parsable_types, **entity_types}
            base_type_name = bt.name.lower()
            if base_type_name in ['string', 'charseq']:
                return StarR(arg1=CC(cc='any'), pure_regex=False)
            elif base_type_name in ['integer', 'float', 'number', 'date', 'time']:
                return MatchType(TypeTag(parsable_types[base_type_name]))
            elif base_type_name in all_entity_types:
                return Or([MatchEntity(EntTag(all_entity_types.get(base_type_name))), MatchQuery(QueryStr(base_type_name), Var())])
            else:
                return MatchQuery(QueryStr(base_type_name), Var())

        if curr_node.name == 'Not':
            raise ValueError("Not node not allow in the partial program so far")

        if isinstance(curr_node, NoneNode):
            return None
        elif isinstance(curr_node, TerminalNode):
            return to_pattern_helper_terminal(curr_node)
        elif isinstance(curr_node, VariableNode):
            if isinstance(curr_node.sym, list):
                # TODO: fix this! bypass this for testing purpose
                pass
            elif curr_node.sym.name == 'r':
                # we can do better than this when we have type information
                if use_type_info:
                    curr_node_type = self.nid_to_goal_type[curr_node.id]
                    if isinstance(curr_node_type, BaseType):
                        return get_over_approx_based_on_type(curr_node_type)
                    elif isinstance(curr_node_type, type_system.Optional):
                        # get internal type
                        base_type = curr_node_type.base
                        return OptionalR(get_over_approx_based_on_type(base_type))
                    else:
                        return StarR(arg1=CC(cc='any'), pure_regex=False)
                else:
                    return StarR(arg1=CC(cc='any'), pure_regex=False)
            else:
                return '?'
                # raise NotImplementedError('curr_node with sym {} is not implemented'.format(curr_node.sym.name))
        else:
            assert isinstance(curr_node, NonterminalNode)

            args_pattern = [self.to_over_approx_helper(arg, use_type_info) for arg in self.get_children(curr_node)]
            # print(curr_node)
            # print(curr_node.sym.name)
            # print(args_pattern)
            if (curr_node.sym.name == 'p' or curr_node.sym.name == 'p1') and '?' in args_pattern:
                return '?'

            elif '?' in args_pattern:
                return StarR(arg1=CC(cc='any'), pure_regex=False)
            else:
                # treat as a normal thing
                return to_pattern_helper_nonterminal(curr_node, args_pattern)

    def to_pattern_helper(self, curr_node: Node) -> Pattern | Predicate | StrFunc | Any:
        """
        helper method to generate the pattern for the sub-tree
        TODO: did not take token_mode and case_tag into consideration so far
        """
        if isinstance(curr_node, NoneNode):
            return None
        elif isinstance(curr_node, TerminalNode):
            return to_pattern_helper_terminal(curr_node)
        elif isinstance(curr_node, HoleNode):
            return to_pattern_helper_hole(curr_node)
        else:
            assert isinstance(curr_node, NonterminalNode)

            args_pattern = [self.to_pattern_helper(arg) for arg in self.get_children(curr_node)]

            return to_pattern_helper_nonterminal(curr_node, args_pattern)

    def get_approximations(self, use_type_info: bool) -> Tuple[Pattern, Pattern]:
        """
        derive the over and under approx pattern for the  program
        """
        over_approx = self.to_over_approx_helper(self.nodes[self.start_node_id], use_type_info)
        under_approx = self.to_under_approx_helper(self.nodes[self.start_node_id])
        under_approx = under_approx if not under_approx == '#' else Null(False, CaseTag.IGNORE, False)
        return over_approx, under_approx

    def to_pattern(self) -> Pattern:
        """
        derive the concrete regex pattern for the program
        """
        return self.to_pattern_helper(self.nodes[self.start_node_id])

    def count_node_size_helper(self, curr_node: Node) -> int:
        if isinstance(curr_node, NoneNode):
            return 0
        elif isinstance(curr_node, TerminalNode):
            return 1
        elif isinstance(curr_node, VariableNode):
            return 1
        else:
            assert isinstance(curr_node, NonterminalNode)
            return 1 + sum([self.count_node_size_helper(arg) for arg in self.get_children(curr_node)])

    def count_node_size(self) -> int:
        return self.count_node_size_helper(self.nodes[self.start_node_id])

    def repr_helper(self, curr_node: Node) -> str:
        if isinstance(curr_node, TerminalNode):
            if curr_node.id_prod is not None and curr_node.id_prod.function_name == 'Str?':
                return '{}?'.format(repr(curr_node))
            else:
                return repr(curr_node)
        elif isinstance(curr_node, NonterminalNode):
            children_node = [self.repr_helper(node) for node in self.get_children(curr_node)]
            return '{}({})'.format(curr_node.name, ','.join(children_node))
        elif isinstance(curr_node, VariableNode):
            if curr_node.id in self.nid_to_goal_type:
                return "{}: ({}, {})".format(repr(curr_node), curr_node.sym, self.nid_to_goal_type[curr_node.id])
            else:
                return "{}: ({})".format(repr(curr_node), curr_node.sym)
        elif isinstance(curr_node, HoleNode):
            return repr(curr_node)
        else:
            raise NotImplementedError(curr_node)

    def __repr__(self):
        return self.repr_helper(self.nodes[self.start_node_id])
