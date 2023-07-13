from enum import Enum
from typing import List, Tuple

from lib.lang.production import Production
from lib.program.node import VariableNode, Node
from lib.program.program import Program
from lib.type.base_type import Type, Any

# This dictionary helps in determining if a program has a redundant nested structure
# For example, StartWith(Endwith(x)) (this can be reduced to Contain(x)).
# If a parent and a child map to the same number, it is a redundant nesting
REDUNDANCY_DICT = {'OptionalR': 0,
                   'StarR': 1, 'Plus': 1,
                   'Startwith': 2, 'Endwith': 2, 'Contain': 2, 'any': 2,
                   'Repeat': 3, 'RepeatRange': 3}


REDUNDANCY_DICT2 = {'OptionalR': 0,
                   'Startwith': 2, 'Endwith': 2, 'Contain': 2, 'StarR': 2, 'Plus': 2,
                   }



class PruneCode(Enum):
    FEASIBLE = 1,
    INFEASIBLE = 2,
    DEPTH = 3


def expand_non_terminal_with_production_helper(curr_p: Program, new_prog_id: int, var_node: VariableNode, prod: Production) \
        -> Tuple[Program, List[VariableNode]]:
    # instantiate the selected node with the proper production
    new_prog = curr_p.duplicate(new_prog_id)
    new_parent_node: Node = new_prog.instantiate_var_node(var_node, prod.function_name, prod.ret_sym, prod)
    new_children_nodes: List[VariableNode] = []

    if prod.function_name == 'AndR' or prod.function_name == 'OrR':
        # FIXME: maybe make this n-ary later
        child_sym = prod.arg_syms[0][0]
        for i in range(2):
            new_node = new_prog.add_variable_node(child_sym)
            new_children_nodes.append(new_node)
        new_prog.set_children(new_parent_node, new_children_nodes)
    else:
        for child_sym in prod.arg_syms:
            new_node = new_prog.add_variable_node(child_sym)
            new_children_nodes.append(new_node)
        new_prog.set_children(new_parent_node, new_children_nodes)

    return new_prog, new_children_nodes


def assign_type_to_arguments(new_prog: Program, new_children_node: List[VariableNode], inferred_arg_type: Type) -> Program:
    """
    Given a just expanded program, assign appropriate type to its children node according to the inferred type
    """
    if isinstance(inferred_arg_type, Any):
        for node in new_children_node:
            new_prog.nid_to_goal_type[node.id] = inferred_arg_type
    else:
        if isinstance(inferred_arg_type, Type):
            if len(new_children_node) == 1:
                new_prog.nid_to_goal_type[new_children_node[0].id] = inferred_arg_type
            else:
                assert len(new_children_node) > 1
                # we need to look at the symbol to decide which type to assign to which node
                # so far the type system only considers those nodes with
                for new_node in new_children_node:
                    new_node_sym = new_node.sym
                    # Question: will we ever have a none symbol node here
                    assert new_node_sym is not None

                    if isinstance(new_node_sym, list):
                        # this branch means we can produce a sequence of values
                        # TODO: for simplicity we are just enumerating one
                        raise NotImplementedError
                    elif new_node_sym.name in ['r', 'x', 's', 't_const', 't_ent', 't_query', 't_type']:
                        new_prog.nid_to_goal_type[new_node.id] = inferred_arg_type
                    elif new_node_sym.name == 'p':
                        new_prog.nid_to_goal_type[new_node.id] = inferred_arg_type
                    else:
                        new_prog.nid_to_goal_type[new_node.id] = Any()

        elif isinstance(inferred_arg_type, Tuple):
            # we get into this branch only when inferring type for AndR and OrR production
            # TODO: refine this mark ones we make both type derivation and expansion n-ary
            assert len(new_children_node) == 2
            for new_node, inferred_type in zip(new_children_node, inferred_arg_type):
                new_prog.nid_to_goal_type[new_node.id] = inferred_type

        else:
            raise TypeError("the type {} is invalid".format(inferred_arg_type.__class__))

    return new_prog


def compute_naive_score(eval_result: Tuple[List[bool], List[bool]]) -> float:
    return sum([1 if res else 0 for res in eval_result[0]] + [0 if res else 1 for res in eval_result[1]]) / (len(eval_result[0]) + len(eval_result[1]))


pq_ranking_func = lambda x: x.depth - (1 / len(x.nodes))

pq_ranking_func_no_type_system = lambda x: x.depth - (1 / len(x.nodes)) if 'Match' not in str(x) else x.depth - (1 / len(x.nodes)) - 1
