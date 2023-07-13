import typing
from typing import Tuple

from lib.lang.production import Production
from lib.program.node import Node
from lib.program.program import Program
from lib.type.base_type import Type
from lib.type.type_system import typing_rules, base_types_to_object, get_type


def infer_type(prog: Program, curr_node_id: int) -> Type:
    """
    A recursive function purely for type-inference
    I am actually not sure how useful this function is lol
    """

    # get current node
    curr_node: Node = prog.nodes[curr_node_id]

    # print("current node: {}".format(curr_node))
    # print("current node.sym: {}".format(curr_node.sym.name))

    if curr_node.sym.name == 'CC' or curr_node.sym.name == 'CONST1':
        return typing_rules['ToCC'].infer()

    elif curr_node.sym.name == 'x':
        # syntactic terminal match
        return typing_rules['ToCC'].infer()

    elif curr_node.sym.name == 's':
        # semantic terminal match
        if curr_node.name == 'Similar':
            return typing_rules[curr_node.name].infer()
        else:
            args = [child.name for child in prog.get_children(curr_node)]
            # print("args: {}".format(args))
            arg_type, _ = get_type(args[0])
            # print("arg_type: {}".format(arg_type))
            if curr_node.name == 'MatchType' and prog.get_children(curr_node)[1].sym.name == 'p1' and prog.get_children(curr_node)[1].name != 'BooleanAtom':
                pred_type = infer_type(prog, prog.get_children(curr_node)[1].id)
                # print('pred_type:', pred_type)
                return typing_rules[curr_node.name].infer([arg_type, pred_type])
            else:
                return typing_rules[curr_node.name].infer([arg_type])

    elif curr_node.sym.name == 'r':
        if curr_node.name in ['NotR', 'Concat', 'StarR', 'Plus', 'Startwith', 'Endwith', 'Contain', 'Plus', 'Repeat', 'RepeatRange']:
            args = [infer_type(prog, child.id) for child in prog.get_children(curr_node)]
            return typing_rules[curr_node.name].infer(args)
        elif curr_node.name in ['OptionalR', 'OrR', 'AndR']:
            args = [infer_type(prog, child.id) for child in prog.get_children(curr_node)]
            return typing_rules[curr_node.name].infer(args)
        else:
            raise NotImplementedError
    elif curr_node.sym.name == 'p1':
        return typing_rules[curr_node.name].infer()


def derive_type(prod: Production, return_type: Type) -> typing.Optional[Type | Tuple[Type, Type]]:
    """
    Given the production and the output type, infer the input type
    """

    if prod.function_name not in typing_rules:
        return base_types_to_object['any']
    else:
        # try to derive the types by calling the function
        return typing_rules[prod.function_name].derive(return_type)
