"""
Sketch is a program with typed holes.
We make a separate class to track some additional information (I am not sure if this is actually necessary)
"""
import itertools
import re
from collections import defaultdict
from typing import Dict, List, Optional

from lark import UnexpectedToken

from lib.config import pd_print
from lib.interpreter.pattern import Pattern, RegexPattern
from lib.lang.constants import DECOMPOSE_SPECIAL_TAGS, CaseTag
from lib.lang.grammar import CFG
from lib.parser.parser import t, parse_program
from lib.program.node import NoneNode, TerminalNode, NonterminalNode, Node, HoleNode
from lib.program.program import Program
from lib.utils.exceptions import SketchParsingException
from lib.utils.pattern_prog_utils import pattern_to_prog
from lib.utils.program_utils import to_pattern_helper_terminal, to_pattern_helper_nonterminal, to_pattern_helper_hole
from lib.utils.sketch_utils import postprocess_gpt3_sketch


class Sketch:
    def __init__(self, pattern: Pattern, program: Program, hole_id_to_type: Dict[int, str], sketch_str: Optional[str] = None):
        self.pattern: Pattern = pattern
        self.program: Program = program
        self.hole_id_to_type: Dict[int, str] = hole_id_to_type
        self.sketch_str = sketch_str

        if len(program.nodes) == 1:
            self.program.start_node_id = list(program.nodes.keys())[0]

        # used for decomposition of the sketch
        self.hole_id_in_optional = None
        self.find_holes_in_optional()

        # used for composing the programs
        self.hole_id_to_prog: Dict[int, List[Program]] = {}

        # used for union of sketches
        self.has_union: bool = False
        self.sketch_to_examples: Dict[int, List[str]] = {}

        # check if it can reject all the negative example already
        self.can_reject_all_negative_example = False

        self.no_type = False

    def union(self, s2: 'Sketch') -> 'Sketch':
        """
        create a new sketch that is union of self and s2
        """
        raise NotImplementedError

    def find_holes_in_optional(self):
        self.hole_id_in_optional: List[int] = []

        # start with the hole node, try to find if there exists an optional from itself to the root
        for hid, hnid in self.program.hole_id_to_hole_nid.items():
            ancestor: List[NonterminalNode] = []
            if hnid != self.program.start_node_id:
                self.program.get_ancestor_helper(hnid, ancestor)
            if any([True if n.name == 'OptionalR' else False for n in ancestor]):
                self.hole_id_in_optional.append(hid)

    def instantiate_hole_with_program(self, hid: int, prog: List[Program]):
        """
        Instantiate holes in the sketch with concrete program
        We use this function when compose the subgoals synthesis results
        """
        self.hole_id_to_prog[hid] = prog

    def get_holes_for_lazy_matching(self) -> List[int]:
        assert isinstance(self.pattern, RegexPattern)

        # figure out which holes we need to let it be lazy matching heuristic: basically lazy match all the holes that should not be lazy matched but next to each other we need to require the hole
        # before the optional node to be lazy matched
        holes_to_be_lazy_matched = []
        for hole_id, hole_type in self.hole_id_to_type.items():
            if hole_type in DECOMPOSE_SPECIAL_TAGS:
                continue
            # if the prev or next hole is a lazy hole as well
            prev_hole_type = self.hole_id_to_type.get(hole_id - 1)
            if prev_hole_type is not None and prev_hole_type not in DECOMPOSE_SPECIAL_TAGS:
                holes_to_be_lazy_matched.append(hole_id)
                continue

            next_hole_type = self.hole_id_to_type.get(hole_id + 1)
            if next_hole_type is not None and next_hole_type not in DECOMPOSE_SPECIAL_TAGS:
                holes_to_be_lazy_matched.append(hole_id)
                continue

        # print('holes_to_be_lazy_matched: {}'.format(holes_to_be_lazy_matched))

        # check whether consecutive in lazy match exists
        holes_to_be_lazy_matched_1 = []
        for i, h in enumerate(holes_to_be_lazy_matched):
            if i == 0:
                holes_to_be_lazy_matched_1.append(h)
                continue

            if h + 1 in self.hole_id_in_optional:
                holes_to_be_lazy_matched_1.append(h)
                continue

            if h - 1 in holes_to_be_lazy_matched_1:
                continue

        # print('holes_to_be_lazy_matched_1: {}'.format(holes_to_be_lazy_matched_1))

        # finally, we add those holes that has the type SPECIAL_DECOMPOSE_TAG
        holes_to_be_lazy_matched_1.extend([h for h, ty in self.hole_id_to_type.items() if ty in DECOMPOSE_SPECIAL_TAGS and h not in holes_to_be_lazy_matched_1])

        return holes_to_be_lazy_matched_1

    def get_overapprox(self, type_str_ctx: Dict[str, List[str]], lazy=True) -> str:

        hole_ctx: Dict[int, List[str]] = defaultdict(list)

        for hole_id, ty in self.hole_id_to_type.items():
            # print('ty', ty)
            if 'optional' in ty.lower():
                ty = ty[9:-1]
            if ty.lower() not in ['string', 'charseq']:
                all_ctx = []
                if type_str_ctx.get(ty.upper()) is not None:
                    all_ctx.extend(type_str_ctx[ty.upper()])
                if type_str_ctx.get(ty.lower()) is not None:
                    all_ctx.extend(type_str_ctx[ty.lower()])
                # print('all_ctx:', all_ctx)
                hole_ctx[hole_id].extend(sorted(all_ctx, key=lambda x: len(x), reverse=True))

        if self.no_type:
            over_approx = self.pattern.to_py_regex([-1], hole_ctx)
        elif lazy:
            over_approx = self.pattern.to_py_regex(self.get_holes_for_lazy_matching(), hole_ctx)
        else:
            over_approx = self.pattern.to_py_regex([], hole_ctx)
        # print("over_approx: {}".format(over_approx))

        # post-process the over-approximation
        pd_print('before post-processing: {}'.format(over_approx))
        if re.search(r'\(\(\.\)\*\?\)\((\?P\<[\w\d]+\>\([^)]+\))\?\)', over_approx) is not None:
            over_approx = re.sub(r'\(\(\.\)\*\?\)\((\?P\<[\w\d]+\>\([^)]+\))\?\)', '((.)*?)(\\1)', over_approx)
        pd_print('after post-processing: {}'.format(over_approx))
        return over_approx

    def to_pattern_helper(self, curr_node: Node):
        """
         helper method to generate the pattern for the sub-tree
         """
        if isinstance(curr_node, NoneNode):
            return None
        elif isinstance(curr_node, TerminalNode):
            return to_pattern_helper_terminal(curr_node)
        elif isinstance(curr_node, HoleNode):
            if len(self.hole_id_to_prog) == 0:
                return to_pattern_helper_hole(curr_node)
            else:
                if curr_node.hole_id in self.hole_id_to_prog:
                    # we are only getting the top-1 program here for each hole
                    sub_prog = self.hole_id_to_prog[curr_node.hole_id][0]
                    return sub_prog.to_pattern()
                else:
                    # The possibility of the hole does not exist exists (when there is no examples associated with it)
                    return to_pattern_helper_terminal(None)
        else:
            assert isinstance(curr_node, NonterminalNode)

            args_pattern = [self.to_pattern_helper(arg) for arg in self.program.get_children(curr_node)]
            return to_pattern_helper_nonterminal(curr_node, args_pattern)

    def to_pattern(self) -> Pattern:
        """
        This function is used when compose sub-programs to a full pattern
        """
        return self.to_pattern_helper(self.program.nodes[self.program.start_node_id])

    def get_concatenated_pattern_helper(self, curr_node: Node, res_list: List[str]):

        if isinstance(curr_node, NonterminalNode):
            if curr_node.name == 'Concat':
                for arg in self.program.get_children(curr_node):
                    self.get_concatenated_pattern_helper(arg, res_list)
            else:
                res_list.append(repr(self.to_pattern_helper(curr_node)))
        elif isinstance(curr_node, HoleNode):
            res_list.append(repr(self.to_pattern_helper(curr_node)))
        else:
            res_list.append(repr(self.to_pattern_helper(curr_node)))

    def get_concatenated_pattern(self) -> List[str]:
        ret_list = []
        self.get_concatenated_pattern_helper(self.program.nodes[self.program.start_node_id], ret_list)
        return ret_list

    def compose_program(self, prog_id: int) -> Program:
        new_prog = self.program.duplicate(prog_id)
        curr_nid = max(list(new_prog.nodes.keys()))
        for hid, progs in self.hole_id_to_prog.items():
            hnid = new_prog.hole_id_to_hole_nid[hid]
            # we need to rename the nid in programs otherwise the information will not be consistent
            # just get the top-1
            curr_prog = progs[0]
            curr_prog.increment_nid(curr_nid)
            curr_nid = max(list(curr_prog.nodes.keys()))

            new_prog.update(curr_prog)

            if hnid == new_prog.start_node_id:
                new_prog.start_node_id = curr_prog.start_node_id

            else:
                # replace the children with the new node
                parent = new_prog.get_node(new_prog.to_parent[hnid])
                new_children = []
                for child in new_prog.get_children(parent):
                    if child.id == hnid:
                        new_children.append(curr_prog.get_node(curr_prog.start_node_id))
                    else:
                        new_children.append(child)
                new_prog.set_children(parent, new_children, True)

        # There might be ignored hid, we also need to handle those
        for hid, hnid in new_prog.hole_id_to_hole_nid.items():
            if hid in self.hole_id_to_prog:
                # already handled
                pass
            else:
                new_child_node = new_prog.add_empty_node()
                parent = new_prog.get_node(new_prog.to_parent[hnid])
                new_children = []
                for child in new_prog.get_children(parent):
                    if child.id == hnid:
                        new_children.append(new_child_node)
                    else:
                        new_children.append(child)
                new_prog.set_children(parent, new_children, True)

        return new_prog

    def get_full_sketch_str(self):
        return self.sketch_str

    def __repr__(self):
        return str(self.pattern)

    def __eq__(self, other):
        if self.sketch_str is not None and other.sketch_str is not None:
            return self.sketch_str == other.sketch_str
        else:
            return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


class SketchWithExamples(Sketch):
    def __init__(self, _type: str, pattern: Pattern, program: Program, hole_id_to_type: Dict[int, str], pos_examples: List[str]):
        super().__init__(pattern, program, hole_id_to_type)
        self.pos_examples = pos_examples

    def __repr__(self):
        return '({}: {})'.format(repr(self.sketch_str), repr(self.pos_examples))


def parse_sketch_from_pattern(parsed_pattern: Pattern, sketch_part: str = None, hole_id_to_type: Dict = None) -> Sketch:
    """
    Turns pattern into sketch directly.
    The sketch_part gets fed into the sketch.sketch_str. If this is none, then str(pattern)
    is used.
    """
    parsed_prog = pattern_to_prog(CFG(), 0, parsed_pattern)
    if sketch_part is None:
        assert hole_id_to_type is not None
        return Sketch(parsed_pattern, parsed_prog, hole_id_to_type, str(parsed_pattern))
    return Sketch(parsed_pattern, parsed_prog, t.hole_id_to_hole_type, sketch_part)


def parse_sketch(prog_str: str, token_mode: bool = False, case_tag: CaseTag = CaseTag.NONE) -> Sketch:
    # some inits for sketch-related stuff
    t.hole_id_counter = itertools.count(start=1)
    t.hole_id_to_hole_type = {}
    return parse_sketch_from_pattern(parse_program(prog_str, token_mode, case_tag), prog_str)


def process_and_parse_sketch(s: str) -> Optional[Sketch]:
    try:
        return parse_sketch(postprocess_gpt3_sketch(s), False)
    except ValueError:
        pd_print('have trouble parsing {}'.format(s))
        return None
    except SketchParsingException:
        pd_print('have trouble parsing {}'.format(s))
        return None
    except UnexpectedToken:
        pd_print('have trouble parsing {}'.format(s))
        return None
    except KeyError:
        pd_print('have trouble parsing {}'.format(s))
        return None
    except AssertionError:
        pd_print('have trouble parsing {}'.format(s))
        return None

