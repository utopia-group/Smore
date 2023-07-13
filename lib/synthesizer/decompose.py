"""
Given a sketch, a set of positive and negative examples, figure out how to decompose the sketch into smaller tasks
"""
import re
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

from lib.config import pd_print
from lib.interpreter.context import StrContext
from lib.program.sketch import Sketch
from lib.utils.exceptions import NoPositiveMatchException


class MatchResult:
    def __init__(self, hole_id: int, hole_type: str, string: StrContext, span: Tuple[int, int]):
        self.hole_id: int = hole_id
        self.hole_type: str = hole_type
        self.string: StrContext = string
        self.span: Tuple[int, int] = span

    def __repr__(self):
        return 'MatchRes(hold_id={}, hole_type={}, string={}, span={})'.format(self.hole_id, self.hole_type, self.string, self.span)


class SubGoal:
    """
    A single goal
    """

    def __init__(self, hole_id: int):
        self.hole_id = hole_id
        self.positive_examples: List[MatchResult] = []
        self.negative_examples: List[MatchResult] = []

    def duplicate(self) -> 'SubGoal':
        new_goal = SubGoal(self.hole_id)
        new_goal.positive_examples = self.positive_examples.copy()
        new_goal.negative_examples = self.negative_examples.copy()

        return new_goal

    def add_positive_examples(self, pe: List[MatchResult]):
        self.positive_examples.extend(pe)

    def add_negative_examples(self, ne: List[MatchResult]):
        self.negative_examples.extend(ne)

    def __repr__(self):
        return "({},{})".format(str([pe.string for pe in self.positive_examples]), str([ne.string for ne in self.negative_examples]))


class Goal:
    def __init__(self, depth: int):
        self.hid_to_sub_goals: Dict[int, SubGoal] = {}
        self.depth = depth

    def duplicate(self) -> 'Goal':
        new_goal = Goal(self.depth)
        for hole_id, sub_goal in self.get_sub_goal_iterator():
            new_goal.add_sub_goal(hole_id, sub_goal.duplicate())

        return new_goal

    def add_sub_goal(self, hid: int, sub_goal: SubGoal):
        self.hid_to_sub_goals[hid] = sub_goal

    def get_sub_goal(self, hid: int) -> Optional[SubGoal]:
        return self.hid_to_sub_goals.get(hid)

    def get_sub_goal_iterator(self) -> List:
        return list(self.hid_to_sub_goals.items())

    def feasible(self) -> bool:
        for sub_goal in self.hid_to_sub_goals.values():
            pos_examples = set([s.string.s for s in sub_goal.positive_examples])
            neg_examples = set([s.string.s for s in sub_goal.negative_examples])
            if len(pos_examples.intersection(neg_examples)) > 0:
                # raise GoalInfeasibleException
                return False
        return True

    def __repr__(self):
        return '{}_{}'.format(self.hid_to_sub_goals, self.depth)


def format_decomposed_results(goal: Goal, decomposed_res: Dict[int, List[MatchResult]], positive: bool):
    for hole_id, matched_results in decomposed_res.items():

        if goal.get_sub_goal(hole_id) is None:
            goal.add_sub_goal(hole_id, SubGoal(hole_id))

        if positive:
            goal.get_sub_goal(hole_id).add_positive_examples(matched_results)
        else:
            goal.get_sub_goal(hole_id).add_negative_examples(matched_results)

    return goal


# TODO: This implementation assume that there only exists one way to match the sub-groups (which in general is not True)
def match_util(regex1: str, context: StrContext, hole_id_to_hole_type: Dict[int, str], positive_only, sketch_object) -> Optional[Dict[int, MatchResult]]:
    # print("regex1:", regex1)
    regex = '^{}$'.format(regex1)
    match_res = re.match(regex, context.s)

    if match_res is None:
        if positive_only:
            raise NoPositiveMatchException(regex1, context, sketch_object, 'Match should be at least successful for positive strings')
        else:
            return None

    hole_to_match_results = {}
    for hole_id, hole_type in hole_id_to_hole_type.items():
        if 'Optional' in hole_type:
            group_name = '{}{}'.format(hole_type[9:-1], hole_id)
        else:
            group_name = '{}{}'.format(hole_type, hole_id)

        if len(group_name.split()) == 1:
            pass
        else:
            group_name = ''.join(group_name.split())
        # print("group_name:", group_name)

        if match_res.group(group_name) is None:
            ms = MatchResult(hole_id=hole_id, hole_type=hole_type, string=StrContext('', None), span=match_res.span(group_name))
        else:
            ms = MatchResult(hole_id=hole_id, hole_type=hole_type, string=context.create_sub_context(match_res.span(group_name)), span=match_res.span(group_name))
        hole_to_match_results[hole_id] = ms

    return hole_to_match_results


def decompose_goal(sketch: Sketch, examples: List[StrContext], positive_only: bool) -> Optional[Dict[int, List[MatchResult]]]:

    hole_to_match_res_all_examples = defaultdict(list)
    update_hole_optional = []
    for example in examples:
        pd_print('example:', example)
        sketch_overapprox = sketch.get_overapprox(dict([(ty, [example.get_substr_regex(span) for span in str_spans]) for ty, str_spans in example.type_to_str_spans.items()]))

        pd_print("sketch_overapprox: {}".format(sketch_overapprox))
        pd_print('decomposing {} with sketch {}'.format(example, sketch_overapprox))

        hole_to_match_res = match_util(sketch_overapprox, example, sketch.hole_id_to_type, positive_only, sketch)
        pd_print('hole_to_match_res: {}'.format(hole_to_match_res))

        if hole_to_match_res is not None:
            for key, value in hole_to_match_res.items():
                if value.string.s == '':
                    if positive_only:
                        # need to update the goal type to optional because it can match an empty string
                        update_hole_optional.append(key)
                hole_to_match_res_all_examples[key].append(value)

    if len(update_hole_optional) > 0:
        for hid in update_hole_optional:
            if 'optional' not in sketch.hole_id_to_type[hid].lower():
                sketch.hole_id_to_type[hid] = 'Optional({})'.format(sketch.hole_id_to_type[hid])

    return hole_to_match_res_all_examples
