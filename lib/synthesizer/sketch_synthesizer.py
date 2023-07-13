from typing import List, Dict

from lib.config import pd_print
from lib.interpreter.executor import Executor
from lib.lang.grammar import CFG
from lib.program.program import Program
from lib.spec.synthesis_spec import SynthesisTask
from lib.synthesizer.decompose import decompose_goal, format_decomposed_results, Goal
from lib.synthesizer.synthesizer import Synthesizer
from lib.synthesizer.top_down_synthesizer import TopDownEnumSynthesizer
from lib.synthesizer.typed_synthesizer import TypedSynthesizer
from lib.type.base_type import BaseType
from lib.type.type_system import get_type, add_base_type
from lib.utils.pq import PriorityQueue
from lib.utils.synth_utils import compute_naive_score, pq_ranking_func, pq_ranking_func_no_type_system
import numpy as np


class SketchSynthesizer(Synthesizer):

    def __init__(self, grammar: CFG, executor: Executor, depth: int, no_type: bool, no_type_system: bool, no_decomp: bool):
        super().__init__(grammar, executor, depth)
        self.no_type = no_type
        self.no_type_system = no_type_system
        self.no_decomp = no_decomp
        if self.no_type_system:
            self.typed_synthesizer = TopDownEnumSynthesizer(grammar, executor, depth, -1)
        else:
            self.typed_synthesizer = TypedSynthesizer(grammar, executor, depth, -1, no_type, no_decomp)

    def init_program(self) -> Program:
        raise NotImplementedError

    def synthesize_no_decomp(self, task: SynthesisTask) -> List[Program]:

        sketch = task.sketch
        program = sketch.program.duplicate(sketch.program.id)
        self.typed_synthesizer.curr_nid = next(sketch.program.node_id_counter)

        # for each hole node in the program, we change it to a variable node, annotated with the type of the hole
        for hid, ty in sketch.hole_id_to_type.items():
            goal_type_object, to_be_added_type_name = get_type(ty)
            if goal_type_object is None:
                add_base_type(to_be_added_type_name, ('gpt3type',))
                goal_type_object, _ = get_type(ty)
            assert isinstance(goal_type_object, BaseType)
            program.instantiate_hole_node(hid, goal_type_object, self.grammar.get_nonterminal_sym('r'))

        worklist = PriorityQueue(priority=pq_ranking_func)
        worklist.put(program)

        programs = self.typed_synthesizer.synthesize(task, worklist)
        return programs

    def synthesize(self, task: SynthesisTask) -> List[Program]:

        sketch = task.sketch
        if self.no_type:
            sketch.no_type = True
        self.typed_synthesizer.curr_nid = next(sketch.program.node_id_counter)

        hole_id_list = sketch.hole_id_to_type.keys()
        num_holes = len(hole_id_list)

        # Priority queue for typed synthesizer of each hole
        if self.no_type_system:
            priority_queues = [PriorityQueue(priority=pq_ranking_func_no_type_system) for _ in range(num_holes)]
        else:
            priority_queues = [PriorityQueue(priority=pq_ranking_func) for _ in range(num_holes)]

        # Now, we get the negative examples
        decomposed_neg_result = decompose_goal(sketch, task.neg_str_context, False)
        pd_print('decomposed neg result: {}'.format(decomposed_neg_result))
        if len(decomposed_neg_result) == 0:
            sketch.can_reject_all_negative_example = True
            pd_print('can reject all negative examples')

        # first decompose the sketch based on positives
        decomposed_pos_result = decompose_goal(sketch, task.pos_str_context, True)
        all_positive_goal = Goal(self.depth_limit)
        format_decomposed_results(all_positive_goal, decomposed_pos_result, True)

        # get subtask list
        # subtask_list[i] is a task containing all the positive examples associated with the ith hole
        # This is our storage system for the positive examples: a list of subtasks, one for each hole.
        subtask_list = [
            task.get_sub_task(
                sketch.hole_id_to_type[hole_id],
                [ex.string for ex in all_positive_goal.get_sub_goal(hole_id).positive_examples], [], self.no_type
            ) for hole_id in hole_id_list
        ]

        # print('subtask_list: ', subtask_list)

        num_negative_examples = 0

        # Each entry of decomposed_negative_examples is the decomposition of one of the negative examples
        # into a hashmap from hole_id to the substring (str) of the negative example corresponding to that
        # hole. This is our storage system for the negative examples.
        decomposed_negative_examples = []

        if len(decomposed_neg_result) != 0:
            num_negative_examples = len(list(decomposed_neg_result.values())[0])
            decomposed_negative_examples = [{} for _ in range(num_negative_examples)]
            for hole_id, neg_example_match_res_list in decomposed_neg_result.items():
                # neg_example_match_res_list is a list of MatchResult objects, the ith one corresponding
                # to the ith negative example.

                for neg_example_dict, neg_example_match_res in zip(decomposed_negative_examples,
                                                                   neg_example_match_res_list):
                    neg_example_dict[hole_id] = neg_example_match_res.string

        # print('decomposed_negative_examples: ', decomposed_negative_examples)

        if num_negative_examples == 0:
            # We need to generate this program on just positive examples
            for hole_id, subtask, worklist in zip(hole_id_list, subtask_list, priority_queues):
                generated_subprogs = self.typed_synthesizer.synthesize(subtask, worklist)
                if len(generated_subprogs) == 0:
                    return []
                sketch.instantiate_hole_with_program(hole_id, generated_subprogs)

            return [sketch.compose_program(next(self.prog_id_counter))]

        # We need to test whether there's a positive/negative conflict
        # If there is, just return immediately
        for curr_decomp_neg_example in decomposed_negative_examples:
            all_holes_collide = True
            for pos_subtask, matched_substr in zip(subtask_list, curr_decomp_neg_example.values()):
                curr_pos_matches = [ex.s for ex in pos_subtask.pos_str_context]
                if matched_substr.s not in curr_pos_matches:
                    all_holes_collide = False
                    break

            if all_holes_collide:  # Unresolvable collision
                return []

        # To keep track of the subprograms synthesized for each hole, we need two structures
        # The first is a list of hashmaps, where the ith hashmap holds the synthesized subprograms
        # for hole i
        curr_synth_subprograms = [{} for _ in range(num_holes)]

        # The second is a special data structure that quickly computes whether a full program is
        # synthesizable based upon the current subprograms.
        subprogram_tracker = SketchSynthesisSubprogramTracker(num_holes, num_negative_examples)

        # We want to initialize each priority queue
        for i, (hole_id, subtask, worklist) in enumerate(zip(hole_id_list, subtask_list, priority_queues)):
            # print('subtask*', subtask)
            # Max iterations: 0
            # We don't want to synthesize, we're just initializing the worklist
            programs = self.typed_synthesizer.synthesize(subtask, worklist, 0)
            # print('subtask* end ')

            if len(programs) == 0:
                continue  # To next hole
            else:
                curr_subprogram = programs[0]  # This should only return 1 program

                bitstring = self._get_negative_example_rejection_bitstring(
                    curr_subprogram, hole_id, decomposed_negative_examples, task.no_context
                )
                # print('bitstring: ', bitstring)
                if subprogram_tracker.add_subprogram(i, bitstring):
                    # If adding the subprogram to the tracker returned true, then this is a new
                    # program
                    curr_synth_subprograms[i][bitstring] = curr_subprogram
                    # print('curr_synth_subprograms: ', curr_synth_subprograms)

        #  Now, we iteratively synthesize

        # all_empty is set to True if all worklists are empty. If this is the case, stop iteration
        all_empty = False

        # new_prog_generated is set to true if the previous iteration of the synthesis loop generated
        # a new program for at least one of the holes. If new_prog_generated is false, there is no need
        # to call subprogram_tracker.has_correct_program() to see if we can synthesize a correct program
        # (more efficient this way).
        new_prog_generated = True

        while not all_empty and (not new_prog_generated or not subprogram_tracker.has_correct_program()):
            all_empty = True
            new_prog_generated = False

            # For each hole, restart synthesis
            for i, (hole_id, subtask, worklist) in enumerate(zip(hole_id_list, subtask_list, priority_queues)):
                # print('hole_id: ', hole_id)
                # print('subtask: ', subtask)

                if worklist.is_empty():
                    continue

                all_empty = False
                #  We cap max iterations at 100
                programs = self.typed_synthesizer.synthesize(subtask, worklist, 100)

                # print("HERE PROGRAMS: ", programs)
                if len(programs) == 0:
                    continue  # To next hole

                curr_subprogram = programs[0]  # This should only return 1 program
                bitstring = self._get_negative_example_rejection_bitstring(
                    curr_subprogram, hole_id, decomposed_negative_examples, task.no_context
                )
                # print('bitstring: ', bitstring)
                if subprogram_tracker.add_subprogram(i, bitstring):
                    # If adding the subprogram to the tracker returned true, then this is a new
                    # program
                    new_prog_generated = True
                    curr_synth_subprograms[i][bitstring] = curr_subprogram
                    # print('curr_synth_subprograms: ', curr_synth_subprograms)

        if not subprogram_tracker.has_correct_program():
            # print('No correct program found')
            return []

        # Now, we synthesize full programs from all the subprograms in curr_synth_subprograms
        full_programs = {0: []}
        for subprograms in curr_synth_subprograms:
            # Combine every program in full_programs with every program in subprograms
            curr_full_programs = {}
            for bitstring1 in full_programs:
                for bitstring2 in subprograms:
                    curr_full_programs[bitstring1 | bitstring2] = (full_programs[bitstring1][::]
                                                                   + [subprograms[bitstring2]])
            full_programs = curr_full_programs
            # print('full_programs: ', full_programs)

        final_subprog_list = full_programs[(1 << num_negative_examples) - 1]
        # print('final_subprog_list: ', final_subprog_list)

        for hole_id, subprog in zip(hole_id_list, final_subprog_list):
            sketch.instantiate_hole_with_program(hole_id, [subprog])

        # We should check that this synthesized program actually does satisfy everything
        pd_print("CHECKING IF ALGO FAILED")
        if compute_naive_score(self.executor.exec_task(task, sketch.to_pattern())) < 1:
            pd_print("ALGO FAILED")  # This is probably not good, something likely went wrong
            # raise Exception("Algorithm failed")
            return []
        return [sketch.compose_program(next(self.prog_id_counter))]

    def _get_negative_example_rejection_bitstring(self,
                                                  subprogram: Program,
                                                  hole_id: int,
                                                  decomposed_negative_examples: List[Dict[int, str]],
                                                  no_context: bool) -> int:
        """
        Treating the given subprogram as a candidate program for filling the specified hole,
        it runs the subprogram against the specified hole in each of the decomposed negative examples.
        It returns a bitstring of the results: if the program rejects the value in the hole of the ith
        given example, the ith bit in the bitstring would be set.
        """

        bitstring = 0
        hole_vals = [decomposed_example[hole_id] for decomposed_example in decomposed_negative_examples]
        result = self.executor.exec_list_of_examples(hole_vals, subprogram.to_pattern(), no_context)
        for i, res in enumerate(result):
            bitstring |= ((not res) << i)
        return bitstring


def _is_sub_bitstring(first_bitstring: int, second_bitstring: int) -> bool:
    """
    Return True if the first bitstring is a submask of the second, False otherwise
    """
    return (first_bitstring & second_bitstring) == first_bitstring


def _merge_synthesis_states(first_state: np.array, second_state: np.array,
                            start_ind: int, state_length: int, res: np.array):
    """
    Merges the synthesis states in first_state and second_state. Puts the result in res,
    starting at position start_ind (that is, res[start_ind] is the first element of the result).
    Function is recursive (on the parameter state_length). state_length must be power of 2 and > 2.
    """
    if state_length == 2:
        # Base case
        res[start_ind] = first_state[0] * second_state[0]
        res[start_ind + 1] = ((first_state[0] + first_state[1]) * (second_state[0] + second_state[1])
                              - res[start_ind])
    else:
        half_state_length = state_length >> 1
        _merge_synthesis_states(first_state, second_state, start_ind, half_state_length, res)  # leading bit 0
        first_state[0:half_state_length] += first_state[half_state_length:state_length]  # leading bit 1
        second_state[0:half_state_length] += second_state[half_state_length:state_length]
        _merge_synthesis_states(first_state, second_state,
                                start_ind + half_state_length, half_state_length, res)
        first_state[0:half_state_length] -= first_state[half_state_length:state_length]
        second_state[0:half_state_length] -= second_state[half_state_length:state_length]
        res[start_ind + half_state_length:start_ind + state_length] -= res[start_ind:start_ind + half_state_length]


class SketchSynthesisSubprogramTracker:
    def __init__(self, num_holes: int, num_negative_examples: int):
        self.num_holes = num_holes
        self.num_negative_examples = num_negative_examples
        self.num_bitstrings = 1 << num_negative_examples
        self.existing_subprograms = np.zeros((num_holes, self.num_bitstrings), dtype=np.uint32)

    def add_subprogram(self, hole_num: int, bitstring: int) -> bool:
        """
        Process a new subprogram for the specified hole number with the specified
        negative example rejection bitstring. Returns true if the subprogram was
        successfully added. Returns false if a program with this bitstring (or a
        superset of the bitstring) already exists for the given hole.
        """
        if self.existing_subprograms[hole_num][bitstring]:
            return False
        subprogs_for_hole = self.existing_subprograms[hole_num]
        for i in range(bitstring + 1):
            if _is_sub_bitstring(i, bitstring) and not subprogs_for_hole[i]:
                subprogs_for_hole[i] = 1
        return True

    def has_correct_program(self) -> bool:
        """
        Return True if it is possible to synthesize a full program from all the subprograms
        processed by the "add_subprogram" function that rejects all negative examples
        """

        # synthesis_state[bitstring] is 0 if no program so far can reject all negative examples
        # corresponding to the bitstring.
        synthesis_state = self.existing_subprograms[0]
        for i in range(1, self.num_holes):
            new_synthesis_state = np.zeros(self.num_bitstrings)
            # Next line combines synthesis_state and the subprograms from the current hole
            # and stores the result into new_synthesis_state
            _merge_synthesis_states(synthesis_state, self.existing_subprograms[i],
                                    0, self.num_bitstrings, new_synthesis_state)
            synthesis_state = new_synthesis_state
            synthesis_state = np.clip(synthesis_state, 0, 1)  # Prevent overflow

        return synthesis_state[self.num_bitstrings - 1] == 1
