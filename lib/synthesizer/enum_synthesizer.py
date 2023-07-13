"""
main enum synthesizer class
"""

from abc import ABC
from math import inf
from typing import Tuple, List, Optional

from lib.config import pd_print
from lib.interpreter.executor import Executor
from lib.lang.grammar import CFG
from lib.lang.production import IdProduction, Production
from lib.lang.symbol import TerminalSymbol
from lib.program.node import VariableNode, ContextNode
from lib.program.program import Program
from lib.spec.spec import Task
from lib.spec.synthesis_spec import SynthesisTask
from lib.synthesizer.synthesizer import Synthesizer
from lib.utils.synth_utils import compute_naive_score, REDUNDANCY_DICT, PruneCode, expand_non_terminal_with_production_helper, REDUNDANCY_DICT2


class EnumSynthesizer(Synthesizer, ABC):

    def __init__(self, grammar: CFG, executor: Executor, depth: int):
        super().__init__(grammar, executor, depth)

    def synthesize(self, task: Task, **args) -> List[Program]:
        raise NotImplementedError

    def eval_approximated_program(self, task: SynthesisTask, program: Program, use_type_info: bool) -> Tuple[List[bool], List[bool]]:
        self.executor.set_context(task.context)
        over_approx, under_approx = program.get_approximations(use_type_info)
        # print('over and under approx: {} {}'.format(over_approx, under_approx))
        if task.no_context:
            return self.executor.exec_list_of_examples([s.s for s in task.pos_str_context], over_approx), \
                   self.executor.exec_list_of_examples([s.s for s in task.neg_str_context], under_approx)
        else:
            return self.executor.exec_list_of_examples(task.pos_str_context, over_approx), \
                self.executor.exec_list_of_examples(task.neg_str_context, under_approx)

    def check_prune_program(self, task: SynthesisTask, curr_p: Program, curr_best: int, pruning: bool, detailed_error: bool = False, use_type_info: bool = True) \
            -> PruneCode | Tuple[PruneCode, Tuple[List[bool], List[bool]]]:
        """
            return True if curr_p should be pruned
            return False if curr_p should not be pruned
        """
        if not pruning:
            return (PruneCode.FEASIBLE, ([], [])) if detailed_error else PruneCode.FEASIBLE

        # print("checking program: {}".format(curr_p))

        # Does it exceed depth limit?
        if curr_p.get_depth() > self.depth_limit:
            return (PruneCode.DEPTH, ([], [])) if detailed_error else PruneCode.DEPTH

        # If it doesn't have a leaf to expand on, no need to prune it
        if not curr_p.contain_leaf:
            # print("the program has no terminal")
            return (PruneCode.FEASIBLE, ([], [])) if detailed_error else PruneCode.FEASIBLE

        # Check for nested redundancies
        # Iterate over (parent, child) pairs
        for parent in curr_p.nodes:
            if parent in curr_p.to_children:
                for child in curr_p.to_children[parent]:
                    parent_name = curr_p.nodes[parent].name
                    child_name = curr_p.nodes[child].name
                    if (parent_name in REDUNDANCY_DICT and child_name in REDUNDANCY_DICT
                            and REDUNDANCY_DICT[parent_name] == REDUNDANCY_DICT[child_name]) or \
                            (parent_name in REDUNDANCY_DICT2 and child_name in REDUNDANCY_DICT2
                             and REDUNDANCY_DICT2[parent_name] == REDUNDANCY_DICT2[child_name]):
                        return (PruneCode.INFEASIBLE, ([], [])) if detailed_error else PruneCode.INFEASIBLE

        exec_result = self.eval_approximated_program(task, curr_p, use_type_info)
        score_ub = compute_naive_score(exec_result)
        if score_ub < curr_best:
            pd_print("prune {} with score={}".format(curr_p, score_ub))
            return (PruneCode.INFEASIBLE, exec_result) if detailed_error else PruneCode.INFEASIBLE
        else:
            return (PruneCode.FEASIBLE, exec_result) if detailed_error else PruneCode.FEASIBLE

    def instantiate_terminal_sym(self, task: SynthesisTask, curr_p: Program, var_node: VariableNode, var_sym: TerminalSymbol,
                                 prod: Optional[IdProduction],
                                 curr_best: int, pruning: bool, new_values: Optional[list] = None) -> List[Program]:
        raise NotImplementedError

    def expand_additional_terminals(self, task: SynthesisTask, new_prog: Program, new_children_nodes: List[VariableNode], curr_prod: Production,
                                    curr_best: int, pruning: bool, use_type_info: bool) -> List[Program]:
        """
        if the new partial program new_prog have terminals to be expanded, we expand them right-away
        this reduces the amount of iteration on the top-level synthesizer, but not sure if it has any cons being this is a bit greedy
        """
        pd_print('here {}'.format(curr_prod.function_name))
        pd_print('new_prog:', new_prog)
        new_programs = []
        sub_new_programs = [new_prog]

        if curr_prod.function_name == 'RepeatRange':
            # do this in another way if the production is repeat range
            n1_var_node = new_children_nodes[1]
            n2_var_node = new_children_nodes[2]
            n_sym = n1_var_node.sym
            assert isinstance(n_sym, TerminalSymbol)

            # this double for-loop can be optimized further
            for i in range(min(n_sym.values), max(n_sym.values) + 1):
                new_prog1 = new_prog.duplicate(next(self.prog_id_counter))
                tmp_progs = self.instantiate_terminal_sym(task, new_prog1, n1_var_node, n_sym, None, curr_best, pruning, new_values=[i])
                assert len(tmp_progs) == 1
                for j in range(i, max(n_sym.values) + 1):
                    if i == j:
                        continue
                    else:
                        new_prog2 = tmp_progs[0].duplicate(next(self.prog_id_counter))
                        tmp_progs_2 = self.instantiate_terminal_sym(task, new_prog2, n2_var_node, n_sym, None, curr_best, pruning, new_values=[j])
                        assert len(tmp_progs_2) <= 1
                        new_programs.extend(tmp_progs_2)

        elif curr_prod.function_name == 'NumMatch':
            # do this all together if the production is num match
            var_sym = new_children_nodes[0].sym
            assert isinstance(var_sym, TerminalSymbol) and len(var_sym.values) == 1
            var_sym_value = var_sym.values[0]
            n_sym = new_children_nodes[1].sym
            assert isinstance(n_sym, TerminalSymbol)

            # we should enumerate the following by instantiating the following templates:
            num_match_templates = [
                lambda n: (var_sym_value, -inf, '>=', n, '<='),  # this is equivalent to <= n2
                lambda n: (var_sym_value, -inf, '>=', n, '<'),  # this is equivalent to < n2
                lambda n: (var_sym_value, n, '>=', inf, '<='),  # this is equivalent to >= n1
                lambda n: (var_sym_value, n, '>', inf, '<='),  # this is equivalent to > n1
                lambda n1, n2: (var_sym_value, n1, '>=', n2, '<='),  # this is equivalent to n1 <= s <= n2
                lambda n1, n2: (var_sym_value, n1, '>', n2, '<='),  # this is equivalent to n1 < s <= n2 (n1 and n2 cannot be equal)
                lambda n1, n2: (var_sym_value, n1, '>=', n2, '<'),  # this is equivalent to n1 <= s < n2 (n1 and n2 cannot be equal)
                lambda n1, n2: (var_sym_value, n1, '>', n2, '<'),  # this is equivalent to n1 < s < n2 (n1 and n2 cannot be equal)
            ]

            for f in num_match_templates:
                if f.__code__.co_argcount == 1:
                    for n_value in n_sym.values:
                        f_n_value = f(n_value)
                        # here we just need to duplicate one time
                        new_prog1 = new_prog.duplicate(next(self.prog_id_counter))
                        for node, value in zip(new_children_nodes, f_n_value):
                            assert isinstance(node.sym, TerminalSymbol)
                            new_prog1.instantiate_var_node(node, value, node.sym, None)
                        prune_check_res = self.check_prune_program(task, new_prog1, curr_best, pruning, use_type_info=use_type_info)
                        if prune_check_res == PruneCode.FEASIBLE:
                            new_programs.append(new_prog1)

                elif f.__code__.co_argcount == 2:
                    for n1_value in n_sym.values:
                        for n2_value in n_sym.values:
                            f_n_value = f(n1_value, n2_value)
                            if f_n_value[2] == '>=' and f_n_value[4] == '<=':
                                if n1_value > n2_value:
                                    continue
                            else:
                                if n1_value >= n2_value:
                                    continue

                            new_prog1 = new_prog.duplicate(next(self.prog_id_counter))
                            for node, value in zip(new_children_nodes, f_n_value):
                                assert isinstance(node.sym, TerminalSymbol)
                                new_prog1.instantiate_var_node(node, value, node.sym, None)
                            prune_check_res = self.check_prune_program(task, new_prog1, curr_best, pruning, use_type_info=use_type_info)
                            if prune_check_res == PruneCode.FEASIBLE:
                                new_programs.append(new_prog1)

                else:
                    raise RuntimeError('Should not reach this branch for f with argument {}'.format(f.__code__.co_varnames))

        elif curr_prod.function_name in ['isYear', 'isMonth', 'isDate', 'btwHour', 'btwMin', 'btwSec']:
            var_sym = new_children_nodes[0].sym  # First arg is a variable
            n_sym = new_children_nodes[1].sym  # Second and third args are the parameters

            assert isinstance(var_sym, TerminalSymbol)
            assert isinstance(n_sym, TerminalSymbol)

            # Instantiate the variable node first
            curr_programs = self.instantiate_terminal_sym(
                task,
                new_prog.duplicate(next(self.prog_id_counter)),
                new_children_nodes[0], var_sym, None,
                curr_best, pruning
            )

            n_values = n_sym.values
            for i in n_values:
                intermediate_progs = [prog.duplicate(next(self.prog_id_counter)) for prog in curr_programs]
                for prog in intermediate_progs:
                    #  Instantiate the first parameter node
                    prog.instantiate_var_node(new_children_nodes[1], i, n_sym)
                for j in n_values:
                    if i <= j:
                        for prog in intermediate_progs:
                            prog_to_add = prog.duplicate(next(self.prog_id_counter))
                            # Instantiate final parameter
                            prog_to_add.instantiate_var_node(new_children_nodes[2], j, n_sym)
                            new_programs.append(prog_to_add)

        elif curr_prod.function_name in ['inCountry', 'inRegion', 'inState']:
            # pd_print('here2')

            possible_terminals = task.context[curr_prod.function_name]  # Gives relevant countries/regions
            pd_print('possible_terminals: {}'.format(possible_terminals))
            var_sym = new_children_nodes[0].sym
            place_sym = new_children_nodes[1].sym
            assert isinstance(var_sym, TerminalSymbol)
            assert isinstance(place_sym, TerminalSymbol)

            curr_programs = self.instantiate_terminal_sym(
                task,
                new_prog.duplicate(next(self.prog_id_counter)),
                new_children_nodes[0], var_sym, None,
                curr_best, pruning
            )

            for prog in curr_programs:
                for terminal in possible_terminals:
                    prog_to_add = prog.duplicate(next(self.prog_id_counter))
                    prog_to_add.instantiate_var_node(new_children_nodes[1], terminal, place_sym)
                    # print('prog_to_add:', prog_to_add)
                    new_programs.append(prog_to_add)
        elif curr_prod.function_name == 'MatchQuery':
            raise ValueError('Synthesis of MatchQuery construct should not be handled inside this function!')
        else:
            for child_node in new_children_nodes:
                child_node_sym = child_node.sym
                if isinstance(child_node_sym, TerminalSymbol):
                    sub_newly_instantiated_programs = []
                    for prog in sub_new_programs:
                        prog2 = self.instantiate_terminal_sym(task, prog, child_node, child_node_sym, None, curr_best, pruning)
                        sub_newly_instantiated_programs.extend(prog2)
                    sub_new_programs = sub_newly_instantiated_programs

            new_programs.extend(sub_new_programs)

        return new_programs

    def expand_match_query(self, task: SynthesisTask, new_prog: Program, new_children_nodes: List[VariableNode], curr_best: int, pruning: bool, use_type_info: bool) -> List[Program]:

        new_programs_tmp = []
        new_programs = []

        for child_node in new_children_nodes:
            child_node_sym = child_node.sym
            if child_node_sym.name == 'IN_CONTEXT':
                # To synthesize matchQuery, sometimes we need to involve in-context examples by utilizing current
                # positive and negative examples

                # this basically requires special instantiation of terminals
                # we should try to involve as few of the in-context examples as possible

                # how it work is:
                # try with no in-context example, see if it can work on the rest
                # if it is all good, we return that as the final program; otherwise, add the first example that fails
                # NOTE that this does not always work since the construct itself has nondeterministic semantic
                #   But ignore this issue for now.
                # TODO: this function needs to be modified given now our program run on cached results
                for prog in new_programs_tmp:

                    prog_tmp = prog.duplicate(next(self.prog_id_counter))
                    # first instantiate the var node
                    new_node = prog_tmp.instantiate_var_node(child_node, 'context', child_node_sym, None)
                    assert isinstance(new_node, ContextNode)

                    while True:
                        check_prune_res, exec_result = self.check_prune_program(task, prog_tmp, curr_best, pruning, detailed_error=True, use_type_info=False)
                        pd_print("exec_result:", exec_result)

                        if check_prune_res == PruneCode.FEASIBLE:
                            new_programs.append(prog_tmp)
                            break
                        else:
                            # find the first example that does not work and add it to the
                            # TODO: since that we don't know the ground truth here (do we?),
                            #   so this implementation might be problematic
                            #   We only know the ground truth when the top-level goal is a GPT3 type
                            #   The an updated implementation would depend on the top-level goal type
                            example_assigned = False
                            for idx, pos_res in enumerate(exec_result[0]):
                                curr_example_str = task.pos_str_context[idx].s
                                if not pos_res and curr_example_str not in new_node.value[0]:
                                    new_node.value[0].append(curr_example_str)
                                    example_assigned = True
                                    break

                            if not example_assigned:
                                for idx, neg_res in enumerate(exec_result[1]):
                                    curr_example_str = task.neg_str_context[idx].s
                                    if neg_res and curr_example_str not in new_node.value[1]:
                                        new_node.value[1].append(curr_example_str)
                                        example_assigned = True
                                        break
                            else:
                                continue

                            if example_assigned:
                                # we check if the current in-context is good enough in the next iteration
                                continue
                            else:
                                # this means we did not add any in-context examples. Time to terminate the loop
                                break
            elif child_node_sym.name == 't_query':
                # This is the first node to be expanded
                assert len(new_programs_tmp) == 0
                prod = self.grammar.get_productions(child_node.sym)[0]
                assert isinstance(prod, IdProduction)
                new_programs_tmp.extend(self.instantiate_id_production(task, new_prog, child_node, prod, curr_best, pruning))

            elif child_node_sym.name == 'f':
                # This is the second node to be expanded
                prev_programs = new_programs_tmp
                new_programs_tmp = []
                for prog_tmp_2 in prev_programs:
                    productions = self.grammar.get_productions(child_node.sym)
                    for prod in productions:
                        if isinstance(prod, IdProduction):
                            new_programs_tmp.extend(self.instantiate_id_production(task, prog_tmp_2, child_node, prod, curr_best, pruning))
                        else:
                            new_prog1, new_children_nodes1 = expand_non_terminal_with_production_helper(prog_tmp_2, next(self.prog_id_counter), child_node, prod)
                            progs = self.expand_additional_terminals(task, new_prog1, new_children_nodes1, prod, curr_best, pruning, use_type_info=use_type_info)
                            new_programs_tmp.extend(progs)
            else:
                # we are going to do some hack here
                raise ValueError('{} should not show up as a child node of MatchQuery'.format(child_node_sym.name))

        return new_programs

    def instantiate_id_production(self, task: SynthesisTask, curr_p: Program, var_node: VariableNode, prod: IdProduction, curr_best_score: int, pruning: bool) -> List[Program]:
        assert len(prod.arg_syms) == 1
        arg_sym = prod.arg_syms[0]
        assert isinstance(arg_sym, TerminalSymbol)
        # ToXXX should never show up in an actual program (just feels inefficient)
        new_prog = curr_p.duplicate(next(self.prog_id_counter))
        progs = self.instantiate_terminal_sym(task, new_prog, var_node, arg_sym, prod, curr_best_score, pruning)

        return progs
