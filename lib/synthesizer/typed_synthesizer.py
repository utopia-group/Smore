import warnings
from typing import List, Optional, Tuple

from lib.config import pd_print
from lib.interpreter.executor import Executor
from lib.lang.grammar import CFG
from lib.lang.production import IdProduction
from lib.lang.symbol import TerminalSymbol
from lib.program.node import VariableNode
from lib.program.program import Program
from lib.spec.synthesis_spec import SynthesisTask
from lib.synthesizer.enum_synthesizer import EnumSynthesizer
from lib.type import base_type
from lib.type.base_type import Any, BaseType, GPT3Type
from lib.type.type_enum import parsable_types, entity_types, extensible_types
from lib.type.type_inference import derive_type
from lib.type.type_system import get_type, add_base_type, get_subtypes
from lib.utils.pq import PriorityQueue
from lib.utils.synth_utils import compute_naive_score, expand_non_terminal_with_production_helper, assign_type_to_arguments, pq_ranking_func, PruneCode
from lib.utils.task_utils import add_place_predicate_context


class TypedSynthesizer(EnumSynthesizer):
    """
    typed synthesizer.
    the type system is provided in type.type_system
    """

    def __init__(self, grammar: CFG, executor: Executor, depth: int, start_nid: int, no_type: bool, no_decomp: bool):
        super().__init__(grammar, executor, depth)
        self.curr_nid: int = start_nid
        self.no_type = no_type
        self.no_decomp = no_decomp

    def init_program(self, goal_type) -> Program:
        curr_p = Program(next(self.prog_id_counter), self.curr_nid)
        new_node = curr_p.add_variable_node(self.grammar.start_sym)
        curr_p.set_start_node(new_node)
        curr_p.nid_to_goal_type[new_node.id] = goal_type

        return curr_p

    def synthesize(self, task: SynthesisTask, worklist: Optional[PriorityQueue] = None,
                   max_iterations: int = 100000) -> List[Program]:
        """
        synthesis procedure works as the following:
            a work-list algorithm, in each iteration:
            -> get the goal type
            -> find valid productions
            -> create new partial programs
            -> infer the goal-type for the variables in the partial programs

        let's just return 1 single program for now
        """

        if all([s.s == '' for s in task.pos_str_context]) and all([s.s == '' for s in task.neg_str_context]):
            any_star_prog = Program(next(self.prog_id_counter), self.curr_nid)
            start_node = any_star_prog.add_nonterminal_node(name='star', symbol=self.grammar.get_nonterminal_sym('r'), prod=self.grammar.name_to_prod['StarR'])
            any_star_prog.set_start_node(start_node)
            any_star_prog.set_children(start_node, [any_star_prog.add_terminal_node(value='any', symbol=self.grammar.get_terminal_sym('CC'), prod=self.grammar.name_to_prod['ToCC'])])
            return [any_star_prog]

        # Should we filter out empty string here?
        new_pos_str_context = []
        for s in task.pos_str_context:
            if s.s.strip() != '':
                new_pos_str_context.append(s)
        task.pos_str_context = new_pos_str_context

        # The function below finds all countries and regions in the grammar, identifies the ones
        # relevant to the task, and adds them to the task context dictionary. This helps with date
        # predicate synthesis.
        add_place_predicate_context(task, self.grammar)
        # print('task.context:', task.context)

        all_synthesized_prog = []

        assert not self.curr_nid == -1

        self.instantiate_terminals_in_grammar(task.pos_str_context, task.neg_str_context)

        if worklist is None:
            worklist = PriorityQueue(priority=pq_ranking_func)

        if worklist.is_empty():
            goal_type_object, to_be_added_type_name = get_type(task.goal_type)
            if goal_type_object is None:
                add_base_type(to_be_added_type_name, ('gpt3type',))
                goal_type_object, _ = get_type(task.goal_type)
            assert goal_type_object is not None
            worklist.put(self.init_program(goal_type_object))

        curr_best_score = 1
        num_iterations = 0
        while not worklist.is_empty() and num_iterations < max_iterations:
            curr_p = worklist.pop()
            num_iterations += 1
            assert isinstance(curr_p, Program)

            pd_print("current prog: {}".format(curr_p))

            if curr_p.is_concrete():
                eval_res = self.eval_program(task, curr_p)
                eval_score = compute_naive_score(eval_res)

                pd_print("current_best_score:", curr_best_score)
                pd_print('eval_score:', eval_score)

                if eval_score == curr_best_score:
                    all_synthesized_prog.append(curr_p)

                    if len(all_synthesized_prog) >= 1:
                        return all_synthesized_prog
                else:
                    continue
            else:
                var_node = curr_p.select_var_node()
                new_programs = self.expand(task, curr_p, var_node, curr_best_score)
                worklist.put_all(new_programs)

        return all_synthesized_prog

    def expand(self, task: SynthesisTask, curr_p: Program, var_node: VariableNode, curr_best_score: int) -> List[Program]:
        """
        type-system-guided expansion
        """

        new_programs: List[Program] = []

        productions = self.grammar.get_productions(var_node.sym)
        for prod in productions:
            pd_print('prod:', prod)
            if isinstance(prod, IdProduction):
                new_programs.extend(self.instantiate_id_production(task, curr_p, var_node, prod, curr_best_score, True))
            else:
                # need to check if using this production can lead to any type-check program
                inferred_arg_type = derive_type(prod, curr_p.nid_to_goal_type[var_node.id])
                pd_print('inferred_arg_type: {}'.format(inferred_arg_type))

                if inferred_arg_type is None:
                    # this means this is an invalid production (according to our type system)
                    pd_print("inferred type is none {}".format(curr_p.nid_to_goal_type[var_node.id]))
                    continue
                elif isinstance(inferred_arg_type, BaseType) and inferred_arg_type.name == 'string' and prod.function_name in ['MatchQuery', 'MatchType', 'MatchEntity']:
                    new_prog, new_children_nodes = expand_non_terminal_with_production_helper(curr_p, next(self.prog_id_counter), var_node, prod)
                    # this branch handles all the special cases when there is no goal type
                    if prod.function_name == 'MatchQuery':
                        # we need to synthesize a type on the fly
                        inferred_new_type = self.executor.nlp_engine.infer_query([ctx.s for ctx in task.pos_str_context], [ctx.s for ctx in task.neg_str_context])
                        # I guess we need to create this new type as well
                        inferred_arg_type = add_base_type(inferred_new_type, ('gpt3type',))
                        new_prog = assign_type_to_arguments(new_prog.duplicate(next(self.prog_id_counter)), new_children_nodes, inferred_arg_type)
                        new_programs.extend(self.expand_match_query(task, new_prog, new_children_nodes, curr_best_score, True, use_type_info=True))
                    elif prod.function_name == 'MatchType' or prod.function_name == 'MatchEntity':
                        if prod.function_name == 'MatchType':
                            candidate_types = [get_type(ty)[0] for ty in parsable_types if get_type(ty)[0] is not None]
                        else:
                            candidate_types = [get_type(ty)[0] for ty in entity_types if get_type(ty)[0] is not None]
                        for ty in candidate_types:
                            new_prog = assign_type_to_arguments(new_prog.duplicate(next(self.prog_id_counter)), new_children_nodes, ty)
                            progs = self.expand_additional_terminals(task, new_prog, new_children_nodes, prod, curr_best_score, True, use_type_info=True)
                            # print('progs:', progs)
                            new_programs.extend(progs)
                    else:
                        raise NotImplementedError('unknown function name: {}'.format(prod.function_name))

                else:
                    # expand the program and assign the appropriate type to the children nodes
                    new_prog, new_children_nodes = expand_non_terminal_with_production_helper(curr_p, next(self.prog_id_counter), var_node, prod)
                    # print("new prog2:", new_prog)
                    if prod.function_name == 'MatchQuery':
                        # we should get all the subtypes of the current types
                        assert isinstance(inferred_arg_type, BaseType)
                        assert inferred_arg_type.name != 'string'
                        # print('inferred_arg_type:', inferred_arg_type)
                        candidate_types = get_subtypes(inferred_arg_type)

                        for ty in candidate_types:
                            # print('candidate_types', ty)
                            new_prog = assign_type_to_arguments(new_prog.duplicate(next(self.prog_id_counter)), new_children_nodes, ty)
                            # print(new_prog)
                            # here we implement the special logic for MatchQuery because we might need to add additional in-context examples
                            new_programs.extend(self.expand_match_query(task, new_prog, new_children_nodes, curr_best_score, True, use_type_info=True))
                    else:
                        # check pruning results and do necessary things depending on whether terminal nodes exists
                        new_prog = assign_type_to_arguments(new_prog, new_children_nodes, inferred_arg_type)
                        check_prune_res = self.check_prune_program(task, new_prog, curr_best_score, True, use_type_info=True)
                        pd_print(check_prune_res)
                        if check_prune_res == PruneCode.FEASIBLE:
                            # if the new_prog contains terminal symbol, then we also expand it
                            progs = self.expand_additional_terminals(task, new_prog, new_children_nodes, prod, curr_best_score, True, use_type_info=True)
                            # print('progs:', progs)
                            new_programs.extend(progs)
                        else:
                            continue

        return new_programs

    def instantiate_terminal_sym(self, task: SynthesisTask, curr_p: Program, var_node: VariableNode, var_sym: TerminalSymbol, prod: Optional[IdProduction],
                                 curr_best: int, pruning: bool, new_values: Optional[list] = None) -> List[Program]:
        """
        Instantiate terminal sym in a type-directed way
        """

        # print('instantiate_terminal_sym')

        new_programs = []
        goal_type = curr_p.nid_to_goal_type[var_node.id]

        # uses goal_type to filter the possible values
        values = []
        if isinstance(goal_type, Any):

            # print(var_node.sym.name)
            # print(var_sym.values)
            values = var_sym.values if new_values is None else new_values
            # print('values:', values)
        elif isinstance(goal_type, BaseType):
            # the following if statements make sure the terminal symbol type-checks
            if goal_type.name == 'string':
                if var_sym.name == 'CC' or var_sym.name == 'CONST1' or var_sym.name == 'CONST2':
                    values = var_sym.values if new_values is None else new_values
                elif var_sym.name == 'TYPE':
                    # this values variable is a list of tuple other than a list of string
                    values = list(set(parsable_types.items()))
                elif var_sym.name == 'ENT':
                    values = list(set(entity_types.values()))
                elif var_sym.name == 'ONT' or var_sym.name == 'BOOL':
                    values = list(var_sym.values)
                elif var_sym.name == 'QUERY':
                    values = [goal_type.name]
                else:
                    warnings.warn('Either {} is not supported for goal type {} yet or this is an bad error'.format(var_sym.name, goal_type.name))
            elif goal_type.name == 'charseq':
                if var_sym.name == 'CC' or var_sym.name == 'CONST1' or var_sym.name == 'CONST2':
                    values = var_sym.values if new_values is None else new_values
                else:
                    warnings.warn('Either {} is not supported for goal type {} yet or this is an bad error'.format(var_sym.name, goal_type.name))
                    pass
            elif goal_type.name in parsable_types or goal_type.name in entity_types:
                if var_sym.name == 'TYPE' or var_sym.name == 'ENT':
                    if goal_type.name in parsable_types:
                        values = [parsable_types[goal_type.name]]
                    elif goal_type.name in entity_types:
                        values = [entity_types[goal_type.name]]
                    else:
                        raise ValueError('Should not reach this branch')
                elif var_sym.name == 'ONT' or var_sym.name == 'QUERY':
                    values = [goal_type.name]
                elif var_sym.name == 'BOOL':
                    if goal_type.name not in ['year', 'month', 'day', 'second', 'minute', 'hour']:
                        values = var_sym.values
                elif var_sym.name == 'VAR':
                    values = var_sym.values
                else:
                    warnings.warn('Either {} is not supported for goal type {} yet or this is an bad error'.format(var_sym.name, goal_type.name))
                    # print('goal_type.name:', goal_type.name)
                    # print('var_sym.name:', var_sym.name)
                    # print('curr_p:', curr_p)
                    pass
            elif isinstance(goal_type, GPT3Type) or goal_type.name in extensible_types:
                if var_sym.name == 'QUERY':
                    values = [goal_type.name]
                else:
                    warnings.warn('Either {} is not supported for goal type {} yet or this is an bad error'.format(var_sym.name, goal_type.name))
                    pass
            else:
                pass
                warnings.warn('Either {} is not supported for goal type {} yet or this is an bad error'.format(var_sym.name, goal_type.name))

        elif isinstance(goal_type, Tuple):
            raise TypeError('Should not have a Tuple goal type for terminal node')
        elif isinstance(goal_type, base_type.Optional):
            pd_print("need to synthesize the Optional construct before getting to the terminal")
            pass
        else:
            raise TypeError('Type {} is not supported / should not be here'.format(goal_type.__class__))

        for value in values:
            new_p = curr_p.duplicate(next(self.prog_id_counter))

            # if we are filling the hole for the first argument for HasType, we can update the type of the second hole here as well
            # TODO: this should be part of the bi-directional type synthesis, we hack this here for now
            if var_sym.name == 'TYPE' and isinstance(value, tuple):
                # decouple the value variable
                type_key, type_value = value
                new_p.instantiate_var_node(var_node, type_value, var_sym, prod)
                # get the second node
                parent_nid = new_p.to_parent[var_node.id]
                children_nids = new_p.to_children[parent_nid]
                assert len(children_nids) == 2
                second_child_nid = new_p.to_children[parent_nid][1]
                type_object, _ = get_type(type_key)
                assert type_object is not None
                new_p.nid_to_goal_type[second_child_nid] = type_object

            else:
                new_p.instantiate_var_node(var_node, value, var_sym, prod)
            pd_print('new_p: {}'.format(new_p))
            check_prune_res = self.check_prune_program(task, new_p, curr_best, pruning, use_type_info=True)
            if check_prune_res == PruneCode.FEASIBLE:
                new_programs.append(new_p)

        return new_programs
