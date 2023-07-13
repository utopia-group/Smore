from typing import List, Optional

from lib.config import pd_print
from lib.interpreter.executor import Executor
from lib.lang.grammar import CFG
from lib.lang.production import IdProduction
from lib.lang.symbol import TerminalSymbol
from lib.program.node import VariableNode
from lib.program.program import Program
from lib.spec.synthesis_spec import SynthesisTask
from lib.synthesizer.enum_synthesizer import EnumSynthesizer
from lib.type.type_inference import infer_type
from lib.type.type_system import get_type, add_base_type, TypeInferenceException, get_subtypes
from lib.utils.pq import PriorityQueue
from lib.utils.synth_utils import compute_naive_score, expand_non_terminal_with_production_helper, PruneCode
from lib.utils.task_utils import add_place_predicate_context


class TopDownEnumSynthesizer(EnumSynthesizer):
    def __init__(self, grammar: CFG, executor: Executor, depth: int, start_nid: int):
        super().__init__(grammar, executor, depth)
        self.curr_nid: int = start_nid
        self.program_limit = -1
        self.type_check = True

    def synthesize(self, task: SynthesisTask, worklist: Optional[PriorityQueue] = None,
                   max_iterations: int = 100000) -> List[Program]:

        # print('TopDownEnumSynthesizer.synthesize()')

        if all([s.s == '' for s in task.pos_str_context]) and all([s.s == '' for s in task.neg_str_context]):
            any_star_prog = Program(next(self.prog_id_counter), self.curr_nid)
            start_node = any_star_prog.add_nonterminal_node(name='star', symbol=self.grammar.get_nonterminal_sym('r'), prod=self.grammar.name_to_prod['StarR'])
            any_star_prog.set_start_node(start_node)
            any_star_prog.set_children(start_node,
                                       [any_star_prog.add_terminal_node(value='any', symbol=self.grammar.get_terminal_sym('CC'), prod=self.grammar.name_to_prod['ToCC'])])
            return [any_star_prog]

        # Should we filter out empty string here?
        new_pos_str_context = []
        for s in task.pos_str_context:
            if s.s.strip() != '':
                new_pos_str_context.append(s)
        task.pos_str_context = new_pos_str_context

        add_place_predicate_context(task, self.grammar)
        assert not self.curr_nid == -1

        self.instantiate_terminals_in_grammar(task.pos_str_context, task.neg_str_context)

        if self.type_check:
            goal_type_str = task.goal_type
            # we need to add the type to the type system
            goal_type_object, to_be_added_type_name = get_type(goal_type_str)
            if goal_type_object is None:
                add_base_type(to_be_added_type_name, ('gpt3type',))
            # we also need to instantiate the MatchQuery tags in the grammar (because we don't have the type system to enforce that here)
            if goal_type_str.lower() not in ['string', 'any', 'charseq']:
                # get the corresponding terminal symbol
                terminal_sym = self.grammar.get_terminal_sym('QUERY')
                terminal_sym.values = []
                assert terminal_sym is not None and isinstance(terminal_sym.values, list)

                # if this type have subtype, we add them to the grammar as well
                candidate_types = get_subtypes(goal_type_str)
                terminal_sym.values.extend([ty.name for ty in candidate_types])

        # if worklist is None:
        #     worklist = PriorityQueue(priority=pq_ranking_func)

        if worklist.is_empty():
            worklist.put(self.init_program())

        synthesized_programs: List[Program] = []
        curr_best_score = 1
        num_iterations = 0

        while not worklist.is_empty() and num_iterations < max_iterations:
            curr_p = worklist.pop()
            num_iterations += 1
            assert isinstance(curr_p, Program)

            pd_print("current prog: {}".format(curr_p))

            if curr_p.is_concrete():

                # if not str(curr_p).startswith('AndR'):
                #     continue

                if self.type_check:
                    # try to type check this program
                    goal_type_object, _ = get_type(task.goal_type)
                    assert goal_type_object is not None
                    try:
                        inferred_type = infer_type(curr_p, curr_p.start_node_id)
                        # print("inferred type:", inferred_type)
                        # print("goal type:", goal_type_object)
                        if issubclass(inferred_type.__class__, goal_type_object.__class__):
                            # print("type check passed")
                            pass
                        else:
                            # print("type check failed")
                            continue
                    except TypeInferenceException:
                        # print("type check failed")
                        continue

                eval_res = self.eval_program(task, curr_p)
                eval_score = compute_naive_score(eval_res)

                pd_print("current_best_score:", curr_best_score)
                pd_print('eval_score:', eval_score)

                # print("current_best_score:", curr_best_score)

                if eval_score == curr_best_score:
                    synthesized_programs.append(curr_p)

                    if len(synthesized_programs) >= 1:
                        # print(worklist.pq)
                        return synthesized_programs
                else:
                    continue
            else:
                var_node = curr_p.select_var_node()
                new_programs = self.expand(task, curr_p, var_node, curr_best_score)
                worklist.put_all(new_programs)

        return synthesized_programs

    def expand(self, task: SynthesisTask, curr_p: Program, var_node: VariableNode, curr_best: int) -> List[Program]:

        new_programs: List[Program] = []

        productions = self.grammar.get_productions(var_node.sym)
        for prod in productions:
            pd_print("expand with production: {}".format(prod))
            if isinstance(prod, IdProduction):
                assert len(prod.arg_syms) == 1
                arg_sym = prod.arg_syms[0]
                assert isinstance(arg_sym, TerminalSymbol)
                # ToXXX should never show up in an actual program (just feels inefficient)
                new_prog = curr_p.duplicate(next(self.prog_id_counter))
                progs = self.instantiate_terminal_sym(task, new_prog, var_node, arg_sym, prod, curr_best, True)
                new_programs.extend(progs)
            else:
                # instantiate the selected variable node with the proper production
                new_prog, new_children_nodes = expand_non_terminal_with_production_helper(curr_p, next(self.prog_id_counter), var_node, prod)

                if prod.function_name == 'MatchQuery':
                    new_programs.extend(self.expand_match_query(task, new_prog.duplicate(next(self.prog_id_counter)), new_children_nodes, curr_best, True, use_type_info=False))
                else:
                    if self.check_prune_program(task, new_prog, curr_best, True, use_type_info=False) == PruneCode.FEASIBLE:
                        # if the new_prog contains terminal symbol, then we also further expand them
                        new_programs.extend(self.expand_additional_terminals(task, new_prog.duplicate(next(self.prog_id_counter)), new_children_nodes, prod, curr_best, True, use_type_info=False))

        pd_print("new programs expanded: {}".format(new_programs))
        # assert False
        return new_programs

    def init_program(self) -> Program:
        curr_p = Program(next(self.prog_id_counter), self.curr_nid)
        new_node = curr_p.add_variable_node(self.grammar.start_sym)
        curr_p.set_start_node(new_node)

        return curr_p

    def instantiate_terminal_sym(self, task: SynthesisTask, curr_p: Program, var_node: VariableNode, var_sym: TerminalSymbol, prod: Optional[IdProduction],
                                 curr_best: int, pruning: bool, new_values: Optional[list] = None) -> List[Program]:
        new_programs = []
        values = var_sym.values if new_values is None else new_values
        # print("values:", values)

        for value in values:
            new_p = curr_p.duplicate(next(self.prog_id_counter))
            new_p.instantiate_var_node(var_node, value, var_sym, prod)

            if self.check_prune_program(task, new_p, curr_best, True, use_type_info=False) == PruneCode.FEASIBLE:
                new_programs.append(new_p)

        return new_programs
