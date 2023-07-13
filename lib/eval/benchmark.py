import re
from typing import List, Tuple, Dict

from lib.lang.constants import CaseTag
from lib.parser.parser import parse_program

from lib.spec.spec import Task


class Benchmark:
    def __init__(self, bid, task: Task):
        self.bid: int = bid
        self.task: Task = task

    def evaluate_program_with_input(self, executor, program, input_str: List) -> List[bool]:

        executor.set_context(self.task.context)

        run_results = []
        for e in input_str:
            print("\t Running example {}:".format(e))
            match_ctx = executor.exec(e, program)
            if match_ctx.success:
                print("\t\t success")
            else:
                print("\t\t failed with match context {}".format(match_ctx))
            run_results.append(match_ctx.success)

        return run_results

    def run_program_on_task(self, executor, program) -> Tuple[List[bool], List[bool]]:
        """
        evaluate the program on positive and negative examples of the task
        """

        print("Evaluating benchmark {}".format(self.__repr__()))

        print("Running positive example: ")
        pos_run_results = self.evaluate_program_with_input(executor, program, self.task.pos_examples)

        print("Running negative example: ")
        neg_run_results = self.evaluate_program_with_input(executor, program, self.task.neg_examples)

        return pos_run_results, neg_run_results

    def run_regex_on_task(self, regex) -> Tuple[List[bool], List[bool]]:
        """
        Given some regex, it should run the regex on the training examples
        """
        print("Evaluating benchmark {} with regex {}".format(self.__repr__(), str(regex)))

        regex = re.compile(r'^' + regex + '$')

        print("Running positive example: ")
        pos_run_results = [re.match(regex, pos) is not None for pos in self.task.pos_examples]

        print("Running negative example: ")
        neg_run_results = [re.match(regex, neg) is not None for neg in self.task.neg_examples]

        return pos_run_results, neg_run_results

    def __repr__(self):
        return 'Benchmark_{}({})'.format(self.bid, self.task)


class EvalBenchmark(Benchmark):
    def __init__(self, bid, task: Task, test_positive: List[str], test_negative: List[str]):
        super().__init__(bid, task)
        self.test_positive: List[str] = [' '.join(pos.split()) for pos in test_positive]
        self.test_negative: List[str] = [' '.join(neg.split()) for neg in test_negative]

    def run_program_on_test(self, executor, program) -> Tuple[List[bool], List[bool]]:
        """
        Given some synthesized program, it should run the program on the testing examples
        """
        print("Evaluating benchmark {} with program {}".format(self.__repr__(), str(program)))

        print("Running positive example: ")
        pos_run_results = self.evaluate_program_with_input(executor, program, self.test_positive)

        print("Running negative example: ")
        neg_run_results = self.evaluate_program_with_input(executor, program, self.test_negative)

        return pos_run_results, neg_run_results

    def run_regex_on_test(self, regex) -> Tuple[List[bool], List[bool]]:
        """
        Given some regex, it should run the regex on the training examples
        """
        print("Evaluating benchmark {} with regex {}".format(self.__repr__(), str(regex)))

        regex = re.compile(r'^' + regex + '$')

        print("Running positive example: ")
        pos_run_results = [re.match(regex, pos) is not None for pos in self.test_positive]

        print("Running negative example: ")
        neg_run_results = [re.match(regex, neg) is not None for neg in self.test_negative]

        return pos_run_results, neg_run_results


class ManualBenchmark(Benchmark):
    def __init__(self, bid, task: Task, program: str, token_mode: bool, case_tag: CaseTag):
        super().__init__(bid, task)
        self.task = task
        self.program_str: str = program
        self.token_mode: bool = token_mode
        self.case_tag: CaseTag = case_tag
        self.program = parse_program(self.program_str, self.token_mode, self.case_tag)

    def __repr__(self):
        return 'Benchmark_{}({}, program={})'.format(self.bid, self.task, self.program_str)


def read_manual_benchmark(b_task: Dict) -> ManualBenchmark:

    task = Task(pos_examples=eval(b_task['train-positive']), neg_examples=eval(b_task['train-negative']),
                context=eval(b_task['context']) if b_task['context'] != '' else None)
    b_ins = ManualBenchmark(bid=b_task['bid'], task=task, program=b_task['program'], token_mode=False if b_task['token_mode'].lower() == 'false' else True,
                            case_tag=CaseTag(b_task['case_tag'].upper()))

    return b_ins
