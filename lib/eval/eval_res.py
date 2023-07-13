from typing import Tuple

from lib.interpreter.pattern import Pattern
from lib.spec.spec import Task


class EvalRes:
    """
    a class that tracks the evaluation results
    """

    def __init__(self, task: Task, pattern: Pattern | None | Tuple[float, float, float] | str, synth_time: float, sketch_explored: int, initial_sketch: str, final_sketch: str, bid=None, sample_num=1):
        self.task = task
        self.pattern: Pattern | None | Tuple[float, float, float] | str = pattern
        self.synth_time: float = synth_time
        self.sketch_explored: int = sketch_explored
        self.initial_sketch: str = initial_sketch
        self.final_sketch: str = final_sketch
        self.bid = bid

        self.sample_num: int = sample_num

    def get_csv_header(self) -> str:
        return ', '.join(['positive', 'negative', 'pattern', 'synth_time', 'sketch_explored', 'initial_sketch', 'final_sketch']) + '\n'

    def to_csv(self):
        return '\"{}\", \"{}\", \"{}\", \"{}\", \"{}\", \"{}\", \"{}\"\n'.format(
            self.task.pos_examples, self.task.neg_examples, str(self.pattern), self.synth_time, self.sketch_explored, self.initial_sketch, self.final_sketch)

    def __repr__(self):
        return "Pattern: " + repr(self.pattern) + ", Sketch: " + repr(self.final_sketch)
