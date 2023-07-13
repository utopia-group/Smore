from collections import defaultdict
from typing import Optional, Dict, List

from lib.interpreter.context import StrContext, EntityPreprocessContext
from lib.nlp.nlp import NLPFunc
from lib.program.sketch import Sketch
from lib.spec.spec import Task
from copy import copy


class SynthesisTask:
    def __init__(self, nlp_executor: NLPFunc, task: Optional[Task], sketch: Optional[Sketch], goal_type: Optional[str]):

        self.executor: NLPFunc = nlp_executor
        self.sketch: Optional[Sketch] = None
        self.goal_type: Optional[str] = goal_type

        self.context: Dict = {}
        self.pos_str_context: List[StrContext] = []
        self.neg_str_context: List[StrContext] = []

        self.no_context: bool = False
        self.no_type: bool = False

        self.type_to_str: Dict[str, List[str]] = defaultdict(list)

        if task is not None:
            assert sketch is not None
            self.sketch: Sketch = sketch
            # print('SKETCH:', self.sketch)
            types_in_sketch = list(self.sketch.hole_id_to_type.values())

            self.context: Dict = task.context
            preprocess_context = EntityPreprocessContext(types_in_sketch, self.executor)
            self.pos_str_context = [StrContext(e, preprocess_context) for e in task.pos_examples]
            self.neg_str_context = [StrContext(e, preprocess_context) for e in task.neg_examples]

            for str_ctx in self.pos_str_context + self.neg_str_context:
                for ty, spans in str_ctx.type_to_str_spans.items():
                    self.type_to_str[ty].extend([str_ctx.get_substr(sp) for sp in spans])

            for ty, strings in self.type_to_str.items():
                # we need to sort from longest to shortest length
                sorted_list = sorted(list(set(self.type_to_str[ty])), key=lambda s: len(s), reverse=True)
                self.type_to_str[ty] = sorted_list

    def get_sub_task(self, goal_type: str, pos_examples: List[StrContext], neg_examples: List[StrContext], no_type: bool) -> 'SynthesisTask':
        new_task = SynthesisTask(self.executor, None, None, goal_type)
        new_task.no_context = self.no_context
        new_task.pos_str_context = pos_examples
        new_task.neg_str_context = neg_examples
        new_task.context = {}

        new_task.no_context = no_type
        new_task.no_type = no_type
        return new_task

    def get_sketch_overapprox(self) -> str:
        return self.sketch.get_overapprox(self.type_to_str)

    def __repr__(self):
        if self.sketch is not None:
            return '{}\t{}\t{}'.format(self.sketch, [s for s in self.pos_str_context], [s for s in self.neg_str_context])
        else:
            return '{}\t{}\t{}'.format(self.goal_type, [s for s in self.pos_str_context], [s for s in self.neg_str_context])
