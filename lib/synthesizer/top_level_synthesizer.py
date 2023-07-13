import time
import traceback
from typing import Optional

from lib.config import pd_print
from lib.eval.eval_res import EvalRes
from lib.interpreter.executor import Executor
from lib.lang.grammar import CFG
from lib.nlp.nlp import NLPFunc
from lib.program.sketch import Sketch
from lib.sketch_gen.sketch_gen import SketchGenerator
from lib.spec.spec import Task
from lib.spec.synthesis_spec import SynthesisTask
from lib.synthesizer.sketch_synthesizer import SketchSynthesizer
from lib.utils.exceptions import SketchInfeasibleException, NoMoreSampleBudgetException, TimeOutException


class TopLevelSynthesizer:
    def __init__(self, executor: Optional[Executor] = None, depth: int = 4, no_type: bool = False, no_type_system: bool = False, no_decomp: bool = False, no_repair: bool = False, prompt_version: str = 'v1'):
        self.mode = 'sketch'
        self.depth = depth
        self.cfg: CFG = CFG()
        self.executor: Executor = Executor() if executor is None else executor
        self.nlp_func: NLPFunc = self.executor.nlp_engine

        self.no_type = no_type
        self.no_type_system = no_type_system
        self.no_decomp = no_decomp
        self.no_repair = no_repair
        self.prompt_version = prompt_version

        self.sketch_synthesizer = SketchSynthesizer(self.cfg, self.executor, self.depth, self.no_type, self.no_type_system, self.no_decomp)
        self.sketch_generator = SketchGenerator(self.mode, self.executor, self.no_type, self.no_decomp, self.no_repair, prompt_version)

    def synthesize(self, task: Task) -> EvalRes:
        sketch_iterator = self.sketch_generator.get_sketch(task)
        synth_start = time.time()
        first_sketch = None
        curr_sketch = None
        repaired_sketch = None
        while True:
            try:
                if repaired_sketch is not None:
                    curr_sketch = repaired_sketch
                    repaired_sketch = None
                else:
                    curr_sketch = next(sketch_iterator)
                first_sketch = curr_sketch if first_sketch is None else first_sketch
            except NoMoreSampleBudgetException:
                return EvalRes(task, None, (time.time() - synth_start), self.sketch_generator.sketch_count, str(first_sketch), str(curr_sketch), None, 10)
            except StopIteration:
                break
            pd_print('SKETCH:', curr_sketch)
            synth_task = self.get_synthesis_task(task, curr_sketch)
            if self.no_type:
                synth_task.no_type = True
                synth_task.no_context = True
            try:
                if self.no_decomp:
                    # print('no decomp')
                    res = self.sketch_synthesizer.synthesize_no_decomp(synth_task)
                else:
                    res = self.sketch_synthesizer.synthesize(synth_task)
                if len(res) > 0:
                    synth_end = time.time()
                    print('synthesis finished with the {}th sketch {}'.format(self.sketch_generator.sketch_count, curr_sketch))

                    # for rebuttal purpose, count the node size here
                    # print("PROGRAM NODE SIZE: ", res[0].count_node_size())
                    return EvalRes(task, res[0].to_pattern(), (synth_end - synth_start), self.sketch_generator.sketch_count, str(first_sketch), str(curr_sketch), None)
                else:
                    raise SketchInfeasibleException()
            except TimeOutException as e:
                raise e
            except Exception as e:
                # print(traceback.format_exc())
                pd_print('SKETCH INFEASIBLE:', curr_sketch)
                if self.no_repair:
                    self.sketch_generator.sample_sketch(task)
                else:
                    repaired_sketch = self.sketch_generator.repair_sketch(synth_task, e)
                    if repaired_sketch is not None:
                        # If we couldn't manually repair, reuse GPT
                        pd_print('MANUAL REPAIRED SKETCH:', repaired_sketch)

                continue

        synth_end = time.time()
        return EvalRes(task, None, (synth_end - synth_start), self.sketch_generator.sketch_count, str(first_sketch), str(curr_sketch), None)

    def synthesize_no_sketch(self, task: Task) -> EvalRes:
        sketch = self.sketch_generator.get_naive_sketch()
        synth_task = self.get_synthesis_task(task, sketch)
        synth_task.no_context = True
        synth_start = time.time()
        try:
            res = self.sketch_synthesizer.synthesize(synth_task)
            synth_end = time.time()
            if len(res) > 0:
                return EvalRes(task, res[0].to_pattern(), (synth_end-synth_start), 1, str(sketch), str(sketch), None)
        except Exception as e:
            pass
            # print(traceback.format_exc())

        synth_end = time.time()
        return EvalRes(task, None, (synth_end - synth_start), self.sketch_generator.sketch_count, str(sketch), str(sketch), None)

    def get_synthesis_task(self, task: Task, sketch: Sketch) -> SynthesisTask:
        task = SynthesisTask(self.nlp_func, task, sketch, None)
        return task
