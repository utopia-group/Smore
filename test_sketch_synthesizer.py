from lib.interpreter.executor import Executor
from lib.lang.grammar import CFG
from lib.program.sketch import parse_sketch
from lib.spec.spec import Task
from lib.spec.synthesis_spec import SynthesisTask
from lib.synthesizer.sketch_synthesizer import SketchSynthesizer
from lib.utils.sketch_utils import postprocess_gpt3_sketch

# not sure if the depth is important here
synthesizer = SketchSynthesizer(CFG(), Executor(), 4, False, False, False)


def get_task(t: Task, s: str):
    return SynthesisTask(synthesizer.executor.nlp_engine, t, parse_sketch(postprocess_gpt3_sketch(s), False), None)


test_cases = [
    get_task(Task(['8.3', '4.6', '10.2'], [], {}), 'float->{??: integer}[.]{??: integer}'),
]


for task in test_cases:
    synth_res = synthesizer.synthesize(task)
    print('synth_res: ', synth_res)
