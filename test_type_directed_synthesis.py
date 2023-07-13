from lib.interpreter.executor import Executor
from lib.lang.grammar import CFG
from lib.program.sketch import parse_sketch
from lib.spec.spec import Task
from lib.spec.synthesis_spec import SynthesisTask
from lib.synthesizer.typed_synthesizer import TypedSynthesizer
from lib.utils.sketch_utils import postprocess_gpt3_sketch

grammar = CFG()
executor = Executor()
synthesizer = TypedSynthesizer(grammar, executor, 4, 0, False, False)


test_cases = [
    ('year', Task(['2020', '2021', '2019'], [], {})),
]


for hole_type, sub_task in test_cases:
    synth_res = synthesizer.synthesize(SynthesisTask(executor.nlp_engine, sub_task, parse_sketch(postprocess_gpt3_sketch('string-> {{??: {}}}'.format(hole_type)), False), hole_type))
    print('synth_res: ', synth_res)
