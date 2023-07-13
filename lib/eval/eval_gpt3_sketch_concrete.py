import re
import time

from lib.cache import cache
from lib.eval.benchmark import Benchmark
from lib.eval.eval_res import EvalRes
from lib.lang.constants import CaseTag
from lib.parser.parser import parse_program, postprocess_gpt_program
from lib.utils.gpt3_utils import generate_concrete_prompt, generate_regex_prompt


@cache(ignore_args=[0])
def run_gpt_concrete_helper(executor, prompt: str, model: str, temperature: float, i: int) -> str:
    res = executor.nlp_engine.call_gpt3_and_get_answer(prompt, '', model, temperature=temperature)
    return res


def run_gpt3_concrete(b: Benchmark, executor, _mode='concrete', model='code', sample_budget: int = 10) -> EvalRes:
    """
    This is how we should run this:
    1. Generate a prompt
    2. Sample a program from the model
    3. Try to parse the program
    4. If the program is parsable and the output matches the target, return the program
    5. If the program is not parsable, go back to step 1
    We need to cache the program we generated because
    """

    assert _mode in ['concrete', 'regex']

    start = time.time()
    if _mode == 'concrete':
        full_prompt = generate_concrete_prompt(b.task.pos_examples, b.task.neg_examples)
    else:
        assert _mode == 'regex'
        full_prompt = generate_regex_prompt(b.task.pos_examples, b.task.neg_examples)

    iteration = 0

    # print("Prompt: {}".format(full_prompt))

    while True:

        if iteration >= sample_budget:
            end = time.time()
            return EvalRes(b.task, None, (end - start), iteration, "", "", b.bid, sample_budget)

        res = run_gpt_concrete_helper(executor, full_prompt, model, 0.7, iteration)
        print("\nIteration {}: {}".format(iteration, res))
        iteration += 1

        try:
            # try to parse the program
            if _mode == 'concrete':
                program = parse_program(postprocess_gpt_program(res), False, CaseTag.NONE)
            else:
                assert _mode == 'regex'
                program = res

        except Exception as e:
            print(e)
            continue

        # if the program is parsable, check if the output matches the target
        if program is not None:
            if _mode == 'concrete':
                pos_ex_res, neg_ex_res = b.run_program_on_task(executor, program)
                if all(pos_ex_res) and not any(neg_ex_res):
                    end = time.time()
                    return EvalRes(b.task, program, (end - start), iteration, "", "", b.bid, iteration + 1)
                else:
                    continue
            else:
                assert _mode == 'regex'
                pos_ex_res, neg_ex_res = b.run_regex_on_task(program)
                if all(pos_ex_res) and not any(neg_ex_res):
                    end = time.time()
                    return EvalRes(b.task, program, (end - start), iteration, "", "", b.bid, iteration + 1)
                else:
                    continue
        else:
            continue

    end = time.time()
    return EvalRes(b.task, None, (end - start), iteration, "", "", b.bid, iteration + 1)
