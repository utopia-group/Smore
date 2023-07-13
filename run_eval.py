"""
main evaluation file
"""
import argparse
import os
import datetime
import signal
import subprocess
import time
import traceback
from collections import defaultdict

from flashgpt_benchmark_processing import generate_flash_gpt3_format
from lib.eval.benchmark_engine import BenchmarkResultEngine
from lib.eval.eval_gpt3_exec import run
from lib.eval.eval_gpt3_sketch_concrete import run_gpt3_concrete
from lib.eval.eval_res import EvalRes
from lib.interpreter.executor import Executor
from lib.synthesizer.top_level_synthesizer import TopLevelSynthesizer
from lib.sketch_gen.sketch_gen import SketchGenerator
from lib.utils.csv_utils import save_dict_to_csv
from lib.utils.exceptions import TimeOutException
from lib.utils.gpt3_utils import generate_sketch_prompt

executor: Executor = Executor()


def handle_timeout(sig, frame):
    raise TimeOutException("Timed out!")


def write_datetime():
    # a date generate in the format YY-MM-DD-HH-MM-SS
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_sketch_gen', action='store_true', help='test sketch generator')
    parser.add_argument('--gpt3_exec', action='store_true', help='use gpt3 as a neural executor')
    parser.add_argument('--chatgpt_exec', action='store_true', help='use chatgpt as a neural executor')
    parser.add_argument('--chatgpt_regex', action='store_true', help='use chatgpt as a neural symbolic regex generator')
    parser.add_argument('--flashgpt_preprocess', action='store_true', help='use flashgpt')
    parser.add_argument('--flashgpt_run', action='store_true', help='use flashgpt')
    parser.add_argument('--flashgpt_postprocess', action='store_true', help='use flashgpt')
    parser.add_argument('--gpt_synth', action='store_true', help='use gpt as a neural synthesizer')
    parser.add_argument('--chatgpt_synth', action='store_true', help='use chatgpt as a neural synthesizer')
    parser.add_argument('--smore', action='store_true', help='run our tool')
    parser.add_argument('--no_sketch', action='store_true', help='synthesizer without sketch')
    parser.add_argument('--no_type', action='store_true', help='synthesizer without type information in the sketch')
    parser.add_argument('--no_type_system', action='store_true', help='synthesizer without type system')
    parser.add_argument('--no_decomp', action='store_true', help='synthesizer without decomposition')
    parser.add_argument('--no_repair', action='store_true', help='synthesizer without sketch repair')
    parser.add_argument('--timeout', type=int, default=60, help='timeout for each program')
    parser.add_argument('--depth', type=int, default=4, help='depth of the synthesized program')
    parser.add_argument('--prompt_version', type=str, default='v1', help='the prompt version to use')
    parser.add_argument('--eval_sheet_name', type=str, default='Semantic Regex Evaluation', help='name of the evaluation sheet')
    parser.add_argument('--eval_worksheet_name', type=str, default=write_datetime(), help='name of the evaluation sheet')
    parser.add_argument('--update_google_sheet', action='store_true', help='update google sheet')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('args: {}'.format(args))

    # read benchmark here first
    benchmark_engine = BenchmarkResultEngine(args.eval_sheet_name, args.eval_worksheet_name, benchmark_size=50)

    if args.test_sketch_gen:
        # test sketch generator with different prompt version and model
        # and output to a csv file
        prompt_versions = ['v1', 'v2']
        # prompt_versions = ['prev_old_prompt']
        results = defaultdict(dict)
        for version in prompt_versions:
            sketch_generator = SketchGenerator('sketch', executor, False, False, prompt_version=version)
            gpt_funcs = sketch_generator.gpt_func
            next_benchmark = benchmark_engine.get_next_benchmark()
            while True:
                try:
                    benchmark = next(next_benchmark)
                    # generate prompt for this benchmark
                    benchmark_prompt = generate_sketch_prompt(benchmark.task.pos_examples, version)
                    for model_name, gpt_func in gpt_funcs.items():
                        res = gpt_func(benchmark_prompt)
                        results[str(benchmark.bid)]['{}-{}'.format(model_name, version)] = res
                except StopIteration:
                    print('No more benchmarks')
                    break

        # convert results to a list of dict including the key
        results = [{**{'bid': bid}, **res} for bid, res in results.items()]

        # save results to a csv file
        save_dict_to_csv('eval_res/sketch_gen_results_{}.csv'.format(args.eval_worksheet_name), results)

    elif args.flashgpt_preprocess:
        # process flashgpt benchmark
        generate_flash_gpt3_format(benchmark_engine.get_next_benchmark())

    elif args.flashgpt_run:
        # run flashgpt evaluation as a system command
        subprocess.run(['dotnet', 'build', 'flashgpt3/FlashGPT3.sln'])
        subprocess.run(['python3', 'run_flashgpt_baseline.py'], cwd='flashgpt3/FlashGPT3/')

    elif args.flashgpt_postprocess:
        # move the result folder to the eval_res folder
        if not os.path.exists('eval_res/flashgpt_results'):
            os.system('cp -r flashgpt3/FlashGPT3/results/semregex_test eval_res/flashgpt_results/')
        benchmark_engine.post_process_flashgpt_results(args.update_google_sheet)

    elif args.gpt_synth or args.chatgpt_synth or args.chatgpt_regex:
        if args.gpt_synth:
            model = 'text-davinci-003'
            eval_mode = 'GPT-3-synth-'
        elif args.chatgpt_synth:
            model = 'gpt-3.5-turbo'
            eval_mode = 'GPT-3.5-synth-'
        elif args.chatgpt_regex:
            model = 'gpt-3.5-turbo'
            eval_mode = 'GPT-3.5-regex-'
        else:
            raise NotImplementedError

        results = []
        next_benchmark = benchmark_engine.get_next_benchmark()
        while True:
            try:
                benchmark = next(next_benchmark)

                # special filtering for the benchmark
                # if benchmark.bid not in [15]:
                #     continue

                try:
                    if '-synth-' in eval_mode:
                        res = run_gpt3_concrete(benchmark, executor, 'concrete', model)
                    elif '-regex-' in eval_mode:
                        res = run_gpt3_concrete(benchmark, executor, 'regex', model)
                    else:
                        raise NotImplementedError

                except Exception:
                    # print exception stack trace
                    traceback.print_exc()
                    res = EvalRes(benchmark.task, 'ERROR', 0, 0, '', '', benchmark.bid)

                results.append(res)
            except StopIteration:
                print('No more benchmarks')
                break

        # post process the results
        benchmark_engine.process_eval_res(eval_mode, executor, results, args.update_google_sheet)

    elif args.gpt3_exec or args.chatgpt_exec:

        if args.gpt3_exec:
            model = 'text-davinci-003'
            eval_mode = 'GPT-3-exec-'
        else:
            model = 'gpt-3.5-turbo'
            eval_mode = 'GPT-3.5-exec-'

        results = []
        next_benchmark = benchmark_engine.get_next_benchmark()
        while True:
            try:
                benchmark = next(next_benchmark)
                signal.alarm(args.timeout)
                try:
                    start = time.time()
                    run_res = run('\n'.join(benchmark.task.pos_examples),
                                  '\n'.join(benchmark.task.neg_examples),
                                  '\n'.join(benchmark.test_positive),
                                  '\n'.join(benchmark.test_negative), model, executor)
                    end = time.time()
                    res = EvalRes(benchmark.task, run_res, (end - start), 0, '', '', benchmark.bid)
                except TimeoutError:
                    res = EvalRes(benchmark.task, 'TIMEOUT', args.timeout, 0, '', '', benchmark.bid)
                finally:
                    signal.alarm(0)

                results.append(res)
            except StopIteration:
                print('No more benchmarks')
                break

        # post process the results
        benchmark_engine.process_eval_res(eval_mode, executor, results, args.update_google_sheet)

    else:
        if args.no_type:
            eval_mode = 'Smore-no-type-'
        elif args.no_type_system:
            eval_mode = 'Smore-no-type-system-'
        elif args.no_decomp:
            eval_mode = 'Smore-no-decomp-'
            args.no_repair = True
        elif args.no_repair:
            eval_mode = 'Smore-no-repair-'
        elif args.no_sketch:
            eval_mode = 'Smore-no-sketch-'
            args.no_repair = True
            args.no_decomp = True
            args.no_type = True
        elif args.smore:
            eval_mode = 'Smore-'
        else:
            raise ValueError('Invalid evaluation mode')

        results = []
        next_benchmark = benchmark_engine.get_next_benchmark()
        while True:
            try:
                benchmark = next(next_benchmark)

                # special filtering for the benchmark
                # if int(benchmark.bid) not in range(14, 15):
                #     continue

                print('=====================')
                print('Evaluating benchmark {}'.format(benchmark.bid))
                synthesizer = TopLevelSynthesizer(executor, args.depth, args.no_type, args.no_type_system, args.no_decomp, args.no_repair, args.prompt_version)
                executor.context = {}
                signal.signal(signal.SIGALRM, handle_timeout)
                signal.alarm(args.timeout)
                try:
                    if args.no_sketch:
                        res = synthesizer.synthesize_no_sketch(benchmark.task)
                    else:
                        res = synthesizer.synthesize(benchmark.task)
                    res.bid = benchmark.bid
                except TimeoutError:
                    res = EvalRes(benchmark.task, 'TIMEOUT', args.timeout, 0, '', '', benchmark.bid)
                    signal.alarm(0)
                except TimeOutException:
                    res = EvalRes(benchmark.task, 'TIMEOUT', args.timeout, 0, '', '', benchmark.bid)
                    signal.alarm(0)
                finally:
                    signal.alarm(0)

                results.append(res)
            except StopIteration:
                print('No more benchmarks')
                break

        # post process the results
        benchmark_engine.process_eval_res(eval_mode, executor, results, args.update_google_sheet)
