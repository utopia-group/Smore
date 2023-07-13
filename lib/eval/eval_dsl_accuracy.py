"""
For each task in our benchmark, evaluate the accuracy of the manually-written program on the training examples
"""
from lib.eval.benchmark import read_manual_benchmark
from lib.interpreter.executor import Executor
from lib.utils.csv_utils import read_csv_to_dict


def eval_dsl_accuracy():
    benchmark_folder = 'benchmarks'
    benchmarks = read_csv_to_dict('{}/benchmarks.csv'.format(benchmark_folder))

    executor = Executor()
    correct_count = 0
    total_count = 0

    error_entry = []

    for b_task in benchmarks:

        print("========== eval {} ============".format(b_task))

        if b_task['token_mode'] == '':
            continue

        b_ins = read_manual_benchmark(b_task)

        pos_ex_res, neg_ex_res = b_ins.run_program_on_task(executor, b_ins.program)
        if all(pos_ex_res) and not any(neg_ex_res):
            correct_count += 1
        else:
            # although I want to print error here but i think error should be shown in evaluate_program
            error = (b_ins, [b_ins.task.pos_examples[i] if not e else 'PASS' for i, e in enumerate(pos_ex_res)],  [b_ins.task.neg_examples[i] if e else 'PASS' for i, e in enumerate(neg_ex_res)])
            error_entry.append(error)
            pass

        total_count += 1

    print("program correct rate: {}/{}={}".format(str(correct_count), str(total_count), str(correct_count/total_count)))
    print()
    print("detailed error: ")
    for entry in error_entry:
        print(entry)


if __name__ == '__main__':
    eval_dsl_accuracy()
