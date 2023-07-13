"""
Generate benchmark files for flash gpt3
"""
import csv
from typing import Iterator

from lib.eval.benchmark import read_manual_benchmark, Benchmark, EvalBenchmark
from lib.utils.csv_utils import read_csv_to_dict


def generate_flash_gpt3_format(benchmark_engine_iterator: Iterator[EvalBenchmark]):
    output_folder = 'flashgpt3/benchmarks/semregex'

    while True:
        try:
            benchmark = next(benchmark_engine_iterator)

            output_list = []

            # these are the training benchmarks
            for ex in benchmark.task.pos_examples:
                output_list.append([ex, ex])
            for ex in benchmark.task.neg_examples:
                output_list.append([ex, ' '])

            # these are the testing benchmarks
            for ex in benchmark.test_positive:
                output_list.append([ex, ex])
            for ex in benchmark.test_negative:
                output_list.append([ex, ' '])

            # opening the csv file in 'w+' mode
            file = open('{}/{}.csv'.format(output_folder, benchmark.bid), 'w+', newline='')

            # writing the data into the file
            with file:
                write = csv.writer(file)
                write.writerows(output_list)

        except StopIteration:
            print('No more benchmarks')
            break
