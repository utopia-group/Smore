import json
import os
from typing import Dict, Iterator, List, Tuple

import numpy as np

from lib.eval.benchmark import EvalBenchmark
from lib.eval.eval_res import EvalRes
from lib.interpreter.pattern import Pattern
from lib.spec.spec import Task
from lib.utils.csv_utils import save_dict_to_csv, read_csv_to_dict
from lib.utils.google_sheet import GoogleSheet


class BenchmarkResultEngine:
    def __init__(self, sheet_name: str, worksheet_name: str, benchmark_size: int, use_google_sheet: bool = False):
        self.sheet_name = sheet_name
        self.worksheet_name = worksheet_name
        self.benchmark_size = benchmark_size
        self.use_google_sheet = use_google_sheet
        if self.use_google_sheet:
            self.google_sheet = GoogleSheet(self.sheet_name, self.worksheet_name, duplicate_from='template')
        else:
            if 'ae' in self.worksheet_name:
                self.sheet = read_csv_to_dict(os.path.join('eval_res', 'ae_results.csv'))
            else:
                self.sheet = read_csv_to_dict(os.path.join('benchmarks', 'benchmarks.csv'))
        self.bid_to_row_idx: Dict[str, int] = {}
        self.bid_to_benchmark: Dict[str, EvalBenchmark] = {}
        self.init_benchmarks()
        print('Initialized benchmark engine')

    def init_benchmarks(self):
        if self.use_google_sheet:
            all_values = self.google_sheet.get_all_records()
        else:
            all_values = self.sheet
        print(len(all_values))

        for row_idx in range(0, self.benchmark_size):
            bid = all_values[row_idx]['bid']
            self.bid_to_row_idx[bid] = row_idx + 2
            task_description = all_values[row_idx]['task']
            train_positive = all_values[row_idx]['train-positive'].split('\n')
            train_negative = all_values[row_idx]['train-negative'].split('\n')
            task = Task(train_positive, train_negative, {}, description=task_description)
            test_positive = all_values[row_idx]['test-positive'].split('\n')
            test_negative = all_values[row_idx]['test-negative'].split('\n')

            benchmark = EvalBenchmark(bid, task, test_positive, test_negative)
            self.bid_to_benchmark[bid] = benchmark

    def get_next_benchmark(self) -> Iterator[EvalBenchmark]:
        for bid, benchmark in self.bid_to_benchmark.items():
            yield benchmark

    def calculate_precision_recall(self, pos_match_res, neg_match_res) -> Tuple[float, float]:

        # count the # of true in pos_match_res and neg_match_res
        pos_match_count = sum(pos_match_res)
        neg_match_count = sum(neg_match_res)

        if pos_match_count + neg_match_count == 0:
            return 0, 0

        # calculate precision and recall
        recall = pos_match_count / len(pos_match_res)
        precision = pos_match_count / (pos_match_count + neg_match_count)

        return precision, recall

    def process_eval_res(self, eval_mode: str, executor, eval_res: List[EvalRes], update_google_sheet: bool):
        """
        Process the evaluation results and update the Google sheet
        if pattern is a tuple, then log them directly
        """

        f1_scores = []
        timeout_count = 0

        if self.use_google_sheet:
            all_records = self.google_sheet.get_all_records()
        else:
            all_records = self.sheet

        for res in eval_res:
            if isinstance(res.pattern, Tuple):
                precision, recall, f1 = res.pattern
                all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + 'precision'] = precision
                all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + 'recall'] = recall

                f1_scores.append(f1)

            elif (isinstance(res.pattern, str) and (res.pattern == 'ERROR' or res.pattern == 'TIMEOUT')) or res.pattern is None:

                all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + 'program'] = res.pattern
                all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + 'precision'] = 0
                all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + 'recall'] = 0
                all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + 'time'] = res.synth_time

                if res.pattern == 'TIMEOUT' or res.pattern is None:
                    timeout_count += 1

                if res.pattern == 'ERROR':
                    f1_scores.append(0)

                if '-regex-' in eval_mode:
                    all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + '#sample'] = res.sample_num
                elif '-synth-' in eval_mode:
                    all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + '#sample'] = res.sample_num
                elif '-no-repair-' in eval_mode:
                    all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + '#sample'] = res.sample_num

                print('result for bid {} is {} with precision={}, recall={} in {} seconds'.format(res.bid, res.pattern, 0, 0, res.synth_time))

            else:
                assert isinstance(res.pattern, Pattern) or isinstance(res.pattern, str)
                print('pattern: {}'.format(res.pattern))
                all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + 'program'] = str(res.pattern)
                all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + 'time'] = res.synth_time

                if '-regex-' in eval_mode:
                    # regex evaluation
                    assert isinstance(res.pattern, str)
                    pos_match_res, neg_match_res = self.bid_to_benchmark[res.bid].run_regex_on_test(res.pattern)
                    all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + '#sample'] = res.sample_num
                    # print('sample num: {}'.format(res.sample_num))
                else:
                    assert isinstance(res.pattern, Pattern)
                    pos_match_res, neg_match_res = self.bid_to_benchmark[res.bid].run_program_on_test(executor, res.pattern)
                    if '-synth-' in eval_mode:
                        all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + '#sample'] = res.sample_num
                    elif '-no-repair-' in eval_mode:
                        all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + '#sample'] = res.sample_num
                        # print('sample num: {}'.format(res.sample_num))

                print('reject positive examples: {}'.format([self.bid_to_benchmark[res.bid].test_positive[i] for i in range(len(pos_match_res)) if not pos_match_res[i]]))
                print('accept negative examples: {}'.format([self.bid_to_benchmark[res.bid].test_negative[i] for i in range(len(neg_match_res)) if neg_match_res[i]]))

                # calculate precision and recall
                precision, recall = self.calculate_precision_recall(pos_match_res, neg_match_res)
                all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + 'precision'] = precision
                all_records[self.bid_to_row_idx[res.bid] - 2][eval_mode + 'recall'] = recall

                f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)

                print('result for bid {} is {} with precision={}, recall={} in {} seconds'.format(res.bid, res.pattern, precision, recall, res.synth_time))

        save_dict_to_csv('eval_res/{}.csv'.format(self.worksheet_name), all_records)

        # print the end-results
        print('====================')
        print('Final results for {} are:'.format(eval_mode))
        print('F1 score: {}'.format(np.mean(f1_scores)))
        print('Timeout count: {}'.format(timeout_count))
        print('====================')

        if update_google_sheet and self.use_google_sheet:
            self.google_sheet.bulk_update(all_records)

    def post_process_flashgpt_results(self, update_google_sheet: bool, res_folder: str = 'eval_res/flashgpt_results/'):

        eval_mode = 'FlashGPT-'

        # for each file in the folder, read the json file and update extract the results
        if self.use_google_sheet:
            all_records = self.google_sheet.get_all_records()
        else:
            all_records = self.sheet

        # need to read all the log files in the folder, and get the second portion of runtime
        bid_to_seconds = {}
        for file in os.listdir(res_folder):
            if file.endswith('.log'):
                with open(res_folder + file) as f:
                    lines = f.readlines()
                    bid, num_of_examples, runtime = -1, -1, -1

                    for line in lines:
                        if line.startswith('>>>'):
                            if bid != -1:
                                bid_to_seconds[bid] = runtime
                                print('bid: {}, runtime: {}'.format(bid, runtime))

                            bid = int(line.split()[-1])
                        if line.startswith('> Finished'):
                            line_split = line.split()
                            num_of_examples = int(line_split[2])
                            runtime = int(line_split[4])

                    bid_to_seconds[bid] = runtime

        print('bid_to_seconds:', bid_to_seconds)

        # walk through the folder
        for file in os.listdir(res_folder):
            if file.endswith('.json'):
                with open(res_folder + file) as f:
                    res = json.load(f)

                    # get the bid
                    bid = file.split('.')[0].split('_')[-1]
                    print('bid: {}'.format(bid))

                    # get the failed index

                    if len(res['failed']) == 0:
                        failed_num_examples = 7
                    else:
                        failed_num_examples = int(res['failed'][0])
                    success_num_examples = failed_num_examples - 1
                    # print(self.bid_to_row_idx)
                    # print(self.bid_to_row_idx[str(bid)])
                    # print(eval_mode + '#ex')
                    # print(all_records)
                    all_records[self.bid_to_row_idx[bid] - 2][eval_mode + '#ex'] = failed_num_examples

                    # synthesis time
                    synth_time = (res['times'][str(success_num_examples)] / 1000) + bid_to_seconds[int(bid)]
                    all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'time'] = synth_time

                    # synthesized program
                    synth_program = res['programs'][str(success_num_examples)][0]['Program']
                    all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'program'] = synth_program

                    # synthesis results
                    synth_res = res['outcomes'][str(success_num_examples)]
                    pos_match_res, neg_match_res = [], []
                    # calculate precision and recall
                    for example_res in synth_res:
                        truth = example_res['Truth'].strip()
                        answer = example_res['Answer'].strip()

                        if truth == answer:
                            if truth == '':
                                neg_match_res.append(False)
                            else:
                                pos_match_res.append(True)
                        else:
                            if truth == '':
                                neg_match_res.append(True)
                            else:
                                pos_match_res.append(False)

                    precision, recall = self.calculate_precision_recall(pos_match_res, neg_match_res)
                    all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'precision'] = precision
                    all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'recall'] = recall

                    print('result for bid {} is {} with precision={}, recall={} in {} seconds'.format(bid, synth_program, precision, recall, synth_time))

        # we have a couple memory issues in a couple benchmark, we manually add the results here
        bid = '13'
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + '#ex'] = 6
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'time'] = 6 + 0.606
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'program'] = 'SubStr(v, SemPos(v, [(2019-05-01 18:00:28, 2019), (2020-05-04 23:43:05, 2020), (2020-05-24 00:39:21, 2020), (2020-05-31 11:59:09,  11), (2020-06-01 22:26:21,  22), (2019-04-30 19:02:14,  19)], "L"), SemPos(v, [(2019-05-01 18:00:28, 28), (2020-05-04 23:43:05, 43:05), (2020-05-24 00:39:21, 21), (2020-05-31 11:59:09, 31 ), (2020-06-01 22:26:21, 01 )    , (2019-04-30 19:02:14, 30 )], "R"))'
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'precision'] = 0
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'recall'] = 0

        bid = '12'
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + '#ex'] = 4
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'time'] = 11 + 0.178
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'program'] = 'SubStr(v, SemPos(v, [(232-E Series 32" Class (32" Diag.) LED 2160p, 232), (4-50" Class (49.5" Diag.) LED 2160p, 4), (825-32" Class (31.5" Diag.) LED 1080p, 825), (227-65" Class (64.5" Diag.) LED 1080p,  1080p)], "L"), SemPos(v, [(232-E Series 32" Class (32" Diag.) LED 2160p, 2160p), (4-50" Class (49.5" Diag.) LED 2160p, 2160p), (825-32" Class (31.5" Diag.) LED 1080p, 1080p), (227-65" Class (64.5" Diag.) LED 1080p, LED )], "R"))'
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'precision'] = 0.42857143
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'recall'] = 0.9

        bid = '33'
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + '#ex'] = 5
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'time'] = 23 + 0.296
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'program'] = 'SubStr(v, SemPos(v, [(near Lake Lindsey, near), (on the shore of Lake Jackson, on), (trib of Meadow Creek, trib), (Gifford Pinchot National Forest,  Forest), (N of Avenue B in town of Apalachicola,  Apalachicola)], "L"), SemPos(v, [(near Lake Lindsey, Lindsey), (on the shore of Lake Jackson, Jackson), (trib of Meadow Creek, Creek), (Gifford Pinch    ot National Forest, National ), (N of Avenue B in town of Apalachicola, town of )], "R"))'
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'precision'] = 0.25
        all_records[self.bid_to_row_idx[bid] - 2][eval_mode + 'recall'] = 0.2

        save_dict_to_csv('eval_res/{}.csv'.format(self.worksheet_name), all_records)
        if update_google_sheet and self.use_google_sheet:
            self.google_sheet.bulk_update(all_records)
