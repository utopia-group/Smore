"""
main synthesizer class
"""

import itertools
from typing import Tuple, List

import dateparser

from lib.config import pd_print
from lib.interpreter.context import StrContext
from lib.interpreter.executor import Executor
from lib.lang.grammar import CFG
from lib.program.program import Program
from lib.spec.synthesis_spec import SynthesisTask
from lib.utils.matcher_utils import num_text_to_num
from lib.utils.task_utils import analyze_examples


class Synthesizer:
    def __init__(self, grammar: CFG, executor: Executor, depth: int = 5):
        self.grammar: CFG = grammar
        self.executor: Executor = executor
        self.depth_limit: int = depth
        self.and_or_limit: int = depth - 1

        self.program_limit: int = 100000

        self.prog_id_counter = itertools.count(start=1)

        # need to cache some terminal info
        self.cc_value = self.grammar.get_terminal_sym('CC').values

    def synthesize(self, **args) -> List[Program]:
        raise NotImplementedError

    def eval_program(self, task: SynthesisTask, curr_p: Program) -> Tuple[List[bool], List[bool]]:
        pt = curr_p.to_pattern()
        return self.executor.exec_task(task, pt)

    def init_program(self, **kwargs) -> Program:
        raise NotImplementedError

    def instantiate_terminals_in_grammar(self, pos_examples: List[StrContext], neg_examples: List[StrContext]):
        """
        update the terminals used in the grammar based on what showed up in the examples
        """
        # print('pos examples: ', [s.s for s in pos_examples])
        # print('neg examples: ', [s.s for s in neg_examples])
        unique_alphabets, unique_nums, unique_syms, min_max_repeat = analyze_examples([s.s for s in pos_examples], [s.s for s in neg_examples])
        # print('unique alphabets: ', unique_alphabets)
        # print('unique nums: ', unique_nums)
        # print('unique syms: ', unique_syms)
        # print('min max repeat: ', min_max_repeat)

        const_sym = self.grammar.get_terminal_sym('CONST1')
        const_sym.values = []
        if len(unique_alphabets) <= 10:
            const_sym.values.extend(unique_alphabets)
        if len(unique_nums) <= 5:
            const_sym.values.extend(unique_nums)
        const_sym.values.extend(unique_syms)

        # constraint terminal space for repeat_num_sym
        repeat_num_sym = self.grammar.get_terminal_sym('NR')
        min_repeat, max_repeat = min_max_repeat
        if max_repeat > 10:
            max_repeat = 10
        if min_repeat > 10:
            min_repeat = 0
        repeat_num_sym.values = list(range(min_repeat, max_repeat + 1))

        # constraint terminal space for num_match
        # the reason we can do this is that we can only synthesize ranges assuming the example hints cover good extreme cases
        num_sym = self.grammar.get_terminal_sym('N')
        year_sym = self.grammar.get_terminal_sym('YEAR')
        month_sym = self.grammar.get_terminal_sym('MONTH')
        day_sym = self.grammar.get_terminal_sym('DATE')
        hour_sym = self.grammar.get_terminal_sym('HOUR')
        minute_sym = self.grammar.get_terminal_sym('MIN')
        second_sym = self.grammar.get_terminal_sym('SEC')

        numbers = []
        years = []
        months = []
        days = []
        hours = []
        minutes = []
        seconds = []

        for example in pos_examples + neg_examples:
            if 'integer' in example.type_to_str_spans:
                numbers.extend([num_text_to_num(example.get_substr(s)) for s in example.type_to_str_spans['integer']])

            if 'float' in example.type_to_str_spans:
                numbers.extend([num_text_to_num(example.get_substr(s)) for s in example.type_to_str_spans['float']])

            if 'year' in example.type_to_str_spans:
                years.extend([num_text_to_num(example.get_substr(s)) for s in example.type_to_str_spans['year']])

            if 'DATE' in example.type_to_str_spans:
                # parse the date and extract the year, month, day
                for s in example.type_to_str_spans['DATE']:
                    date_str = example.get_substr(s).strip()

                    # same thing, parse in specific way if we know the format, if not just call dateparser
                    if date_str.isdigit():
                        if len(date_str) == 2:
                            for year_prefix in ['13', '14', '15', '16', '17', '18', '19', '20']:
                                year = int(year_prefix + date_str)
                                if 0 < year < 3000:
                                    years.append(year)
                        elif len(date_str) == 4:

                            # 1. this might be a year
                            year = int(date_str)
                            if 0 < year < 3000:
                                years.append(year)

                            # 2. this might be a month + day
                            month = int(date_str[:2])
                            day = int(date_str[2:])
                            if 0 < month < 13 and 0 < day < 32:
                                months.append(month)
                                days.append(day)

                        elif len(date_str) == 6:
                            year_suffix = date_str[:2]
                            for year_prefix in ['13', '14', '15', '16', '17', '18', '19', '20']:
                                year = int(year_prefix + year_suffix)
                                if 0 < year < 3000:
                                    years.append(year)

                            month = int(date_str[2:4])
                            day = int(date_str[4:])
                            if 0 < month < 13 and 0 < day < 32:
                                months.append(month)
                                days.append(day)

                        elif len(date_str) == 8:
                            year = int(date_str[:4])
                            if 0 < year < 3000:
                                years.append(year)

                            month = int(date_str[4:6])
                            day = int(date_str[6:])
                            if 0 < month < 13 and 0 < day < 32:
                                months.append(month)
                                days.append(day)
                        else:
                            pass

                    elif date_str.count('-') == 2 or date_str.count('/') == 2 or date_str.count('.') == 2 or date_str.count(' ') == 2:
                        parsed_date = dateparser.parse(date_str)
                        if parsed_date is not None:
                            year = parsed_date.year
                            month = parsed_date.month
                            day = parsed_date.day
                            years.append(year)
                            months.append(month)
                            days.append(day)
                    elif date_str.count('-') == 1 or date_str.count('/') == 1 or date_str.count('.') == 1 or date_str.count(' ') == 1:
                        parsed_date = dateparser.parse(date_str)
                        if parsed_date is not None:
                            if date_str.split('-')[0].isdigit() and len(date_str.split('-')[0]) == 4 or \
                                    date_str.split('/')[0].isdigit() and len(date_str.split('/')[0]) == 4 or \
                                    date_str.split('.')[0].isdigit() and len(date_str.split('.')[0]) == 4 or \
                                    date_str.split(' ')[0].isdigit() and len(date_str.split(' ')[0]) == 4:
                                year = parsed_date.year
                                years.append(year)
                                month = parsed_date.month
                                months.append(month)
                            else:
                                month = parsed_date.month
                                day = parsed_date.day
                                months.append(month)
                                days.append(day)
                    else:
                        parsed_date = dateparser.parse(date_str)
                        if parsed_date is not None:
                            year = parsed_date.year
                            years.append(year)
                            month = parsed_date.month
                            months.append(month)
                            day = parsed_date.day
                            days.append(day)

            if 'TIME' in example.type_to_str_spans:
                # parse the time and extract the hour, minute, second
                for s in example.type_to_str_spans['TIME']:
                    time_str = example.get_substr(s).strip()
                    parsed_time = dateparser.parse(time_str)
                    if parsed_time is not None:
                        hour = parsed_time.hour
                        hours.append(hour)
                        minute = parsed_time.minute
                        minutes.append(minute)
                        second = parsed_time.second
                        seconds.append(second)

        num_sym.values = list(set([n for n in numbers if n is not None]))
        year_sym.values = [0, 2023] + list(set([n for n in years if n is not None]))
        month_sym.values = list(set([n for n in months if n is not None]))
        day_sym.values = list(set([n for n in days if n is not None]))
        hour_sym.values = list(set([n for n in hours if n is not None]))
        minute_sym.values = list(set([n for n in minutes if n is not None]))
        second_sym.values = list(set([n for n in seconds if n is not None]))

        # print('num_sym.values', num_sym.values)
        # print('year_sym.values', year_sym.values)
        # print('month_sym.values', month_sym.values)
        # print('day_sym.values', day_sym.values)
        # print('hour_sym.values', hour_sym.values)
        # print('minute_sym.values', minute_sym.values)
        # print('second_sym.values', second_sym.values)

        cc_sym = self.grammar.get_terminal_sym('CC')
        prev_cc_sym_value = self.cc_value
        cc_sym_remove = []
        # remove letter if unique_alphabets is 0
        if len(unique_nums) == 0:
            cc_sym_remove.append('num')
        if len(unique_alphabets) == 0:
            cc_sym_remove.append('let')
            cc_sym_remove.append('cap')
            cc_sym_remove.append('low')
            cc_sym_remove.append('word')
        if len(unique_syms) == 0:
            pass
        # print('cc_sym_remove', cc_sym_remove)
        cc_sym.values = [v for v in prev_cc_sym_value if v not in cc_sym_remove]
        pd_print('cc_sym.values {}'.format(cc_sym.values))
