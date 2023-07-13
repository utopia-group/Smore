from typing import List, Tuple

from lib.lang.grammar import CFG
from lib.spec.synthesis_spec import SynthesisTask
from lib.utils.matcher_utils import check_regions, check_countries


def add_place_predicate_context(task: SynthesisTask, grammar: CFG):
    """
    Finds all countries and regions in the provided grammar that appear in the string st.
    Adds these countries and regions to the task context dictionary (to the entries 'inCountry',
    'inRegion', and 'inState' respectively).
    """

    if 'processed' not in task.context:
        task.context['processed'] = set()

    processed_strs = task.context['processed']
    new_strings = [s for s in task.pos_str_context + task.neg_str_context if s.s not in processed_strs]
    if len(new_strings) == 0:
        return

    processed_strs.update([s.s for s in new_strings])

    if task.no_context:

        regions_in_grammar = grammar.get_terminal_sym('REGION').values
        relevant_regions = []
        for region in regions_in_grammar:
            if any([check_regions(s.s, [region]) for s in new_strings]):
                relevant_regions.append(region)

        task.context['inRegion'] = relevant_regions

        countries_in_grammar = grammar.get_terminal_sym('COUNTRY').values
        relevant_countries = []
        for country in countries_in_grammar:
            if any([check_countries(s.s, [country]) for s in new_strings]):
                relevant_countries.append(country)

        task.context['inCountry'] = relevant_countries

        task.context['inState'] = grammar.get_terminal_sym('STATE').values
    else:

        def find_entity(ty: str) -> List[str]:
            relevant_entity = []
            for s_ctx in new_strings:
                for span_ctx in s_ctx.str_span_to_predicate_context.values():
                    if ty in span_ctx and span_ctx.get(ty) not in relevant_entity:
                        relevant_entity.extend(span_ctx.get(ty))
            return list(set(relevant_entity))

        task.context['inRegion'] = find_entity('region')
        task.context['inCountry'] = find_entity('country')
        task.context['inState'] = find_entity('state')


def analyze_examples(pos_examples: List[str], neg_examples: List[str]) -> Tuple[List[str], List[str], List[str], Tuple[int, int]]:
    """
    given a task, analyze its information to prepare for the synthesis task
    for example, find out what const strings are necessary
    This function should be conditioned on token mode
    """
    unique_alphabets = []
    unique_nums = []
    unique_symbols = []

    for string in pos_examples + neg_examples:
        for c in string:
            if c.isalpha():
                unique_alphabets.append(c)
            elif c.isdigit():
                unique_nums.append(c)
            elif c.isascii():
                unique_symbols.append(c)
            else:
                pass

    repeat_min = min(len(s) for s in pos_examples + neg_examples)
    repeat_max = max(len(s) for s in pos_examples + neg_examples)

    return list(set(unique_alphabets)), list(set(unique_nums)), list(set(unique_symbols)), (repeat_min, repeat_max)
