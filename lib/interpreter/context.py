import re
from typing import Set, List, Tuple, Dict, Optional

import fuzzywuzzy.fuzz

from lib.config import pd_print, PIPELINE_DEBUG
from lib.interpreter.span import MatchSpan
from lib.lang.constants import ENT_TAGS
from lib.nlp.nlp import NLPFunc, date_template_match3
from lib.type.type_enum import parsable_types, entity_types
from lib.type.type_system import get_subtypes
from lib.utils.context_utils import context_tokenize_similarity_matching
from lib.utils.matcher_utils import get_number_spans, int_float_ent_matcher_helper, ent_matcher_helper, merge_entity_match_query, get_state_from_city, split_entity_match_query, \
    filepath_related_helpers


class Context:
    """
    Trace the intermediate results of execution
    """

    def __init__(self):
        pass


class EntityPreprocessContext(Context):
    """
    context for preprocessing entity
    """
    def __init__(self, types: List[str], nlp_func: NLPFunc):
        super().__init__()
        self.types: List[str] = types
        self.nlp_func = nlp_func

    def process(self, s: str, ty: str) -> List[MatchSpan]:
        spans: List[MatchSpan] = []
        match ty:
            case 'charseq' | 'string':
                # no entity in CharSeq
                pass
            case 'integer' | 'float' | 'Int' | 'int' | 'INT' | 'FLOAT' | 'Integer':
                # call integer/float match
                if ty.lower() == 'int' or ty.lower() == 'integer':
                    ty = 'integer'
                if ty.lower() == 'float':
                    ty = 'float'
                num_to_type_map = {'integer': int, 'float': float}
                spans.extend(get_number_spans(s, num_to_type_map[ty.lower()]))
                cardinality_entities: List = self.nlp_func.get_entity_with_tag(s, 'CARDINAL')
                int_float_ent_matcher_helper(spans, cardinality_entities, num_to_type_map[ty.lower()])
                pd_print('number spans: {}'.format(spans))
            case 'DATE' | 'TIME':
                entities = self.nlp_func.get_entity_with_tag(s, ty)
                ent_matcher_helper(spans, entities)

                # special cases for date and time
                if entities is None and re.search(date_template_match3, s):
                    date_str = re.search(date_template_match3, s).group(0)

                    # let's verify this is a proper date
                    year = int(date_str[:2])
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])

                    if year < 0 or year > 99 or month < 1 or month > 12 or day < 1 or day > 31:
                        pass
                    else:
                        # get the start index for date
                        date_start_idx = s.index(date_str)
                        date_span = (date_start_idx, date_start_idx + len(date_str))
                        ent_matcher_helper(spans, [(date_str, MatchSpan(date_span))])

                # print('date time entities:', entities)
            case ent if ent in ENT_TAGS:
                entities = self.nlp_func.get_entity_with_tag(s, ent)
                ent_matcher_helper(spans, entities)
            case _:
                # call gpt3 only
                ty_gpt3_res = self.nlp_func.query_entity_generation(s, ty)
                # print('ty_gpt3_res:', ty_gpt3_res)

                # ugh manual filtering
                # for some reason gpt think date such as 1880-1980 is a location
                # if ty in ['nationality', 'location']:
                #     new_ty_gpt3_res = []
                #     for ent in ty_gpt3_res:
                #         if all(c.isdigit() for c in ent[:4]) and all(c.isdigit() for c in ent[-4:]):
                #             pass
                #         else:
                #             new_ty_gpt3_res.append(ent)
                #     ty_gpt3_res = new_ty_gpt3_res

                # print('ty_gpt3_res after:', ty_gpt3_res)
                merged_entities = merge_entity_match_query(s, ty_gpt3_res, self.nlp_func)
                # print('merged entities:', merged_entities)
                spans.extend(merged_entities)
                # ty_chatgpt_res = self.nlp_func.query_entity_generation_chatgpt(s, ty)
                # merged_entities_chatgpt = merge_entity_match_query(s, ty_chatgpt_res, self.nlp_func)
                # print('merged entities chatgpt:', merged_entities_chatgpt)
                # spans.extend(merged_entities_chatgpt)
                split_entities = split_entity_match_query(s, ty_gpt3_res, ty, self.nlp_func)
                # print('split entities:', split_entities)
                spans.extend(split_entities)

                # we need a special entity parser for url-related entities
                if ty.lower() in ['directory', 'path', 'file name', 'file']:
                    additional_entities = filepath_related_helpers(s, ty)
                    # print('additional entities:', additional_entities)
                    spans.extend(additional_entities)

                pd_print('all entities after post-processing: {}'.format([s[sp.start:sp.end] for sp in spans]))

        spans = list(set(spans))
        return spans

    def process_predicate(self, test_str: str, ty: str) -> Dict[str, list]:
        """
        preprocess the predicate for places
        """
        res: Dict[str, List] = {}

        # we start with city
        city_name = self.nlp_func.place_context_generation(test_str, 'city')
        # print('city name:', city_name)
        if city_name is None:
            pass
        else:
            # check the fuzzy ratio
            if city_name.lower() == test_str.lower():
                res['city'] = [city_name]
                test_str = city_name
            elif 80 <= fuzzywuzzy.fuzz.token_sort_ratio(city_name, test_str) < 100:
                test_str = 'SPELLING_ERROR'
            elif city_name[0].lower() != test_str[0].lower():
                test_str = 'SPELLING_ERROR'
            else:
                res['city'] = [city_name]
                test_str = city_name

        # print('test_str:', test_str)

        # we then use the previous results to check us state
        state_name_gpt = self.nlp_func.place_context_generation(test_str, 'US state')
        # print('state name:', state_name_gpt)
        state_name_symbolic = get_state_from_city(test_str)
        # print('state name symbolic:', state_name_symbolic)
        if state_name_gpt is not None and state_name_gpt not in state_name_symbolic:
            state_name_symbolic.append(state_name_gpt)

        if len(state_name_symbolic) > 0:
            res['state'] = state_name_symbolic
        else:
            pass

        # we then use the previous results to check country
        # note that since state is us only, in this case we can just query test_str directly
        country_name = self.nlp_func.place_context_generation(test_str, 'country')
        if country_name is not None:
            res['country'] = [country_name]
            if 'state' in res and len(res['state']) > 0 and 'United States' not in res['country']:
                res['country'].append('United States')
        else:
            if 'state' in res and len(res['state']) > 0:
                res['country'] = ['United States']

        # we finally check continent
        continent_names = []
        if 'country' in res:
            for country in res['country']:
                continent_name = self.nlp_func.place_context_generation(country, 'continent')
                if continent_name is not None:
                    continent_names.append(continent_name)

        continent_name = self.nlp_func.place_context_generation(test_str, 'continent')
        if continent_name is not None and continent_name not in continent_names:
            continent_names.append(continent_name)

        if len(continent_names) > 0:
            res['region'] = continent_names
        else:
            pass

        return res


class StrContext(Context):
    """
    current progress have been made on a string
    NOTE: this should have a data structure that can be easily copied
    """

    def __init__(self, s: str, preprocess_context: Optional[Context], token_mode: bool = False):
        super().__init__()
        self.s: str = s
        self.s_orig: str = "None"

        self.space_spans: List[MatchSpan] = []
        self.empty_spans: List[MatchSpan] = []
        self.span_universe: Set[MatchSpan] = set()

        self.type_to_str_spans: Dict[str, list[MatchSpan]] = {}
        self.str_span_to_predicate_context: Dict[MatchSpan, Dict[str, List[str]]] = {}

        self.preprocess(preprocess_context, token_mode=token_mode)
        # pd_print(self)

    def preprocess(self, preprocess_context: Optional[Context], token_mode: bool):
        """
        init the preprocessing fields in the context
        preprocess includes:
            - standardize white_space if token_mode is enabled
            - generate all space_spans
            - generate all empty_spans
            - generate the universe of spans (used for complement)
        """
        self.s_orig = self.s

        if token_mode:
            """
            if token_mode, then normalize the string (i.e. one space between each token)
            tokenization is simple, just split by space (or multiple space)
            """
            # clean all the comma
            self.s = self.s.replace(',', ' ')
            self.s = self.s.replace(';', ' ')
            self.s = self.s.replace('|', ' | ')
            self.s = ' '.join(self.s.split())

        from lib.utils.matcher_utils import regex_match
        from lib.utils.matcher_utils import get_space_pattern
        self.space_spans = regex_match(self.s, get_space_pattern())
        self.empty_spans = [MatchSpan(i, i) for i in range(len(self.s) + 1)]

        self.span_universe = set()
        for i in range(len(self.s) + 1):
            for j in range(i, len(self.s) + 1):
                self.span_universe.add(MatchSpan(i, j))

        if isinstance(preprocess_context, StrContext):
            type_to_str_spans_new = {}
            for key, spans in preprocess_context.type_to_str_spans.items():
                spans_new = []
                for span in spans:
                    span_str = preprocess_context.get_substr(span)
                    if span_str in self.s:
                        spans_new.append(MatchSpan(self.s.index(span_str), self.s.index(span_str) + len(span_str)))
                type_to_str_spans_new[key] = list(set(spans_new))
            self.type_to_str_spans = type_to_str_spans_new

            str_span_to_predicate_context_new = {}
            for span, context in preprocess_context.str_span_to_predicate_context.items():
                span_str = preprocess_context.get_substr(span)
                if span_str in self.s:
                    str_span_to_predicate_context_new[MatchSpan(self.s.index(span_str), self.s.index(span_str) + len(span_str))] = context
            self.str_span_to_predicate_context = str_span_to_predicate_context_new

        elif isinstance(preprocess_context, EntityPreprocessContext):
            for ty in preprocess_context.types:
                # print('processing ', ty)
                all_entity_types = {**parsable_types, **entity_types}
                if ty in ['integer', 'float', 'number', 'int']:
                    if ty == 'int':
                        ty = 'integer'
                    self.type_to_str_spans[ty] = preprocess_context.process(self.s, ty)
                elif ty in ['date', 'time']:
                    self.type_to_str_spans[ty.upper()] = preprocess_context.process(self.s, ty.upper())
                elif ty in ['place', 'city', 'country', 'continent', 'state', 'nationality', 'org']:

                    if ty == 'place':
                        entity_tag, type_tags = 'GPE', ['city', 'country', 'state', 'continent', 'place']
                    elif ty == 'nationality':
                        entity_tag, type_tags = 'NORP', ['nationality']
                    elif ty == 'org':
                        entity_tag, type_tags = 'ORG', ['organization']
                    else:
                        entity_tag, type_tags = None, [ty]

                    outputs = []
                    if entity_tag is not None and entity_tag != 'NORP':
                        outputs.extend(preprocess_context.process(self.s, entity_tag))

                    for type_tag in type_tags:
                        outputs.extend(preprocess_context.process(self.s, type_tag))
                    self.type_to_str_spans[ty.upper()] = list(set(outputs))

                    # print('self.type_to_str_spans:', self.type_to_str_spans)

                    explored_spans = []
                    for span in self.type_to_str_spans[ty.upper()]:
                        if span in explored_spans:
                            continue
                        else:
                            s = self.get_substr(span)
                            self.str_span_to_predicate_context[span] = preprocess_context.process_predicate(s, ty)
                    # print('self.str_span_to_predicate_context:', self.str_span_to_predicate_context)

                elif ty in all_entity_types:
                    # this means we can run both entity and gpt3 on the string
                    entity_tag = all_entity_types.get(ty)
                    self.type_to_str_spans[entity_tag] = list(set(preprocess_context.process(self.s, entity_tag)))
                    self.type_to_str_spans[ty] = list(set(preprocess_context.process(self.s, ty)))

                else:
                    # string that matched by a certain type is the union of strings matched itself and all its subtypes
                    if ty.lower() in ['charseq', 'string']:
                        pass
                    else:
                        super_type = ty
                        subtypes = get_subtypes(ty)
                        super_type_matched_str = []
                        if len(subtypes) > 1:
                            for subty in subtypes:
                                spans = preprocess_context.process(self.s, subty.name)
                                self.type_to_str_spans[subty.name] = list(set(spans))
                                super_type_matched_str.extend(list(set(spans)))
                        else:
                            spans = preprocess_context.process(self.s, super_type)
                            super_type_matched_str.extend(list(set(spans)))

                        self.type_to_str_spans[super_type] = super_type_matched_str
        else:
            pass

    def get_end_index(self) -> int:
        """
        get the end index of this string (to create a MatchSpan)
        """
        return len(self.s)

    def get_substr(self, span: MatchSpan) -> str:
        return self.s[span.start:span.end]

    def get_substr_regex(self, span: MatchSpan) -> str:
        return ''.join([c if c.isalnum() else '\{}'.format(c) for c in self.get_substr(span)])

    def create_sub_context(self, span: MatchSpan | Tuple[int, int]) -> 'StrContext':
        """
        given a span, create a new StrContext that captures the span only
        """
        # print("create_sub_context: {} {}".format(self.s, span))
        if isinstance(span, MatchSpan):
            assert span.start <= len(self.s) and span.end <= len(self.s)
            new_str = self.s[span.start: span.end]
        else:
            assert isinstance(span, Tuple) and len(span) == 2
            assert span[0] <= len(self.s) and span[1] <= len(self.s)
            new_str = self.s[span[0]: span[1]]

        return StrContext(new_str, preprocess_context=self)

    def subspan_gen_for_similarity_matching(self) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        generate all the possible subspan of this current context so that we can use it for similarity matching
        TODO: this should be cached
        """

        # first tokenize the context according to the string similarity matching criteria
        context_tokenize = context_tokenize_similarity_matching(self.s)

        # then create a list that's len(tokenize_context) * len(tokenize_context) size that consists of all possible sub-spans of the context
        all_subspan_context = []
        all_subspan_range = []
        for start_idx in range(len(context_tokenize)):
            for end_idx in range((start_idx + 1), len(context_tokenize) + 1):
                tokenized_context_selected = context_tokenize[start_idx:end_idx]
                all_subspan_context.append(' '.join([token[0] for token in tokenized_context_selected]))
                all_subspan_range.append((tokenized_context_selected[0][1], tokenized_context_selected[-1][2]))

        return all_subspan_context, all_subspan_range

    def __repr__(self):
        if self.s == self.s_orig:
            if PIPELINE_DEBUG:
                return 'StrContext(s="{}", context={}, {})'.format(self.s, self.type_to_str_spans, self.str_span_to_predicate_context)
            else:
                return self.s
        else:
            if PIPELINE_DEBUG:
                return 'StrContext(s="{}", s_orig="{}")'.format(self.s, self.s_orig)
            else:
                return self.s
