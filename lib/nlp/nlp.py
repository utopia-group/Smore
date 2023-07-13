"""
Wrapper function for all the NLP functionalities
"""
import os
import re
from collections import defaultdict

import spacy

from typing import Dict, Any, List, Tuple, Optional

import wikipediaapi
from sentence_transformers import SentenceTransformer, util
from datamuse import Datamuse
from nltk.corpus import wordnet as wn
from spacy.matcher import DependencyMatcher
from tokenizers.pre_tokenizers import ByteLevel

from lib.cache import cache
from lib.config import pd_print, config
from lib.lang.constants import ENT_TAGS
from lib.interpreter.span import MatchSpan
from lib.nlp.gpt import GPT, get_response
from lib.utils.eval_utils import save_response
from lib.utils.matcher_utils import double_check_ent, find_date, sub_date_time_detect
from lib.utils.gpt3_utils import parse_gpt3_output_for_entity_classification, generate_prompt_for_entity_classification, get_inferred_categories_pattern, \
    generate_prompt_for_infer, generate_prompt_for_entity_generation, parse_gpt3_output_for_entity_generation, generate_prompt_for_place_context, \
    parse_gpt3_output_for_place_context, generate_exec_prompt, parse_gpt3_output_for_exec
from lib.utils.wiki_utils import get_wiki_template_dict


date_template_match1 = r'(^|[^-])\d{4}-(\d{4}|\d{2})($|[^\d-])'
date_template_match2 = r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}'
date_template_match3 = r'\d{6}'


def update_ent_dict_with_entities(new_entities: Dict[str, List], ents_dict: Dict[str, List[Tuple[str, MatchSpan]]], start_idx: int) \
        -> Dict[str, List[Tuple[str, MatchSpan]]]:
    if len(new_entities) > 0:
        for ent_label, ent_lists in new_entities.items():
            for el in ent_lists:
                ents_dict[ent_label].append((el[0], el[1].update_span(start_idx)))

    return ents_dict


class NLPFunc:
    def __init__(self, disable=False, debug=False):
        self.disable = disable
        self.device = 'cpu'  # init this to cuda later if we are running this on gpu
        self.debug = debug
        self.spacy = self.load_spacy()
        self.spacy_matcher = DependencyMatcher(self.spacy.vocab)
        self.spacy_matcher.add("PARSE_GPT3_INFER", [get_inferred_categories_pattern])

        self.sbert = self.load_sbert()
        self.sbert_cache: Dict[str, Any] = {}
        self.byte_tokenizer = ByteLevel()
        self.gpt3 = GPT(os.getenv("OPENAI_API_KEY"))
        self.gpt3_entity_model = 'text-davinci-003'
        self.datamuse = Datamuse()
        # self.wiki = wikipediaapi.Wikipedia('en')
        self.wiki = None

        self.spacy_cache: Dict[str, Any] = {}
        self.datamuse_cache: Dict[str, Any] = {}
        self.wiki_cache: Dict[str, Any] = {}
        self.wordnet_cache: Dict[str, Any] = {}
        self.gpt3_cache: Dict[str, Dict] = {}

    def d_print(self, *args):
        if self.debug:
            print(args[0])

    def read_cache(self):
        raise NotImplementedError

    def write_cache(self):
        raise NotImplementedError

    """
    Following are gpt-3 related
    probably need to add something else 
    """
    def call_gpt(self, prompt: str, temperature=0, max_token=256, stop=None, model='text-davinci-003') -> Dict:
        if temperature == 0:
            return self.call_gpt_deterministic(prompt, max_token, stop, model)
        else:
            return self.gpt3.call(prompt, max_token, temperature, stop, model)

    @cache(ignore_args=[0])
    def call_gpt_deterministic(self, prompt: str, max_token: int, stop, model: str) -> Dict:
        if not config.AE:
            response = self.gpt3.call(prompt, max_token, 0, stop, model)
        else:
            response = None
        # we cannot cache response
        if response is None:
            raise Exception("GPT-3 call failed")
        return response

    def query_gpt_exec(self, positive_str: List[str], negative_str: List[str], query: str, model: str) -> bool:
        prompt = generate_exec_prompt(positive_str, negative_str, query)
        response = self.call_gpt(prompt, max_token=16, stop=['\\n', 'Matched?'], model=model)

        return parse_gpt3_output_for_exec(get_response(response, model))

    def query_entity_classification(self, test_str: str, query: str, positive_str: Optional[List[str]] = None, negative_str: Optional[List[str]] = None) -> bool:
        # query entity to gpt3 as a classification task

        prompt = generate_prompt_for_entity_classification(test_str, query, positive_str, negative_str)
        # print("query_entity generated prompt: {}".format(prompt))
        response = self.call_gpt(prompt, max_token=16, stop=['\\n'], model=self.gpt3_entity_model)

        return parse_gpt3_output_for_entity_classification(get_response(response, self.gpt3_entity_model))

    def query_entity_generation(self, test_str: str, query: str, positive_str: Optional[List[str]] = None, negative_str: Optional[List[str]] = None) -> List[str]:
        # an initial version of given a string, find all the substring of a certain entity according to gpt3

        prompt = generate_prompt_for_entity_generation(test_str, query, positive_str, negative_str)
        # print("query_entity generated prompt: {}".format(prompt))
        response = self.call_gpt(prompt, max_token=256, stop=['\\n'], model=self.gpt3_entity_model)

        result = parse_gpt3_output_for_entity_generation(get_response(response, self.gpt3_entity_model), test_str)
        pd_print("query_entity_generation result for entity {}: {}".format(query, result))
        return result

    def query_entity_generation_chatgpt(self, test_str: str, query: str, positive_str: Optional[List[str]] = None, negative_str: Optional[List[str]] = None) -> List[str]:
        # an initial version of given a string, find all the substring of a certain entity according to gpt3

        prompt = generate_prompt_for_entity_generation(test_str, query, positive_str, negative_str)
        # print("query_entity generated prompt: {}".format(prompt))
        response = self.call_gpt(prompt, max_token=256, stop=['\\n'], model='gpt-3.5-turbo')

        result = parse_gpt3_output_for_entity_generation(get_response(response, 'gpt-3.5-turbo'), test_str)
        pd_print("query_entity_generation result for entity {}: {}".format(query, result))
        return result

    def query_entity_generation_verifier(self, s: str, test_str: str, entity_type: str) -> bool:
        def prompt_generator(s1: str, test_str1: str, entity_type1: str, label: str = '') -> str:
            if label == '':
                return 'Q: Does \'{}\' represent a {} name in the string \'{}\'?\nA:'.format(test_str1, entity_type1, s1)
            else:
                return 'Q: Does \'{}\' represent a {} name in the string \'{}\'?\nA: {}.\n'.format(test_str1, entity_type1, s1, label)

        prompt = ''
        prompt += prompt_generator('VINTAGE  2 METER FOLDING RULER', 'Folding Ruler', 'product', 'Yes') + '\n'
        prompt += prompt_generator('BURGER KING 4525', 'Burger King', 'person', 'No') + '\n'
        prompt += prompt_generator(s, test_str, entity_type)

        response = self.call_gpt(prompt, max_token=256, stop=['\\n'], model=self.gpt3_entity_model)
        return parse_gpt3_output_for_exec(get_response(response, self.gpt3_entity_model))

    def infer_query(self, positive_str: List[str], negative_str: List[str]) -> str:
        """
        infer the query that will be used to run query_entity
        this is a synthesizer function
        """
        prompt = generate_prompt_for_infer(positive_str, negative_str)
        # print("infer_query generated prompt: {}".format(prompt))
        response = self.call_gpt(prompt, max_token=256, stop=['\\n', '.'], model=self.gpt3_entity_model)
        # print("infer_query response:", response)

        return get_response(response, self.gpt3_entity_model).strip().lower()

    def call_gpt3_and_get_answer(self, prompt: str, mode: str, openai_mode: str, stop: List[str] = ['\n', 'Summarize', 'Find'], temperature: int = 0) -> str:
        try:
            response = self.call_gpt(prompt, temperature=temperature, stop=stop, model=openai_mode)
            # print(response)
            answer = save_response(get_response(response, openai_mode))
        except Exception as e:
            if not config.AE:
                print("Exception: {}".format(e))
            return ''

        return answer.strip()

    def place_context_generation(self, test_str: str, query: str) -> Optional[str]:
        prompt = generate_prompt_for_place_context(test_str, query)
        response = self.call_gpt(prompt, temperature=0, stop=['\\n'], model=self.gpt3_entity_model)

        return parse_gpt3_output_for_place_context(get_response(response, self.gpt3_entity_model))

    """
    Following functions are tokenizer-related
    """

    def tokenize(self, context: str) -> List[Tuple[str, Tuple]]:
        """
        need to format the output here
        need to think
        (1) but the output should definitely switch from List[Tuple[str, Tuple[int, int]]] to List[Tuple[str, Span]]
        (2) and we need to think about how to deal with the starting characters.

        merging:
        - do not tokenize for fractional number
        """

        def process_buffered_token(buffer_tokens: List) -> Tuple[str, Tuple]:
            # merge buffer_tokens
            merged_token = ''.join([t for t, _ in buffer_tokens])
            merged_span = (buffer_tokens[0][1][0], buffer_tokens[-1][1][1])

            # despite the beginning Ġ, remove the rest of the Ġ with space
            new_merged_token = merged_token[0] + merged_token[1:].replace('Ġ', ' ')

            return new_merged_token, merged_span

        tokenzied_str = self.byte_tokenizer.pre_tokenize_str(context)
        # print("pre tokenized_str:", tokenzied_str)

        # merge fractional/decimal, then merge integer + fractionals (may exist space)
        # examples are: 15.4, 15/4, 15 1/4
        postprocessed_tokenized_str = []
        idx = 0
        buffered_token = []
        while idx < len(tokenzied_str):
            # need to use look-a-head
            token, span = tokenzied_str[idx]
            # print("token, span:", token, span)

            if token.startswith('Ä '):
                if token[1:].isdigit():
                    buffered_token.append((token, span))
                elif token[1:] == '.':
                    buffered_token.append((token, span))
                else:
                    if len(buffered_token) > 0:
                        postprocessed_tokenized_str.append(process_buffered_token(buffered_token))
                        buffered_token = []
                    postprocessed_tokenized_str.append((token, span))
            elif token.startswith('Â'):
                if len(buffered_token) > 0:
                    postprocessed_tokenized_str.append(process_buffered_token(buffered_token))
                    buffered_token = []
                postprocessed_tokenized_str.append((token.replace('Â', 'Ġ'), span))
            else:
                if len(buffered_token) > 0:
                    if token.isdigit() or token == '.' or token == '/':
                        buffered_token.append((token, span))
                    else:
                        postprocessed_tokenized_str.extend(buffered_token)
                        buffered_token = []
                        postprocessed_tokenized_str.append((token, span))
                else:
                    postprocessed_tokenized_str.append((token, span))

            # print("postprocessed_token_sr:", postprocessed_tokenized_str)
            # print("buffered_token:", buffered_token)

            idx += 1

        # print(postprocessed_tokenized_str)
        return postprocessed_tokenized_str

    """
    Following functions are onelook/wordnet-related
    """

    def find_synonym(self, context: str, mode: str, _max: int = 20) -> List[str]:

        key = str((context, mode))

        if self.datamuse_cache.get(key) is None:

            if mode == 'ONELOOK':

                query_res = self.datamuse.words(ml=context, max=_max)
                self.datamuse_cache[key] = [word['word'] for word in query_res]

            elif mode == 'WORDNET':

                word = wn.sysnets(context)
                word_hypernym = word[0].hypernyms[0]
                self.datamuse_cache[key] = list(set([w for s in word_hypernym.closure(lambda s: s.hyponyms()) for w in s.lemma_names()]))

            else:
                raise NotImplementedError

        return self.datamuse_cache.get(key)

    """
    Following functions are wikipedia-related
    """

    def find_wiki_related_terms(self, context: str, rank: str = 'last') -> List[str]:
        """
        heuristic: smallest: template with the smallest number of entry
                    largest: ... largest ...
                    all: get strings in all related templates
        """

        raise NotImplementedError

        # if self.wiki_cache.get(context) is None:
        #
        #     page_py = self.wiki.page(context)
        #     terms = []
        #
        #     # find templates
        #     templates_pages = [entry for entry in page_py.links.items() if 'template' in entry[0].lower() and entry[1].namespace == 10]
        #     print(templates_pages)
        #     template_entries = [list(get_wiki_template_dict(template[0].replace(' ', '_'))) for template in templates_pages]
        #
        #     if len(template_entries) == 0:
        #         return terms
        #
        #     if rank == 'all':
        #         templates_entry = [e for template in template_entries for e in template]
        #     elif rank == 'last':
        #         templates_entry = template_entries[-1]
        #     elif rank == 'smallest':
        #         templates_entry = list(sorted(template_entries, key=len))[0]
        #     elif rank == 'largest':
        #         templates_entry = list(sorted(template_entries, key=len, reverse=True))[0]
        #     else:
        #         raise ValueError('rank={} is not supported'.format(rank))
        #
        #     # post-process entry
        #     self.d_print("find_wiki_related_terms before post-process {}".format(templates_entry))
        #
        #     for string in templates_entry:
        #         # 1. get rid of all the brackets
        #         string_split = string.split('(')
        #
        #         # 2. only get those with len 1 of token (i.e. no space)
        #         if ' ' not in string_split[0]:
        #             terms.append(string_split[0])
        #
        #     self.d_print("find_wiki_related_terms returns {}".format(terms))
        #
        #     self.wiki_cache[context] = terms
        #
        # return self.wiki_cache[context]

    """
    Following functions are sbert related
    """

    def load_sbert(self):
        if not self.disable:
            sbert_model = SentenceTransformer('all-mpnet-base-v2')
            # sbert_model = SentenceTransformer('sentence-t5-xxl')      # according to huggingface this is fine-tuned to work for sentence similarity
            return sbert_model

    def get_subspan_similarity(self, keyword: str, threshold: float, context: Tuple[str, Tuple], similarity_fuc=util.cos_sim) -> List[MatchSpan]:
        """
        find the span of words in context that passes similarity threshold to keyword using similarity_func
        FYI, sbert use character-level tokenization
        Note: need to find all possible span of the words
            The one implemented for now has O(n^2) complexity algorithm for now, optimize later
        """

        s, (all_subspan_context, all_subspan_range) = context

        if s not in self.sbert_cache:
            self.sbert_cache[s] = self.sbert.encode(all_subspan_context)

        if keyword not in self.sbert_cache:
            self.sbert_cache[keyword] = self.sbert.encode(keyword)

        # then we feed all subspan_context to model.encode()
        keyword_embedding = self.sbert_cache[keyword]
        subspan_context_embeddings = self.sbert_cache[s]
        cosine_sim_with_keyword = similarity_fuc(keyword_embedding, subspan_context_embeddings).squeeze().tolist()
        # wrap this in a list if the string has only one element
        if isinstance(cosine_sim_with_keyword, float):
            cosine_sim_with_keyword = [cosine_sim_with_keyword]
        # print('cosine_sim_with_keyword:', cosine_sim_with_keyword)
        all_index_geq_threshold = list(x[0] for x in enumerate(cosine_sim_with_keyword) if x[1] > threshold)

        # similar_subspans = [(
        #     context[all_subspan_range[idx][0]: all_subspan_range[idx][1]],
        #     MatchSpan(all_subspan_range[idx][0], all_subspan_range[idx][1])) for idx in all_index_geq_threshold]

        similar_subspans = [MatchSpan(all_subspan_range[idx][0], all_subspan_range[idx][1]) for idx in all_index_geq_threshold]
        return similar_subspans

    """
    Following functions are spacy-related
    """

    def load_spacy(self):
        if not self.disable:
            spacy_model = spacy.load("en_core_web_md")
            return spacy_model

    def get_spacy_context(self, context: str):
        if context not in self.spacy_cache:
            self.spacy_cache[context] = self.spacy(context)

        return self.spacy_cache[context]

    def get_entity(self, context: str) -> Dict[str, List[Tuple[str, MatchSpan]]]:
        """
        return format of this function: ENT_LABEL: [(ENT_TEXT, ENT_SPAN), ...]
        Here is the workflow of entity matching
        Try the whole string, if get rid of the sub-string that has entity information, then keep match the rest
        123 USA ->(no) 123 CARD
                    -> USA Ent
        123USA ->(no)  123 USA ->(no) 123 CARD
                                    USA Ent
        FIXME: the current workflow is very not efficient but at least it does the thing I want
        TODO: add special matcher for DATETIME
        """

        self.d_print("\nnlp.pt get_entity_func for context: {}".format(context))

        ents_dict = defaultdict(list)
        context_spacy = self.get_spacy_context(context)

        matched_spans: List[Tuple] = []

        self.d_print("original entity: {}".format([(ent.text, ent.label_) for ent in context_spacy.ents]))

        for ent in context_spacy.ents:
            ent_span = (ent.start_char, ent.end_char)

            # we need to filter out some over-used tags for precision
            if ent.label_ == 'CARDINAL' or ent.label_ == 'PRODUCT':
                # we need to filter out some over-used tags for precision
                pass_check, alternative_label = double_check_ent(ent.text)
                if not pass_check:
                    if alternative_label is None:
                        continue
                    else:
                        ent.label_ = alternative_label
            elif ent.label_ not in ENT_TAGS:
                continue

            matched_spans.append(ent_span)
            ents_dict[ent.label_].append((ent.text, MatchSpan(ent_span)))

            if ent.label_ == 'ORG':
                # for some reason, entities start with '-' should be filtered out
                org_text = ent.text
                if org_text.startswith('-'):
                    org_text = context.replace('-', '')
                    org_start_idx = context.index(org_text)
                    org_span = (org_start_idx, org_start_idx + len(org_text))
                    ents_dict[ent.label_].append((org_text, MatchSpan(org_span)))

            if ent.label_ == 'DATE':
                # we need to extract substrings just in case the match is not perfect
                # for example, if '1833-1910|American' is DATE, then we should add '1833' and '1910' as DATE entity
                # NOTE: add more hacks in the future
                processed_context = context.replace('|', ' ')
                # print('processed_context:', processed_context)

                template_1_match = re.finditer(date_template_match1, processed_context)
                template_2_match = re.finditer(date_template_match2, processed_context)
                template_3_match = re.finditer(date_template_match3, processed_context)
                template_match = False

                # sorry need to hack a bit here because this is very difficult to get things right
                if template_1_match is not None:
                    template_match = True
                    for res in template_1_match:
                        new_processed_context = res.group(0)
                        processed_context_split = new_processed_context.split('-')
                        year_1_str = processed_context_split[0].strip()
                        year_2_str = processed_context_split[1].strip()
                        # print('year_1_str:', year_1_str)
                        # print('year_2_str:', year_2_str)

                        # get the start index for year 1
                        year_1_start_idx = context.index(year_1_str)
                        year_1_span = (year_1_start_idx, year_1_start_idx + len(year_1_str))
                        matched_spans.append(year_1_span)
                        ents_dict['DATE'].append((year_1_str, MatchSpan(year_1_span)))

                        # get the start index for year 2
                        year_2_start_idx = context.index(year_2_str)
                        year_2_span = (year_2_start_idx, year_2_start_idx + len(year_2_str))
                        matched_spans.append(year_2_span)
                        ents_dict['DATE'].append((year_2_str, MatchSpan(year_2_span)))

                if template_2_match is not None:
                    template_match = True
                    for res in template_2_match:
                        # get the match result
                        new_processed_context = res.group(0)
                        # print('new_processed_context:', new_processed_context)

                        # split string
                        processed_context_split = new_processed_context.split(' ')
                        date_str = processed_context_split[0]
                        time_str = processed_context_split[1]

                        # get the start index for date
                        date_start_idx = context.index(date_str)
                        date_span = (date_start_idx, date_start_idx + len(date_str))
                        matched_spans.append(date_span)
                        ents_dict['DATE'].append((date_str, MatchSpan(date_span)))

                        # get the start index for time
                        time_start_idx = context.index(time_str)
                        time_span = (time_start_idx, time_start_idx + len(time_str))
                        matched_spans.append(time_span)
                        ents_dict['TIME'].append((time_str, MatchSpan(time_span)))

                if template_3_match is not None:
                    # get the match result
                    template_match = True
                    for res in template_3_match:
                        date_str = res.group(0)

                        # let's verify this is a proper date
                        year = int(date_str[:2])
                        month = int(date_str[2:4])
                        day = int(date_str[4:6])

                        if year < 0 or year > 99 or month < 1 or month > 12 or day < 1 or day > 31:
                            continue

                        # get the start index for date
                        date_start_idx = context.index(date_str)
                        date_span = (date_start_idx, date_start_idx + len(date_str))
                        matched_spans.append(date_span)
                        ents_dict['DATE'].append((date_str, MatchSpan(date_span)))

                if not template_match:
                    # do the old way
                    for date_res in find_date(processed_context):
                        # print('date_res:', date_res)
                        date_str = date_res[0]
                        date_start_idx = context.index(date_str)
                        date_span = (date_start_idx, date_start_idx + len(date_str))
                        matched_spans.append(date_span)
                        ents_dict['DATE'].append((date_str, MatchSpan(date_span)))

        # additional date time matcher if no date or time is found
        if 'DATE' not in ents_dict or 'TIME' not in ents_dict:
            date_match, time_match, results = sub_date_time_detect(context)
            if 'DATE' not in ents_dict and date_match:
                date_str = find_date(context)[0][0]
                date_start_idx = context.index(date_str)
                date_span = (date_start_idx, date_start_idx + len(date_str))
                matched_spans.append(date_span)
                ents_dict['DATE'].append((date_str, MatchSpan(date_span)))
            if 'TIME' not in ents_dict and time_match:
                if '-' in context and ':' in context:
                    split_string = context.split(' ')
                    time_str = find_date(split_string[1])[0][0]
                    time_start_idx = context.index(time_str)
                    time_span = (time_start_idx, time_start_idx + len(time_str))
                    matched_spans.append(time_span)
                    ents_dict['TIME'].append((time_str, MatchSpan(time_span)))
                else:
                    time_str = find_date(context)[0][0]
                    time_start_idx = context.index(time_str)
                    time_span = (time_start_idx, time_start_idx + len(time_str))
                    matched_spans.append(time_span)
                    ents_dict['TIME'].append((time_str, MatchSpan(time_span)))

        # print('ents_dict', ents_dict)

        self.d_print("matched entities for {}: {}".format(context, ents_dict))

        # find if there's any span left:
        # given a word-span, find out which are the stuff that's left
        span_set = set(range(len(context)))
        self.d_print("span_set before: {}".format(span_set))

        for ms in matched_spans:
            span_set = span_set.difference(set(range(ms[0], ms[1])))

        self.d_print("span_set after: {}".format(span_set))

        if len(span_set) == 0:
            # if the entire context are recognized as entity, we are done
            pass
        elif len(span_set) == len(context):
            # this means none of the string in the context is recognized as an entity, then we need to use other way to do this
            # tokenizes rules:
            # I think for the rest everything is fine, but we need to treat number a bit differently
            #  i.e. if we have number such as fraction 15 1/4 -> they should be the same token
            # print(context)

            # this string triggers some recursion error
            if context == 'Подарочные карты | Магазин Google Merchandise':
                pass
            else:
                tokenized_context = self.tokenize(context)
                self.d_print("tokenized_context: {}".format(tokenized_context))
                if len(tokenized_context) > 1:
                    for idx, (token, span) in enumerate(tokenized_context):
                        self.d_print("token: {}".format(token))
                        self.d_print("span: {}".format(span))
                        if 'Ġ' in token:
                            token = token.replace('Ġ', '')

                            if idx > 0:
                                # for all token that contains the Ġ but is not the first token, we need to increment by 1 (since tokenizer skip the space)
                                start_idx = span[0] + (len(token) - len(token.lstrip()) + 1)
                            else:
                                start_idx = span[0] + (len(token) - len(token.lstrip()))
                        else:
                            start_idx = span[0] + (len(token) - len(token.lstrip()))
                            token = token.lstrip()
                        # print("new start_idx:", start_idx)

                        ents_dict = update_ent_dict_with_entities(self.get_entity(token), ents_dict, start_idx)

        else:
            # sort spans
            # print("branch 3")
            span_set = sorted(span_set)
            idx = 0
            lower = span_set[idx]
            while True:

                self.d_print("idx: {}".format(idx))

                # check if the current idx is the last idx of the span_set list
                if idx == (len(span_set) - 1):
                    upper = span_set[idx] + 1
                    ents_dict = self.get_entity_helper(context, lower, upper, ents_dict)
                    break
                else:
                    # check if the current element and the next element are consecutive
                    if not span_set[idx] + 1 == span_set[idx + 1]:
                        # if not consecutive, update the upper bound, restart the lower bound
                        upper = span_set[idx]
                        ents_dict = self.get_entity_helper(context, lower, upper, ents_dict)
                        lower = span_set[idx + 1]

                idx += 1

        return ents_dict

    def get_entity_with_tag(self, context: str, ent_tag: str) -> List[Tuple[str, MatchSpan]]:
        return self.get_entity(context).get(ent_tag)

    """
    Helper methods here
    """

    def get_entity_helper(self, context: str, lower: int, upper: int, ents_dict: Dict[str, List[Tuple[str, MatchSpan]]]) \
            -> Dict[str, List[Tuple[str, MatchSpan]]]:
        """
        Given a new_context, update the ents_dict with the proper span given the entities found in new_context
        """

        new_context = context[lower: upper]
        # the starting index of how the entities should be count is # current lower idx + idx of the first non-empty space character
        start_idx = lower + (len(new_context) - len(new_context.lstrip()))
        new_context = new_context.lstrip()
        ents_dict = update_ent_dict_with_entities(self.get_entity(new_context), ents_dict, start_idx)

        return ents_dict
