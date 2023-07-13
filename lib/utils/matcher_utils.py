import os
import re
import traceback
from fractions import Fraction

import numpy as np
import pycountry
import pycountry_convert as pc
from geonamescache import GeonamesCache

from typing import List, Optional, Tuple, Dict, Set

from word2number import w2n
from dateparser.search import search_dates
from lib.interpreter.span import MatchSpan
from lib.lang.constants import CaseTag, SPECIAL_CHARS
from lib.utils.csv_utils import read_csv_to_dict
from lib.utils.gpt3_utils import tokenize
from lib.utils.regex_utils import create_matrix_representation, create_span_representation


def get_number_spans(s: str, num_type: Optional[type]) -> Set[MatchSpan]:
    if num_type is None:
        return get_float_spans(s) | get_integer_spans(s)
    if num_type == float:
        return get_float_spans(s)
    elif num_type == int:
        return get_integer_spans(s)


def get_integer_spans(s: str) -> Set[MatchSpan]:
    spans = set()
    for i, ch in enumerate(s):
        curr_is_digit = ch.isdigit()
        if curr_is_digit:
            spans.add(MatchSpan(i, i + 1))
        if curr_is_digit or ch == '+' or ch == '-':
            j = i + 1
            has_digit_chars = True
            while j < len(s) and has_digit_chars:
                if s[j].isdigit():
                    spans.add(MatchSpan(i, j + 1))
                else:
                    has_digit_chars = False
                j += 1
    return spans


def get_float_spans(s: str) -> Set[MatchSpan]:
    spans = set()

    def add_to_spans(bi, pi):
        assert len(bi) > 0
        if s[bi[0]] == '+' or s[bi[0]] == '-':
            spans.add(MatchSpan(bi[1], bi[-1] + 1))
            # print('add {}'.format((s[bi[1]:bi[-1] + 1])))

        if pi > -1:
            if pi == 0:
                spans.add(MatchSpan(bi[pi + 1], bi[-1] + 1))
                # print('add {}'.format(s[bi[pi + 1]:bi[-1] + 1]))
            elif pi == len(bi) - 1:
                spans.add(MatchSpan(bi[0], bi[-1]))
                # print('add {}'.format(s[bi[0]:bi[-1]]))
            else:
                spans.add(MatchSpan(bi[0], bi[pi]))
                spans.add(MatchSpan(bi[pi], bi[-1] + 1))
                spans.add(MatchSpan(bi[pi + 1], bi[-1] + 1))
                # print('add {}'.format(s[bi[0]:bi[pi]]))
                # print('add {}'.format(s[bi[pi]:bi[-1] + 1]))
                # print('add {}'.format(s[bi[pi + 1]:bi[-1] + 1]))

        spans.add(MatchSpan(bi[0], bi[-1] + 1))
        # print('add {}'.format(s[bi[0]:bi[-1] + 1]))

    buffer_i = []
    period_idx = -1
    for i, ch in enumerate(s):
        # print(i, ch)
        curr_is_digit = ch.isdigit()
        if curr_is_digit:
            buffer_i.append(i)
        elif ch == '+' or ch == '-':
            if len(buffer_i) > 0:
                add_to_spans(buffer_i, period_idx)
                buffer_i = []
                period_idx = -1

            # one lookahead
            if i + 1 < len(s) and s[i + 1].isdigit():
                buffer_i.append(i)
        elif ch == '.':
            if len(buffer_i) > 0 and period_idx != -1:
                add_to_spans(buffer_i, period_idx)

            # lookahead
            if i + 1 < len(s) and s[i + 1].isdigit():
                period_idx = len(buffer_i)
                buffer_i.append(i)
        else:
            if len(buffer_i) > 0:
                add_to_spans(buffer_i, period_idx)
                buffer_i = []
                period_idx = -1
    return spans


def regex_match(s: str, pattern: str, case_tag: CaseTag = CaseTag.NONE) -> List[MatchSpan]:
    """
    Regex matching
    """

    # print("regex_match args: s: {}, pattern: {}".format(s, pattern))
    assert case_tag == case_tag.NONE or case_tag == CaseTag.IGNORE or case_tag == CaseTag.CAP or case_tag == CaseTag.LOW

    if case_tag == CaseTag.IGNORE:
        pattern_compiled = re.compile(pattern, flags=re.IGNORECASE)
    else:
        pattern_compiled = re.compile(pattern)
    matches = pattern_compiled.finditer(s)

    spans = []
    for match in matches:
        span = match.span(0)
        if case_tag == CaseTag.CAP:
            if not any(i.islower() for i in s[span[0]:span[1]]):
                spans.append(MatchSpan(span))
        elif case_tag == CaseTag.LOW:
            if not any(i.isupper() for i in s[span[0]:span[1]]):
                spans.append(MatchSpan(span))
        else:
            spans.append(MatchSpan(span))
        # print('matcher_utils match span res:', span)

    return spans


def get_space_pattern() -> str:
    """
    get the very simple space regex
    """
    return r'[\s]+'


def get_integer_pattern() -> str:
    """
    very simple integer regex copied from here: https://stackoverflow.com/questions/8586346/python-regex-for-integer
    """
    return r'[-+]?[0-9]+'


def get_float_pattern() -> str:
    """
    very simple float regex copied from here:
    https://stackoverflow.com/questions/12929308/python-regular-expression-that-matches-floating-point-numbers
    """
    return r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'


def get_tag_value_pattern(tag_values: List[str]) -> str:
    return r'|'.join(tag_values)


def word_to_num(word: str) -> int | float:
    return w2n.word_to_num(word)


def is_fraction_num(s):
    if ' ' in s:
        s_split = s.split()
        for s1 in s_split:
            if '/' in s1:
                s11 = s1.split('/')
                if not (len(s11) == 2 and all(i.isdigit() for i in s11)):
                    return False
            else:
                if not s1.isdigit():
                    return False
    return True


def parse_fraction_to_float(s: str) -> float:
    assert '/' in s

    s_split = s.split()
    assert len(s_split) == 1 or len(s_split) == 2

    if len(s_split) == 1:
        return float(Fraction(s_split[0]))
    elif len(s_split) == 2:
        return float(s_split[0]) + float(Fraction(s_split[1]))


def int_float_ent_matcher_helper(smc, cardinality_entities: List, _type):
    if cardinality_entities is not None:
        for ent_match in cardinality_entities:
            if ent_match[0][0].isdigit() and ent_match[0][-1].isdigit() and '-' in ent_match[0]:
                continue
            # print("ent_match:", ent_match)
            try:
                parse_num = _type(ent_match[0])
                if isinstance(parse_num, _type):
                    if isinstance(smc, list):
                        smc.append(ent_match[1])
                    else:
                        smc.match_spans.add(ent_match[1])
                    continue
            except ValueError as e:
                pass
                # print(str(e))

            try:
                if isinstance(word_to_num(ent_match[0]), _type):
                    if isinstance(smc, list):
                        smc.append(ent_match[1])
                    else:
                        smc.match_spans.add(ent_match[1])
                    continue
            except ValueError as e:
                pass
                # print(str(e))

            # we need to add another parsing mechanism to float
            if _type == float:
                s = ent_match[0]
                if is_fraction_num(s):
                    if isinstance(smc, list):
                        smc.append(ent_match[1])
                    else:
                        smc.match_spans.add(ent_match[1])
                    continue

    return smc


def num_text_to_num(s: str) -> Optional[int | float]:
    if '/' in s:
        if '-' in s and not s.startswith('-'):
            s_tmp = s.replace('-', ' ')
        else:
            s_tmp = s
        s_num = parse_fraction_to_float(s_tmp)
    else:
        try:
            s_num = float(s)
        except ValueError:
            try:
                s_num = float(word_to_num(s))
            except ValueError as e:
                # print(str(e))
                return None
        except Exception as e:
            # print(str(e))
            return None

    return s_num


def ent_matcher_helper(smc, matched_entities: List):
    if matched_entities is not None:
        for ent_match in matched_entities:
            if isinstance(smc, list):
                smc.append(ent_match[1])
            else:
                smc.match_spans.add(ent_match[1])


def double_check_ent(context: str) -> Tuple[bool, Optional[str]]:
    """
    since spacy might mis-identify context with other tags to be CARDINAL tag, here we double-check if context is truly cardinal
    if it is, return (True, None)
    if it is not, return (False, Alternative_tag) if we can find a Alternative_tag
    """

    # first check date, then check time
    union_res = []
    union_res.extend(find_date(context, {'REQUIRE_PARTS': ['year']}))
    union_res.extend(find_date(context, {'REQUIRE_PARTS': ['month']}))
    union_res.extend(find_date(context, {'REQUIRE_PARTS': ['date']}))

    # print("union_res:", union_res)

    if len(union_res) > 0 and any([text == context and text_dt.year <= 2022 for text, text_dt in union_res]):
        return False, 'DATE'

    find_time_res = find_date(context)
    if len(find_time_res) > 0 and any([text == context and text_dt.year <= 2022 for text, text_dt in find_time_res]):
        return False, 'TIME'

    # oops i am doing hacks again
    if context.endswith('x'):
        return False, None

    return True, None


"""
data-time specific matching functions (copied from WebQA so this might be a overkill)
"""

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
add_hoc_date_pattern = re.compile(r"((January|February|March|April|May|June|July|August|September|October|November"
                                  r"|December) \d{1,2})")
add_hoc_date_pattern_2 = re.compile(r"(\d{1,2}[/]\d{1,2}([/]\d{2,4})?[, ]+\d{1,2}[:]\d{1,2}(am|pm)([ -]?\d{1,2}[:]\d{1,"
                                    r"2}(am|pm))?)")
add_hoc_date_pattern_3 = re.compile(r"((Monday|Tuesday|Wednesday|Thursday|Friday)( ("
                                    r"Monday|Tuesday|Wednesday|Thursday|Friday))*([:])?[ ]\d{1,2}([:]\d{1,"
                                    r"2}(am|AM|pm|PM)?)?([ ]["
                                    r"-][ ]\d{1,2}([:]\d{1,2}(am|AM|pm|PM)?)?)?)")
add_hoc_date_pattern_4 = re.compile(r"((Monday|Tuesday|Wednesday|Thursday|Friday) (\d{1,2} ("
                                    r"January|February|March|April|May|June|July|August|September|October|November"
                                    r"|December))([,]?[ ]?\d{1,2}[:]\d{1,2} - \d{1,2}[:]\d{1,2})?)")


def find_additional_date_after(string):
    res = []
    ad_hoc_matches = re.findall(add_hoc_date_pattern, string)
    if ad_hoc_matches is not None:
        # print("ad_hoc matches:", ad_hoc_matches)
        res.extend([(m[0], None) for m in ad_hoc_matches])

    ad_hoc_matches_2 = re.findall(add_hoc_date_pattern_2, string)
    if ad_hoc_matches_2 is not None:
        # print("ad_hoc matches:", ad_hoc_matches_2)
        res.extend([(m[0], None) for m in ad_hoc_matches_2])
    return res


def find_additional_date_before(string):
    res = []
    new_string = string
    # print("new_string:", new_string)
    try:
        ad_hoc_matches_4 = re.findall(add_hoc_date_pattern_4, string)
    except Exception:
        ad_hoc_matches_4 = None

    if add_hoc_date_pattern_4 is not None and ad_hoc_matches_4 is not None:
        for r in ad_hoc_matches_4:
            cr = r[0]
            res.append((cr, None))
            new_string.replace(cr, "")

    try:
        ad_hoc_matches_3 = re.findall(add_hoc_date_pattern_3, new_string)
    except Exception:
        ad_hoc_matches_3 = None
    if ad_hoc_matches_3 is not None:
        # print("ad_hoc matches:", ad_hoc_matches)
        res.extend([(m[0], None) for m in ad_hoc_matches_3])
    return res


def filter_valid_date_string(res, string):
    res_filtered = []
    new_string = string
    for r in res:
        text, _ = r

        # print("curr text:", text)
        if not any(char.isdigit() for char in text):
            if text in weekdays:
                pass
            else:
                continue

        res_filtered.append(r)
        new_string = new_string.replace(text, "")

    return res_filtered, new_string


def find_date(string: str, _setting: Optional[Dict] = None):
    """
    Return whether the string can be interpreted as a date.
    :param string: str, string to check for date
    """

    res_filtered = []

    r = find_additional_date_before(string)
    res, string = filter_valid_date_string(r, string)
    res_filtered.extend(r)

    try:
        if _setting is None:
            r = search_dates(string, languages=['en'])
        else:
            r = search_dates(string, languages=['en'], settings=_setting)
        if r is None:
            r = []
    except Exception:
        return res_filtered

    if len(r) == 0:
        res_filtered.extend(find_additional_date_after(string))
        return res_filtered

    res, new_string = filter_valid_date_string(r, string)
    res_filtered.extend(res)
    res_filtered.extend(find_additional_date_after(new_string))

    return res_filtered


def sub_date_time_detect(string: str):
    required_parts = ['year', 'month', 'day', 'hour', 'minute', 'second']
    required_part_matched = []

    for pt in required_parts:
        res = find_date(string, {'REQUIRE_PARTS': [pt]})
        if len(res) > 0:
            required_part_matched.append(True)
        else:
            required_part_matched.append(False)

    res = find_date(string)

    for r in res:
        if r[1].hour == 0 and r[1].minute == 0:
            required_part_matched.append(False)
            required_part_matched.append(False)
            required_part_matched.append(False)
            break
        else:
            required_part_matched.append(True)
            required_part_matched.append(True)
            required_part_matched.append(True)
            break

    return any(required_part_matched[:3]), any(required_part_matched[3:]), [e for idx, e in enumerate(required_parts) if required_part_matched[idx]]


"""
place-predicate-related matching functions
"""

# not sure if this is the best place to do this
print(os.getcwd())
norp_country_list = read_csv_to_dict('resources/demonyms.csv', fieldname=['norp', 'country'])
norp_to_country: Dict[str, str] = dict([(entry['norp'].lower(), entry['country'].lower()) for entry in norp_country_list])
country_to_norp: Dict[str, str] = dict([(entry['country'].lower(), entry['norp'].lower()) for entry in norp_country_list])
geonames = GeonamesCache()
country_search = pycountry.countries


def parse_place_str(s: str) -> Optional[Tuple[str, List[str]]]:
    """
    parse the str that can be potentially be a place
    if it is in the norp keys, then return its country
    if it is in the country keys, then return itself
    if it is not in anything, check if there is any city and return the country corresponds to it
    """

    def lookup_country(name: str) -> Optional[str]:
        try:
            res = pycountry.countries.search_fuzzy(name)
            if 'America' in name and len(res) > 1:
                return 'United States'
            elif len(res) > 1:
                raise Exception('Resolve country ambiguity')
            else:
                assert len(res) == 1
                return res[0].name
        except:
            return None

    # preprocess

    if s.lower().startswith('north '):
        s = s.lower().replace('north ', '').strip()
    elif s.lower().startswith('south '):
        s = s.lower().replace('south ', '').strip()
    elif s.lower().startswith('west '):
        s = s.lower().replace('west ', '').strip()
    elif s.lower().startswith('east '):
        s = s.lower().replace('east ', '').strip()

    if s.lower() in norp_to_country:
        s = norp_to_country[s.lower()]

    s_country = lookup_country(s)

    if s_country is None:
        candidate_cities = geonames.search_cities(s, case_sensitive=True)
        if len(candidate_cities) == 0:
            return None
        else:
            # just get the first results
            candidates = geonames.search_cities(s)
            candidates_countries = [geonames.get_countries()[city['countrycode']]['name'] for city in candidates]

            return s, candidates_countries
    else:
        return s, [s_country]


def check_regions(s: str, regions: List[str]) -> bool:
    def get_continent(_country: str):
        country_code = pc.country_name_to_country_alpha2(_country, cn_name_format="lower")
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        country_continent_name = pc.convert_continent_code_to_continent_name(continent_code)

        return country_continent_name

    regions_lower = [r.lower() for r in regions]
    res = parse_place_str(s)

    if res is None:
        return False

    _, countries = res

    # need to catch exception
    continents = []
    for country in countries:
        try:
            cont = get_continent(country).lower()
            continents.append(cont)
        except:
            pass

    # print('continents:', continents)

    return any([r in continents for r in regions_lower])


def check_countries(s: str, countries: List[str]) -> bool:
    res = parse_place_str(s)
    countries_lower = [c.lower() for c in countries]

    if res is None:
        return False

    _, countries_out = res

    return any([country.lower() in countries_lower for country in countries_out])


us_city_list = read_csv_to_dict('resources/us_cities_states_counties.txt', delimiter='|')


def get_city_from_state(key: str = 'State short', value: str = 'CA') -> List:
    city_list = []

    for entry in us_city_list:
        if entry[key] == value:
            city_list.append(entry['City'])

    return list(set(city_list))


def get_state_from_city(city_name: str) -> List:
    state_list = []

    for entry in us_city_list:
        if entry['City'].lower() == city_name.lower():
            state_list.append(entry['State full'])

    return list(set(state_list))


def merge_entity_match_query(s: str, candidate_entities: List[str], nlp_func) -> List[MatchSpan]:
    # print(s)
    # print(candidate_entities)
    merged_entities = []

    # for each entity found, find out if the no-space version is a substring of s
    for ent in candidate_entities:
        if len(ent.split()) > 1:
            ent_s = ent.replace(' ', '').lower()
            if ent_s in s.lower():
                match_idx = [m.start() for m in re.finditer(ent_s, s.lower())]
                merged_entities.extend([MatchSpan(start, start + len(ent_s)) for start in match_idx])
        else:
            # I don't know why this is necessary but it seems there is some bug
            try:
                if ent.lower() in s.lower():
                    # print(ent.lower())
                    ent_p = ent.replace('|', '\|').lower()
                    match_idx = [m.start() for m in re.finditer(ent_p, s.lower())]
                    merged_entities.extend([MatchSpan(start, start + len(ent)) for start in match_idx])
                    # print(merged_entities)
            except Exception:
                pass

    # print(merged_entities)

    # now we work on the tokenized version of the problem
    tokenized_s, tokenize_idx_to_s_idx_s = tokenize(s.lower(), SPECIAL_CHARS)
    # print(tokenized_s, tokenize_idx_to_s_idx_s)

    if len(tokenized_s) > 1:
        for ent in candidate_entities:
            tokenized_ent, _ = tokenize(ent.lower(), SPECIAL_CHARS)
            # print(tokenized_ent)
            merged_entities.extend(merge_entity_match_query_helper(tokenized_s, tokenize_idx_to_s_idx_s, tokenized_ent, False))

        ent_merged = ' '.join(candidate_entities)
        tokenized_ent, _ = tokenize(ent_merged.lower(), SPECIAL_CHARS)
        # print(tokenized_ent)
        merged_entities.extend(merge_entity_match_query_helper(tokenized_s, tokenize_idx_to_s_idx_s, tokenized_ent, True))
    else:

        def format_tokenizer_res(res: List[Tuple[str, Tuple[int, int]]]) -> Tuple[List[str], Dict[int, int]]:
            ts, ts_idx = [], {}
            for idx, (token, span) in enumerate(res):
                if 'Ġ' in token:
                    token = token.replace('Ġ', '')
                ts.append(token)
                ts_idx[idx] = span[0]

            return ts, ts_idx

        tokenized_s, tokenize_idx_to_s_idx_s = format_tokenizer_res(nlp_func.byte_tokenizer.pre_tokenize_str(s.lower()))
        # print(tokenized_s, tokenize_idx_to_s_idx_s)

        for ent in candidate_entities:
            tokenized_ent, _ = format_tokenizer_res(nlp_func.byte_tokenizer.pre_tokenize_str(ent.lower()))
            # print(tokenized_ent)
            merged_entities.extend(merge_entity_match_query_helper(tokenized_s, tokenize_idx_to_s_idx_s, tokenized_ent, False))

    # another pass to take space as part of the entity
    # if the entity starts at the first non-word char of the string, we include it as well
    # print('merge_entities', merged_entities)
    merged_entities_new = []
    for ent_span in merged_entities:
        if ent_span in merged_entities_new:
            continue
        else:
            merged_entities_new.append(ent_span)

            if ent_span.start > 0 and not any(s[i].isalnum() for i in range(0, ent_span.start)):
                merged_entities_new.append(MatchSpan(0, ent_span.end))

            if ent_span.end < (len(s) - 1) and not any(s[i].isalnum() for i in range(ent_span.end, len(s))):
                merged_entities_new.append(MatchSpan(ent_span.start, len(s)))

            if ent_span.start > 0:
                if s[ent_span.start - 1] == ' ':
                    merged_entities_new.append(MatchSpan(ent_span.start - 1, ent_span.end))
                    if ent_span.end < len(s) - 1:
                        if s[ent_span.end] == ' ':
                            merged_entities_new.append(MatchSpan(ent_span.start, ent_span.end + 1))
                            merged_entities_new.append(MatchSpan(ent_span.start - 1, ent_span.end + 1))
                        elif s[ent_span.end] == '.':
                            merged_entities_new.append(MatchSpan(ent_span.start, ent_span.end + 1))
                            merged_entities_new.append(MatchSpan(ent_span.start - 1, ent_span.end + 1))
                        elif s[ent_span.end] == ')':
                            merged_entities_new.append(MatchSpan(ent_span.start, ent_span.end + 1))
                            merged_entities_new.append(MatchSpan(ent_span.start - 1, ent_span.end + 1))
            elif ent_span.end < len(s) - 1:
                if s[ent_span.end] == ' ':
                    merged_entities_new.append(MatchSpan(ent_span.start, ent_span.end + 1))
                elif s[ent_span.end] == '.':
                    merged_entities_new.append(MatchSpan(ent_span.start, ent_span.end + 1))
                elif s[ent_span.end] == ')':
                    merged_entities_new.append(MatchSpan(ent_span.start, ent_span.end + 1))
            else:
                pass
    # print('merge_entities_new', merged_entities_new)

    # another pass to merge all the concatenations
    string_size: int = len(s)
    arg1_matrix: np.array = create_matrix_representation(string_size, merged_entities_new)
    arg2_matrix: np.array = create_matrix_representation(string_size, merged_entities_new)
    res_matrix: np.array = arg1_matrix @ arg2_matrix
    spans1 = create_span_representation(string_size, res_matrix)

    # handle the 'and' case as well
    key = None if not ('and' in s or '&' in s) else ('and' if 'and' in s else '&')
    if key is not None:
        # find match span for and
        and_spans = [MatchSpan(i, i + len(key)) for i in range(len(s)) if s.startswith(key, i)]
        string_size: int = len(s)
        arg1_matrix: np.array = create_matrix_representation(string_size, merged_entities_new)
        arg3_matrix: np.array = create_matrix_representation(string_size, and_spans)
        arg2_matrix: np.array = create_matrix_representation(string_size, merged_entities_new)
        res_matrix: np.array = arg1_matrix @ arg3_matrix @ arg2_matrix
        spans2 = create_span_representation(string_size, res_matrix)
        return merged_entities_new + spans1 + spans2
    else:
        return merged_entities_new + spans1


def merge_entity_match_query_helper(tokenized_s: List[str], tokenize_idx_to_s_idx_s: Dict[int, int], tokenized_ent: List[str], merge: bool) -> \
        List[MatchSpan]:
    # print('tokenized_ent:', tokenized_ent)

    merged_entities = []

    if len(tokenized_ent) == 0:
        return merged_entities

    if not merge:
        e0_s_idx = [idx for idx, t in enumerate(tokenized_s) if t == tokenized_ent[0]]
        for e_0_i in e0_s_idx:
            if all([tokenized_ent[e_i] == tokenized_s[e_0_i + e_i] if (e_0_i + e_i) < len(tokenized_s) else False for e_i in range(0, len(tokenized_ent))]):
                # print(tokenized_ent, s[tokenize_idx_to_s_idx_s[e_0_i]:(tokenize_idx_to_s_idx_s[e_0_i + len(tokenized_ent) - 1] + len(tokenized_s[e_0_i + len(tokenized_ent) - 1]))])
                merged_entities.append(
                    MatchSpan(tokenize_idx_to_s_idx_s[e_0_i], tokenize_idx_to_s_idx_s[e_0_i + len(tokenized_ent) - 1] + len(tokenized_s[e_0_i + len(tokenized_ent) - 1])))
    else:
        curr_e0 = 0
        while curr_e0 < len(tokenized_ent):
            e0_s_idx = [idx for idx, t in enumerate(tokenized_s) if t == tokenized_ent[curr_e0]]
            chosen_len = -1
            chosen_e0_s = -1
            chosen_e0_e = -1
            for e_0_i in e0_s_idx:
                # print(e_0_i)
                for i in range(0, len(tokenized_ent)):
                    e_i = curr_e0 + i
                    if e_i >= len(tokenized_ent):
                        break

                    if (e_0_i + i) < len(tokenized_s) and tokenized_ent[e_i] == tokenized_s[e_0_i + i]:
                        continue
                    else:
                        curr_e_len = i
                        if curr_e_len >= chosen_len:
                            chosen_len = curr_e_len
                            chosen_e0_s = e_0_i
                            chosen_e0_e = curr_e0
                        break
                if chosen_len == -1:
                    chosen_len = len(tokenized_ent) - curr_e0
                    chosen_e0_s = e_0_i
                    chosen_e0_e = curr_e0
                    break

            if chosen_len > 0:
                curr_e0 = chosen_e0_e + chosen_len
                merged_entities.append(
                    MatchSpan(tokenize_idx_to_s_idx_s[chosen_e0_s], tokenize_idx_to_s_idx_s[chosen_e0_s + chosen_len - 1] + len(tokenized_s[chosen_e0_s + chosen_len - 1])))
            else:
                curr_e0 += 1

    return merged_entities


def split_entity_match_query(s: str, candidate_entities: List[str], entity_type: str, nlp_func) -> List[MatchSpan]:
    # for each candidate entities_try to split them and see if some substring of them is also a candidate entity
    # if so, then we split the entity into two parts
    if len(candidate_entities) > 5:
        # heuristic: if there are more than 2 candidate entities_try, then we don't split them
        return []

    results = set()

    for ent in candidate_entities:
        tokenized_ent = ent.split()

        # we need to do some hacks here, because gpt is really stupid
        if entity_type.lower() == 'product':
            if len(tokenized_ent) > 4 and tokenized_ent[0].lower() == 'set':
                if tokenized_ent[1].lower() == 'of':
                    tokenized_ent = tokenized_ent[3:]
                else:
                    tokenized_ent = tokenized_ent[2:]

                sub_token = ' '.join(tokenized_ent).strip()
                match_idx = [m.start() for m in re.finditer(sub_token.lower(), s.lower())]
                for idx in match_idx:
                    if idx + len(sub_token) > len(s):
                        continue
                    if s[idx:(idx + len(sub_token))] == sub_token:
                        results.add(MatchSpan(idx, idx + len(sub_token)))
                continue
        elif entity_type.lower() == 'organization':
            if len(tokenized_ent) >= 3 and any(t.lower() == 'of' for t in tokenized_ent):
                # print('here12')
                # find the index of and split the entity into two parts
                sub_token = ' '.join(tokenized_ent)
                tokenized_ents = re.split('of', sub_token, flags=re.IGNORECASE)
                tokenized_ent0 = tokenized_ents[0].strip()
                match_idx = [m.start() for m in re.finditer(tokenized_ent0.lower(), s.lower())]
                for idx in match_idx:
                    if idx + len(tokenized_ent0) > len(s):
                        continue
                    if s[idx:(idx + len(tokenized_ent0))] == tokenized_ent0:
                        results.add(MatchSpan(idx, idx + len(tokenized_ent0)))
                continue

        if len(tokenized_ent) > 5 or len(tokenized_ent) == 0 or len(tokenized_ent) == 1:
            # heuristic: if the entity is longer than 3 or smaller or equal to 1, then we don't split them
            continue
        elif ent.lower() not in s.lower():
            # heuristic: if the entity is not in the string, then we don't split them
            continue
        else:
            # tokenize the entity
            ent_sub_tokens = []
            # get subset tokens of the entity
            for i in range(0, len(tokenized_ent)):
                for j in range(i + 1, i + 3):
                    if j <= len(tokenized_ent):
                        ent_sub_tokens.append(' '.join(tokenized_ent[i:j]))

            # verify if the subset tokens are also of the type entity_type
            for sub_token in ent_sub_tokens:
                if nlp_func.query_entity_generation_verifier(s, sub_token, entity_type):
                    # generate a matchspan
                    # print('split entity:', sub_token, 'from entity:', ent, 'in string:', s)

                    # sorry, more hacks to filter out some random behavior of gpt stuff
                    if entity_type == 'location':
                        if sub_token.lower() in ['science']:
                            continue

                    match_idx = [m.start() for m in re.finditer(preprocess_str_for_regex(sub_token).lower(), s.lower())]
                    for idx in match_idx:
                        if idx + len(sub_token) > len(s):
                            continue
                        if s[idx:(idx + len(sub_token))].lower() == sub_token.lower():
                            results.add(MatchSpan(idx, idx + len(sub_token)))

    return list(results)


def preprocess_str_for_regex(s: str) -> str:
    if '(' in s and '\(' not in s:
        s = s.replace('(', '\(')
    if ')' in s and '\)' not in s:
        s = s.replace(')', '\)')
    if '[' in s and '\[' not in s:
        s = s.replace('[', '\[')
    if ']' in s and '\]' not in s:
        s = s.replace(']', '\]')
    if '{' in s and '\{' not in s:
        s = s.replace('{', '\{')
    if '}' in s and '\}' not in s:
        s = s.replace('}', '\}')
    if '.' in s and '\.' not in s:
        s = s.replace('.', '\.')
    if '?' in s and '\?' not in s:
        s = s.replace('?', '\?')
    if '*' in s and '\*' not in s:
        s = s.replace('*', '\*')
    if '^' in s and '\^' not in s:
        s = s.replace('^', '\^')
    if '+' in s and '\+' not in s:
        s = s.replace('+', '\+')
    if '$' in s and '\$' not in s:
        s = s.replace('$', '\$')
    if '|' in s and '\|' not in s:
        s = s.replace('|', '\|')
    if '/' in s and '\/' not in s:
        s = s.replace('/', '\/')
    return s


def filepath_related_helpers(s: str, ty: str) -> List[MatchSpan]:
    # first split by ' '
    # for each word, if it is a url, then split it by '/'
    # else skip
    ty = ty.lower()

    # split by ' '
    url_parts1 = s.split(' ')
    # print(url_parts1)
    results = []
    for part in url_parts1:
        if '/' in part:
            url_parts = part.split('/')
            # print(url_parts)
            if ty == 'directory':

                if url_parts[0] == '':
                    # version 1: /a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z
                    for i in range(0, len(url_parts) - 1):
                        results.append('/'.join(url_parts[:i + 1]))
                        results.append('/'.join(url_parts[:i + 1]) + '/')

                    if '.' not in url_parts[-1]:
                        results.append('/'.join(url_parts))
                        results.append('/'.join(url_parts) + '/')

                    url_parts = url_parts[1:]

                # version 2: a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z/
                for i in range(0, len(url_parts) - 1):
                    results.append('/'.join(url_parts[:i + 1]))
                    results.append('/'.join(url_parts[:i + 1]) + '/')

                if '.' not in url_parts[-1]:
                    results.append('/'.join(url_parts))
                    results.append('/'.join(url_parts) + '/')

            elif ty == 'path':
                # sorry this is much longer...

                # I think we need to remove the '?' in the file name, which is the last part
                if '?' in url_parts[-1]:
                    url_parts[-1] = url_parts[-1].split('?')[0]

                if url_parts[0] == '':
                    # version 1: /a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z
                    for i in range(0, len(url_parts) - 1):
                        for j in range(i + 1, len(url_parts)):
                            results.append('/'.join(url_parts[i:j + 1]))
                            results.append('/'.join(url_parts[i:j + 1]) + '/')

                    url_parts = url_parts[1:]

                # version 2: a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z/
                for i in range(0, len(url_parts) - 1):
                    for j in range(i + 1, len(url_parts)):
                        results.append('/'.join(url_parts[i:j+ 1]))
                        results.append('/'.join(url_parts[i:j + 1]) + '/')

            elif ty in ['file', 'file name']:
                if '.' in url_parts[-1]:
                    results.append(url_parts[-1])
                    split_file = url_parts[-1].split('.')
                    results.append(split_file[0])

                    # in rare cases like website, it might have 'index.php?....'
                    if '?' in split_file[0]:
                        results.append(split_file[0].split('?')[0])

    # print(results)
    # get match span
    match_spans = []
    for result in results:
        if result in ['/', '']:
            continue
        match_idx = [m.start() for m in re.finditer(preprocess_str_for_regex(result).lower(), s.lower())]
        for idx in match_idx:
            if idx + len(result) > len(s):
                continue
            if s[idx:(idx + len(result))].lower() == result.lower():
                match_spans.append(MatchSpan(idx, idx + len(result)))

    return match_spans
