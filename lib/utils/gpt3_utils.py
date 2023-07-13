import warnings
from typing import Dict, Optional, List, Tuple

from lib.lang.constants import SPECIAL_CHARS
from lib.utils.prompt_training_utils import TRAINING_EXAMPLES, ENTITY_TRAINING_EXAMPLES, TRAINING_EXAMPLES_PREV, ENTITY_INFER_EXAMPLES

"""
utility function for gpt3
"""

"""
Following are prompt engineering related to sketch generation and its corresponding baselines stuff
"""


def single_prompt_sketch_gen(positive_examples: List[str], version: str) -> str:
    prompt = ''
    if 'v1' in version:
        prompt += 'Summarize the structure of the following positive examples in the form of a regular expression sketch. Use {??: <semantic type>} to represent the unknown part of the ' \
                  'sketch.\nPositive examples:\n'
    elif 'v2' in version:
        prompt += 'Find out the pattern of the following input strings. You can use regular expression syntax to describe the patterns. You can also use {concept:X} to denote some strings ' \
                  'representing a certain concept X.\nInput Strings:\n'
    elif 'old_prompt' in version:
        prompt += "/* Generate a regular expression sketch such that an instantiation of the sketch will accept the positive examples {}. Use ?? to represent the unknown part of the sketch. */".format(
            str(positive_examples))
    else:
        raise ValueError('Prompt for version {} is not implemented'.format(version))

    if 'old_prompt' not in version:
        for example in positive_examples:
            prompt += '- {}\n'.format(example)

    if 'v1' in version:
        prompt += 'Sketch:\n-'
    elif 'v2' in version:
        prompt += 'Pattern:\n-'
    elif 'old_prompt' in version:
        prompt += '\n'
    else:
        raise ValueError('Prompt for version {} is not implemented'.format(version))

    return prompt


def single_prompt_regex_gen(positive_examples: List[str], negative_examples: List[str]) -> str:
    prompt = 'Find a program using the semantic regular expression such that the program can match the positive examples and reject the negative examples.\nPositive examples:\n'
    for example in positive_examples:
        prompt += '- {}\n'.format(example)

    prompt += 'Negative examples:\n'

    for example in negative_examples:
        prompt += '- {}\n'.format(example)

    prompt += 'Program:\n-'
    return prompt


def single_prompt_concrete_gen(positive_examples: List[str], negative_examples: List[str]) -> str:
    prompt = 'Find a program using the semantic regular expression such that the program can match the positive examples and reject the negative examples.\nPositive examples:\n'
    for example in positive_examples:
        prompt += '- {}\n'.format(example)

    prompt += 'Negative examples:\n'

    for example in negative_examples:
        prompt += '- {}\n'.format(example)

    prompt += 'Program:\n-'
    return prompt


def generate_prompt_sketch_training(version: str) -> str:
    if 'prev' in version:
        set_of_training_examples = TRAINING_EXAMPLES_PREV
    else:
        set_of_training_examples = TRAINING_EXAMPLES

    if 'prev' in version:
        index_version = 'prev'
    else:
        index_version = version

    prompt = ''
    for prompt_example in set_of_training_examples:
        if 'v1' in version:
            prompt += '{} {}\n'.format(single_prompt_sketch_gen(prompt_example.positive, version), prompt_example.sketches[index_version])
        elif 'v2' in version:
            prompt += '{} {}\n'.format(single_prompt_sketch_gen(prompt_example.positive, version), prompt_example.sketches[index_version])
        elif 'old_prompt' in version:
            prompt += '{}{}\n'.format(single_prompt_sketch_gen(prompt_example.positive, version), prompt_example.sketches[index_version])
        else:
            raise ValueError('Prompt for version {} is not implemented'.format(version))
        prompt += '\n'

    return prompt


def generate_prompt_concrete_training() -> str:
    prompt = ''
    for prompt_example in TRAINING_EXAMPLES:
        prompt += '{} {}\n'.format(single_prompt_concrete_gen(prompt_example.positive, prompt_example.negative), prompt_example.concrete)
        prompt += '\n'

    return prompt


def generate_prompt_regex_training() -> str:
    prompt = ''
    for prompt_example in TRAINING_EXAMPLES:
        prompt += '{} {}\n'.format(single_prompt_regex_gen(prompt_example.positive, prompt_example.negative), prompt_example.regex)
        prompt += '\n'

    return prompt


SKETCH_TRAINING_PROMPT = {
    'old_prompt': generate_prompt_sketch_training('prev_old_prompt'),
    'v1': generate_prompt_sketch_training('v1'),
    'v2': generate_prompt_sketch_training('v2')
}
CONCRETE_TRAINING_PROMPT = generate_prompt_concrete_training()
REGEX_TRAINING_PROMPT = generate_prompt_regex_training()


def generate_sketch_prompt(positive_examples: List[str], version: str):
    if 'v1' in version:
        index_version = 'v1'
    elif 'v2' in version:
        index_version = 'v2'
    elif 'old_prompt' in version:
        index_version = 'old_prompt'
    else:
        index_version = ''

    if index_version not in SKETCH_TRAINING_PROMPT:
        raise ValueError('Prompt for version {} is not implemented'.format(index_version))
    else:
        prompt = '{}{}'.format(SKETCH_TRAINING_PROMPT[index_version], single_prompt_sketch_gen(positive_examples, version))

        return prompt


def generate_concrete_prompt(positive_examples: List[str], negative_examples: List[str]):
    prompt = '''\'\'\'
Semantic Regex Syntax:
r ::= constant | cc
    | {<type> -> f} | {<type_b> -> p} | {<type_b>}
    | r? | r* | r+ | r{n} | r{n1,n2}
    | rr | r|r | r & r
f ::= x | toUpper | toLower
    | substring[number1, number2] | abbreviate[string]
p ::= True | ~p | p|p | p&p | NumMatch(number1, sym, number2, sym) 
    | isYear(year1, year2) | isMonth(month1, month2) | isDate(date1, date2)
    | btwHour(n1, n2) | btwMin(n1, n2) | btwSec(n1, n2) | isMorning | isAfternoon | isEvening
    | inRegion(continent) | inCountry(country) | inState(state)
cc ::= ANY | LET | NUM | CAP
type_b ::= Person | Organization | Product | Event | Work of Art
        | Number | Integer | Float
        | Date | Year | Month | Day
        | Time | Hour | Minute | Second
        | Place | Location | Nationality | Country | City
 \'\'\'\n\n'''
    prompt += '{}{}'.format(CONCRETE_TRAINING_PROMPT, single_prompt_concrete_gen(positive_examples, negative_examples))
    prompt += '\n'

    return prompt


def generate_regex_prompt(positive_examples: List[str], negative_examples: List[str]):
    prompt = '{}{}'.format(REGEX_TRAINING_PROMPT, single_prompt_regex_gen(positive_examples, negative_examples))

    return prompt


def generate_exec_training_prompt(positive_examples: List[str], negative_examples: List[str]):
    prompt = '{}\n'.format("/* Given the string below, output 'Yes' if this string should be matched, 'No' if this string should not be matched.*/")

    for pos in positive_examples:
        prompt += '\n{}\nMatched? Yes\n'.format(pos)

    for neg in negative_examples:
        prompt += '\n{}\nMatched? No\n'.format(neg)

    return prompt


def generate_exec_prompt(positive_examples: List[str], negative_examples: List[str], query: str):
    training_prompt = generate_exec_training_prompt(positive_examples, negative_examples)
    prompt = '{}\n{}\nMatched?'.format(training_prompt, query)
    # print('prompt:', prompt)

    return prompt


def parse_gpt3_output_for_exec(response: str) -> bool:
    # print('res:', response)
    res = response.strip().lower()

    if 'yes' in res:
        return True
    elif 'no' in res:
        return False
    else:
        raise Exception("GPT3 returns un-parsable result {}".format(res))


"""
Following are prompt engineering related to category inference/entity classification
"""

# TODO: this might not be generalizable enough
get_inferred_categories_pattern = [
    {
        "RIGHT_ID": "root",
        "RIGHT_ATTRS": {"POS": "AUX", "DEP": "ROOT"}
    },
    {
        "LEFT_ID": "root",
        "REL_OP": ".*",
        "RIGHT_ID": "1",
        "RIGHT_ATTRS": {"DEP": "attr"}
    },
    {
        "LEFT_ID": "1",
        "REL_OP": ".*",
        "RIGHT_ID": "pobj1",
        "RIGHT_ATTRS": {"DEP": "pobj"}
    },
    {
        "LEFT_ID": "1",
        "REL_OP": "$++",
        "RIGHT_ID": "2",
        "RIGHT_ATTRS": {"DEP": "advcl"}
    },
    {
        "LEFT_ID": "2",
        "REL_OP": ".*",
        "RIGHT_ID": "pobj2",
        "RIGHT_ATTRS": {"DEP": "pobj"}
    },
]


def generate_prompt_for_infer(positive_str: List[str], negative_str: List[str]) -> str:

    prompt = '\n'.join(["Q: What is the semantic entity that describes {} but does not describe {}? Give me an answer in one or two words.\nA:{}\n"
                       .format(str(entity_training.pos), str(entity_training.neg), entity_training.output) for entity_training in ENTITY_INFER_EXAMPLES])

    prompt += '\n'
    prompt += "Q: What is the semantic entity that describes {} but does not describe {}? Give me an answer in one or two words.\nA:".format(str(positive_str), str(negative_str))
    return prompt


def generate_prompt_for_entity_classification(test_str: str, query: str, positive_str: Optional[List[str]], negative_str: Optional[List[str]]):
    # TODO: the argument of this function is subject to change. i have not thought about how this thing work overall

    positive_prompt = '\n'.join(['Input:{}\nClassify:yes\n'.format(s) for s in positive_str]) if positive_str is not None else ''

    negative_prompt = '\n'.join(['Input:{}\nClassify:no\n'.format(s) for s in negative_str]) if negative_str is not None else ''

    prompt = 'Classify if the following input string is a {} and is not anything else. Respond in yes or no.\n\n'.format(query)

    if not positive_prompt == '':
        prompt += '{}\n'.format(positive_prompt)

    if not negative_prompt == '':
        prompt += '{}\n'.format(negative_prompt)

    prompt += 'Input:{}\nClassify:'.format(test_str)

    return prompt


def generate_prompt_for_entity_generation(test_str: str, query: str, positive_str: Optional[List[str]], negative_str: Optional[List[str]]):
    query = query[0].capitalize() + query[1:].lower()
    positive_prompt = '\n'.join(['{0}\n{1}: [{0}]\n'.format(s, query) for s in positive_str]) if positive_str is not None else ''
    negative_prompt = '\n'.join(['{0}\n{1}: none\n'.format(s, query) for s in negative_str]) if negative_str is not None else ''

    # if query.lower().endswith('name'):
    #     # prompt = 'Identify all the longest substrings of the given input that is likely to be a name. Output none if you are not confident enough.\n\n'.format()
    #     prompt = 'Identify all possible substrings of the given input that means a {} using a comma-separated string. Output none if you are not confident enough.\n\n'.format(query)
    # else:
    #     prompt = 'Identify all possible substrings of the given input that means a {} using a comma-separated string. Output none if you are not confident enough.\n\n'.format(query)
    #     # prompt = 'Identify all the longest substrings of the given input that is likely to be a {} name. Output none if you are not confident enough.\n\n'.format(query)

    prompt = 'Identify all possible substrings of the given input that has the specified semantic. Output none if you are not confident enough.\n\n'

    # in-context examples
    prompt += '\n\n'.join(['{}\n{}: {}'.format(ex.input, ex.label, ';'.join(ex.output)) for ex in ENTITY_TRAINING_EXAMPLES]) + '\n\n'

    if not positive_prompt == '':
        prompt += '{}\n'.format(positive_prompt)

    if not negative_prompt == '':
        prompt += '{}\n'.format(negative_prompt)

    prompt += '{}\n{}:'.format(test_str, query)

    # if query.lower().endswith('name'):
    #     prompt += 'Input:{}\n{}:'.format(test_str, query)
    # else:
    #     prompt += 'Input:{}\n{}:'.format(test_str, query)
    # print('prompt:', prompt)

    return prompt


def generate_prompt_for_place_context(test_str: str, query: str):
    prompt = "Q: What is the {} name corresponding to the string '{}'? Respond in one or few words and return None if not applicable.\nA:".format(query.lower(), test_str)
    return prompt


def parse_gpt3_output_for_place_context(response: str) -> Optional[str]:
    res = response.strip()

    if 'none' in res.lower():
        return None
    elif ',' in res:
        return res.split(',')[0].strip()
    else:
        return res.replace('.', '')


def parse_gpt3_output_for_entity_classification(response: str) -> bool:
    # print("parse_output_for_entity input:", response)

    # get the first result
    res = response.strip().lower()

    # print("res: {}".format(res))

    if 'yes' in res:
        return True
    elif 'no' in res:
        return False
    else:
        raise Exception("GPT3 returns un-parsable result {}".format(res))


def parse_gpt3_output_for_entity_generation(response: str, input_str: str) -> List[str]:
    # get the first result
    res = response.strip()
    # print('input:', input_str)
    # print("res: {}".format(res))

    if res == '':
        return []
    else:
        res = res.replace('\n', ';')
        # we need to filter out anything that is hallucinated
        res_list = []
        for line in res.split('\n'):
            for r in line.split(';'):
                r = r.strip()
                if r == '':
                    continue
                elif r.lower() == 'none':
                    continue
                elif '[' in r and ':' in r.split('[')[0]:
                    continue
                else:
                    if r.startswith('['):
                        res_list.append(r[1:-1])
                    else:
                        res_list.append(r)
        # print('res_list:', res_list)

        # we need do additional filtering to find out if there is strings that is not exactly the same as the input
        res_list_2 = []
        tokenized_s, tokenize_idx_to_s_idx_s = tokenize(input_str.lower(), SPECIAL_CHARS)
        # print('tokenized_s:', tokenized_s)
        for r in res_list:
            tokenized_r, _ = tokenize(r.lower(), SPECIAL_CHARS)
            # print('tokenized_r:', tokenized_r)
            if len(tokenized_r) == 0:
                res_list_2.append(r)
            else:
                e0_s_idx = [idx for idx, t in enumerate(tokenized_s) if t == tokenized_r[0]]
                for e0_s_idx in e0_s_idx:
                    for e_i in range(0, len(tokenized_r)):
                        if e_i + e0_s_idx >= len(tokenized_s):
                            break
                        elif tokenized_s[e0_s_idx + e_i] != tokenized_r[e_i]:
                            # print('here1')
                            # we append up to the e_i - 1 index
                            if e_i > 0:
                                s_start_idx = tokenize_idx_to_s_idx_s[e0_s_idx]
                                s_end_idx = tokenize_idx_to_s_idx_s[e0_s_idx + e_i - 1] + len(tokenized_s[e0_s_idx + e_i - 1])
                                # print(tokenized_s[e0_s_idx], tokenized_s[e0_s_idx + e_i], tokenized_s[e0_s_idx + e_i - 1],  s_start_idx, s_end_idx)
                                res_list_2.append(input_str[s_start_idx:s_end_idx])
                            break
                        elif e_i == len(tokenized_r) - 1:
                            res_list_2.append(r)
                        else:
                            continue

        # print("parse_output_for_entity input {}: {}".format(input_str, res_list_2))
        return res_list_2


def tokenize(string: str, split_symbol: list) -> Tuple[List[str], Dict[int, int]]:
    # return a tokenized list, and the mapping from each token and the start of the string
    tokens = []
    token_idx_to_string_idx = {}
    token_idx = 0
    idx = 0
    buffer = ''
    buffer_start_idx = -1

    while True:
        if idx == len(string):
            if buffer != '':
                tokens.append(buffer)
                token_idx_to_string_idx[token_idx] = buffer_start_idx
                token_idx += 1
            break

        if string[idx] in split_symbol:
            if buffer != '':
                tokens.append(buffer)
                token_idx_to_string_idx[token_idx] = buffer_start_idx
                token_idx += 1
                buffer = ''
                buffer_start_idx = -1
        else:
            if buffer == '':
                buffer_start_idx = idx
            buffer += string[idx]

        idx += 1

    return tokens, token_idx_to_string_idx
