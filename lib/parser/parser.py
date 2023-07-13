import itertools
import re
from math import inf
from typing import List, Dict

import lark
from lark import Transformer, Token

from lib.config import pd_print
from lib.interpreter.pattern import RegexPattern, SemPattern, CC, Const, StarR, OptionalR, Concat, Repeat, RepeatRange, \
    Or, PredicateFormula, NumPredicate, Contain, TimePredicate, DatePredicate, \
    PlacePredicate, Startwith, Endwith, And, NotR, Similar, TypeTag, EntTag, OntTag, MatchEntity, MatchType, MatchQuery, \
    QueryStr, ConstStr, Hole, Predicate, Plus
from lib.interpreter.str_function import StrFunc, Var, Str, Append, Capitalize, LowerCase, SubStr, SplitConcat
from lib.lang.constants import CaseTag, CC_REGEX_MAP, CONST_REGEX_MAP
from lib.utils.exceptions import ProgramParsingException
from lib.utils.sketch_utils import format_const_or_cc

GRAMMAR = """
?start : unionexp
        | 

?unionexp : interexp "||" unionexp                                       -> top_or
            | interexp "|" unionexp                                       -> top_or
            | interexp

?interexp : concatexp "&" interexp                                     -> top_and
            | concatexp

?concatexp : repeatexp concatexp                                        -> top_concat
            | repeatexp

?repeatexp : repeatexp "?"                                              -> top_optional
            | repeatexp "*"                                             -> top_star
            | repeatexp "+"                                             -> top_repeat_1
            | repeatexp "{" NUMBER "}"                                  -> top_repeat
            | repeatexp "{" NUMBER "," NUMBER "}"                       -> top_repeatr
            | complexp
            
?complexp : "~" charclassexp                                            -> top_not
            | charclassexp

?charclassexp : pr
            | "[ ]"                                                     -> whitespace                                     
            | "<" c ">"                                                 -> top_const
            | "[" c "]"                                                 -> top_const
            | "{??:" CNAME "}"                                          -> hole
            | simplexp

?simplexp : "(" unionexp ")"
            | "Contain(" concatexp ")"		                            -> top_contain
            | "EW(" concatexp ")"		                                -> top_ew
            | "SW(" concatexp ")"		                                -> top_sw
            | "Sep(" concatexp "," CONST "," NUMBER "," NUMBER ")"      -> top_sep_1
            | "Sep(" concatexp "," CONST "," NUMBER "," ")"             -> top_sep_2	
            
?pr: "{" tag "->" pred "}"                              -> phrase
    | "{" CONST "->" str_extract "}"                    -> match_query
    | "{" tag "}"                                       -> phrase_no_pred    
    
?str_extract : "x"                                      -> extract_id 
    | "<" c ">"                                         -> extract_str
    | "<" c ">?"                                        -> extract_optional
    | str_extract str_extract                           -> extract_append
    | "Cap(" str_extract ")"                            -> extract_cap
    | "Low(" str_extract ")"                            -> extract_low
    | "SubStr(" str_extract "," NUMBER "," NUMBER ")"   -> extract_substr
    | "SplitConcat(" str_extract "," CONST "," CONST ")" -> extract_concatsplit

?tag: "<" TYPE_TAG ">"                                  -> type_tag 						
    | "<" ENT ">"                                       -> ent_tag 
    | "<" CONST ">"                                     -> const
    | "Similar(" CONST "," NUMBER ")"                   -> word_vec
    | "Similar(" CONST "," SIM_TAG ")"                  -> word_sim
    
?pred: pred2                                            
    | pred "||" pred                                    -> pred_or
    | pred "&" pred                                     -> pred_and
    
?pred2 : pred3                                          -> pred
    | "~" pred2                                         -> pred_neg
    
?pred3: BOOL
    | "NumMatch(" NUMBER "," EQ ")"	                    -> pred_num_1
    | "NumMatch(" NUMBER "," EQ "," NUMBER "," EQ ")"	-> pred_num_2 
    | pt_date               	                        
    | pt_time	                                        
    | pt_place        
    | "(" pred ")"                                      -> pred                                  
    
?pt_date: "isYear(" NUMBER "," NUMBER ")"               -> date_year
    | "isMonth(" NUMBER "," NUMBER ")"                  -> date_month
    | "isDate(" NUMBER "," NUMBER ")"                   -> date_date

?pt_time: "btwHour(" NUMBER "," NUMBER ")"              -> time_hour
    | "btwMin(" NUMBER "," NUMBER ")"                   -> time_min
    | "btwSec(" NUMBER "," NUMBER ")"                   -> time_sec
    | "isMorning"                                       -> time_morning
    | "isAfternoon"                                     -> time_afternoon
    | "isEvening"                                       -> time_evening
    
?pt_place: "inRegion(" CONST ")"                        -> place_region
    |   "inCountry(" CONST ")"                          -> place_country
    |   "inState(" CONST ")"                            -> place_state
    
?c: CC                                                  -> cc
    | CONST                                             -> const
    
VL: "list"                                             
    | "split"                                          

CC: "ANY" | "LET" | "NUM" | "CAP" | "WORD" | "QUOTE" | "ALNUM" | "/" | "." | "," | "+" | "?" | "5" | "0" | "1" | "2" | "3" | "4" | "6" | "7" | "8" | "9" | "-" | ":" | "|" | "x" | ")" | "(" | "@" | "#" | "&" | "*" 
    | "%" | "$" | "k" | "K" | ";" | "N" | "S" | "E" | "W" | "_" | "'" | "–" | "H" | "h" | "s" | "=" | "D" 

EQ: "<" | ">" | "==" | ">=" | "<="

CONST: ESCAPED_STRING

CNAME: (LETTER|DIGIT|WS)+

CASE_TAG: "IGNORE" | "NONE" | "LOW" | "CAP"

SIM_TAG: "WIKI" | "ONELOOK" | "WORDNET"

ENT: "Person" | "Organization" | "Company"

TYPE_TAG: "Integer" | "Float" | "Date" | "Time" | "Place" | "Nationality" | "City" | "Country" | "State" | "Region" | "Year" | "Month" | "Second" | "Day" | "Hour" | "Minute"   

BOOL: "True" | "False"

%import common.WS
%import common.NUMBER
%import common.ESCAPED_STRING
%import common.LETTER
%import common.DIGIT
%ignore WS
""".strip()


class SemRegexTree(Transformer):
    """
    This is essentially the visitor class
    """
    def __init__(self, token_mode: bool, case_tag: CaseTag):
        super().__init__()
        self.token_mode = token_mode
        self.case_tag = case_tag

        # the following fields are for tracking holes only
        self.hole_id_counter = None
        self.hole_id_to_hole_type: Dict[int, str] = {}

    def top_optional(self, regexes: List[RegexPattern]) -> RegexPattern:
        assert len(regexes) == 1
        return OptionalR(regexes[0], self.token_mode, self.case_tag)

    def top_star(self, regexes: List[RegexPattern]) -> RegexPattern:
        assert len(regexes) == 1
        return StarR(regexes[0], self.token_mode, self.case_tag)

    def top_not(self, regexes: List[RegexPattern]) -> RegexPattern:
        assert len(regexes) == 1
        return NotR(regexes[0], self.token_mode, self.case_tag)

    def top_repeat(self, args: List) -> RegexPattern:
        assert len(args) == 2
        assert isinstance(args[0], RegexPattern) or isinstance(args[0], SemPattern)
        assert isinstance(args[1], Token)
        return Repeat(args[0], int(args[1].value), self.token_mode, self.case_tag)

    def top_repeat_1(self, args: List) -> RegexPattern:
        assert len(args) == 1
        assert isinstance(args[0], RegexPattern) or isinstance(args[0], SemPattern)
        return Plus(args[0], self.token_mode, self.case_tag)

    def top_repeatr(self, args: List) -> RegexPattern:
        assert len(args) == 3
        assert isinstance(args[0], RegexPattern) or isinstance(args[0], SemPattern)
        assert isinstance(args[1], Token)
        assert isinstance(args[2], Token)
        return RepeatRange(args[0], int(args[1].value), int(args[2].value), self.token_mode, self.case_tag)

    def top_contain(self, regexes: List[RegexPattern]) -> RegexPattern:
        assert len(regexes) == 1
        return Contain(regexes[0], self.token_mode, self.case_tag)

    def top_sw(self, regexes: List[RegexPattern]) -> RegexPattern:
        assert len(regexes) == 1
        return Startwith(regexes[0], self.token_mode, self.case_tag)

    def top_ew(self, regexes: List[RegexPattern]) -> RegexPattern:
        assert len(regexes) == 1
        return Endwith(regexes[0], self.token_mode, self.case_tag)

    def top_sep_1(self, args: List) -> RegexPattern:
        assert len(args) == 4
        arg1, arg2, arg3, arg4 = args[0], args[1], args[2], args[3]
        assert isinstance(arg1, RegexPattern) or isinstance(arg1, SemPattern)
        assert isinstance(arg2, Token)
        assert isinstance(arg3, Token)
        assert isinstance(arg4, Token)
        n1 = int(arg3)
        n2 = int(arg4)
        delimiter_regex = CC(arg2.value[1:-1], case_tag=CaseTag.IGNORE)

        return Concat(arg1, RepeatRange(Concat(delimiter_regex, arg1, self.token_mode), (n1-1), (n2-1), self.token_mode), self.token_mode, self.case_tag)

    def top_sep_2(self, args: List) -> RegexPattern:
        assert len(args) == 3
        arg1, arg2, arg3 = args[0], args[1], args[2]
        assert isinstance(arg1, RegexPattern) or isinstance(arg1, SemPattern)
        assert isinstance(arg2, Token)
        assert isinstance(arg3, Token)
        n1 = int(arg3)
        n2 = inf
        assert n2 >= n1 >= 1 and n2 >= 2
        delimiter_regex = CC(arg2.value[1:-1], case_tag=CaseTag.IGNORE)

        return Concat(arg1, RepeatRange(Concat(delimiter_regex, arg1, self.token_mode), (n1 - 1), inf, self.token_mode), self.token_mode, self.case_tag)

    def top_concat(self, regexes: List[RegexPattern]) -> RegexPattern:
        assert len(regexes) == 2
        return Concat(regexes[0], regexes[1], self.token_mode, self.case_tag)

    def top_or(self, regexes: List[RegexPattern]) -> RegexPattern:
        assert len(regexes) == 2

        new_args: List[RegexPattern] = []
        arg1, arg2 = regexes[0], regexes[1]
        if isinstance(arg1, Or):
            new_args.extend(arg1.args)
        else:
            new_args.append(arg1)

        if isinstance(arg2, Or):
            new_args.extend(arg2.args)
        else:
            new_args.append(arg2)

        return Or(new_args, self.token_mode, self.case_tag)

    def top_and(self, regexes: List[RegexPattern]) -> RegexPattern:
        assert len(regexes) == 2

        new_args: List[RegexPattern] = []
        arg1, arg2 = regexes[0], regexes[1]
        if isinstance(arg1, And):
            new_args.extend(arg1.args)
        else:
            new_args.append(arg1)

        if isinstance(arg2, And):
            new_args.extend(arg2.args)
        else:
            new_args.append(arg2)

        return And(new_args, self.token_mode, self.case_tag)

    def top_const(self, args: List):
        assert len(args) == 1
        arg1 = args[0]
        assert isinstance(arg1, Const) or isinstance(arg1, CC)
        arg1.case_tag = self.case_tag

        return arg1

    def whitespace(self, args: List):
        return Const(' ', self.token_mode, self.case_tag)

    def phrase(self, args: List) -> MatchType:
        assert len(args) == 2

        type_tag, pred_prog = args[0], args[1]

        assert isinstance(type_tag, TypeTag)
        assert isinstance(pred_prog, Predicate)

        sem_prog = MatchType(type_tag, pred_prog)

        return sem_prog

    def phrase_no_pred(self, args: List) -> SemPattern:
        assert len(args) == 1

        arg1 = args[0]
        if isinstance(arg1, EntTag) or isinstance(arg1, OntTag):
            sem_prog = MatchEntity(arg1)
        elif isinstance(arg1, TypeTag):
            if arg1.tag == 'YEAR':
                sem_prog = MatchType(TypeTag('DATE'), DatePredicate('isYear', [0, 2023]))
            elif arg1.tag == 'MONTH':
                sem_prog = MatchType(TypeTag('DATE'), DatePredicate('isMonth', [1, 12]))
            elif arg1.tag == 'DAY':
                sem_prog = MatchType(TypeTag('DATE'), DatePredicate('isDate', [1, 31]))
            elif arg1.tag == 'HOUR':
                sem_prog = MatchType(TypeTag('TIME'), TimePredicate('btwHour', [0, 23]))
            elif arg1.tag == 'MINUTE':
                sem_prog = MatchType(TypeTag('TIME'), TimePredicate('btwMin', [0, 59]))
            elif arg1.tag == 'SECOND':
                sem_prog = MatchType(TypeTag('TIME'), TimePredicate('btwSec', [0, 59]))
            else:
                sem_prog = MatchType(arg1)
        elif isinstance(arg1, Similar):
            sem_prog = arg1
        elif isinstance(arg1, QueryStr):
            return MatchQuery(arg1, Var())
        else:
            raise ValueError(arg1, type(arg1))
        return sem_prog

    def match_query(self, args: List) -> MatchQuery:
        assert len(args) == 2

        arg1, arg2 = args[0], args[1]
        assert isinstance(arg1, Token) and isinstance(arg2, StrFunc)

        if arg1.value.startswith('"'):
            query_str = arg1.value[1:-1]
        else:
            query_str = arg1.value

        return MatchQuery(QueryStr(query_str.lower()), arg2)

    def hole(self, args: List) -> Hole:
        assert len(args) == 1

        arg1 = args[0]
        assert isinstance(arg1, Token)
        hole_id = next(self.hole_id_counter)
        ret = Hole(arg1.value.strip(), hole_id, self.token_mode, self.case_tag, False)
        self.hole_id_to_hole_type[hole_id] = arg1.value.strip()
        return ret

    def extract_id(self, args: List) -> StrFunc:
        return Var()

    def extract_str(self, args: List) -> StrFunc:
        assert len(args) == 1

        arg1 = args[0]
        assert isinstance(arg1, CC) or isinstance(arg1, Const)

        return Str(arg1.value) if isinstance(arg1, Const) else Str(CC_REGEX_MAP[arg1.cc])

    def extract_optional(self, args: List) -> StrFunc:
        assert len(args) == 1

        arg1 = args[0]
        assert isinstance(arg1, CC) or isinstance(arg1, Const)

        if isinstance(arg1, Const):
            return Str(arg1.value, optional=True)
        else:
            if arg1.cc in CC_REGEX_MAP:
                return Str(CC_REGEX_MAP[arg1.cc], optional=True)
            elif arg1.cc in CONST_REGEX_MAP:
                return Str(CONST_REGEX_MAP[arg1.cc], optional=True)
            else:
                raise ValueError('{} is not supported'.format(arg1.cc))

    def extract_append(self, args: List) -> StrFunc:
        assert len(args) == 2

        arg1, arg2 = args[0], args[1]
        assert isinstance(arg1, StrFunc) and isinstance(arg2, StrFunc)

        return Append(arg1, arg2)

    def extract_cap(self, args: List) -> StrFunc:
        assert len(args) == 1

        arg1 = args[0]
        assert isinstance(arg1, StrFunc)

        return Capitalize(arg1)

    def extract_low(self, args: List) -> StrFunc:
        assert len(args) == 1

        arg1 = args[0]
        assert isinstance(arg1, StrFunc)

        return LowerCase(arg1)

    def extract_substr(self, args: List) -> StrFunc:
        assert len(args) == 3
        arg1 = args[0]
        assert isinstance(arg1, StrFunc)
        return SubStr(arg1, int(args[1].value), int(args[2].value))

    def extract_concatsplit(self, args: List) -> StrFunc:
        assert len(args) == 3
        arg1, arg2, arg3 = args[0], args[1], args[2]
        assert isinstance(arg1, StrFunc) and isinstance(arg2, Token) and isinstance(arg3, Token)

        return SplitConcat(arg1, arg2.value[1:-1], arg3.value[1:-1])

    def type_tag(self, args: List) -> TypeTag:
        assert len(args) == 1
        arg = args[0]
        assert isinstance(arg, Token)

        if arg.value.startswith('"'):
            tag = arg.value[1:-1]
        else:
            tag = arg.value

        if tag.lower() == 'integer':
            tag = 'int'

        return TypeTag(tag.upper())

    def ent_tag(self, args: List) -> QueryStr:
        assert len(args) == 1
        arg = args[0]
        assert isinstance(arg, Token)

        if arg.value.startswith('"'):
            tag = arg.value[1:-1]
        else:
            tag = arg.value

        return QueryStr(tag.lower())
        # return EntTag(tag)

    def ont_tag(self, args: List) -> OntTag:
        assert len(args) == 1
        arg = args[0]
        assert isinstance(arg, Token)

        if arg.value.startswith('"'):
            tag = arg.value[1:-1]
        else:
            tag = arg.value

        return OntTag(tag)

    def word_vec(self, args: List):
        assert len(args) == 2
        arg1, arg2 = args[0], args[1]
        assert isinstance(arg1, Token) and isinstance(arg2, Token)
        return Similar(ConstStr(arg1.value[1:-1]), 'VEC', float(arg2.value))

    def word_sim(self, args: List):
        assert len(args) == 2
        arg1, arg2 = args[0], args[1]
        assert isinstance(arg1, Token) and isinstance(arg2, Token)
        return Similar(ConstStr(arg1.value[1:-1]), arg2.value)

    def pred(self, pred: List) -> Predicate:
        assert len(pred) == 1
        return pred[0]

    def pred_neg(self, pred: List) -> Predicate:
        return PredicateFormula(pred, 'not')

    def pred_or(self, preds: List) -> Predicate:
        return PredicateFormula(preds, 'or')

    def pred_and(self, preds: List) -> Predicate:
        return PredicateFormula(preds, 'and')

    def pred_num_1(self, args: List):
        assert len(args) == 2
        arg1, arg2 = args[0], args[1]
        assert isinstance(arg1, Token) and isinstance(arg2, Token)
        n = float(arg1.value)
        eq = arg2.value

        if eq == '==':
            return NumPredicate(n, '>=', n, '<=')
        elif '>' in eq:
            return NumPredicate(n, eq, inf, '<=')
        elif '<' in eq:
            return NumPredicate(-inf, '>=', n, eq)
        else:
            raise ValueError

    def pred_num_2(self, args: List):
        assert len(args) == 4
        arg1, arg2, arg3, arg4 = args[0], args[1], args[2], args[3]
        assert isinstance(arg1, Token) and isinstance(arg2, Token) and isinstance(arg3, Token) and isinstance(arg4, Token)
        n_low = float(arg1.value)
        eq_low = arg2.value
        n_high = float(arg3.value)
        eq_high = arg4.value

        return NumPredicate(n_low, eq_low, n_high, eq_high)

    def date_year(self, args: List[Token]):
        assert len(args) == 2
        return DatePredicate('isYear', [int(args[0].value), int(args[1].value)])

    def date_month(self, args: List[Token]):
        assert len(args) == 2
        return DatePredicate('isMonth', [int(args[0].value), int(args[1].value)])

    def date_date(self, args: List[Token]):
        assert len(args) == 2
        return DatePredicate('isDate', [int(args[0].value), int(args[1].value)])

    def time_hour(self, args: List[Token]):
        assert len(args) == 2
        return TimePredicate('btwHour', [int(args[0].value), int(args[1].value)])

    def time_min(self, args: List[Token]):
        assert len(args) == 2
        return TimePredicate('btwMin', [int(args[0].value), int(args[1].value)])

    def time_sec(self, args: List[Token]):
        assert len(args) == 2
        return TimePredicate('btwSec', [int(args[0].value), int(args[1].value)])

    def time_morning(self, tree):
        return TimePredicate('isMorning', [])

    def time_afternoon(self, tree):
        return TimePredicate('isAfternoon', [])

    def time_evening(self, tree):
        return TimePredicate('isEvening', [])

    def place_region(self, args: List[Token]):
        assert len(args) == 1
        arg1 = args[0].value.strip()[1:-1]
        candidates = [e.strip() for e in arg1.split(',')]
        return PlacePredicate('inRegion', candidates)

    def place_country(self, args: List[Token]):
        assert len(args) == 1
        arg1 = args[0].value.strip()[1:-1]
        candidates = [e.strip() for e in arg1.split(',')]
        return PlacePredicate('inCountry', candidates)

    def place_state(self, args: List[Token]):
        assert len(args) == 1
        arg1 = args[0].value.strip()[1:-1]
        candidates = [e.strip() for e in arg1.split(',')]
        return PlacePredicate('inState', candidates)

    def cc(self, values: List[Token]):
        assert len(values) == 1
        return CC(values[0].value, self.token_mode, self.case_tag)

    def const(self, values: List[Token]):
        assert len(values) == 1
        return Const(values[0].value[1:-1], self.token_mode, self.case_tag)


parser = lark.Lark(GRAMMAR, parser='lalr')
t = SemRegexTree(token_mode=False, case_tag=CaseTag.NONE)


def parse_program(prog_str: str, token_mode: bool, case_tag: CaseTag = CaseTag.NONE) -> RegexPattern | SemPattern:
    parsed_tree = parser.parse(prog_str)
    t.token_mode = token_mode
    t.case_tag = case_tag
    t.hole_id_counter = itertools.count(start=1)
    t.hole_id_to_hole_type = {}
    prog = t.transform(parsed_tree)
    # print("finished parsing {} \n to {}".format(parsed_tree, prog))
    pd_print("finished parsing to {} from {}".format(prog, prog_str))
    return prog


def postprocess_gpt_program(prog_str: str) -> str:

    error_string = [
        'There is no',
        'No program',
        'Sorry',
        'Unfortunately',
        'Not possible to provide',
        'None provided',
        'N/A',
    ]

    prog_str = prog_str.strip()

    if any([s.lower() in prog_str.lower() for s in error_string]):
        raise ProgramParsingException

    if '\\b' in prog_str:
        raise ProgramParsingException()

    print("postprocessing {}".format(prog_str))

    prog_str = re.sub(r"\{<(Integer)>([+]|[*])?\}", r'{<Integer>}\2', prog_str, 0, re.IGNORECASE)
    prog_str = re.sub(r"\{?<(Alpha)>([+]|[*])?\}?", r'[A-Za-z]\2', prog_str, 0, re.IGNORECASE)
    prog_str = re.sub(r"\{?<(Word)>([+]|[*])?\}?", r'[A-Za-z0-9_]\2', prog_str, 0, re.IGNORECASE)
    prog_str = re.sub(r"\{?<(Alphanum)>([+]|[*])?\}?", r'[A-Za-z0-9]\2', prog_str, 0, re.IGNORECASE)
    prog_str = re.sub(r"\{?<(Alnum)>([+]|[*])?\}?", r'[A-Za-z0-9]\2', prog_str, 0, re.IGNORECASE)
    prog_str = re.sub(r"\{?<(Any)>([+]|[*])?\}?", r'(.\2)', prog_str, 0, re.IGNORECASE)
    prog_str = re.sub(r"\{?<(cc)>([+]|[*])?\}?", r'(.\2)', prog_str, 0, re.IGNORECASE)
    prog_str = re.sub(r"\{?<(constant)>([+]|[*])?\}?", r'(.\2)', prog_str, 0, re.IGNORECASE)
    prog_str = re.sub(r"{<Integer>}( ?-> ?NumMatch\([^)]+\))", r'{<Integer>\1}', prog_str, 0, re.IGNORECASE)
    prog_str = re.sub(r"(NumMatch[(][^)]+, )(=)(,[^)]+[)])", r'\1==\3', prog_str, 0, re.IGNORECASE)
    prog_str = re.sub(r"{<(NumMatch\([^)]+\))>}", r'{<Float> ->\1}', prog_str, 0, re.IGNORECASE)
    prog_str = prog_str.replace("InYear", "isYear")
    prog_str = prog_str.replace("btwYear", "isYear")
    prog_str = re.sub(r"isDate\((\d{4},\d{4})\)", r'isYear(\1)', prog_str, 0, re.IGNORECASE)
    prog_str = re.sub(r"isYear\((\d{4}),\)", r'isYear(\1, 2023)', prog_str, 0, re.IGNORECASE)
    prog_str = re.sub(r"isYear}", r'isYear(0, 2023)}', prog_str, 0, re.IGNORECASE)
    prog_str = prog_str.replace('<Fraction>', '<Float>')
    prog_str = re.sub(r'\\d', '[0-9]', prog_str)
    prog_str = re.sub(r'\\w', '[A-Za-z ]', prog_str)
    prog_str = re.sub(r'\\s', ' ', prog_str)
    prog_str = prog_str.replace("'", "[']")
    prog_str = prog_str.replace('"', '[\\"]')
    prog_str = prog_str.replace('|)', ')?')
    prog_str = re.sub(r'\\([\W_])', r'[\1]', prog_str)
    prog_str = prog_str.replace('&|', '[&]|')
    prog_str = prog_str.replace('|&', '|[&]')
    prog_str = re.sub(r'(\{[^}]+\})[?]', r'(\1)?', prog_str)
    prog_str = re.sub(r'\{(\<[^>]+\>)[?]\}', r'({\1})?*', prog_str)
    prog_str = prog_str.replace('trib of', 'trib of|of')

    print("postprocessed1 {}".format(prog_str))

    in_hole = False
    hole_content_buffer = ''
    in_buffer = False
    buffer = ''
    in_cc = False

    new_prog_str = ''

    for char in prog_str:

        # print("char: {}".format(char))
        # print(in_hole, in_buffer)
        # print("buffer: {}".format(buffer))

        if char == '{':
            if in_buffer:
                new_prog_str += format_const_or_cc(buffer)
                in_buffer = False
                buffer = ''
            in_hole = True
            new_prog_str += char
        elif char == '}':
            assert in_hole
            in_hole = False
            hole_type = hole_content_buffer.split('->')[0]
            # modify the new_sketch_content so that if name is in it, remove it
            # this is a heuristic to handle weird behavior of recent gpt3 models
            if any([s in hole_type for s in ['first name', 'last name', '??: name']]):
                pass
            else:
                hole_type = hole_type.replace(' name', '')
            # print("hole_type: {}".format(hole_type))
            if hole_type.startswith('<') and hole_type.endswith('>'):
                if hole_type.lower()[1:-1].strip() in ['int', 'integer', 'float', "date", "time", "place", "nationality", "city", "country", "state", "region", "year", "month", "second", "day", "hour", "minute"]:
                    new_prog_str += hole_content_buffer
                else:
                    hole_type = hole_type.replace('<', '"')
                    hole_type = hole_type.replace('>', '"')
                    if len(hole_content_buffer.split('->')) == 1:
                        new_prog_str += hole_type + ' -> x'
                    else:
                        new_prog_str += hole_type + ' -> ' + hole_content_buffer[1]
            else:
                new_prog_str += hole_content_buffer

            hole_content_buffer = ''
            new_prog_str += char
        elif char == '[':
            if in_buffer:
                # flush the constant
                new_prog_str += format_const_or_cc(buffer)
                in_buffer = False
                buffer = ''
            in_buffer = True
            in_cc = True
        elif char == ']':
            new_prog_str += format_const_or_cc(buffer)
            in_buffer = False
            in_cc = False
            buffer = ''
        elif char == '?' or char == '*' or char == '+' or char == '&' or char == '|' or char == '(' or char == ')':
            if in_cc:
                buffer += char
            else:
                if in_buffer:
                    # flush the constant
                    if buffer == '.':
                        # dot is the special case
                        new_prog_str += '<ANY>'
                    else:
                        new_prog_str += format_const_or_cc(buffer)
                    in_buffer = False
                    buffer = ''
                    new_prog_str += char
                elif in_hole:
                    hole_content_buffer += char
                else:
                    new_prog_str += char
        else:
            if in_hole:
                # new_prog_str += char
                hole_content_buffer += char
            else:
                in_buffer = True
                buffer += char

    if len(buffer) > 0:
        new_prog_str += format_const_or_cc(buffer)

    new_prog_str = new_prog_str.replace('>?', '> ?')
    new_prog_str = new_prog_str.replace('["]', '<QUOTE>')

    print("postprocessed {}".format(new_prog_str))

    return new_prog_str


if __name__ == '__main__':
    parse_program("{<INT> -> ~ True || ~ False}", True)
