"""
goal of this: input the grammar in the form of a string and then instantiate a grammar class
"""
from collections import defaultdict
from typing import List, Dict, Optional

from lib.config import pd_print
from lib.interpreter.pattern import NumPredicate, PlacePredicate, DatePredicate, TimePredicate, CC, Const, OptionalR, \
    StarR, Repeat, Plus, \
    RepeatRange, Concat, Or, And, MatchQuery, MatchEntity, MatchType, ConstStr, QueryStr, EntTag, TypeTag, OntTag, \
    Contain, Startwith, Endwith, PredicateFormula, BooleanAtom
from lib.interpreter.str_function import Str, SubStr, Capitalize, LowerCase, Append, SplitConcat, Var
from lib.lang.constants import ENT_TAGS, CUSTOMIZE_TAGS, CC_REGEX_MAP, CONST_REGEX_MAP
from lib.lang.production import Production, IdProduction
from lib.lang.symbol import NonterminalSymbol, TerminalSymbol, Symbol


class CFG:
    def __init__(self):
        self.terminal_syms: List[TerminalSymbol] = []
        self.nonterminal_syms: List[NonterminalSymbol] = []
        self.start_sym: Optional[NonterminalSymbol] = None

        self.name_to_sym: Dict[str, Symbol] = {}
        self.name_to_prod: Dict[str, Production] = {}

        self.ret_symbol_to_productions: defaultdict[Symbol, List[Production]] = defaultdict(list)

        self.parse()

    def parse(self, grammar: Optional[str] = None):
        """
        TODO: The function should unify with the GRAMMAR variable in the parser in the future
        """

        # nonterminal
        r_sym = self.add_nonterminal_sym('r')  # top-level regex
        x_sym = self.add_nonterminal_sym('x')  # syntactic_match
        s_sym = self.add_nonterminal_sym('s')  # semantic match
        t_sym_const = self.add_nonterminal_sym('t_const')  # tag sym
        t_sym_ent = self.add_nonterminal_sym('t_ent')  # tag sym
        t_sym_query = self.add_nonterminal_sym('t_query')  # tag sym
        t_sym_type = self.add_nonterminal_sym('t_type')  # tag sym
        f_sym = self.add_nonterminal_sym('f')  # str extraction function symbol
        p_sym = self.add_nonterminal_sym('p')  # pred sym - recursive
        p1_sym = self.add_nonterminal_sym('p1')  # pred sym - base

        # terminal
        v_sym = self.add_terminal_sym('VAR', values=['x'])  # Var used for predicate function
        x_var_sym = self.add_terminal_sym('XVAR', values=['x'])  # Var used for the string transformation
        object_sym = self.add_terminal_sym('TYPE', values=['TIME', 'DATE', 'INT', 'FLOAT', 'CITY', 'COUNTRY', 'STATE', 'NATIONALITY', 'REGION', 'PLACE'])
        ent_sym = self.add_terminal_sym('ENT', values=ENT_TAGS)
        ont_sym = self.add_terminal_sym('ONT', values=CUSTOMIZE_TAGS.keys())  # this needs to be initialized depending on the context
        sim_sym = self.add_terminal_sym('SIM', values=[[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                                                       'WIKI', 'ONELOOK'])
        op_sym = self.add_terminal_sym('OP', values=['>', '<', '>=', '<='])
        bool_sym = self.add_terminal_sym('BOOL', values=['TRUE'])

        # the following symbol should be initialized based on context
        num_sym = self.add_terminal_sym('N', values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        repeat_num_sym = self.add_terminal_sym('NR', values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        year_sym = self.add_terminal_sym('YEAR', values=[0, 2023])
        month_sym = self.add_terminal_sym('MONTH', values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        hour_sym = self.add_terminal_sym('HOUR', values=list(range(0, 24)))
        minute_sym = self.add_terminal_sym('MIN', values=[0, 15, 30, 45, 60])
        second_sym = self.add_terminal_sym('SEC', values=[0, 15, 30, 45, 60])
        date_sym = self.add_terminal_sym('DATE', values=list(range(1, 32)))
        region_sym = self.add_terminal_sym('REGION', values=['NORTH AMERICA', 'SOUTH AMERICA', 'EUROPE', 'ASIA', 'AUSTRALIA', 'AFRICA'])
        country_sym = self.add_terminal_sym('COUNTRY', values=['United States', 'Canada', 'Britain', 'Italy', 'Germany', 'France', 'Spain', 'China', 'India', 'Japan', 'Australia', 'South Africa'])
        state_sym = self.add_terminal_sym('STATE', values=['California', 'Florida', 'Texas', 'New York', 'Pennsylvania', 'Illinois', 'Ohio', 'Georgia', 'North Carolina', 'Michigan', 'New Jersey', 'Virginia', 'Washington', 'Arizona', 'Massachusetts', 'Tennessee', 'Indiana', 'Missouri', 'Maryland', 'Wisconsin', 'Minnesota', 'Colorado', 'Alabama', 'South Carolina', 'Louisiana', 'Kentucky', 'Oregon', 'Oklahoma', 'Connecticut', 'Puerto Rico', 'Iowa', 'Mississippi', 'Arkansas', 'Utah', 'Kansas', 'Nevada', 'Nebraska', 'New Mexico', 'West Virginia', 'Idaho', 'Hawaii', 'New Hampshire', 'Maine', 'Montana', 'Rhode Island', 'Delaware', 'South Dakota', 'North Dakota', 'Alaska', 'District of Columbia', 'Vermont', 'Wyoming'])
        cc_sym = self.add_terminal_sym('CC', values=list(CC_REGEX_MAP.keys()))
        const1_sym = self.add_terminal_sym('CONST1', values=list(CONST_REGEX_MAP.keys()))
        const2_sym = self.add_terminal_sym('CONST2', values=[])
        query_sym = self.add_terminal_sym('QUERY', values=[])
        delim_sym = self.add_terminal_sym('DELIM', values=[' ', ''])  # delim symbol for SplitConcat
        custom_tag_sym = self.add_terminal_sym('CUSTOM_TAG', values=[])
        in_context_sym = self.add_terminal_sym('IN_CONTEXT', values=[])

        # set the start symbol
        self.start_sym = r_sym

        # adding the productions
        self.ret_symbol_to_productions[r_sym] = self.generate_regex_productions_with_proper_sym(r_sym, x_sym)
        self.ret_symbol_to_productions[x_sym] = self.generate_regex_bc_productions_with_proper_sym(x_sym) \
                                                    + [self.add_production(IdProduction('ToS', ret_sym=x_sym, arg_syms=[s_sym], constructor=None))]

        self.ret_symbol_to_productions[s_sym] = [
                                                # self.add_production(Production('Similar', ret_sym=s_sym, arg_syms=[t_sym_const, sim_sym], constructor=Similar)),
                                                 self.add_production(Production('MatchQuery', ret_sym=s_sym, arg_syms=[t_sym_query, f_sym, in_context_sym], constructor=MatchQuery)),
                                                 self.add_production(Production('MatchEntity', ret_sym=s_sym, arg_syms=[t_sym_ent], constructor=MatchEntity)),
                                                 self.add_production(Production('MatchType', ret_sym=s_sym, arg_syms=[t_sym_type, p_sym], constructor=MatchType))
                                                 ]

        self.ret_symbol_to_productions[t_sym_const] = [self.add_production(IdProduction('ConstStr', ret_sym=t_sym_const, arg_syms=[const2_sym], constructor=ConstStr))]

        self.ret_symbol_to_productions[t_sym_query] = [self.add_production(IdProduction('QueryStr', ret_sym=t_sym_query, arg_syms=[query_sym], constructor=QueryStr))]

        self.ret_symbol_to_productions[t_sym_ent] = [self.add_production(IdProduction('EntTag', ret_sym=t_sym_ent, arg_syms=[ent_sym], constructor=EntTag)),
                                                     # self.add_production(IdProduction('OntTag', ret_sym=t_sym_ent, arg_syms=[ont_sym], constructor=OntTag))
                                                     ]

        self.ret_symbol_to_productions[t_sym_type] = [self.add_production(IdProduction('TypeTag', ret_sym=t_sym_type, arg_syms=[object_sym], constructor=TypeTag))]

        self.ret_symbol_to_productions[f_sym] = [self.add_production(IdProduction('ToVar', ret_sym=f_sym, arg_syms=[x_var_sym], constructor=Var)),
                                                 # self.add_production(IdProduction('Str', ret_sym=f_sym, arg_syms=[const1_sym], constructor=Str)),
                                                 # self.add_production(IdProduction('Str?', ret_sym=f_sym, arg_syms=[const1_sym], constructor=lambda x: Str(x, optional=True))),
                                                 # self.add_production(Production('SubStr', ret_sym=f_sym, arg_syms=[f_sym, repeat_num_sym, repeat_num_sym], constructor=SubStr)),
                                                 # self.add_production(Production('Capitalize', ret_sym=f_sym, arg_syms=[f_sym], constructor=Capitalize)),
                                                 # self.add_production(Production('LowerCase', ret_sym=f_sym, arg_syms=[f_sym], constructor=LowerCase)),
                                                 # self.add_production(Production('Append', ret_sym=f_sym, arg_syms=[f_sym, f_sym], constructor=Append)),
                                                 # self.add_production(Production('SplitConcat', ret_sym=f_sym, arg_syms=[f_sym, delim_sym, delim_sym], constructor=SplitConcat))
                                                 ]

        self.ret_symbol_to_productions[p_sym] = [self.add_production(IdProduction('ToAtom', ret_sym=p_sym, arg_syms=[p1_sym], constructor=lambda x: x)),
                                                 self.add_production(Production('NotPred', ret_sym=p_sym, arg_syms=[p1_sym], constructor=lambda x: PredicateFormula([x], 'not'))),
                                                 self.add_production(Production('OrPred', ret_sym=p_sym, arg_syms=[p_sym, p_sym], constructor=lambda x1, x2: PredicateFormula([x1, x2], 'or'))),
                                                 self.add_production(Production('AndPred', ret_sym=p_sym, arg_syms=[p_sym, p_sym], constructor=lambda x1, x2: PredicateFormula([x1, x2], 'and'))),
                                                 ]

        self.ret_symbol_to_productions[p1_sym] = [self.add_production(Production('BooleanAtom', ret_sym=p1_sym, arg_syms=[v_sym, bool_sym], constructor=lambda v, b: BooleanAtom(b))),
                                                  self.add_production(Production('NumMatch', ret_sym=p1_sym, arg_syms=[v_sym, num_sym, op_sym, num_sym, op_sym], constructor=lambda v, num1, op1, num2, op2: NumPredicate(num1, op1, num2, op2))),
                                                  self.add_production(Production('inRegion', ret_sym=p1_sym, arg_syms=[v_sym, region_sym], constructor=lambda v, x: PlacePredicate('inRegion', x))),
                                                  self.add_production(Production('inCountry', ret_sym=p1_sym, arg_syms=[v_sym, country_sym], constructor=lambda v, x: PlacePredicate('inCountry', x))),
                                                  self.add_production(Production('inState', ret_sym=p1_sym, arg_syms=[v_sym, state_sym], constructor=lambda v, x: PlacePredicate('inState', x))),
                                                  self.add_production(
                                                      Production('isYear', ret_sym=p1_sym, arg_syms=[v_sym, year_sym, year_sym], constructor=lambda v, x1, x2: DatePredicate('isYear', [x1, x2]))),
                                                  self.add_production(
                                                      Production('isMonth', ret_sym=p1_sym, arg_syms=[v_sym, month_sym, month_sym], constructor=lambda v, x1, x2: DatePredicate('isMonth', [x1, x2]))),
                                                  self.add_production(
                                                      Production('isDate', ret_sym=p1_sym, arg_syms=[v_sym, date_sym, date_sym], constructor=lambda v, x1, x2: DatePredicate('isDate', [x1, x2]))),
                                                  self.add_production(
                                                      Production('btwHour', ret_sym=p1_sym, arg_syms=[v_sym, hour_sym, hour_sym], constructor=lambda v, x1, x2: TimePredicate('btwHour', [x1, x2]))),
                                                  self.add_production(
                                                      Production('btwMin', ret_sym=p1_sym, arg_syms=[v_sym, minute_sym, minute_sym], constructor=lambda v, x1, x2: TimePredicate('btwMin', [x1, x2]))),
                                                  self.add_production(
                                                      Production('btwSec', ret_sym=p1_sym, arg_syms=[v_sym, second_sym, second_sym], constructor=lambda v, x1, x2: TimePredicate('btwSec', [x1, x2]))),
                                                  self.add_production(Production('isMorning', ret_sym=p1_sym, arg_syms=[v_sym], constructor=lambda v: TimePredicate('isMorning'))),
                                                  self.add_production(Production('isAfternoon', ret_sym=p1_sym, arg_syms=[v_sym], constructor=lambda v: TimePredicate('isAfternoon'))),
                                                  self.add_production(Production('isEvening', ret_sym=p1_sym, arg_syms=[v_sym], constructor=lambda v: TimePredicate('isEvening')))
                                                  ]

        pd_print("Grammar initialization finished")

    def add_terminal_sym(self, name, values: Optional = None) -> TerminalSymbol:
        assert name not in self.name_to_sym
        sym = TerminalSymbol(name, values)
        self.name_to_sym[name] = sym
        return sym

    def add_nonterminal_sym(self, name) -> NonterminalSymbol:
        assert name not in self.name_to_sym
        sym = NonterminalSymbol(name)
        self.name_to_sym[name] = sym

        return sym

    def add_production(self, prod: Production) -> Production:
        assert prod.function_name not in self.name_to_prod
        self.name_to_prod[prod.function_name] = prod
        return prod

    def get_terminal_sym(self, name: str) -> TerminalSymbol:
        sym = self.name_to_sym[name]
        assert isinstance(sym, TerminalSymbol)
        return sym

    def get_nonterminal_sym(self, name: str) -> NonterminalSymbol:
        sym = self.name_to_sym[name]
        assert isinstance(sym, NonterminalSymbol)
        return sym

    def get_id_production(self, name: str) -> IdProduction:
        prod = self.name_to_prod[name]
        assert isinstance(prod, IdProduction)
        return prod

    """
    Following are helper functions for parse()
    """

    def generate_regex_bc_productions_with_proper_sym(self, a_symbol: NonterminalSymbol) -> List[Production]:
        cc_sym = self.name_to_sym['CC']
        const_sym = self.name_to_sym['CONST1']

        return [
            self.add_production(IdProduction('ToCC', ret_sym=a_symbol, arg_syms=[cc_sym], constructor=CC)),
            self.add_production(IdProduction('ToConst', ret_sym=a_symbol, arg_syms=[const_sym], constructor=Const)),
        ]

    def generate_regex_productions_with_proper_sym(self, r_symbol: NonterminalSymbol, a_symbol: NonterminalSymbol) -> List[Production]:
        num_sym = self.name_to_sym['NR']
        optional_suffix = '1' if '1' in r_symbol.name else ''

        return [
            self.add_production(IdProduction('ToA' + optional_suffix, ret_sym=r_symbol, arg_syms=[a_symbol], constructor=None)),
            self.add_production(Production('OptionalR' + optional_suffix, ret_sym=r_symbol, arg_syms=[r_symbol], constructor=OptionalR)),
            self.add_production(Production('StarR' + optional_suffix, ret_sym=r_symbol, arg_syms=[r_symbol], constructor=StarR)),
            self.add_production(Production('Plus' + optional_suffix, ret_sym=r_symbol, arg_syms=[r_symbol], constructor=Plus)),
            self.add_production(Production('Startwith' + optional_suffix, ret_sym=r_symbol, arg_syms=[r_symbol], constructor=Startwith)),
            self.add_production(Production('Endwith' + optional_suffix, ret_sym=r_symbol, arg_syms=[r_symbol], constructor=Endwith)),
            self.add_production(Production('Contain' + optional_suffix, ret_sym=r_symbol, arg_syms=[r_symbol], constructor=Contain)),
            # self.add_production(Production('NotR' + optional_suffix, ret_sym=r_symbol, arg_syms=[r_symbol], constructor=NotR)),                       # TODO: add it back later with contain only
            self.add_production(Production('Repeat' + optional_suffix, ret_sym=r_symbol, arg_syms=[r_symbol, num_sym], constructor=Repeat)),
            # self.add_production(Production('RepeatRange' + optional_suffix, ret_sym=r_symbol, arg_syms=[r_symbol, num_sym, num_sym], constructor=RepeatRange)),
            self.add_production(Production('Concat' + optional_suffix, ret_sym=r_symbol, arg_syms=[r_symbol, r_symbol], constructor=Concat)),
            self.add_production(Production('OrR' + optional_suffix, ret_sym=r_symbol, arg_syms=[[r_symbol]], constructor=Or)),
            self.add_production(Production('AndR' + optional_suffix, ret_sym=r_symbol, arg_syms=[[r_symbol]], constructor=And)),
        ]

    """
    here are helper function for get productions
    """

    def get_meaningful_id_production(self, prod: IdProduction) -> List[Production]:

        assert len(prod.arg_syms) == 1

        if isinstance(prod.arg_syms[0], TerminalSymbol):
            return [prod]
        else:
            tmp_prods = []
            for prod1 in self.ret_symbol_to_productions[prod.arg_syms[0]]:
                if isinstance(prod1, IdProduction):
                    tmp_prods.extend(self.get_meaningful_id_production(prod1))
                else:
                    tmp_prods.append(prod1)
            return tmp_prods

    def get_productions(self, sym: Symbol) -> List[Production]:

        assert isinstance(sym, NonterminalSymbol)

        productions: List[Production] = self.ret_symbol_to_productions[sym]
        ret_productions: List[Production] = []

        # need to process production here
        for prod in productions:
            if isinstance(prod, IdProduction):
                # need to keep tracing
                ret_productions.extend(self.get_meaningful_id_production(prod))
            else:
                ret_productions.append(prod)

        return ret_productions
