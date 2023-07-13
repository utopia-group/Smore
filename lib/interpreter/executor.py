"""
Given a program, execute the program

Match Functions: This is the interpreter for a given string and a pattern
"""
from typing import List, Dict, Tuple

from lib.config import pd_print
from lib.lang.constants import ENT_TAGS, CUSTOMIZE_TAGS, CaseTag, ADDITIONAL_TAGS
from lib.interpreter.context import StrContext, EntityPreprocessContext
from lib.interpreter.span import MatchSpan
from lib.interpreter.match_contexts import ObjectMatchContext, TagMatchContext, RegexMatchContext, \
    CustomSemMatchContext, SimilarMatchContext, QueryMatchContext, MatchContext
from lib.interpreter.pattern import Pattern, RegexPattern, Concat, \
    Repeat, Or, RepeatRange, OptionalR, StarR, Contain, Endwith, Startwith, And, NotR, SemPattern, Tag, MatchType, \
    Similar, MatchEntity, MatchQuery, Plus, Null, CC, Const, Hole
from lib.nlp.nlp import NLPFunc
from lib.spec.spec import Task
from lib.spec.synthesis_spec import SynthesisTask
from lib.utils.matcher_utils import regex_match, get_tag_value_pattern, merge_entity_match_query, split_entity_match_query, filepath_related_helpers
from lib.utils.pattern_prog_utils import get_all_semantic_types


class Executor:

    def __init__(self, debug: bool = False):
        self.debug: bool = debug
        self.nlp_engine = NLPFunc(debug=self.debug)
        self.context: Dict = {}

    def set_context(self, context: Dict):
        """
        set the context given the task
        may need to do more complicated thing in the future
        """
        self.context = context if context is not None else {}

    def d_print(self, *args):
        """
        Using formatted string as argument is required
        """
        if self.debug:
            print(args[0])

    def exec_task(self, task: Task | SynthesisTask, pt: Pattern) -> Tuple[List[bool], List[bool]]:
        self.set_context(task.context)
        if isinstance(task, Task):
            return self.exec_list_of_examples(task.pos_examples, pt), self.exec_list_of_examples(task.neg_examples, pt)
        else:
            assert isinstance(task, SynthesisTask)
            if task.no_context:
                return self.exec_list_of_examples([s.s for s in task.pos_str_context], pt), self.exec_list_of_examples([s.s for s in task.neg_str_context], pt)
            else:
                return self.exec_list_of_examples(task.pos_str_context, pt), self.exec_list_of_examples(task.neg_str_context, pt)

    def exec_list_of_examples(self, examples: List[str | StrContext], pt: Pattern, no_context: bool = False) -> List[bool]:
        pd_print("pattern: {}".format(pt))
        # print('pattern:', pt)
        pd_print("examples: {}".format(examples))
        # print("examples: {}".format(examples))

        if no_context:
            # print("no context")
            pd_print("exec_list_of_examples: {}".format([self.exec(e.s, pt).success for e in examples]))
            return [self.exec(e.s, pt).success for e in examples]
        else:
            # print("context")
            pd_print("exec_list_of_examples: {}".format([self.exec(e, pt).success for e in examples]))
            return [self.exec(e, pt).success for e in examples]

    def exec(self, input_str: str | StrContext, pt: Pattern) -> MatchContext:
        """
        execute the given string over a pattern
        """
        # print(type(input_str), type(pt))
        if isinstance(input_str, str):
            # enable preprocess context
            # print(get_all_semantic_types(pt))
            preprocess_context = EntityPreprocessContext(get_all_semantic_types(pt), self.nlp_engine)
            str_context = StrContext(input_str, preprocess_context, token_mode=pt.token_mode)
        else:
            assert isinstance(input_str, StrContext)
            str_context = input_str
        # print("str_context:", str_context.type_to_str_spans)
        return self.match(str_context, pt)

    def match(self, s: StrContext, pt: Pattern) -> MatchContext:

        """
        should I say something here???
        """
        assert isinstance(pt, RegexPattern) or isinstance(pt, SemPattern)
        mc = self.regex_match(s, pt)
        mc.check_success()
        self.d_print("Top level match success: {}".format(mc.success))

        return mc

    def object_match(self, s: StrContext, pt: MatchType) -> ObjectMatchContext:
        """
        overall logic is the following:
            we first invoke SemPattern, let it return something (ideally it should be all possible match),
            and for each possible match, we determine if one of the possible match also match syntactically
            The returned PhraseMatchContext should only contain those that both pass semantic check and predicate check
            The success criteria for passing both checks are different for ANY tag and the rest of the tags (as reflected in the code)
        """

        self.d_print(">> object_match {} for {}".format(pt, s))

        omc: ObjectMatchContext = ObjectMatchContext(s, pt)
        smc: TagMatchContext = self.tag_match(s, pt.tag)

        for sem_match_span in smc.match_spans:
            pred_success = pt.formula.exec(s.create_sub_context(sem_match_span)) if pt.formula is not None else True

            if pred_success:
                omc.match_spans.add(sem_match_span)

        self.d_print("<< object_match returns {}".format(omc))
        return omc

    def tag_match(self, s: StrContext, pt: Tag) -> TagMatchContext:
        """
        Do semantic matching based on the tag
        """
        smc = TagMatchContext(s, pt)

        self.d_print(">> tag_match {} for {}".format(pt, s))

        if pt.tag == 'ANY':
            """
            The tag is Any, then returns the whole string span, it is essentially just a regex matching problem
            TODO: let's assume this is the case, revisit later
            """
            smc.match_spans.add(MatchSpan(0, s.get_end_index()))
        elif pt.tag == 'INT':
            """
            There is a couple way we can identify a integer
            e.g. 23, +23, twenty three
            The first two case be use as an regex, while the last need a spacy package to handle it (by detecting cardinality)   
            """
            if 'integer' in s.type_to_str_spans:
                smc.match_spans.update(s.type_to_str_spans.get('integer'))

            # smc.match_spans.update(get_integer_spans(s.s))
            #
            # get cardinality entity
            # cardinality_entities = self.nlp_engine.get_entity_with_tag(s.s, 'CARDINAL')
            # int_float_ent_matcher_helper(smc, cardinality_entities, int)

        elif pt.tag == 'FLOAT':
            """
            Pretty much the same rules as the INT tag
            """
            if 'float' in s.type_to_str_spans:
                smc.match_spans.update(s.type_to_str_spans.get('float'))

            # smc.match_spans.update(get_float_spans(s.s))
            #
            # cardinality_entities = self.nlp_engine.get_entity_with_tag(s.s, 'CARDINAL')
            # int_float_ent_matcher_helper(smc, cardinality_entities, float)

        elif pt.tag in ENT_TAGS or pt.tag in ADDITIONAL_TAGS:
            # print(pt.tag)
            if pt.tag in s.type_to_str_spans:
                smc.match_spans.update(s.type_to_str_spans.get(pt.tag))

            # entities = self.nlp_engine.get_entity_with_tag(s.s, pt.tag)
            # self.d_print("entities for {}: {}".format(s.s, entities))
            # ent_matcher_helper(smc, entities)

        elif pt.tag in CUSTOMIZE_TAGS or pt.tag in self.context:

            # change smc from SemMatchContext
            smc.__class__ = CustomSemMatchContext
            assert isinstance(smc, CustomSemMatchContext)
            if pt.tag in self.context:
                smc.tag_values = self.context[pt.tag]
            else:
                smc.tag_values = CUSTOMIZE_TAGS[pt.tag]

            # let's do a pure string match of the strings the values in the tag
            smc.match_spans.update(set(regex_match(s.s, get_tag_value_pattern(smc.tag_values), case_tag=CaseTag.IGNORE)))

        else:
            raise NotImplementedError(pt.tag)

        self.d_print("<< sem_match returns {}".format(smc))

        return smc

    def similar_match(self, s: StrContext, pt: Similar) -> SimilarMatchContext:

        smc = SimilarMatchContext(s, pt)

        self.d_print(">> similar_match {} for {}".format(pt, s))

        if pt.mode == 'VEC':
            if not s.s.strip() == '':
                similar_subspans = self.nlp_engine.get_subspan_similarity(pt.keyword.tag, pt.threshold, (s.s, s.subspan_gen_for_similarity_matching()))
                smc.match_spans.update(set(similar_subspans))
        else:
            # call the nlp function with the appropriate mode
            # how this internally works should be very similar to customized_tag
            # let's also switch this to a CustomSemMatchContext since the mechanism is similar here
            smc.__class__ = CustomSemMatchContext
            assert isinstance(smc, CustomSemMatchContext)
            smc.mode = pt.mode

            if pt.mode == 'WIKI':
                smc.tag_values = self.nlp_engine.find_wiki_related_terms(pt.keyword.tag)
            elif pt.mode == 'ONELOOK' or pt.mode == 'WORDNET':
                smc.tag_values = self.nlp_engine.find_synonym(pt.keyword.tag, pt.mode)
            else:
                raise NotImplementedError('pt.mode={} is not supported'.format(pt.mode))

            smc.tag_values.append(pt.keyword.tag)
            smc.match_spans.update(set(regex_match(s.s, get_tag_value_pattern(smc.tag_values), case_tag=CaseTag.IGNORE)))

        return smc

    def query_match(self, s: StrContext, pt: MatchQuery) -> QueryMatchContext:

        qmc = QueryMatchContext(s, pt)

        self.d_print(">> query_match {} for {}".format(pt, s))
        # print('pt.query.tag:', pt.query.tag)
        # print('s_context:', s.str_span_to_predicate_context)

        if pt.query.tag in CUSTOMIZE_TAGS or pt.query.tag in self.context:

            tag_values = self.context[pt.query.tag] if pt.query.tag in self.context else CUSTOMIZE_TAGS[pt.query.tag]

            # for each tag_values, call pt.f to do string transformation
            # regex_match_res = regex_match(s.s, get_tag_value_pattern(qmc.tag_values))

            # from my experience, let's first sort this list!
            val_regexes = []
            tag_values = sorted(tag_values, key=lambda x: len(x.split()), reverse=True)

            for value in tag_values:
                val_regexes.append(pt.func.to_regex(value))

            regex = '|'.join(val_regexes)
            qmc.match_spans.update(set(regex_match(s.s, regex, case_tag=CaseTag.IGNORE)))

        elif pt.query.tag.lower() in s.type_to_str_spans:
            qmc.match_spans.update(s.type_to_str_spans[pt.query.tag.lower()])
        else:
            # generate the prompt
            spans = []
            candidate_entities = self.nlp_engine.query_entity_generation(s.s, pt.query.tag,
                                                                         pt.in_context_examples[0] if len(pt.in_context_examples[0]) > 0 else None,
                                                                         pt.in_context_examples[1] if len(pt.in_context_examples[1]) > 0 else None)
            merged_entities = merge_entity_match_query(s.s, candidate_entities, self.nlp_engine)
            spans.extend(merged_entities)
            split_entities = split_entity_match_query(s.s, candidate_entities, pt.query.tag, self.nlp_engine)
            pd_print('all entities after post-processing: {}'.format([s.get_substr(sp) for sp in spans]))
            # print("split_entities: ", split_entities)
            spans.extend(split_entities)

            # we need a special entity parser for url-related entities
            if pt.query.tag.lower() in ['directory', 'path', 'file name', 'file']:
                additional_entities = filepath_related_helpers(s.s, pt.query.tag.lower())
                spans.extend(additional_entities)
            
            qmc.match_spans.update(set(spans))

        return qmc

    # @cache(ignore_args=[0], in_memory=True)
    def regex_match(self, s: StrContext, pt: RegexPattern | SemPattern) -> MatchContext:
        """
        Regex matcher
        """
        self.d_print(">> regex_match {} for {} with pure_regex={} and token_mode={}".format(pt, s, pt.pure_regex, pt.token_mode))

        rmc = RegexMatchContext(s, pt)
        string_size = len(s.s)

        if pt.pure_regex and ((not pt.token_mode) or isinstance(pt, CC) or isinstance(pt, Const)):
            # directly call regex matching
            # we should not enter this mode if pt is in token_mode since normal regex does not support token-style regex matching
            assert isinstance(pt, RegexPattern)
            regex_pattern = pt.to_py_regex()
            # print('regex_pattern:', regex_pattern)
            # print(s.s)
            # since we now have '|' as well, we need to add a group here
            if isinstance(pt, Const) and '|' in regex_pattern:
                candidate = regex_pattern[1:-1].split('|')
                for c in candidate:
                    # print(c)
                    if c in s.s:
                        regex_pattern = c
                        rmc.match_spans.update(set(regex_match(s.s, regex_pattern, pt.case_tag)))
            else:
                self.d_print("------ regex pattern: {}".format(regex_pattern))
                rmc.match_spans.update(set(regex_match(s.s, regex_pattern, pt.case_tag)))

        else:
            if isinstance(pt, Null):
                # do nothing
                pass
            elif isinstance(pt, Concat):
                # we first need to get the arguments matching results (this is a recursive call)
                arg1_ctx: MatchContext = self.regex_match(s, pt.arg1)
                arg2_ctx: MatchContext = self.regex_match(s, pt.arg2)

                self.d_print("arg1_ctx: {}".format(arg1_ctx))
                self.d_print("arg2_ctx: {}".format(arg2_ctx))
                self.d_print("space_spans: {}".format(s.space_spans))

                res_span = pt.exec(string_size, arg1_ctx.match_spans, arg2_ctx.match_spans, s.space_spans)

                rmc.match_spans.update(res_span)
            elif isinstance(pt, Repeat) or isinstance(pt, RepeatRange):
                arg1_ctx: MatchContext = self.regex_match(s, pt.arg1)
                self.d_print("arg1_ctx: {}".format(arg1_ctx))
                self.d_print("space_spans: {}".format(s.space_spans))

                res_span = pt.exec(string_size, arg1_ctx.match_spans, s.space_spans)
                rmc.match_spans.update(res_span)

            elif isinstance(pt, OptionalR):
                arg1_ctx: MatchContext = self.regex_match(s, pt.arg1)
                self.d_print("arg1_ctx: {}".format(arg1_ctx))
                self.d_print("empty_spans: {}".format(s.empty_spans))

                res_span = pt.exec(arg1_ctx.match_spans, s.empty_spans)
                rmc.match_spans.update(res_span)

            elif isinstance(pt, Contain) or isinstance(pt, Endwith) or isinstance(pt, Startwith):
                arg_ctx: MatchContext = self.regex_match(s, pt.arg)
                self.d_print("arg1_ctx: {}".format(arg_ctx))

                res_span = pt.exec(string_size, arg_ctx.match_spans)
                rmc.match_spans.update(res_span)

            elif isinstance(pt, StarR):
                arg1_ctx: MatchContext = self.regex_match(s, pt.arg1)
                self.d_print("arg1_ctx: {}".format(arg1_ctx))
                self.d_print("space_spans: {}".format(s.space_spans))
                self.d_print("empty_spans: {}".format(s.empty_spans))

                res_span = pt.exec(string_size, arg1_ctx.match_spans, s.space_spans)
                rmc.match_spans.update(res_span)

            elif isinstance(pt, Plus):
                arg1_ctx: MatchContext = self.regex_match(s, pt.arg)
                self.d_print("arg1_ctx: {}".format(arg1_ctx))
                self.d_print("space_spans: {}".format(s.space_spans))

                res_span = pt.exec(string_size, arg1_ctx.match_spans, s.space_spans)
                rmc.match_spans.update(res_span)

            elif isinstance(pt, NotR):
                arg1_ctx: MatchContext = self.regex_match(s, pt.arg)
                self.d_print("arg1_ctx: {}".format(arg1_ctx))

                res_span = pt.exec(arg1_ctx.match_spans, s.span_universe)
                rmc.match_spans.update(res_span)

            elif isinstance(pt, Or) or isinstance(pt, And):
                narg_ctx: List = []
                for arg in pt.args:
                    narg_ctx.append(self.regex_match(s, arg).match_spans)

                self.d_print("narg_ctx: {}".format(narg_ctx))

                res_span = pt.exec(*narg_ctx)
                rmc.match_spans.update(res_span)

            elif isinstance(pt, MatchType):
                rmc = self.object_match(s, pt)

            elif isinstance(pt, MatchEntity):
                rmc = self.tag_match(s, pt.ent)

            elif isinstance(pt, Similar):
                rmc = self.similar_match(s, pt)

            elif isinstance(pt, MatchQuery):
                rmc = self.query_match(s, pt)

            elif isinstance(pt, Hole):
                rmc.match_spans.update(
                    set(s.type_to_str_spans[pt.type] if pt.type in s.type_to_str_spans else [])
                )

            else:
                raise NotImplementedError("pattern pt {} not supported".format(pt))

        self.d_print("<< regex_match returns {}".format(rmc))
        return rmc
