import re
from typing import List, Iterator, Dict, Tuple, Optional

from lib.cache import cache
from lib.config import pd_print
from lib.interpreter.executor import Executor
from lib.program.sketch import Sketch, process_and_parse_sketch
from lib.spec.spec import Task
from lib.spec.synthesis_spec import SynthesisTask
from lib.utils.exceptions import NoPositiveMatchException, SketchInfeasibleException, NoMoreSampleBudgetException
from lib.utils.gpt3_utils import generate_sketch_prompt
from lib.utils.matcher_utils import get_number_spans
from lib.utils.pq import PriorityQueue
from lib.utils.sketch_repair_utils import check_date_repair, check_time_repair, locate_error, check_general_semantic_repair, generalize_error_regex, decompose_regex, \
    get_error_component_type, decompose_sketch
from lib.utils.sketch_utils import get_semantic_holes


@cache(ignore_args=[0])
def sample_sketch_helper(executor, prompt: str, model: str, temperature: float, i: int) -> str:
    res = executor.nlp_engine.call_gpt3_and_get_answer(prompt, '', model, temperature=temperature)
    return res


class SketchGenerator:
    def __init__(self, mode: str, executor: Executor, no_type: bool, no_decomp: bool, no_repair: bool, prompt_version: str):
        self.mode = mode
        self.prompt_version: str = prompt_version
        self.sketch_ranking_func = lambda x: (-get_semantic_holes(x.sketch_str), -len(x.sketch_str))
        self.executor = executor
        self.nlp_func = executor.nlp_engine
        self.gpt_func = {
            'code-davinci-002': lambda x: self.nlp_func.call_gpt3_and_get_answer(x, self.mode, 'code-davinci-002'),
            'text-davinci-003': lambda x: self.nlp_func.call_gpt3_and_get_answer(x, self.mode, 'text-davinci-003'),
            'gpt-3.5-turbo': lambda x: self.nlp_func.call_gpt3_and_get_answer(x, self.mode, 'gpt-3.5-turbo')
        }

        self.no_type = no_type
        self.no_decomp = no_decomp
        self.no_repair = no_repair

        self.sketch_to_prompt: Dict[str, str] = {}
        self.worklist: PriorityQueue = PriorityQueue(self.sketch_ranking_func)

        self.handled_new_examples: List[str] = []

        self.sketch_count: int = 0

    def get_sketch(self, task: Task) -> Iterator[Sketch]:
        """
        A worklist algorithm of keep providing sketches lol
        """
        self.worklist: PriorityQueue = PriorityQueue(self.sketch_ranking_func)
        self.sketch_to_prompt: Dict[str, str] = {}
        self.sketch_count = 0
        self.worklist.put_all(self.init(task))

        while not self.worklist.is_empty():

            if self.no_repair and self.sketch_count >= 10:
                raise NoMoreSampleBudgetException()

            s = self.worklist.pop()
            self.sketch_count += 1
            yield s

    def get_naive_sketch(self) -> Sketch:
        sketch_str = '{??: String}'
        sketch = process_and_parse_sketch(sketch_str)
        return sketch

    def sample_sketch(self, task: Task):
        full_prompt = generate_sketch_prompt(task.pos_examples, self.prompt_version)

        res = []
        for model in self.gpt_func:
            sketch_str = sample_sketch_helper(self.executor, full_prompt, model, 0.7, self.sketch_count)
            if 'N/A' in sketch_str:
                continue
            if '{??: }' in sketch_str:
                continue
            if 'No positive examples' in sketch_str:
                continue
            if 'Sketch:' in sketch_str:
                continue

            sketch = process_and_parse_sketch(sketch_str)
            if sketch is not None and str(sketch) not in self.sketch_to_prompt:
                res.append(sketch)
                self.sketch_to_prompt[str(sketch)] = full_prompt
        self.worklist.put_all(list(set(res)))

    def repair_sketch(self, task: SynthesisTask, exception: Exception) -> Optional[Sketch]:
        # repair the sketch based on whatever and add the repaired version to the worklist
        pd_print('REPAIRING SKETCH ', task.sketch)

        def regenerate_sketch(p: str) -> List[Sketch]:
            np = '{}\n{}\n\n{}'.format(p, task.sketch.get_full_sketch_str(), "Give me a sketch that is different from the last one.")
            # pd_print('new_prompt: ', np)
            ns = self.call_gpt_and_parse(np)
            return ns

        def regenerate_abstract_sketch(p: str) -> List[Sketch]:
            np = '{}\n{}\n\n{}'.format(p, task.sketch.get_full_sketch_str(), "/*Regenerate the sketch for the last task that is more abstract.*/")
            # pd_print('new_prompt: ', np)
            ns = self.call_gpt_and_parse(np)
            return ns

        def date_time_repair(p: str) -> List[Sketch]:
            assert isinstance(exception, NoPositiveMatchException)
            np = '{}\n{}\n\n{}'.format(p, task.sketch.get_full_sketch_str(), "Give me a different sketch using entities such as date and time.")
            # pd_print('new_prompt: ', np)
            ns = self.call_gpt_and_parse(np)
            return ns

        def number_repair(p: str) -> List[Sketch]:
            assert isinstance(exception, NoPositiveMatchException)
            np = '{}\n{}\n\n{}'.format(p, task.sketch.get_full_sketch_str(), "Give me a different sketch using more abstract types such as float or integer.")
            # pd_print('new_prompt: ', np)
            ns = self.call_gpt_and_parse(np)
            return ns

        def general_semantic_repair(rc: List[str], ec: int, s: str, s1: str) -> List[Sketch]:
            """
            TODO: there are a couple scenarios to handle
                1. current sketch work on some of the examples -> we keep the working sketch as well and do a union of the new and old sketch
                2. current sketch does not work on any of the examples -> replace the old sketch with the new sketch
            """
            assert isinstance(exception, NoPositiveMatchException)
            valid_examples, new_positive_examples, eid = generate_new_examples(task, ec, (s, s1), located_mode)
            # if the current examples have already been handled before, we don't want to add them again
            if str(new_positive_examples) in self.handled_new_examples:
                return []
            else:
                self.handled_new_examples.append(str(new_positive_examples))
            # print('new_examples:', new_positive_examples)
            if any(e.strip() == '' for e in new_positive_examples) and not all(e.strip() == '' for e in new_positive_examples):
                add_optional = True
            else:
                add_optional = False

            new_prompt = generate_sketch_prompt(new_positive_examples, self.prompt_version)
            sketch_to_keep, removed_sketch = generate_sketch_to_keep(task.sketch, rc, eid, located_mode, add_optional)
            if all(e.strip() == '' for e in new_positive_examples):
                ns = self.call_gpt_and_parse(new_prompt, sketch_to_keep, removed_sketch)
            else:
                ns = self.call_gpt_and_parse(new_prompt, sketch_to_keep)
            # pd_print('new_prompt:', new_prompt)
            # pd_print('ns:', ns)
            return ns

        def optional_repair(opt_rc_i: List[int]) -> List[Sketch]:

            sketch_decomposed: List[str] = decompose_sketch(repr(task.sketch.to_pattern()))

            # print('sketch_decomposed:', sketch_decomposed)
            # print('regex_components:', regex_components)
            assert len(sketch_decomposed) == len(regex_components)

            for i in opt_rc_i:
                sketch_decomposed[i] = '{}?'.format(sketch_decomposed[i])

            new_sketch_str = ''.join(sketch_decomposed)
            # print('new_sketch_str:', new_sketch_str)
            sketch = process_and_parse_sketch(new_sketch_str)
            if sketch is not None and str(sketch) not in self.sketch_to_prompt:
                self.sketch_to_prompt[str(sketch)] = ''
            else:
                return []

            return [sketch]

        if isinstance(exception, NoPositiveMatchException):
            if task.sketch.has_union:
                raise NotImplementedError

            # find out what type of error is the error component is
            # print(self.sketch_to_prompt.keys())
            # print(str(task.sketch))
            prompt = self.sketch_to_prompt.get(str(task.sketch))
            assert prompt is not None

            # try to locate the error
            locate_error_res = locate_error(exception)
            # print('locate_error_res:', locate_error_res)

            if locate_error_res is None:
                # case 1: The sketch is failed completely
                #   i.e. the first component is not valid and the last component is not valid
                #   some components in the middle might be valid, but we don't handle this case

                new_sketches = regenerate_sketch(prompt)
                self.worklist.put_all(new_sketches)
            else:
                # case 2: the sketch is not completely failed
                regex_components, error_cid, remained_str, located_mode, optional_rc_i = locate_error_res
                error_component = get_error_component_type(regex_components, error_cid)

                if isinstance(error_component, tuple):
                    # situation 1: the hole itself is wrong -> then we handle based on the hole type

                    error_hole_type, error_hole_id = error_component
                    # This is a heuristic
                    if check_date_repair(error_hole_type) or check_time_repair(error_hole_type):
                        # scenario 1: date/time checker
                        # Ask another sketch using some other concepts
                        new_sketches = date_time_repair(prompt)
                        self.worklist.put_all(new_sketches)

                    elif check_general_semantic_repair(error_hole_type):
                        # scenario 2: general semantic type error
                        # Once we find out the error hole, we need to generate the new examples and query gpt3 again
                        new_sketches = general_semantic_repair(regex_components, error_cid, exception.context.s, remained_str)
                        self.worklist.put_all(new_sketches)

                    else:
                        # scenario 3: integer/floating number checker
                        # Check if the remaining string parsable. If not, just ask for another sketch; if it is, then we need to handle differently
                        num_spans = get_number_spans(remained_str, None)
                        if any(span.start == 0 and span.end == len(remained_str) for span in num_spans):
                            new_sketches = number_repair(prompt)
                            self.worklist.put_all(new_sketches)
                        else:
                            new_sketches = general_semantic_repair(regex_components, error_cid, exception.context.s, remained_str)
                            self.worklist.put_all(new_sketches)
                else:
                    if error_component == 'concrete':
                        # situation 2: a concrete part of the regex is wrong, then we handle using general semantic repair
                        new_sketches = general_semantic_repair(regex_components, error_cid, exception.context.s, remained_str)
                        self.worklist.put_all(new_sketches)
                    elif error_component == 'append':
                        # situation 3: a new component need to be generated at the end of the regex
                        new_sketches = general_semantic_repair(regex_components, error_cid, exception.context.s, remained_str)
                        self.worklist.put_all(new_sketches)
                    elif error_component == 'optional':
                        # making the component in optional_rc optional
                        new_sketches = optional_repair(optional_rc_i)
                        self.worklist.put_all(new_sketches)
                    else:
                        raise ValueError('{} should not show up here'.format(error_component))

        elif isinstance(exception, SketchInfeasibleException):
            # This branch handles the cases where the synthesis task is not feasible
            #   i.e. the over-approx satisfy the positive examples but the synthesis failed
            prompt = self.sketch_to_prompt.get(str(task.sketch))
            assert prompt is not None
            new_sketches = regenerate_abstract_sketch(prompt)
            self.worklist.put_all(new_sketches)
        elif isinstance(exception, NotImplementedError):
            # something is wrong, at this point we just ignore (because I don't have time to fix thing now)
            pass
        elif isinstance(exception, TimeoutError):
            raise exception
        else:
            # TODO: Not-yet handled exceptions, not sure what else are there, at this point we just ignore
            pass
            # raise NotImplementedError

        return None

    def init(self, task: Task) -> List[Sketch]:
        full_prompt = generate_sketch_prompt(task.pos_examples, self.prompt_version)
        all_sketches = self.call_gpt_and_parse(full_prompt)
        # current heuristic in deciding how to choose the sketch: how much semantic holes and the length of the sketch
        return all_sketches

    def call_gpt_and_parse(self, prompt, sketch_to_keep: Optional[str] = None, add_optional: Optional[str] = None) -> List[Sketch]:
        res = []
        for func in self.gpt_func.values():
            output = func(prompt)
            if 'N/A' in output:
                continue
            if '{??: }' in output:
                continue
            if 'No positive examples' in output:
                continue
            if 'Sketch:' in output:
                continue
            pd_print('output:', output)
            # processed_output = postprocess_gpt3_sketch(output)

            if self.no_type:
                output = re.sub(r'{[?][?]: ([\w ]+)}', '{??: string}', output)

            if sketch_to_keep is not None:
                new_sketch = output.strip()
                if add_optional is not None:
                    output = sketch_to_keep.replace('<REPLACE>', '(' + add_optional + ')?')
                else:
                    output = sketch_to_keep.replace('<REPLACE>', '(' + new_sketch + ')')
            sketch = process_and_parse_sketch(output)
            if sketch is not None and str(sketch) not in self.sketch_to_prompt:
                if self.no_type:
                    sketch.no_type = True
                else:
                    sketch.no_type = False

                res.append(sketch)
                self.sketch_to_prompt[str(sketch)] = prompt
        return list(set(res))


def generate_new_examples(task: SynthesisTask, error_cid: int, processed_str: Tuple[str, str], mode: str) -> Tuple[List[str], List[str], int]:
    """
    generate a set of new examples to generate sketches for the error part of the sketch
    """

    def get_new_example(s, e):
        regex = task.sketch.get_overapprox(dict([(ty, [s.get_substr_regex(span) for span in str_spans]) for ty, str_spans in s.type_to_str_spans.items()]), lazy=False)
        pd_print('regex: {}'.format(regex))
        # modify the regex such that every part after error_hid is .*
        decomposed_regex = decompose_regex(regex)
        regex1, gn = generalize_error_regex(decomposed_regex, e, mode)
        regex2 = '^{}$'.format(regex1)
        pd_print('regex2: {}'.format(regex2))
        res = re.match(regex2, s.s)

        return res, gn

    eid = error_cid
    decomposed_sketch = decompose_sketch(repr(task.sketch.to_pattern()))
    # print('decomposed_sketch:', decomposed_sketch)

    if mode == 'forward':
        while eid >= 0:
            # print('eid:', eid)
            new_res = []  # examples that don't pass the current sketch
            old_res = []  # examples that pass the current sketch

            if ('charseq' in decomposed_sketch[eid-1] or decomposed_sketch == '((.)*)') and eid != 0:
                eid -= 1

            for s_ctx in task.pos_str_context:
                match_res, group_name = get_new_example(s_ctx, eid)
                if match_res is None:
                    eid -= 1
                    break
                else:
                    new_res.append(match_res.group(group_name))

            pd_print('new_res: {}'.format(new_res))

            if len(new_res) == len(task.pos_str_context):
                return old_res, new_res, eid
    else:
        assert mode == 'backward'
        while eid < len(decomposed_sketch):
            pd_print('eid: {}'.format(eid))
            new_res = []  # examples that don't pass the current sketch
            old_res = []  # examples that pass the current sketch

            if eid != len(decomposed_sketch) - 1 and ('charseq' in decomposed_sketch[eid+1] or decomposed_sketch == '((.)*)'):
                eid += 1

            for s_ctx in task.pos_str_context:
                match_res, group_name = get_new_example(s_ctx, eid)
                if match_res is None:
                    eid += 1
                    break
                else:
                    new_res.append(match_res.group(group_name))

            pd_print('new_res: {}'.format(new_res))

            if len(new_res) == len(task.pos_str_context):
                return old_res, new_res, eid


def generate_sketch_to_keep(sketch: Sketch, regex_components: List[str], error_cid: int, mode: str = 'forward', add_optional: bool = False):
    """
    given a sketch, figure out which part of the sketch need to be kept
    """

    sketch_decomposed: List[str] = decompose_sketch(repr(sketch.to_pattern()))

    # print('sketch_decomposed:', sketch_decomposed)
    # print('regex_components:', regex_components)
    assert len(sketch_decomposed) == len(regex_components)

    if mode == 'forward':
        sketch_split = sketch_decomposed[:error_cid]
        if add_optional:
            new_sketch_str = '{}{}'.format(''.join(sketch_split), '(<REPLACE>)?')
        else:
            new_sketch_str = '{}{}'.format(''.join(sketch_split), '<REPLACE>')
        # print('new_sketch_str:', new_sketch_str)
        return new_sketch_str, ''.join(sketch_decomposed[error_cid:])
    else:
        assert mode == 'backward'
        sketch_split = sketch_decomposed[(error_cid + 1):]
        if add_optional:
            new_sketch_str = '{}{}'.format('<REPLACE>?', ''.join(sketch_split))
        else:
            new_sketch_str = '{}{}'.format('<REPLACE>', ''.join(sketch_split))
        # print('new_sketch_str:', new_sketch_str)
        return new_sketch_str, ''.join(sketch_decomposed[:(error_cid + 1)])
