from lib.interpreter.executor import Executor
from lib.interpreter.pattern import *
from lib.program.sketch import Sketch, parse_sketch_from_pattern
from lib.spec.synthesis_spec import SynthesisTask


def make_optional(pattern: Pattern,
                  examples: List[StrContext],
                  min_inds: List[int], executor: Executor) -> Pattern:
    """
    This function recursively traverses "Concat" nodes until it hits nodes that are
    not of type "Concat". It traverses these nodes from left to right and makes nodes
    that break the examples optional.

    Min_inds[i] represents the lowest index in the ith example such that examples[0 to i]
    matches the pattern upto this point in the traversal.
    """
    assert isinstance(pattern, RegexPattern | SemPattern)

    if not isinstance(pattern, Concat):
        # We have a non-Concat node
        # Evaluate it

        new_min_inds = []
        make_node_optional = False
        for i, (example, min_ind) in enumerate(zip(examples, min_inds)):
            spans = executor.match(example, pattern).match_spans
            # Filter out all spans that begin before the min_ind of this example so far
            spans = [span for span in spans if span.start >= min_ind]
            # if len(spans) == 0 and 'float' in str(pattern):
            #     # This node needs to become optional
            #     make_node_optional = True
            #     break
            if len(spans) == 0:
                # This node needs to become optional
                make_node_optional = True
                break

            # If spans isn't empty, we can update min_inds
            new_min_inds.append(min([span.end for span in spans]))

        if make_node_optional:
            return OptionalR(pattern)
        else:
            for i, new_min_ind in enumerate(new_min_inds):
                min_inds[i] = new_min_ind
            return pattern

    left_pt = make_optional(pattern.args[0], examples, min_inds, executor)
    pattern.args[0] = left_pt
    pattern.arg1 = left_pt
    right_pt = make_optional(pattern.args[1], examples, min_inds, executor)
    pattern.args[1] = right_pt
    pattern.arg2 = right_pt
    return pattern


def pad_pattern(pattern: Pattern,
                example: StrContext, executor: Executor,
                start_inds: List[int] = None) -> Tuple[Optional[Pattern], Optional[List[int]]]:
    """
    This function pads elements of the pattern with ".*" nodes in order to ensure that the
    pattern matches all examples. The tokens are potentially inserted between adjacent nodes
    in the pattern.

    Can return null if the function is unable to pad in a way that makes a match.
    """
    assert isinstance(pattern, RegexPattern | SemPattern)

    if start_inds is None:
        start_inds = [0]

    if not isinstance(pattern, Concat):
        # Non-Concat node
        spans = executor.match(example, pattern).match_spans
        perfect_span_matches = [span for span in spans if span.start in start_inds]
        if len(perfect_span_matches) > 0:
            # We don't need to pad here.
            new_start_inds = [span.end for span in perfect_span_matches]
            return pattern, new_start_inds
        else:
            # We have to settle for anything that's to the right of a start_ind
            min_start_ind = min(start_inds)
            imperfect_span_matches = [span for span in spans if span.start >= min_start_ind]
            if len(imperfect_span_matches) == 0:
                return None, None
            return Concat(StarR(CC('any')), pattern), [span.end for span in imperfect_span_matches]

    left_pt, start_inds_after_left = pad_pattern(pattern.args[0], example, executor, start_inds)
    if left_pt is None:
        return None, None
    pattern.args[0] = left_pt
    pattern.arg1 = left_pt

    right_pt, start_inds_after_right = pad_pattern(pattern.args[1], example, executor, start_inds_after_left)
    if right_pt is None:
        return None, None
    pattern.args[1] = right_pt
    pattern.arg2 = right_pt

    return pattern, start_inds_after_right


def manual_sketch_repair(task: SynthesisTask, executor: Executor) -> Optional[Sketch]:
    """
    Given a failed sketch, try to manually repair it to accept the positive examples.
    This does not try to create a new sketch by reprompting GPT. It just detects segments
    of the sketch that don't get matched and marks them optional, as well as padding segments
    of the sketch with "catch all" Regex parts (.*). If the synthesis fails, return None.

    This takes in an executor in order to run sample repairs (it is expensive to build a new
    executor every time this is run).
    """
    pattern = task.sketch.pattern.duplicate()
    examples = task.pos_str_context
    num_examples = len(examples)
    pattern = make_optional(pattern, examples, [0] * num_examples, executor)
    for example in examples:
        pattern = pad_pattern(pattern, example, executor)[0]
        if pattern is None:
            # If this happens, we need to make our algorithm more sophisticated than one-pass
            return None

    if not all([executor.exec(example, pattern).success for example in examples]):
        # This doesn't match everything.
        # A potential reason is because we're missing a .* token at the end
        assert isinstance(pattern, SemPattern | RegexPattern)
        pattern = Concat(pattern, StarR(CC('any')))
        if not all([executor.exec(example, pattern) for example in examples]):
            return None

    return parse_sketch_from_pattern(pattern, sketch_part=None, hole_id_to_type=task.sketch.hole_id_to_type)

