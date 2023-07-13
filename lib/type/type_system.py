import typing
from collections import namedtuple
from typing import Dict, List, Tuple

from lib.type.base_type import BaseType, construct_base_type, Type, Union, Any, Optional, GPT3Type
from lib.type.type_enum import base_types, parsable_types, entity_types


class TypeInferenceException(Exception):
    pass


"""
First instantiate all the base types
"""

base_types_to_class: Dict[str, type] = {
    'any': Any,
    'basetype': BaseType,
    'gpt3type': GPT3Type
}
base_types_to_object: Dict[str, BaseType] = {
    'any': Any(),
    'basetype': BaseType('top'),
    'gpt3type': GPT3Type()
}


def add_base_type(type_name: str, superclasses: tuple) -> Type:
    superclasses_classes = tuple([base_types_to_class.get(e) for e in superclasses])
    assert all(v is not None for v in superclasses_classes)

    new_class, new_object = construct_base_type(type_name, superclasses_classes)  # type: ignore
    base_types_to_class[type_name] = new_class
    base_types_to_object[type_name] = new_object

    return new_object


def instantiate_base_type():
    """
    Create all the base type classes and objects automatically
    """
    for type_name, superclasses in base_types:
        add_base_type(type_name, superclasses)

    assert base_types_to_object.keys() == base_types_to_class.keys()


instantiate_base_type()


def get_type(name: str) -> Tuple[typing.Optional[Type], typing.Optional[str]]:
    """
    Return the type if found the type
    Return none if not found the type and return what type should be added to the base type system
    NOTE: According to how this function is implemented, this function should at most recurse once
    """
    name = name.lower()

    if name == 'int':
        name = 'integer'
    elif name == 'org':
        name = 'organization'
    elif name == 'loc':
        name = 'location'

    if name.startswith('optional'):
        assert '(' in name and ')' in name
        arg_name = name[(name.index('(') + 1): name.index(')')]
        arg_type, return_name = get_type(arg_name)

        if arg_type is None:
            return None, return_name
        else:
            assert isinstance(arg_type, BaseType)
            return Optional(arg_type), None
    elif name in base_types_to_object:
        return base_types_to_object[name], None
    else:
        return None, name


def get_subtypes(ty: BaseType | str) -> List[BaseType]:
    if isinstance(ty, str):
        ty, to_be_add_ty = get_type(ty)
        if ty is None:
            return []

    subtypes = []
    for type_object in base_types_to_object.values():
        if isinstance(type_object, ty.__class__):
            subtypes.append(type_object)

    return subtypes


"""
Given a certain syntax of the language, output the type
Rules for type inference 
"""


def infer_type_cc_const() -> BaseType:
    return base_types_to_object['charseq']


def derive_type_cc_const(return_type: Type) -> typing.Optional[BaseType]:
    if return_type == base_types_to_object['string'] or return_type == base_types_to_object['charseq']:
        assert isinstance(return_type, BaseType)
        return return_type

    return None


def infer_type_similar() -> BaseType:
    return base_types_to_object['string']


def derive_type_similar(return_type: Type) -> typing.Optional[BaseType]:
    if isinstance(return_type, BaseType):
        return return_type

    return None


def infer_type_matchQuery(args: List) -> BaseType:
    arg_type = args[0]
    assert isinstance(arg_type, BaseType)
    if arg_type.name in base_types_to_object:
        return base_types_to_object[arg_type.name]
    elif isinstance(arg_type, GPT3Type):
        return arg_type
    else:
        return base_types_to_object['string']


def derive_type_matchQuery(return_type: Type) -> typing.Optional[BaseType]:
    if isinstance(return_type, BaseType):
        if return_type.name not in parsable_types or return_type.name == 'string':
            return return_type
        elif isinstance(return_type, GPT3Type):
            return return_type

    return None


def infer_type_matchEntity(args: List) -> BaseType:
    arg_type = args[0]
    assert isinstance(arg_type, BaseType)
    if arg_type.name in base_types_to_object:
        return base_types_to_object[arg_type.name]
    else:
        raise ValueError('{} is not a entity tag'.format(args[0]))


def derive_type_matchEntity(return_type: Type) -> typing.Optional[BaseType]:
    if isinstance(return_type, BaseType):
        # if return_type.name in entity_types or return_type.name == 'string':
        #     return return_type
        if return_type.name == 'string':
            return return_type

    return None


def infer_type_hasType(args: List) -> BaseType:
    arg_type = args[0]
    assert isinstance(arg_type, BaseType)
    if arg_type.name in parsable_types:
        if len(args) == 1:
            return arg_type
        else:
            assert len(args) == 2
            pred_type = args[1]
            assert isinstance(pred_type, BaseType)
            if isinstance(pred_type, arg_type.__class__):
                return pred_type
            elif isinstance(arg_type, pred_type.__class__):
                return arg_type
            else:
                raise TypeInferenceException('The predicate type {} is not a subtype of the arg type {}'.format(pred_type.name, arg_type.name))
    else:
        raise ValueError('{} is not a parsable tag'.format(args[0]))


def derive_type_hasType(return_type: Type) -> typing.Optional[BaseType]:
    if isinstance(return_type, BaseType):
        if return_type.name in parsable_types or return_type.name == 'string':
            return return_type

    return None


def infer_type_not() -> BaseType:
    return base_types_to_object['string']


def derive_type_not(return_type: Type) -> typing.Optional[BaseType]:
    if return_type == base_types_to_object['string'] or return_type == base_types_to_object['charseq']:
        assert isinstance(return_type, BaseType)
        return return_type
    else:
        return None


def infer_type_concat(args: List) -> BaseType:
    not_charseq = False
    for arg in args:
        if isinstance(arg, BaseType):
            if arg.name == 'charseq':
                continue
            else:
                not_charseq = True
                break
    if not not_charseq:
        return base_types_to_object['charseq']
    else:
        return base_types_to_object['string']


def derive_type_concat(return_type: Type) -> typing.Optional[BaseType]:
    if return_type == base_types_to_object['string'] or return_type == base_types_to_object['charseq']:
        assert isinstance(return_type, BaseType)
        return return_type
    else:
        return None


def infer_type_star(args: List) -> Optional:
    assert len(args) == 1
    if isinstance(args[0], BaseType):
        if args[0].name == 'charseq':
            return Optional(base_types_to_object['charseq'])

    return Optional(base_types_to_object['string'])


def derive_type_star(return_type: Type) -> typing.Optional[BaseType]:
    if isinstance(return_type, Optional):
        if return_type.base == base_types_to_object['string'] or return_type == base_types_to_object['charseq']:
            return return_type.base

    return None


def infer_type_optional(args: List[Type]) -> Optional:
    arg_type = args[0]
    if isinstance(arg_type, BaseType):
        return Optional(arg_type)
    elif isinstance(arg_type, Optional):
        return arg_type
    else:
        raise NotImplementedError


def derive_type_optional(return_type: Type) -> typing.Optional[BaseType]:
    if isinstance(return_type, Optional):
        return return_type.base
    else:
        return None


def infer_type_or(args: List[Type]) -> Union:
    assert len(args) > 0
    new_types = []

    for arg in args:
        if isinstance(arg, Union):
            new_types.extend(arg.init_types)
        else:
            new_types.append(arg)

    return Union(new_types)


def derive_type_or(return_type: Type) -> Tuple[Type, Type]:
    if isinstance(return_type, Union):
        if len(return_type.init_types) == 2:
            return return_type.init_types[0], return_type.init_types[1]
        else:
            return return_type.init_types[0], Union(return_type.init_types[1:])
    else:
        return return_type, return_type


def infer_type_and(args: List[Type]) -> Type:
    assert len(args) > 0
    most_precise_type = args[0]

    for arg in args[1:]:
        # print(arg)

        if isinstance(arg, BaseType):
            if arg.name == 'charseq':
                continue

            if issubclass(arg.__class__, most_precise_type.__class__):
                most_precise_type = arg
            elif issubclass(most_precise_type.__class__, arg.__class__):
                pass
            else:
                raise TypeInferenceException('Cannot infer type for and')
        elif isinstance(arg, Optional):

            if arg.base.name == 'charseq':
                continue

            if isinstance(most_precise_type, Optional):

                if issubclass(arg.base.__class__, most_precise_type.base.__class__):
                    most_precise_type = arg
                elif issubclass(most_precise_type.base.__class__, arg.base.__class__):
                    pass
                else:
                    raise TypeInferenceException('Cannot infer type for and')
            else:
                if issubclass(arg.base.__class__, most_precise_type.__class__):
                    most_precise_type = arg.base
                elif issubclass(most_precise_type.__class__, arg.base.__class__):
                    pass
                else:
                    raise TypeInferenceException('Cannot infer type for and')
        else:
            raise NotImplementedError('Cannot infer type for and ({})'.format(type(arg)))

    return most_precise_type


def derive_type_and(return_type: Type) -> Tuple[Type, Type]:
    """
    TODO: This rule seems problematic
    """
    return return_type, base_types_to_object['charseq']


def infer_type_repeat_class(args: List) -> BaseType:
    assert len(args) >= 1
    if isinstance(args[0], BaseType):
        if args[0].name == 'charseq':
            return base_types_to_object['charseq']

    return base_types_to_object['string']


def derive_type_repeat_class(return_type: Type) -> typing.Optional[BaseType]:
    if return_type == base_types_to_object['string'] or return_type == base_types_to_object['charseq']:
        assert isinstance(return_type, BaseType)
        return return_type
    else:
        return None


def infer_type_numMatch() -> BaseType:
    return base_types_to_object['number']


def derive_type_numMatch(return_type: Type) -> typing.Optional[BaseType]:
    if return_type.__class__ in base_types_to_object['integer'].__class__.mro() or return_type.__class__ in \
            base_types_to_object['float'].__class__.mro():
        assert isinstance(return_type, BaseType)
        if isinstance(return_type, base_types_to_object['number'].__class__):
            return return_type
        else:
            return base_types_to_object['number']

    return None


def infer_type_inRegion() -> BaseType:
    return base_types_to_object['place']


def derive_type_inRegion(return_type: Type) -> typing.Optional[BaseType]:
    assert isinstance(return_type, BaseType)

    to_return = base_types_to_object['region']
    if return_type.__class__ in base_types_to_object['region'].__class__.mro():
        return to_return

    if (return_type.name in parsable_types
            and parsable_types[return_type.name] in ['GPE', 'NORP']):
        return to_return

    return None


def infer_type_inCountry() -> BaseType:
    return base_types_to_object['place']


def derive_type_inCountry(return_type: Type) -> typing.Optional[BaseType]:
    assert isinstance(return_type, BaseType)

    to_return = base_types_to_object['country']
    if return_type.__class__ in base_types_to_object['country'].__class__.mro():
        return to_return

    if (return_type.name in parsable_types
            and parsable_types[return_type.name] in ['GPE', 'NORP']):
        return to_return

    return None


def infer_type_inState() -> BaseType:
    return base_types_to_object['place']


def derive_type_inState(return_type: Type) -> typing.Optional[BaseType]:
    assert isinstance(return_type, BaseType)

    to_return = base_types_to_object['state']
    if return_type.__class__ in base_types_to_object['state'].__class__.mro():
        return to_return

    if (return_type.name in parsable_types
            and parsable_types[return_type.name] in ['PLACE', 'NATIONALITY', 'CITY', 'STATE', 'CONTINENT', 'COUNTRY']):
        return to_return

    return None


def infer_type_isYear() -> BaseType:
    return base_types_to_object['year']


def derive_type_isYear(return_type: Type) -> typing.Optional[BaseType]:
    assert isinstance(return_type, BaseType)
    if return_type.__class__ in base_types_to_object['year'].__class__.mro():
        return base_types_to_object['year']

    return None


def infer_type_isMonth() -> BaseType:
    return base_types_to_object['month']


def derive_type_isMonth(return_type: Type) -> typing.Optional[BaseType]:
    assert isinstance(return_type, BaseType)
    if return_type.__class__ in base_types_to_object['month'].__class__.mro():
        return base_types_to_object['month']

    return None


def infer_type_isDate() -> BaseType:
    return base_types_to_object['day']


def derive_type_isDate(return_type: Type) -> typing.Optional[BaseType]:
    assert isinstance(return_type, BaseType)
    if return_type.__class__ in base_types_to_object['day'].__class__.mro():
        return base_types_to_object['day']

    return None


def infer_type_btwHour() -> BaseType:
    return base_types_to_object['hour']


def derive_type_btwHour(return_type: Type) -> typing.Optional[BaseType]:
    assert isinstance(return_type, BaseType)
    if return_type.__class__ in base_types_to_object['hour'].__class__.mro():
        return base_types_to_object['hour']

    return None


def infer_type_btwMin() -> BaseType:
    return base_types_to_object['minute']


def derive_type_btwMin(return_type: Type) -> typing.Optional[BaseType]:
    assert isinstance(return_type, BaseType)
    if return_type.__class__ in base_types_to_object['minute'].__class__.mro():
        return base_types_to_object['minute']

    return None


def infer_type_btwSec() -> BaseType:
    return base_types_to_object['second']


def derive_type_btwSec(return_type: Type) -> typing.Optional[BaseType]:
    assert isinstance(return_type, BaseType)
    if return_type.__class__ in base_types_to_object['second'].__class__.mro():
        return base_types_to_object['second']

    return None


def infer_type_isMorning() -> BaseType:
    return base_types_to_object['time']


def derive_type_isMorning(return_type: Type) -> typing.Optional[BaseType]:
    assert isinstance(return_type, BaseType)
    if return_type.__class__ in base_types_to_object['time'].__class__.mro():
        return base_types_to_object['time']

    return None


def infer_type_isEvening() -> BaseType:
    return base_types_to_object['time']


def derive_type_isEvening(return_type: Type) -> typing.Optional[BaseType]:
    assert isinstance(return_type, BaseType)
    if return_type.__class__ in base_types_to_object['time'].__class__.mro():
        return base_types_to_object['time']

    return None


def infer_type_isAfternoon() -> BaseType:
    return base_types_to_object['time']


def derive_type_isAfternoon(return_type: Type) -> typing.Optional[BaseType]:
    assert isinstance(return_type, BaseType)
    if return_type.__class__ in base_types_to_object['time'].__class__.mro():
        return base_types_to_object['time']

    return None


def infer_type_pred(args: List[Type]) -> Type:
    return args[0]


def derive_type_pred(return_type: Type) -> Type:
    return return_type


def infer_type_bool() -> Type:
    return base_types_to_object['any']


def derive_type_bool(return_type: Type) -> typing.Optional[Type]:
    assert isinstance(return_type, BaseType)
    if return_type.name not in ['year', 'month', 'day', 'second', 'minute', 'hour']:
        return return_type
    return None


TypingRule = namedtuple('TypingRule', ['infer', 'derive'])

typing_rules: Dict[str, TypingRule] = {
    'ToCC': TypingRule(infer=infer_type_cc_const, derive=derive_type_cc_const),
    'ToConst': TypingRule(infer=infer_type_cc_const, derive=derive_type_cc_const),
    'Similar': TypingRule(infer=infer_type_similar, derive=derive_type_similar),
    'MatchQuery': TypingRule(infer=infer_type_matchQuery, derive=derive_type_matchQuery),
    'MatchEntity': TypingRule(infer=infer_type_matchEntity, derive=derive_type_matchEntity),
    'MatchType': TypingRule(infer=infer_type_hasType, derive=derive_type_hasType),
    'OptionalR': TypingRule(infer=infer_type_optional, derive=derive_type_optional),
    'StarR': TypingRule(infer=infer_type_star, derive=derive_type_star),
    'Plus': TypingRule(infer=infer_type_repeat_class, derive=derive_type_repeat_class),
    'Startwith': TypingRule(infer=infer_type_concat, derive=derive_type_concat),
    'Endwith': TypingRule(infer=infer_type_concat, derive=derive_type_concat),
    'Contain': TypingRule(infer=infer_type_concat, derive=derive_type_concat),
    'Not': TypingRule(infer=infer_type_not, derive=derive_type_not),
    'Repeat': TypingRule(infer=infer_type_repeat_class, derive=derive_type_repeat_class),
    'RepeatRange': TypingRule(infer=infer_type_repeat_class, derive=derive_type_repeat_class),
    'Concat': TypingRule(infer=infer_type_concat, derive=derive_type_concat),
    'OrR': TypingRule(infer=infer_type_or, derive=derive_type_or),
    'AndR': TypingRule(infer=infer_type_and, derive=derive_type_and),
    'NumMatch': TypingRule(infer=infer_type_numMatch, derive=derive_type_numMatch),
    'inRegion': TypingRule(infer=infer_type_inRegion, derive=derive_type_inRegion),
    'inCountry': TypingRule(infer=infer_type_inCountry, derive=derive_type_inCountry),
    'inState': TypingRule(infer=infer_type_inState, derive=derive_type_inState),
    'isYear': TypingRule(infer=infer_type_isYear, derive=derive_type_isYear),
    'isMonth': TypingRule(infer=infer_type_isMonth, derive=derive_type_isMonth),
    'isDate': TypingRule(infer=infer_type_isDate, derive=derive_type_isDate),
    'btwHour': TypingRule(infer=infer_type_btwHour, derive=derive_type_btwHour),
    'btwMin': TypingRule(infer=infer_type_btwMin, derive=derive_type_btwMin),
    'btwSec': TypingRule(infer=infer_type_btwSec, derive=derive_type_btwSec),
    'isMorning': TypingRule(infer=infer_type_isMorning, derive=derive_type_isMorning),
    'isEvening': TypingRule(infer=infer_type_isEvening, derive=derive_type_isEvening),
    'isAfternoon': TypingRule(infer=infer_type_isAfternoon, derive=derive_type_isAfternoon),
    'AndPred': TypingRule(infer=infer_type_pred, derive=derive_type_pred),
    'OrPred': TypingRule(infer=infer_type_pred, derive=derive_type_pred),
    'NotPred': TypingRule(infer=infer_type_pred, derive=derive_type_pred),
    'BooleanAtom': TypingRule(infer=infer_type_bool, derive=derive_type_bool)
}
