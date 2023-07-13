"""
enumerate all the types
separate file so that it is sort of independent of the type system
"""
from typing import List, Tuple, Dict

# a list of name, and a tuple of superclass name pairs (in order from the most general to the least general)
# the list also ranks from the most general to the least general
base_types: List[Tuple[str, Tuple[str]]] = [
    ('any', ('any',)),
    ('string', ('basetype',)),
    ('charseq', ('string',)),
    ('number', ('string',)),
    ('integer', ('number',)),
    ('float', ('number',)),
    ('person', ('string',)),
    ('organization', ('string',)),
    ('company', ('organization',)),
    ('restaurant', ('organization',)),
    ('institution', ('organization',)),
    ('datetime', ('string',)),
    ('date', ('datetime',)),
    ('time', ('datetime',)),
    ('year', ('date',)),
    ('month', ('date',)),
    ('day', ('date',)),
    ('hour', ('time',)),
    ('minute', ('time',)),
    ('second', ('time',)),
    ('place', ('string',)),
    ('nationality', ('place',)),
    ('city', ('nationality',)),
    ('state', ('nationality',)),
    ('country', ('nationality',)),
    ('region', ('nationality',)),
    ('name', ('string',)),
    ('last name', ('name', )),
    ('first name', ('name', ))
]

"""
parsable types and the type tag they map to in the language
"""
parsable_types: Dict[str, str] = {
    'integer': 'INT',
    'float': 'FLOAT',
    'date': 'DATE',
    'time': 'TIME',
    'year': 'DATE',
    'month': 'DATE',
    'day': 'DATE',
    'hour': 'TIME',
    'minute': 'TIME',
    'second': 'TIME',
    'place': 'PLACE',
    'city': 'CITY',
    'state': 'STATE',
    'country': 'COUNTRY',
    'region': 'REGION',
    'nationality': 'NATIONALITY',
}

entity_types: Dict[str, str] = {
    'organization': 'ORG',
    'company': 'ORG',
    'restaurant': 'ORG',
    'institution': 'ORG',
    'person': 'PERSON',
}

extensible_types: Dict[str, str] = {
    'name': 'PERSON',
    'first name': 'PERSON',
    'last name': 'PERSON',
}