import string
from enum import Enum

ENT_TAGS = ['ORG', 'PERSON', 'DATE', 'TIME']
ADDITIONAL_TAGS = ['CITY', 'STATE', 'COUNTRY', 'REGION', 'NATIONALITY', 'PLACE']
# PARSABLE_TAGS = ['GPE', 'LOC', 'ORG', 'PERSON', 'DATE', 'TIME', 'EVENT', 'PRODUCT', 'CARDINAL', "NORP"]
CUSTOMIZE_TAGS = {'CITYNAME': ['San Francisco', 'Daly City', 'South San Francisco', 'Austin'],
                  'CITY1': ['San Francisco', 'Daly City', 'South San Francisco', 'Austin'],
                  'DEPARTMENT': ['R&D', 'Finance & Operations']}

CC_REGEX_MAP = {'any': '.',
                'cap': '[A-Z]',
                'low': '[a-z]',
                'let': '[A-Za-z]',
                'word': '[A-Za-z_]',
                'words': '[A-Za-z _-]',
                'alnum': '[A-Za-z0-9]',
                'num': '[0-9]',
                'quote': '[\\"]',
                }

CONST_REGEX_MAP = {'/': '[\/]',
                   '.': '[.]',
                   ',': '[,]',
                   '+': '[+]',
                   '*': '[*]',
                   '%': '[%]',
                   '-': '[-]',
                   '?': '[?]',
                   '_': '[_]',
                   '|': '[|]',
                   ' ': '[ ]',
                   "'": "[']",
                   'x': '[x]',
                   ':': '[:]',
                   ';': '[;]',
                   '(': '[(]',
                   ')': '[)]',
                   '[': '[[]',
                   ']': '[]]',
                   '@': '[@]',
                   '#': '[#]',
                   '&': '[&]',
                   '$': '[$]',
                   '+-': '[+-]',
                   'V|L': '[V|L]',
                   '0-9': '[0-9]'
                   }

for i in range(0, 10):
    CONST_REGEX_MAP[str(i)] = '[{}]'.format(str(i))

for c in string.ascii_letters:
    CONST_REGEX_MAP[c] = '[{}|{}]'.format(c.lower(), c.upper())

DECOMPOSE_SPECIAL_TAGS = ['year', 'month', 'day', 'decimal', 'int', 'integer', 'float', 'hour', 'minute', 'second', 'id', 'charseq', 'string']

SPECIAL_CHARS = [' ', '.', ',', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '?', '+', '-', '=', '_', '[', ']', '\\', '|', ':', ';', '<', '>', '/', '\n']


class CaseTag(Enum):
    NONE = 'NONE'
    IGNORE = 'IGNORE'
    LOW = 'LOW'
    CAP = 'CAP'
