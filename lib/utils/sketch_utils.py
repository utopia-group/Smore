import re

from lib.config import pd_print
from lib.utils.exceptions import SketchParsingException


def format_const_or_cc(buffer: str):
    if '??' in buffer:
        raise ValueError('?? should not be part of either constant or cc')

    if len(buffer) == 1:
        return '[{}]'.format(buffer)
    elif len(buffer) > 1:
        if buffer == 'A-Z':
            return '<CAP>'
        elif buffer == 'a-z':
            return '<LOW>'
        elif buffer == '0-9':
            return '<NUM>'
        elif buffer == 'A-Za-z':
            return '<LET>'
        elif buffer == 'A-Za-z ':
            return '<WORD>'
        elif buffer == 'A-Za-z0-9':
            return '<ALNUM>'
        elif buffer.lower() in ['num', 'let', 'low', 'cap']:
            return '<{}>'.format(buffer)
        else:
            return '<"{}">'.format(buffer)
    else:
        return buffer


def postprocess_gpt3_sketch(sketch: str) -> str:
    """
    The following lists the postprocessing rules (heuristic-based):

    ande returns the corresponding token_mode of case_mode ( not sure if i should do that)

    datetime-> {??: year}-{??: month}-{??: day} {??: hour}:{??: minute}:{??: second}
    road-name-> {??: road}[ ]{??: int}[.]{??: int}[ ]{??: direction}[ ]{??: int}[ ]{??: direction}[ ]{??: road}([,] {??: location})?
    string-> {??: int}[ ]{??: int}[/]{??: int}[ ]x[ ]{??: int}[ ]{??: int}[/]{??: int}[ ]x[ ]{??: int}[ ]{??: int}[/]{??: int}[ ]in[.]([(]{??: int}[ ]x[ ]{??: int}[ ]x[ ]{??: int}[ ]cm[)])?
    string-> {??: int}([ ]{??: int}[/]{??: int})?([ x][ ]{??: int}([ ]{??: int}[/]{??: int})?)?([ x][ ]{??: int}([ ]{??: int}[/]{??: int})?)?([ ]{??: string})
    artist-> ({??: nationality}, {??: year}-present|{??: year}-{??: year})([|]{??: nationality}, {??: year}-{??: year})?
    """
    if sketch.strip() == '':
        raise SketchParsingException()

    if 'Type of' in sketch:
        raise SketchParsingException()

    if '\\b' in sketch:
        raise SketchParsingException()

    if ': optional}' in sketch.lower():
        raise SketchParsingException()

    # for some reason gpt might return sketch with multiple lines, raise exception if that is the case
    sketch_new_line = sketch.split('\n')
    if len(sketch_new_line) > 1 and any([l.strip() != '' for l in sketch_new_line]):
        raise SketchParsingException()

    # print('sketch: {}'.format(sketch))

    sketch_content = sketch.strip()
    # sketch_type = split_sketch[0].strip()
    # sketch_content = split_sketch[1].strip()

    pd_print("sketch_content: {}".format(sketch_content))

    in_hole = False
    in_buffer = False
    buffer = ''
    in_cc = False

    new_sketch_content = ''
    sketch_content = sketch_content.replace('in.', 'in\.')
    sketch_content = sketch_content.replace('({??: Float}|{??: Integer})', '{??: Float}')
    sketch_content = sketch_content.replace('{??: Float}', '{??: float}')
    sketch_content = sketch_content.replace('{??: Date Range}', '{??: DATE}-{??: DATE}')
    sketch_content = sketch_content.replace('trib of', 'trib of|of')
    sketch_content = sketch_content.replace('\s', ' ')

    sketch_content = re.sub(r'{[?][?]: ([\w ]+([|][\w ]+)+)}', lambda m: '(' + m.group(1) + ')', sketch_content)
    sketch_content = re.sub(r'\\([\W_])', r'[\1]', sketch_content)
    sketch_content = re.sub(r'{[?][?]: ([\w ]+)}', lambda m: '{??: ' + m.group(1).lower() + '}', sketch_content)
    # print(sketch_content)
    sketch_content = re.sub(r'[(]?\\d[)|+]?', '{??: integer}', sketch_content)
    sketch_content = re.sub(r'[(]?\\w[)|+]?', '({??: charseq}|{??: integer})', sketch_content)
    # sketch_content = re.sub('\\\\d\+', '{??: integer}', sketch_content)
    # sketch_content = re.sub('\\\\w\+', '({??: charseq}|{??: integer})', sketch_content)
    sketch_content = re.sub('\{0,1\}', '+', sketch_content)

    sketch_content = sketch_content.replace('&|', '[&]|')
    sketch_content = sketch_content.replace('|&', '|[&]')
    sketch_content = sketch_content.replace("'", "[']")
    sketch_content = sketch_content.replace('"', '[\\"]')
    sketch_content = sketch_content.replace('|)', ')?')
    sketch_content = sketch_content.replace('–', '-')

    if '{(' in sketch_content and ')}' in sketch_content:
        sketch_content = sketch_content.replace('{(', '( ')
        sketch_content = sketch_content.replace(')}', ')')

    for char in sketch_content:

        # print("char: {}".format(char))
        # print(in_hole, in_buffer)
        # print("buffer: {}".format(buffer))

        if char == '{':
            if in_buffer:
                new_sketch_content += format_const_or_cc(buffer)
                in_buffer = False
                buffer = ''
            in_hole = True
            new_sketch_content += char
        elif char == '}':
            assert in_hole
            in_hole = False
            # modify the new_sketch_content so that if name is in it, remove it
            # this is a heuristic to handle weird behavior of recent gpt3 models
            if any([s in new_sketch_content for s in ['first name', 'last name', '??: name']]):
                pass
            else:
                new_sketch_content = new_sketch_content.replace(' name', '')
            new_sketch_content += char
        elif char == '[':
            if in_buffer:
                # flush the constant
                new_sketch_content += format_const_or_cc(buffer)
                in_buffer = False
                buffer = ''
            in_buffer = True
            in_cc = True
        elif char == ']':
            new_sketch_content += format_const_or_cc(buffer)
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
                        new_sketch_content += '<ANY>'
                    else:
                        new_sketch_content += format_const_or_cc(buffer)
                    in_buffer = False
                    buffer = ''
                new_sketch_content += char
        else:
            if in_hole:
                new_sketch_content += char
            else:
                in_buffer = True
                buffer += char

    if len(buffer) > 0:
        new_sketch_content += format_const_or_cc(buffer)

    # a few heuristic fixes
    new_sketch_content = new_sketch_content.replace('city name}', 'city}')
    new_sketch_content = new_sketch_content.replace('int}', 'integer}')
    new_sketch_content = new_sketch_content.replace('birthdate}', 'date}')
    new_sketch_content = new_sketch_content.replace('birth year}', 'year}')
    new_sketch_content = new_sketch_content.replace('birthplace}', 'place}')
    new_sketch_content = new_sketch_content.replace('birth place}', 'place}')
    new_sketch_content = new_sketch_content.replace('deathdate}', 'date}')
    new_sketch_content = new_sketch_content.replace('death year}', 'year}')
    new_sketch_content = new_sketch_content.replace('deathplace}', 'place}')
    new_sketch_content = new_sketch_content.replace('death place}', 'place}')
    new_sketch_content = new_sketch_content.replace('digit}', 'integer}')
    new_sketch_content = new_sketch_content.replace('digits}', 'integer}')
    new_sketch_content = new_sketch_content.replace('fraction}', 'float}')
    new_sketch_content = new_sketch_content.replace('char}', 'charseq}')
    new_sketch_content = new_sketch_content.replace('character}', 'charseq}')
    new_sketch_content = new_sketch_content.replace('word}', 'charseq}')
    new_sketch_content = new_sketch_content.replace('{,}', '[,]')
    new_sketch_content = new_sketch_content.replace('Road/Street}', 'road}')

    pd_print("new_sketch_content: {}".format(new_sketch_content))
    # before we do this, we need to make sure that this hole is not already wrapped in a Contain(hole)
    if re.search(r'\(?\(?\<ANY\>\)?\*\)?{[?][?]: [\w ]*(description|Description)[\w ]*}\??[(]+\<ANY\>\)?\*\)?', new_sketch_content) is None:
        new_sketch_content = re.sub(r'{[?][?]: [\w ]*(description|Description)[\w ]*}', r'<ANY>*\g<0><ANY>*', new_sketch_content)
    if re.search(r'\(?\(?\<ANY\>\)?\*\)?{[?][?]: [\w ]*(additional info)[\w ]*}\??[(]+\<ANY\>\)?\*\)?', new_sketch_content) is None:
        new_sketch_content = re.sub(r'{[?][?]: [\w ]*(additional info)[\w ]*}', r'<ANY>*\g<0><ANY>*', new_sketch_content)
    if re.search(r'\(?\(?\<ANY\>\)?\*\)?{[?][?]: [\w ]*(memory)[\w ]*}\??[(]+\<ANY\>\)?\*\)?', new_sketch_content) is None:
        new_sketch_content = re.sub(r'{[?][?]: [\w ]*(memory)[\w ]*}', r'<ANY>*?{??: integer}<ANY>*?', new_sketch_content)
    if re.search(r'\(?\(?\<ANY\>\)?\*\)?{[?][?]: [\w ]*(size)[\w ]*}\??[(]+\<ANY\>\)?\*\)?', new_sketch_content) is None:
        new_sketch_content = re.sub(r'{[?][?]: [\w ]*(size)[\w ]*}', r'<ANY>*?{??: float}<ANY>*?', new_sketch_content)
    if re.search(r'{[?][?]: [\w ]*(unit)[\w ]*}\??[(]+\<ANY\>\)?\*\)?\)?', new_sketch_content) is None:
        new_sketch_content = re.sub(r'{[?][?]: [\w ]*(unit)[\w ]*}', r'{??: unit}<ANY>*?', new_sketch_content)

    # for some reason we need to add a space between '>?'
    new_sketch_content = new_sketch_content.replace('>?', '> ?')
    # print(new_sketch_content)
    return new_sketch_content


def get_semantic_holes(sketch_str: str):
    # print(sketch_str)
    semantic_hole_count = 0
    maybe_in_hole = False
    in_hole = False
    type_buffer = ''
    for char in sketch_str:
        if char == '{':
            maybe_in_hole = True
            continue
        elif char == '}':
            if in_hole:
                # print(type_buffer)
                if type_buffer.strip().lower() not in ['string', 'charseq', 'int', 'integer', 'dates', 'birth', 'death', 'type']:
                    semantic_hole_count += 1
            in_hole = False
            type_buffer = ''
            continue
        elif char == '?':
            if maybe_in_hole:
                in_hole = True
        else:
            if in_hole:
                if char == ':':
                    continue
                else:
                    type_buffer += char

        maybe_in_hole = False

    # print(semantic_hole_count)
    return semantic_hole_count
