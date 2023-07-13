from typing import List, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup as bs

"""
These are the utility functions for parsing wikipedia template
"""


def get_wiki_template_page(template_name: str):
    """
    example
    https://en.wikipedia.org/w/api.php?action=expandtemplates&text={{Template:Graphics_file_formats}}&prop=wikitext&title=Page%20Title
    """

    s = requests.Session()

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "expandtemplates",
        "text": '{{' + template_name + '}}',
        "prop": "wikitext",
        "format": "json"
    }

    r = s.get(url=url, params=params)
    data = r.json()

    return data


def get_wiki_template_table(template_name: str):
    html_text = get_wiki_template_page(template_name)['expandtemplates']['wikitext']
    tables = bs(html_text, 'lxml').find_all('table')

    if len(tables) == 0:
        return None
    else:
        return tables[0]


def get_wiki_template_dict(template_name: str, keep_all_texts: bool = False, has_header: bool = True) -> List[str] | List[Tuple[str, str]]:
    # print("get_wiki_template_dict with template name: {}".format(template_name))
    table_html = get_wiki_template_table(template_name)
    if table_html is not None:
        return parse_wiki_table(str(table_html))
    else:
        return []


def parse_wiki_table(url, keep_full_text: bool = False) -> List[str] | List[Tuple[str, str]]:

    def parse_cell(s: str):
        s = s.strip()

        # first split sub-item
        if '%' in s:
            s_split = s.split('%')
            if len(s_split) >= 1:
                return parse_cell(s_split[0])
        else:

            if s.startswith('[['):
                s1 = s[2:-2]

                if '|' in s1:
                    s11 = s1.split('|')
                    if keep_full_text:
                        if len(s11) == 2:
                            return s11[0].strip(), s11[1].strip()
                        elif len(s11) == 1:
                            return s11[0].strip(), s11[0].strip()
                    else:
                        if len(s11) == 2:
                            return s11[1].strip()
                        elif len(s11) == 1:
                            return s11[0].strip()
                else:
                    if keep_full_text:
                        return s1.strip(), s1.strip()
                    else:
                        return s1.strip()
            else:
                if keep_full_text:
                    return s, s
                else:
                    return s

    res_list = []

    tables = pd.read_html(url)

    # logic is here:
    # to keep things simple
    # just to get all the text starts with *
    for table in tables:
        # print("table:", table)
        if table.shape[1] < 2:
            continue
        # let's only focus on column 2
        column2 = table.iloc[:, 1]
        # print("column2:", column2)
        for row in column2:
            # print("row:", row)
            # first fix ** -> % (otherwise there is some splitting issue)
            row1 = row.replace('**', '%')
            row1_split = row1.split('*')
            for cell in row1_split:
                # don't do anything if the string is a comma-separated string, too hard
                # print("cell:", cell)
                if ',' in cell:
                    continue
                else:
                    parse_res = parse_cell(cell)
                    if isinstance(parse_res, Tuple):
                        if parse_res[0] == '' or '[[' in parse_res[0] or '§' in parse_res[0]:
                            continue
                    elif isinstance(parse_res, str):
                        if parse_res == '' or '[[' in parse_res or '§' in parse_res:
                            continue
                    res_list.append(parse_cell(cell))

    return res_list
