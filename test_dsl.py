"""
Testing file for the DSL syntax and execution
"""
from typing import List, Dict, Tuple, Pattern

from lib.interpreter.executor import Executor
from lib.interpreter.pattern import SemPattern
from lib.parser.parser import parse_program

executor = Executor()


def testing_func(prog: Pattern, pos_strs: List[str], neg_strs: List[str]):
    for ps in pos_strs:
        print("ps: {}".format(ps))
        assert executor.exec(ps, prog).success
        print()

    for ns in neg_strs:
        print("ns: {}".format(ns))
        assert not executor.exec(ns, prog).success
        print()


"""
test cases format: test_id -> [PhrasePattern Program, positive_examples, negative_examples]
"""

test_cases_regex_dt: Dict[str, Tuple[SemPattern, List[str], List[str]]] = {
    # 'test_1': (parse_program("{<CITYNAME> -> StrMatch(<ANY>*, IGNORE) || StrMatch(genpt(split, substr(0,1) <.>?), IGNORE)}", token_mode=True), ["D.C", "D.C.", "Daly City", "DALY CITY"], ["94115", "Daly", "APT.#2"]),
    # 'test_2': (parse_program('({<ORG> -> True} | Contain({<ANY> -> StrMatch(<"LLC">, NONE)})) {<ANY> -> StrMatch(</>, NONE)}? {<PERSON> -> True}+', token_mode=True), ["Ariel Ventures, LLC / Roseller Tolentino", "O'Neill's Irish Pub SF, LLC / Kevin Kynoch, Doug Sinclair"], ["Cornerstone Academy", "George N Shatara"]),
    # 'test_3': (parse_program('({<ORG> -> True} Contain({<INT> -> True})) | {<ORG> -> StrMatch(<ANY>* <NUM>+, NONE)}', token_mode=True), ["CHEVRON #1512", "AT&T - WILLIE MAYS PLAZA BAR [145087]", "Burger King 9365",  "PUERTO ALEGRE NO. 2"], ["AT&T Park - Food Cart P2", "24 Hour Fitness, Inc.", "1428 Haight"]),
    # 'test_4': (parse_program('Contain({Similar("Desktop Intel Core", 0.7) -> True}) Contain({<INT> -> NumMatch(8, >=)}) {<ANY> -> StrMatch(<"gb memory">, IGNORE)}', token_mode=True), ["268-ENVY Desktop  Intel Core i7  16GB Memory", "286-Desktop  Intel Core i5  8GB Memory"], ["118-iPhone 6s 16GB  Gold (AT&T)", '470-43" Class (42.5" Diag.)  LED  1080p', "278-Inspiron Desktop  Intel Pentium  4GB Memory"]),
    # 'test_7': (parse_program('Contain({Similar("Gift of", 0.35) -> True}){<INT> -> NumMatch(1950, >=, 2000, <=)}', token_mode=True), ["Gift of Marcia and William Goodman, 1981", "The Florence I. Balasny-Barnes Collection, Gift of Florence I. Balasny-Barnes, in memory of her parents, Elizabeth C. and Joseph Balasny, 1991"],["Purchase, Joseph Pulitzer Bequest, 1938", "The Cloisters Collection, 1950", "Rogers Fund, 1947", "Purchase, Mr. and Mrs. Robert G. Goelet Gift, 2012"]),
    # 'test_8': (parse_program('Contain({<NORP> -> PlaceMatch(inRegion("EUROPE"))} | {<GPE> -> PlaceMatch(inRegion("EUROPE"))} )', token_mode=True), ['1853-1905|American (born Austria), Vienna 1867-1915 New York', 'North Netherlandish, active Strasbourg, ca. 1462-died 1473 Vienna', 'British, ca. 1753-1836|1759-present'], ['died 1801', 'American, New York, 1829-35']),
    # 'test_9': (parse_program('Sep({<FLOAT> -> NumMatch(10, >=, 50, <=)},"x",3,3){<ANY> -> StrMatch(<"in">, IGNORE)} Contain({<ANY> -> StrMatch(<"cm">, IGNORE)})', token_mode=True), ["15 1/4 x 20 3/4 x 15 1/4 in. (38.7 x 52.7 x 38.7 cm)", "44 1/4 x 17 1/4 x 10 3/4 in., 66 lb. (112.4 x 43.8 x 27.3 cm, 29.9 kg)"], ["17 15/16 x 29 1/4 x 3/8 in. (45.6 x 74.3 x 1 cm)"]),
    # 'test_10': (parse_program("Contain({<INT> -> NumMatch(1400, >=, 1500, <)} <->)", token_mode=True), ["North Netherlandish, active Strasbourg, ca. 1462-died 1473 Vienna", "German, 1480-1542"], ["British, London 1714-1798 London", "American, New York 1848-1933 New York|1902-32"]),
    # 'test_11': (parse_program('Sep({<NORP> -> True}?{<GPE> -> True}?{<ANY> -> StrMatch(<NUM>{4}, NONE)}{<ANY> -> StrMatch(<->, IGNORE)}({<ANY> -> StrMatch(<NUM>{4}, NONE)} | {<ANY> -> StrMatch(<"present">, IGNORE)}),"|",2,)', token_mode=True), ["American, 1831-present|1864-1926", "1881-1965|1865-1941","1833-1910|American, New York, 1881-1892"], ["British, London 1714-1798 London", "1803-1879|American (born Ireland), 1810-1861 Paris|1796-1854", "American, active 1904-22|American, 1865-1946"]),
    # 'test_12': (parse_program('{Similar("Sir", WIKI) -> True} {<PERSON> -> True}', token_mode=True), ["Mr. Hector", "Ms. Diego", "Sir Leann"], ["Roman Mahoney", "Dianne Todd", "Lora Mullen Jr."]),
    # 'test_15': (parse_program('Contain({<FLOAT> -> True} {Similar("mile", ONELOOK) -> True}) SW({<ANY> -> StrMatch(<"of">, IGNORE)} {Similar("US 101", 0.4) -> True})', token_mode=True), ["Upper Hoh Rd 8.3 kilometer E of US 101", "FS Rd 25 4.6 mi S of FS Rd 300, Gifford Pinchot National Forest", "Slate Creek Rd 10.2 mi E of US 95, Nez Perce National Forest"], ["FL; Bay Co.; outside fence row bordering roadside dep.", "near Lake Lindsey.", "Copalinga Private Reserve-blue trail (1050m-C)"]),
    # 'test_16': (parse_program('Contain({Similar("Photo", 0.4) -> True}) | EW({Similar("JPG",WIKI) -> True})', token_mode=False), ["/images/phocagallery/thumbs/phoca_thumb_l_winterfoto.jpg", "/components/com_simplephotogallery/js/main.js"], ["/apache-log/access_150727.log", "/backup/backup.zip", "/old/wp-admin/"]),
    # 'test_17': (parse_program('{Similar("Director of department", 0.45)-> True} {<ANY> -> StrMatch(<,>, NONE)} {<DEPARTMENT> -> True} {<ANY> -> StrMatch(<,>, NONE)} Sep({<INT> -> True}?,",",6, 6)', token_mode=False), ["Director of DevOps,R&D,54,53,53,16,63,17", "Head of People Ops,Finance & Operations,,10,10,2,4,2"], ["Manager,DevOps,R&D,,6,6,1,9,1", "Demand Generation Manager,Marketing,6,2,2,1,,"]),
}

test_cases_debug = {
    'test': (parse_program('{<Integer>}[-]{"product" -> x}[ ]{"description" -> x}[ ]{<Integer> -> NumMatch(8.0, >=)}<"GB Memory">', token_mode=False), ['274-IdeaCentre 300s Desktop Intel Core i5 8GB Memory', '281-Pavilion Desktop Intel Core i3 8GB Memory', '286-Desktop Intel Core i5 8GB Memory'], [])
}


if __name__ == '__main__':
    # test_case_selected = test_cases_comb
    # test_case_selected = test_cases_phrase
    # test_case_selected = test_cases_template
    # test_case_selected = test_cases_wv
    # test_case_selected = test_cases_top_regex
    # test_case_selected = test_cases_regex_pred
    # test_case_selected = test_cases_regex_dt
    # test_case_selected = test_cases_phrase | test_cases_comb
    test_case_selected = test_cases_debug

    for test_id, test_case in test_case_selected.items():
        print("================{}===============".format("Running test case {}".format(test_id)))
        testing_func(test_case[0], test_case[1], test_case[2])
        print()
