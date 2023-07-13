"""
This is a convenient script for me running some gpt stuff as an executor
"""
from typing import Optional, Tuple

from lib.nlp.nlp import NLPFunc

nlp = NLPFunc()


def run_single_query(positive_training_str: str, negative_training_str: str, query_str: str, model: str):
    positive_training = positive_training_str.split('\n')
    negative_training = negative_training_str.split('\n')

    res = nlp.query_gpt_exec(positive_training, negative_training, query_str, model)
    return res


def run(positive_training_str: str, negative_training_str: str, positive_testing_str: str, negative_testing_str: str, model: str, executor: Optional = None) -> Tuple[float, float, float]:

    if executor is not None:
        nlp2 = executor.nlp_engine
    else:
        nlp2 = nlp

    positive_training = positive_training_str.split('\n')
    negative_training = negative_training_str.split('\n')

    positive_testing = positive_testing_str.split('\n')
    negative_testing = negative_testing_str.split('\n')

    pos_correct = 0
    pos_incorrect_str = []
    neg_correct = 0
    neg_incorrect_str = []

    for pos in positive_testing:
        print('pos: ', pos)
        res = nlp2.query_gpt_exec(positive_training, negative_training, pos, model)
        print("res: ", res)
        if res:
            pos_correct += 1
        else:
            pos_incorrect_str.append(pos)

    for neg in negative_testing:
        print('neg: ', neg)
        res = nlp2.query_gpt_exec(positive_training, negative_training, neg, model)
        print('res: ', res)
        if not res:
            neg_correct += 1
        else:
            neg_incorrect_str.append(neg)

    recall = pos_correct/len(positive_testing)
    if pos_correct + (len(negative_testing) - neg_correct) == 0:
        precision = 0
    else:
        precision = pos_correct/(pos_correct + (len(negative_testing) - neg_correct))

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    print('precision: {}, recall: {}, f1: {}'.format(precision, recall, f1))
    print('pos_incorrect_str:', pos_incorrect_str)
    print('neg_incorrect_str:', neg_incorrect_str)

    return precision, recall, f1


if __name__ == '__main__':
    # model = 'gpt-3.5-turbo'
    model = 'text-davinci-003'

    # positive_training_str = 'Daly City\nSan Francisco\nS.F.'
    # negative_training_str = 'DEERFIELD\nSeattle\nDay City'
    # positive_testing_str = 'Yountville\nSouth Pasadena\nSAN CARLOS\nVacaville\nMill Valley\nDublin\nBurlingame\nHercules\nSf.\nRedlands'
    # negative_testing_str = 'Ponte Vedra Beach\n        WINTER PARK\nHonolulu\nEl Sobrante\n94117\nTiburon\nSan Frfancisco\nALPHARETTA\n        DANVILLE\nSANF FRANCISCO'

    # positive_training_str = '12 Tribes Kosher Foods, Inc. / Rebecca Joseph\n1234 Polk St., Inc. / Ibrahim M. Al Haj\nAndersen America, Inc. / Nobumitsu Kobayashi and Tad Saito'
    # negative_training_str = '100 Brannan LLC\nAAA Vegi, Inc\nALEX MIRETSKY'
    # positive_testing_str = '4505 Grove & Divisadero LLC / Ryan Farr\nBattambang Market, Inc. / Michael Tarbox, Kim Tarbox, Kem Chea\nBasco LLC / Danel de Betelu\nBlue Polk, LLC / Nathan D. Valentine\nAriel Ventures, LLC / Roseller Tolentino\nAlcyone LLC / Dennis Leary, Eric Passetti\n3D Hospitality Group, LLC / Daniel R. McKinney\nBently Reserve, LP / Christopher Bently and Mark T. Mayfield\nAshbury Taverns, LLC / John Lillis, Jennifer Lillis\nBacano Life, Inc. / Laverne Matias'
    # negative_testing_str = "Anthony's Cookies, LLC\nAsadullah Amini\n550 Market Inc   Arsalan Najmabadi\nAlatorre Bruce Enterprises\nCINEMASF, LLC\nAnand Mishigdorj\nCLEMENT SEAFOOD CENTER, INC.\nBEVERAGES & MORE!\nBSISSO, KETAM\nAbuelrous Hany M"

    # positive_training_str = '24 HOUR FITNESS, INC. #547\n7 Eleven #2366-24139C\nBURGER KING 4525'
    # negative_training_str = '1601 Bar & Kitchen\n24 Hour Fitness, Inc\nBar 587'
    # positive_testing_str = "24 Hour Fitness, #273\n7-Eleven, Store 2366-21389F\nBNC #49\nAFC SUSHI at Mollie Stone's #10\n7-Eleven, Store 2231-33006B\nBaysubway 02\nBacon Bacon #2\nAlbertsons #7122\nBig Joe's Broiller #2\n7-ELEVEN #20473"
    # negative_testing_str = "17th & Noe Market\nBeachside Coffee Bar & Kitchen\nBAYSIDE MARKET\nBenjarong Thai Cuisine\nBuddha Lounge\n19th Ave Shell\nBoulette's Larder\n3rd Cousin\nBoudin Petit Cafe\nAT&T - (J-2A) BEN & JERRY'S"

    positive_training_str = '17TH & COLE MARKET\n2227 Irving Seafood Market Inc\nClement Street Farmers Market'
    negative_training_str = '21ST AMENDMENT BREWERY CAFE\n19th Ave Shell\nA B C MARKET'
    positive_testing_str = '17th & Noe Market\n17th & Balboa Market\n23rd & Guerrero Market & Deli\n50 Fremont Farmers Market\n22ND & IRVING MARKET\nBernal Heights Market\nClement Mini Market\nCalifornia & Lyon Market\nFort Mason Market and Deli\nGreen & Polk Produce'
    negative_testing_str = "BAYSIDE MARKET\nBeachside Coffee Bar & Kitchen\nBenjarong Thai Cuisine\n3rd Cousin\nBoudin Petit Cafe\nAT&T - (J-2A) BEN & JERRY'S\nA. TARANTINO & SONS INC.\nAT&T PARK - 134 HOT CORNER BAR\nAT&T PARK - CLUB LEMONADE\nBOOGALOOS"

    run(positive_training_str, negative_training_str, positive_testing_str, negative_testing_str, model)