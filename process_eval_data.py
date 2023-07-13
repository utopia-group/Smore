"""
Script for post-processing the evaluation data.
"""
import argparse
from collections import defaultdict
from typing import List, Dict

from lib.utils.csv_utils import read_csv_to_dict
from lib.utils.google_sheet import GoogleSheet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ae', action='store_false', help='disable ae mode')
    parser.add_argument('--mode', type=int, help='generate results for different sections of the evaluation')
    parser.add_argument('--sheet_name', type=str, default='ae_results', help='name of the evaluation sheet')
    parser.add_argument('--google_sheet', action='store_true', help='whether to output the results to a google sheet')

    return parser.parse_args()


def output_ablation_figure_tikz(args, worksheet_name='Ablation Time Calculation'):
    """
    This is for generating the tikz code for the ablation figure in the paper.
    This code assume we have already run the evaluation and have the processed results in a Google sheet.
    """

    assert args.google_sheet

    google_sheet = GoogleSheet(args.sheet_name, worksheet_name)

    header_of_interest = [h for h in google_sheet.headers if 'cumulative' in h]
    column_to_data = {}
    legend_options = [{'mark': 'square', 'color': 'yellow'},
                      {'mark': 'triangle', 'color': 'darkblue'},
                      {'mark': 'x', 'color': 'darkpink'},
                      {'mark': 'diamond', 'color': 'organgered'},
                      {'mark': 'pentagon', 'color': 'darkpurple'},]
    column_to_legend = defaultdict(dict)
    for col_header in header_of_interest:
        print(col_header)
        col_data = google_sheet.get_col_by_header(col_header)
        print(col_data)
        column_to_data[col_header] = list(enumerate(col_data[1:], 1))
        column_to_legend[col_header] = legend_options.pop(0)

    # output tikz code
    code = r'''
\begin{tikzpicture}[scale=1.0]
\begin{axis}[
    ymax=500,
    y=0.01cm,
    x=0.18cm,
    legend cell align = left,
    legend pos = outer north east,
    legend style = {
        nodes={scale=0.8, transform shape},
        at={(0.22,0.98)},
        legend columns=1,
        anchor=north,
    },
    xlabel style={yshift=1mm},
    ylabel = Time(s),
    xlabel = \# Completed Benchmarks,
    xmax = 55,
    xmin = -5
]
    '''
    code += r'\legend{' + ','.join([r'{\sc ' + h + r'}' for h in column_to_data.keys()]) + '}\n'
    for col_header, col_data in column_to_data.items():
        code += r'\addplot[smooth, line width=0.4mm, mark=' + column_to_legend[col_header]['mark'] + r', mark options={fill=' + column_to_legend[col_header]['color'] + \
                r'}, mark size=0.8pt, ' + column_to_legend[col_header]['color'] + '] coordinates {\n'
        for row_idx, row_data in col_data:
            code += f'({row_idx}, {row_data}) \n'
        code += '};\n'
    code += r'\end{axis}\end{tikzpicture}'

    print(code)


def data_helper(d, toolname, header, ignore_error=False):
    num_finished = 0
    data_all = defaultdict(list)
    for h in header:
        if h == 'time' and 'exec' in toolname:
            continue
        if h == 'f1':
            continue
        for i in d:
            if toolname + '-time' in i and i[toolname + '-' + 'time'] == '60':
                continue
            if toolname + '-program' in i and i[toolname + '-program'] == 'TIMEOUT':
                continue
            if toolname + '-program' in i and i[toolname + '-program'] == 'ERROR' and not ignore_error:
                continue
            if toolname + '-program' in i and i[toolname + '-program'] == '':
                continue
            if toolname == 'FlashGPT' and i['FlashGPT-#ex'] != '7':
                continue
            data_all[h].append(float(i[toolname + '-' + h]))
            if h == 'time':
                num_finished += 1

    # compute f1
    data_all['f1'] = [2 * p * r / (p + r) if p + r > 0 else 0 for p, r in zip(data_all['precision'], data_all['recall'])]
    # compute average preision, recall, f1 and time
    print_data = {}
    for h in header:
        print_data[h] = sum(data_all[h]) / len(data_all[h]) if len(data_all[h]) > 0 else 0
    print_data['finished'] = num_finished

    return data_all, print_data


def generate_table_7_1(d: List[Dict]):
    """
    The table looks like this:
    toolname #finished p r f1 synth_time
    with the following tool:
    ChatGPT-Regex-Synth
    ChatGPT-Exec
    FlashGPT
    Smore
    """

    toolname = ['GPT-3.5-regex', 'GPT-3.5-exec', 'FlashGPT', 'Smore']
    header = ['precision', 'recall', 'time', 'f1']

    # pretty print the result as a table
    print("========================================")
    print("Table 7.1")
    print("========================================")
    print("Toolname & #Finished & Precision & Recall & F1 & Time \\\\")
    for t in toolname:
        _, print_data = data_helper(d, t, header, ignore_error=True)
        t = t.replace('GPT-3.5', 'ChatGPT')
        if 'exec' in t:
            print("{} & - & {:.2f} & {:.2f} & {:.2f} & - \\\\".format(t, print_data['precision'], print_data['recall'], print_data['f1']))
        elif 'regex' in t:
            print("{} & {} & {:.2f} & {:.2f} & {:.2f} & - \\\\".format(t, print_data['finished'], print_data['precision'], print_data['recall'], print_data['f1']))
        else:
            print("{} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(t, print_data['finished'], print_data['precision'], print_data['recall'], print_data['f1'], print_data['time']))
    print("========================================")


def generate_table_7_2(d: List[Dict]):
    """
    The table looks like this:
    toolname #finished p r f1 synth_time
    with the following tool:
    GPT-3.5-synth
    Smore-no-sketch
    Smore
    """
    toolname = ['GPT-3.5-synth', 'Smore-no-sketch', 'Smore']
    header = ['precision', 'recall', 'time', 'f1']

    # pretty print the result as a table
    print("========================================")
    print("Table 7.2")
    print("========================================")
    print("Toolname & #Finished & Precision & Recall & F1 & Time \\\\")
    for t in toolname:
        _, print_data = data_helper(d, t, header)
        t = t.replace('GPT-3.5', 'ChatGPT')
        if 'synth' in t:
            print("{} & {} & {:.2f} & {:.2f} & {:.2f} & - \\\\".format(t, print_data['finished'], print_data['precision'], print_data['recall'], print_data['f1']))
        else:
            print("{} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(t, print_data['finished'], print_data['precision'], print_data['recall'], print_data['f1'], print_data['time']))
    print("========================================")


def generate_figure_7_3(d: List[Dict]):
    toolname = ['Smore', 'Smore-no-type', 'Smore-no-type-system', 'Smore-no-decomp', 'Smore-no-repair']
    header = ['time']
    tool_to_cumulative_time = defaultdict(list)
    for t in toolname:
        data_all, _ = data_helper(d, t, header)
        time_data = data_all['time']
        # sort the list from low to high, then compute the cumulative sum for every value so far
        time_data.sort()
        cumulative_time = 0
        for td in time_data:
            cumulative_time += td
            tool_to_cumulative_time[t].append(cumulative_time)

    # plot the figure as a line chart, x-axis is the cumulative number of solved instances and y-axis is the cumulative time
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 3))
    plt.xlabel('Cumulative number of solved instances')
    plt.ylabel('Cumulative time (s)')
    plt.plot(tool_to_cumulative_time['Smore'], label='Smore')
    plt.plot(tool_to_cumulative_time['Smore-no-type'], label='Smore-no-type')
    plt.plot(tool_to_cumulative_time['Smore-no-type-system'], label='Smore-no-type-system')
    plt.plot(tool_to_cumulative_time['Smore-no-decomp'], label='Smore-no-decomp')
    plt.plot(tool_to_cumulative_time['Smore-no-repair'], label='Smore-no-repair')
    plt.legend()
    plt.tight_layout()
    plt.show()






if __name__ == '__main__':
    args = parse_args()

    if args.ae:

        data = read_csv_to_dict('eval_res/{}.csv'.format(args.sheet_name))

        if args.mode == 1:
            generate_table_7_1(data)
        elif args.mode == 2:
            generate_table_7_2(data)
        elif args.mode == 3:
            generate_figure_7_3(data)
        else:
            raise NotImplementedError('mode not supported')

    # output_ablation_figure_tikz()
