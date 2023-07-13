import pickle
from datetime import datetime
from typing import Dict


def get_data_csv(dataset_name, column) -> str:
    return 'benchmarks/data/{}_{}_sampled.csv'.format(dataset_name, column.strip().lower().replace(' ', '_'))


def curr_time_to_formatted_timestamp() -> str:
    current_time = datetime.now()
    str_date_time = current_time.strftime("%Y%m%d%H%M%S")
    return str_date_time


def save_response(response: str) -> str:
    # folder = 'prompting_examples_{}'.format(mode)
    # folder_text = 'text'
    # folder_pkl = 'pkl'
    # tid = curr_time_to_formatted_timestamp()
    # uid = '{}-{}-{}'.format(openai_mode, tid, response['id'])
    # # top_answer = response['choices'][0]['text']
    #
    # with open('{}/{}/{}'.format(folder, folder_text, uid), 'w+') as text_f:
    #     format_str = '''{}\n------ THE FOLLOWING ARE RETURNED BY GPT3 ------\n{} '''.format(prompt, top_answer)
    #     text_f.write(format_str)
    #
    # # with open('{}/{}/{}'.format(folder, folder_pkl, uid), 'wb+') as pkl_f:
    # #     response['prompt'] = prompt
    # #     pickle.dump(response, pkl_f)

    return response
