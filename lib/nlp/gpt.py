from time import sleep
from typing import Dict, List

import openai


class GPT:
    def __init__(self, key):
        self.gpt3_default_args = {'top_p': 1, 'n': 1}
        if key is not None and not key == '':
            self.exist_key = True
            openai.api_key = key
        else:
            self.exist_key = False
        openai.api_requestor.TIMEOUT_SECS = 5
        self.max_attempt = 60

    def call(self, prompt: str, max_token: int, temperature: float, stop, model: str) -> Dict:

        if not self.exist_key:
            raise Exception('OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.')

        for i in range(self.max_attempt):
            try:
                if model in ['gpt-3.5-turbo', 'gpt-4']:
                    # call run_chat, terminate if not finished in 5 seconds and if it times out, try again
                    return self.run_chat(prompt, max_token, temperature, stop, model)

                else:
                    return self.run_base(prompt, max_token, temperature, stop, model)
            except openai.error.RateLimitError as e:
                print('RateLimitError: ', e)
                print('Sleep for 1 sec and then try again')
                sleep(1)
            except openai.error.APIError as e:
                print('APIError: ', e)
            except openai.error.Timeout as e:
                print('Timeout: ', e)
            except openai.error.APIConnectionError as e:
                print('APIConnectionError: ', e)
            except openai.error.ServiceUnavailableError as e:
                print('ServiceUnavailableError: ', e)

    def run_chat(self, prompt: str, max_token: int, temperature: float, stop, model: str) -> Dict:

        message = format_chat_input(prompt)
        if stop is not None:
            response = openai.ChatCompletion.create(model=model, messages=message, temperature=temperature, max_tokens=max_token, stop=stop, **self.gpt3_default_args)
        else:
            response = openai.ChatCompletion.create(model=model, messages=message, temperature=temperature, max_tokens=max_token, **self.gpt3_default_args)

        return response

    def run_base(self, prompt: str, max_token: int, temperature: float, stop, model: str) -> Dict:
        # print('base arguments:', prompt, max_token, temperature, stop, model)
        if stop is not None:
            response = openai.Completion.create(model=model, prompt=prompt, temperature=temperature, max_tokens=max_token, stop=stop, **self.gpt3_default_args)
        else:
            response = openai.Completion.create(model=model, prompt=prompt, temperature=temperature, max_tokens=max_token, **self.gpt3_default_args)

        return response


def format_chat_input(prompt: str) -> List[Dict]:
    return [
        {'role': 'user', 'content': prompt}
    ]


def get_response(response: Dict, model) -> str:
    # print('response:', response)
    if model in ['gpt-3.5-turbo', 'gpt-4']:
        return get_chat_response(response)
    else:
        return get_base_response(response)


def get_chat_response(response: Dict) -> str:
    return response['choices'][0]['message']['content']


def get_base_response(response: Dict) -> str:
    return response['choices'][0]['text']