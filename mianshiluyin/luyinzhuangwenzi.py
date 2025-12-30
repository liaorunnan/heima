import json
import os
import sys
from http import HTTPStatus

import dashscope
from dashscope.audio.qwen_asr import QwenTranscription
from dashscope.api_entities.dashscope_response import TranscriptionResponse
from conf import settings


# run the transcription script
if __name__ == '__main__':

    dashscope.api_key = settings.qw_api_key


    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
    task_response = QwenTranscription.async_call(
        model='qwen3-asr-flash-filetrans',
        file_url='https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/sensevoice/rich_text_example_1.wav',
        language="zh",
        enable_itn=False
        #corpus= {
        #    "text": ""
        #}
    )
    print(f'task_response: {task_response}')
    print(task_response.output.task_id)
    query_response = QwenTranscription.fetch(task=task_response.output.task_id)
    print(f'query_response: {query_response}')
    task_result = QwenTranscription.wait(task=task_response.output.task_id)
    print(f'task_result: {task_result}')