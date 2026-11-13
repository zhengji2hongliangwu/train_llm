# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os
import requests
import subprocess
import time
import unittest

from evalscope.backend.opencompass import OpenCompassBackendManager
from evalscope.run import run_task
from evalscope.summarizer import Summarizer
from evalscope.utils.import_utils import is_module_installed
from evalscope.utils.logger import get_logger


logger = get_logger(__name__)

DEFAULT_CHAT_MODEL_URL = 'http://127.0.0.1:8000/v1/chat/completions'
DEFAULT_BASE_MODEL_URL = 'http://127.0.0.1:8001/v1/completions'

model_name = "qwen-plus"

def main():
    # Prepare the config
    task_cfg = dict(
        eval_backend='OpenCompass',
        eval_config={
            'datasets': ['cmb', 'bbh', 'ceval', 'ARC_e', 'gsm8k'],
            'models': [
                {
                    'path': model_name,
                    'openai_api_base': DEFAULT_CHAT_MODEL_URL,
                    'batch_size': 8,
                    'openai_api_key': 'YOUR_API_KEY'
                },
            ],
            'work_dir': 'outputs/evaloutput/llama3_eval_result',
            'reuse': None,  # string, `latest` or timestamp, e.g. `20230516_144254`, default to None
            'limit': 5,  # string or int or float, e.g. `[2:5]`, 5, 5.0, default to None, it means run all examples
        },
    )

    # Check the service status
    data = {'model': self.model_name, 'messages': [{'role': 'user', 'content': 'who are you?'}]}
    assert self.check_service_status(
        DEFAULT_CHAT_MODEL_URL, data=data), f'Failed to check service status: {DEFAULT_CHAT_MODEL_URL}'

    # Submit the task
    logger.info(f'Start to run UT with cfg: {task_cfg}')
    run_task(task_cfg=task_cfg)

    # Get the final report with summarizer
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    logger.info(f'>>The report list:\n{report_list}')

    assert len(report_list) > 0, f'Failed to get report list: {report_list}'


if __name__ == '__main__':
    main()
