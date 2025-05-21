import json
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .data_structures import LongResponseQASample, TaskAttributes


class Task(ABC):

    EVALUATION_CRITERIAS: list[Any] = []
    TASK_ATTRIBUTES: list[TaskAttributes] = []

    @abstractmethod
    def get_qa_samples(
        self, api_response: dict[Any, Any]
    ) -> list[LongResponseQASample]:
        """
        Get a list of Long Response QA samples given an API response
        """
        raise NotImplementedError

    def evaluate_task(self, qa_task: LongResponseQASample) -> list[bool]:
        """
        Evaluate the task results using the gold and predicted answers.
        The EVALUATION_CRITERIAS variable defines the list of evaluation metrics
        """

        assert (
            len(self.EVALUATION_CRITERIAS) > 0
        ), "Evaluation criterias not set for task"

        eval_metrics = [
            eval_criteria(qa_task) for eval_criteria in self.EVALUATION_CRITERIAS
        ]

        return eval_metrics

    def evaluate(self, qa_tasklist: list[LongResponseQASample]) -> np.ndarray:
        result = [self.evaluate_task(qa_task=qa_task) for qa_task in qa_tasklist]
        result_avg = np.average(result, axis=0)
        return result_avg

    def get_prompt(self, qa_sample: LongResponseQASample) -> str:

        prompt_template = (
            "You are given a response from an API call (in JSON format). "
            "Answer the question based on the information provided in the API response.\n\n"
            "```json\n{api_response}\n```\n\n"
            "Question: {question}\n\n"
            "Only respond with the answer. Do not include any other text or json in the response."
            "Do not rephrase the answer or write it in complete sentence, return exactly as is from the JSON.\n\n"
            "Answer:"
        )

        prompt = prompt_template.format(
            api_response=json.dumps(qa_sample.api_response, indent=4),
            question=qa_sample.question,
        )

        return prompt
