import re
from typing import Any

from ..large_response_utils import generate

from .data_structures import LongResponseQASample


def accuracy_string(task: LongResponseQASample, normalize: bool = True) -> bool:

    if not isinstance(task.gold_answer, str) or not isinstance(task.pred_answer, str):
        return False

    if normalize:
        gold_ans = task.gold_answer.strip().lower()
        pred_ans = task.pred_answer.strip().lower()
    else:
        gold_ans = task.gold_answer
        pred_ans = task.pred_answer

    if pred_ans.endswith("."):
        pred_ans = pred_ans[:-1]

    return gold_ans == pred_ans


def approx_number_match(task: LongResponseQASample, rounding: bool = True) -> bool:
    """We expect the gold answer to be a single number, but the model output can have multiple numbers.
    The match will return True if any of the numbers from the predicted answer matches the gold answer, with or without rounding.
    Likely that the model output will not match the exact number, can be rounded, with a $ sign etc.

    Parameters
    ----------
    task : LongResponseQASample
    rounding : bool, optional
        whether to account for rounding for the match, by default True

    Returns
    -------
    bool
        whether any of the numbers in the predicted answer matches the gold answer
    """
    if not isinstance(task.gold_answer, str) or not isinstance(task.pred_answer, str):
        return False

    pred_answers = re.findall(
        r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", task.pred_answer.strip()
    )
    # if the task.pred_answer is "USD 1.8 is greater than 1.65", the pred_answers is [('1.8', '.8', ''), ('1.65', '.65', '')]
    for pred_ans_matches in pred_answers:
        pred_ans: str = pred_ans_matches[0]
        gold_ans = task.gold_answer.strip()
        if rounding:
            if abs(float(pred_ans) - (float(gold_ans))) <= 1:
                return True
        else:
            return pred_ans == gold_ans

    return False  # no number was found in the predicted answer


def unordered_list_str_match(
    task: LongResponseQASample, normalize: bool = True, deduplicate: bool = True
) -> bool:
    """The gold answer and predicted answer are assumed to be comma separated lists of strings.
    This metric checks if the unordered list matches.

    Parameters
    ----------
    task : LongResponseQASample
    normalize : bool, optional
        strip and ignore-case, by default True

    Returns
    -------
    bool
        match or not
    """
    if not isinstance(task.gold_answer, str) or not isinstance(task.pred_answer, str):
        return False

    def normalize_list_elements(elements: list[str]) -> list[str]:
        normalized_elements = []
        for element in elements:
            element = element.strip().lower()
            if element.endswith("."):
                element = element[:-1]
            normalized_elements.append(element)
        return normalized_elements

    if task.pred_answer.endswith("."):
        task.pred_answer = task.pred_answer[:-1]

    if normalize:
        gold_ans = task.gold_answer.strip().lower()
        pred_ans = task.pred_answer.strip().lower()
    else:
        gold_ans = task.gold_answer
        pred_ans = task.pred_answer

    gold_ans_elements = gold_ans.split(",")
    pred_ans_elements = pred_ans.split(",")

    if normalize:
        gold_ans_elements = normalize_list_elements(gold_ans_elements)
        pred_ans_elements = normalize_list_elements(pred_ans_elements)

    if deduplicate:
        return set(sorted(gold_ans_elements)) == set(sorted(pred_ans_elements))
    else:
        return sorted(gold_ans_elements) == sorted(pred_ans_elements)


def llm_as_a_judge(
    task: LongResponseQASample, eval_llm: Any, eval_model_name: str
) -> bool:

    EVAL_PROMPT_TEMPLATE = """You are given two inputs, one is the ground truth and one is predicted answer.
    These are two strings. Your task is to output either "True" or "False" indicating whether the
    predicted answer matches the ground truth.
    Please follow the instructions listed below:
    1. As these are free text outputs, the match does not need to be exact.
    2. Please allow for paraphrases or verbalizations.
    3. Please output only "True" or "False".

    ground_truth:{gold_answer}
    predicted_answer:{predicted_answer}

    Does the predicted_answer match the ground_truth?: """

    eval_prompt = EVAL_PROMPT_TEMPLATE.format(
        gold_answer=task.gold_answer, predicted_answer=task.pred_answer
    )

    llm_as_a_judge_outputs = generate(
        llm=eval_llm, model_name=eval_model_name, prompts=eval_prompt
    )
    if llm_as_a_judge_outputs[0].strip().lower().startswith("true"):
        llm_as_a_judge_output = True
    else:
        llm_as_a_judge_output = False
    return llm_as_a_judge_output

def contains(task: LongResponseQASample, normalize: bool = True) -> bool:

    if not isinstance(task.gold_answer, str) or not isinstance(task.pred_answer, str):
        return False

    if normalize:
        gold_ans = task.gold_answer.strip().lower()
        pred_ans = task.pred_answer.strip().lower()
    else:
        gold_ans = task.gold_answer
        pred_ans = task.pred_answer

    if pred_ans.endswith("."):
        pred_ans = pred_ans[:-1]

    return (gold_ans in pred_ans)

if __name__ == "__main__":
    task = LongResponseQASample(
        api_response={},
        question="",
        gold_answer="40.0",
        pred_answer="10.0  ",
    )
    print(approx_number_match(task))
