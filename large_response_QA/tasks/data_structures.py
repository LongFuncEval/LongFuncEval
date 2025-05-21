from dataclasses import dataclass
from enum import Enum
from typing import Any, Union


class TaskAttributes(Enum):
    EXTRACTIVE = "EXTRACTIVE"  # Simple extraction of answers from the API response
    FILTERING = "FILTERING"  # Filtering of the API response based on some conditions, similar to a select operator in relational algebra
    AGGREGRATION = "AGGREGRATION"  # Aggregation over the API response elements, such as lowest price, latest album


@dataclass
class LongResponseQASample:
    api_response: dict[Any, Any]
    question: str
    gold_answer: Any
    pred_answer: Any = None
    metrics: Any = None
    task_type: Union[list[TaskAttributes], None] = None
