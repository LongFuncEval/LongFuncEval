"""\
This script contains functions to create data subsets for different JSON QA tasks
Each individual function (for example, create_booking_rooms_with_availability_across_hotels)

Usage: 
"""

import json
import os
from typing import Any, Dict, List, Tuple

from large_response_QA.tasks.task_list import (
    BookingGetAvailabilityTaskList,
    BookingGetRoomListWithAvailabilityTaskList,
    BookingSearchCarRentalsTaskList,
    BookingSearchFlightsMultiStopsTaskList,
    BookingGetSeatMapTaskList
)


def filter_dataset(
    dataset: Dict[int, Any], num_data_samples_to_select: int
) -> List[Tuple[int, int]]:
    random_seeds_num_entities_list = []
    random_seeds_num_entities_dict = {}
    for key, value in dataset.items():
        num_entities = value["num_entities"]
        random_seeds_num_entities_dict[key] = num_entities
    sorted_random_seeds_num_entities_dict = dict(
        sorted(
            random_seeds_num_entities_dict.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    random_seeds_num_entities_list = list(
        sorted_random_seeds_num_entities_dict.items()
    )[0:num_data_samples_to_select]
    return random_seeds_num_entities_list


if __name__ == "__main__":
    #This is the tokenizer used
    model_name = "meta-llama/llama-3.1-70b-instruct"
    token_limits = [80000, 40000, 20000, 10000]  # descending order of token_limits
    num_data_samples_to_select = 10
    task_lists = [
        BookingGetAvailabilityTaskList,
        BookingGetRoomListWithAvailabilityTaskList,
        BookingSearchCarRentalsTaskList,
        BookingSearchFlightsMultiStopsTaskList,
        BookingGetSeatMapTaskList
    ]
    random_seeds_num_entities_list: List[Any] = []
    for task_list in task_lists:
        for token_limit in token_limits:
            dataset = {}
            filtered_dataset = {}

            host = task_list.host
            endpoint_name = task_list.endpoint_name
            if len(random_seeds_num_entities_list) == 0:
                num_iterations = [j + 1 for j in range(100)]
            else:
                num_iterations = [
                    random_seed[0] for random_seed in random_seeds_num_entities_list
                ]
            for i in num_iterations:
                task_list_obj = task_list(
                    os.path.join(
                        os.path.dirname(__file__),
                        f"../data/{host}_{endpoint_name}.json",
                    ),
                )
                output_data_dict, num_entities = task_list_obj.create_data_subsets(
                    token_limit,
                    random_seed=i,
                    model_name=model_name,
                    min_entities=None,
                )
                output_data_dict["num_entities"] = num_entities
                if output_data_dict is not None:
                    dataset[i] = output_data_dict
            # # Filter the dataset to keep only num_data_samples_to_select samples with the maximum possible number of entities
            # # and it returns the random_seeds corresponding to that selection
            if len(random_seeds_num_entities_list) == 0:
                random_seeds_num_entities_list = filter_dataset(
                    dataset, num_data_samples_to_select
                )
                print(f"{token_limit}: {random_seeds_num_entities_list}")
                for random_seed, _ in random_seeds_num_entities_list:
                    filtered_dataset[random_seed] = dataset[random_seed]
            else:
                filtered_dataset = dataset
            for key, value in filtered_dataset.items():
                del value["num_entities"]
            json.dump(
                filtered_dataset,
                open(
                    os.path.join(
                        os.path.dirname(__file__),
                        f"../data/data_subsets_for_lim_experiments/{host}_{endpoint_name}_subset_{token_limit}.json",
                    ),
                    "w",
                ),
                indent=4,
            )
