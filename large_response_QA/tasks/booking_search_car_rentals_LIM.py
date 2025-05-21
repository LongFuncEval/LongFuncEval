from typing import Any

from ..large_response_utils import manipulate_response

from . import evals
from .base import Task
from .data_structures import LongResponseQASample, TaskAttributes
import json

class GetCleanlinessRating(Task):
    EVALUATION_CRITERIAS = [evals.accuracy_string]
    TASK_ATTRIBUTES = [TaskAttributes.EXTRACTIVE]

    def get_question(self, vehicle_id: str) -> str:
        return f'What is the cleanliness rating of "{vehicle_id}"?'

    def get_answer(self, api_response: dict[Any, Any], vehicle_id: str) -> str:
        for query_args, query_result in api_response.items():
            for car in query_result["data"]["search_results"]:
                if car["vehicle_id"].strip().lower() == vehicle_id.strip().lower():
                    return str(car["rating_info"]["cleanliness"])
        return "None"

    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:
        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            query_result = api_response[query_args]
            api_response = manipulate_response(api_response, index)

            vehicle_ids = []
            for car in query_result["data"]["search_results"]:
                vehicle_id = car["vehicle_id"]
                if vehicle_id not in vehicle_ids:
                    vehicle_ids.append(vehicle_id)
                    question = self.get_question(vehicle_id=vehicle_id)
                    answer = self.get_answer(
                        api_response=api_response, vehicle_id=vehicle_id
                    )

            if answer is not None and answer != "None":
                task = LongResponseQASample(
                    api_response=api_response, question=question, gold_answer=answer
                )
                qa_samples.append(task)
        except BaseException:
            pass
        return qa_samples


class GetFuelPolicy(Task):
    EVALUATION_CRITERIAS = [evals.accuracy_string]
    TASK_ATTRIBUTES = [TaskAttributes.EXTRACTIVE]

    def get_question(self, vehicle_id: str) -> str:
        return f'What is the fuel policy of "{vehicle_id}"?'

    def get_answer(self, api_response: dict[Any, Any], vehicle_id: str) -> str:
        for query_args, query_result in api_response.items():
            for car in query_result["data"]["search_results"]:
                if car["vehicle_id"].strip().lower() == vehicle_id.strip().lower():
                    return str(car["vehicle_info"]["fuel_policy"])
        return "None"

    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:

        vehicle_ids = []
        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            query_result = api_response[query_args]
            api_response = manipulate_response(api_response, index)
            for car in query_result["data"]["search_results"]:
                vehicle_id = car["vehicle_id"]
                if vehicle_id not in vehicle_ids:
                    vehicle_ids.append(vehicle_id)
                    question = self.get_question(vehicle_id=vehicle_id)
                    answer = self.get_answer(
                        api_response=api_response, vehicle_id=vehicle_id
                    )

            if answer is not None and answer != "None":
                task = LongResponseQASample(
                    api_response=api_response, question=question, gold_answer=answer
                )
                qa_samples.append(task)
        except BaseException:
            pass
        return qa_samples


class ListCarInCurrency(Task):
    EVALUATION_CRITERIAS = [evals.unordered_list_str_match]
    TASK_ATTRIBUTES = [TaskAttributes.FILTERING]

    def get_question(
        self,
        currency: str,
        pick_up_latitude: str,
        pick_up_longitude: str,
        pick_up_date: str,
    ) -> str:
        return f'Show me cars with prices in "{currency}" for cars that can be picked up from "{pick_up_latitude}", "{pick_up_longitude}" on "{pick_up_date}"? Output a comma separated list of vehicle IDs.'

    def get_answer(
        self,
        api_response: dict[Any, Any],
        currency: str,
        pick_up_latitude: str,
        pick_up_longitude: str,
        pick_up_date: str,
    ) -> str:
        vehicle_list = []
        at_least_one = False
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            if (
                query_args_dict["pick_up_latitude"] == pick_up_latitude
                and query_args_dict["pick_up_longitude"] == pick_up_longitude
                and query_args_dict["pick_up_date"] == pick_up_date
            ):
                for car in query_result["data"]["search_results"]:
                    try:
                        if car["pricing_info"]["base_currency"] == currency:
                            vehicle_list.append(car["vehicle_id"])
                            at_least_one = True
                    except BaseException:
                        continue
        if at_least_one:
            return ", ".join(vehicle_list)
        else:
            return "None"

    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:
        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            query_result = api_response[query_args]
            api_response = manipulate_response(api_response, index)

            query_args_dict = json.loads(query_args.replace("'", '"'))
            pick_up_latitude = query_args_dict["pick_up_latitude"]
            pick_up_longitude = query_args_dict["pick_up_longitude"]
            pick_up_date = query_args_dict["pick_up_date"]
            combined_criteria = []
            for car in query_result["data"]["search_results"]:
                currency = car["pricing_info"]["base_currency"]
                data_combined_criteria = (
                    f"{currency}_{pick_up_latitude}_{pick_up_longitude}_{pick_up_date}"
                )
                if data_combined_criteria not in combined_criteria:
                    combined_criteria.append(data_combined_criteria)
                    question = self.get_question(
                        currency=currency,
                        pick_up_latitude=pick_up_latitude,
                        pick_up_longitude=pick_up_longitude,
                        pick_up_date=pick_up_date,
                    )
                    answer = self.get_answer(
                        api_response=api_response,
                        currency=currency,
                        pick_up_latitude=pick_up_latitude,
                        pick_up_longitude=pick_up_longitude,
                        pick_up_date=pick_up_date,
                    )

                    if answer is not None and answer != "None":
                        task = LongResponseQASample(
                            api_response=api_response,
                            question=question,
                            gold_answer=answer,
                        )
                        qa_samples.append(task)
        except BaseException:
            pass
        return qa_samples


class ListCarFreeCancellation(Task):
    EVALUATION_CRITERIAS = [evals.unordered_list_str_match]
    TASK_ATTRIBUTES = [TaskAttributes.FILTERING]

    def get_question(
        self, pick_up_latitude: str, pick_up_longitude: str, pick_up_date: str
    ) -> str:
        return f'List all cars with a free cancellation policy for cars that can be picked up from "{pick_up_latitude}", "{pick_up_longitude}" on "{pick_up_date}"? Output a comma separated list of vehicle IDs.'

    def get_answer(
        self,
        api_response: dict[Any, Any],
        pick_up_latitude: str,
        pick_up_longitude: str,
        pick_up_date: str,
    ) -> str:
        vehicle_list = []
        at_least_one = False
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            if (
                query_args_dict["pick_up_latitude"] == pick_up_latitude
                and query_args_dict["pick_up_longitude"] == pick_up_longitude
                and query_args_dict["pick_up_date"] == pick_up_date
            ):
                for car in query_result["data"]["search_results"]:
                    try:
                        if car["vehicle_info"]["free_cancellation"] == 1:
                            vehicle_list.append(car["vehicle_id"])
                            at_least_one = True
                    except BaseException:
                        continue
        if at_least_one:
            return ", ".join(vehicle_list)
        else:
            return "None"

    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:
        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            api_response = manipulate_response(api_response, index)

            query_args_dict = json.loads(query_args.replace("'", '"'))
            pick_up_latitude = query_args_dict["pick_up_latitude"]
            pick_up_longitude = query_args_dict["pick_up_longitude"]
            pick_up_date = query_args_dict["pick_up_date"]

            question = self.get_question(
                pick_up_latitude=pick_up_latitude,
                pick_up_longitude=pick_up_longitude,
                pick_up_date=pick_up_date,
            )
            answer = self.get_answer(
                api_response=api_response,
                pick_up_latitude=pick_up_latitude,
                pick_up_longitude=pick_up_longitude,
                pick_up_date=pick_up_date,
            )

            if answer is not None and answer != "None":
                task = LongResponseQASample(
                    api_response=api_response, question=question, gold_answer=answer
                )
                qa_samples.append(task)
        except BaseException:
            pass
        return qa_samples


class CountCarsByTransmission(Task):
    EVALUATION_CRITERIAS = [evals.approx_number_match]
    TASK_ATTRIBUTES = [TaskAttributes.AGGREGRATION]

    def get_question(
        self,
        transmission_type: str,
        pick_up_latitude: str,
        pick_up_longitude: str,
        pick_up_date: str,
    ) -> str:
        return f'How many cars have an "{transmission_type}" transmission for cars that can be picked up from "{pick_up_latitude}", "{pick_up_longitude}" on "{pick_up_date}"?'

    def get_answer(
        self,
        api_response: dict[Any, Any],
        transmission_type: str,
        pick_up_latitude: str,
        pick_up_longitude: str,
        pick_up_date: str,
    ) -> str:
        car_count = 0
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            if (
                query_args_dict["pick_up_latitude"] == pick_up_latitude
                and query_args_dict["pick_up_longitude"] == pick_up_longitude
                and query_args_dict["pick_up_date"] == pick_up_date
            ):
                for car in query_result["data"]["search_results"]:
                    if (
                        car["vehicle_info"]["transmission"].strip().lower()
                        == transmission_type.strip().lower()
                    ):
                        car_count += 1
        return str(car_count)

    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:
        transmission_type_list = []
        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            api_response = manipulate_response(api_response, index)

            query_result = api_response[query_args]
            query_args_dict = json.loads(query_args.replace("'", '"'))
            pick_up_latitude = query_args_dict["pick_up_latitude"]
            pick_up_longitude = query_args_dict["pick_up_longitude"]
            pick_up_date = query_args_dict["pick_up_date"]

            for car in query_result["data"]["search_results"]:
                transmission_type = car["vehicle_info"]["transmission"]
                if transmission_type not in transmission_type_list:
                    transmission_type_list.append(transmission_type)
                    question = self.get_question(
                        transmission_type=transmission_type,
                        pick_up_latitude=pick_up_latitude,
                        pick_up_longitude=pick_up_longitude,
                        pick_up_date=pick_up_date,
                    )
                    answer = self.get_answer(
                        api_response=api_response,
                        transmission_type=transmission_type,
                        pick_up_latitude=pick_up_latitude,
                        pick_up_longitude=pick_up_longitude,
                        pick_up_date=pick_up_date,
                    )

            if answer is not None and answer != "None":
                task = LongResponseQASample(
                    api_response=api_response, question=question, gold_answer=answer
                )
                qa_samples.append(task)
        except BaseException:
            pass
        return qa_samples


class CheapestCar(Task):
    EVALUATION_CRITERIAS = [evals.approx_number_match]
    TASK_ATTRIBUTES = [TaskAttributes.AGGREGRATION]

    def get_question(
        self, pick_up_latitude: str, pick_up_longitude: str, pick_up_date: str
    ) -> str:
        return f'What is the cheapest base price available for cars that can be picked up from "{pick_up_latitude}", "{pick_up_longitude}" on "{pick_up_date}"?'

    def get_answer(
        self,
        api_response: dict[Any, Any],
        pick_up_latitude: str,
        pick_up_longitude: str,
        pick_up_date: str,
    ) -> str:
        car_price = []
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            if (
                query_args_dict["pick_up_latitude"] == pick_up_latitude
                and query_args_dict["pick_up_longitude"] == pick_up_longitude
                and query_args_dict["pick_up_date"] == pick_up_date
            ):
                for car in query_result["data"]["search_results"]:
                    car_price.append(car["pricing_info"]["base_price"])
        return str(min(car_price))

    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:
        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            api_response = manipulate_response(api_response, index)
            query_args_dict = json.loads(query_args.replace("'", '"'))
            pick_up_latitude = query_args_dict["pick_up_latitude"]
            pick_up_longitude = query_args_dict["pick_up_longitude"]
            pick_up_date = query_args_dict["pick_up_date"]

            question = self.get_question(
                pick_up_latitude=pick_up_latitude,
                pick_up_longitude=pick_up_longitude,
                pick_up_date=pick_up_date,
            )
            answer = self.get_answer(
                api_response=api_response,
                pick_up_latitude=pick_up_latitude,
                pick_up_longitude=pick_up_longitude,
                pick_up_date=pick_up_date,
            )

            if answer is not None and answer != "None":
                task = LongResponseQASample(
                    api_response=api_response, question=question, gold_answer=answer
                )
                qa_samples.append(task)
        except BaseException:
            pass
        return qa_samples


if __name__ == "__main__":
    import json
    import os

    len_qa_pairs = []
    dataset = json.load(
        open(
            os.path.expanduser(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../data/data_subsets_for_lim_experiments/booking-com15.p.rapidapi.com_Search_Car_Rentals_subset_80000.json"
                )
            )
        )
    )
    task_obj: Any = None
    num_qa_pairs = 0
    for random_seed, api_responses in dataset.items():
        for app, endpoint_info in api_responses.items():
            for endpoint, query_info in endpoint_info.items():
                task_obj = GetCleanlinessRating()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                len_qa_pairs.append(len(qa_pairs))

                task_obj = GetFuelPolicy()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                len_qa_pairs.append(len(qa_pairs))

                task_obj = ListCarInCurrency()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                len_qa_pairs.append(len(qa_pairs))

                task_obj = ListCarFreeCancellation()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                len_qa_pairs.append(len(qa_pairs))

                task_obj = CountCarsByTransmission()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                len_qa_pairs.append(len(qa_pairs))

                task_obj = CheapestCar()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                len_qa_pairs.append(len(qa_pairs))
    print(f"num_qa_pairs: {num_qa_pairs}")