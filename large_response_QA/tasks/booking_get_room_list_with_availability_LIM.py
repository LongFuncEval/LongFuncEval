import json
from typing import Any, Union

import numpy as np
from ..large_response_utils import manipulate_response

from . import evals
from .base import Task
from .data_structures import LongResponseQASample, TaskAttributes


class GetRoomCount(Task):
    """Get the number of available rooms for Get_Room_List_With_Availability.
    The tool spec for Get_Room_List_With_Availability can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.accuracy_string]
    TASK_ATTRIBUTES = [TaskAttributes.EXTRACTIVE]

    # example answer: "20"
    def get_answer(self, api_response: dict[Any, Any], name: str, hotel_id: str) -> str:
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            resp_hotel_id = query_args_dict["hotel_id"]
            if resp_hotel_id.strip().lower() == hotel_id.strip().lower():
                available_rooms = query_result["available"]

                for room_kind in available_rooms:
                    room_kind_name = room_kind["name"]
                    if room_kind_name.strip().lower() == name.strip().lower():
                        return str(room_kind["room_count"])

        return "None"

    # example question: "What is the total number of available rooms of the kind Executive Twin Room - Free cancellation in hotel 12345?""
    def get_question(self, name: str, hotel_id: str) -> str:
        return f"What is the total number of available rooms of the kind {name} in hotel {hotel_id}?"

    # This signature deviates from the base class as we added index. Should we add it to the base?
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
            hotel_id = query_args_dict["hotel_id"]
            available_rooms = query_result["available"]
            considered_room_kinds = []
            for room_kind in available_rooms:
                room_kind_name = room_kind["name"]
                if room_kind_name not in considered_room_kinds:
                    considered_room_kinds.append(room_kind_name)
                    question = self.get_question(name=room_kind_name, hotel_id=hotel_id)
                    answer = self.get_answer(
                        api_response=api_response,
                        name=room_kind_name,
                        hotel_id=hotel_id,
                    )

                    if answer is not None and answer != "None" and len(qa_samples) < 5:
                        task = LongResponseQASample(
                            api_response=api_response,
                            question=question,
                            gold_answer=answer,
                        )
                        qa_samples.append(task)
        except BaseException:
            pass

        return qa_samples


class GetRoomArea(Task):
    """Get the square feet area for Get_Room_List_With_Availability.
    The tool spec for Get_Room_List_With_Availability can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.accuracy_string]
    TASK_ATTRIBUTES = [TaskAttributes.EXTRACTIVE]

    # example answer: "301.3894912"
    def get_answer(self, api_response: dict[Any, Any], name: str, hotel_id: str) -> str:
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            resp_hotel_id = query_args_dict["hotel_id"]
            if resp_hotel_id.strip().lower() == hotel_id.strip().lower():
                available_rooms = query_result["available"]
                for room_kind in available_rooms:
                    room_kind_name = room_kind["name"]
                    if room_kind_name.strip().lower() == name.strip().lower():
                        try:
                            return str(room_kind["room_surface_in_feet2"])
                        except KeyError:
                            return "None"
        return "None"

    # example question: "What is the area in square feet of Executive Twin Room - Free cancellation?""
    def get_question(self, name: str, hotel_id: str) -> str:
        return f"What is the area in square feet of {name} in hotel {hotel_id}? Include just the number and not the unit."

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
            hotel_id = query_args_dict["hotel_id"]
            available_rooms = query_result["available"]
            considered_room_kinds = []
            for room_kind in available_rooms:
                room_kind_name = room_kind["name"]
                if room_kind_name not in considered_room_kinds:
                    considered_room_kinds.append(room_kind_name)
                    question = self.get_question(name=room_kind_name, hotel_id=hotel_id)
                    answer = self.get_answer(
                        api_response=api_response,
                        name=room_kind_name,
                        hotel_id=hotel_id,
                    )

                    if answer is not None and answer != "None" and len(qa_samples) < 5:
                        task = LongResponseQASample(
                            api_response=api_response,
                            question=question,
                            gold_answer=answer,
                        )
                        qa_samples.append(task)
        except BaseException:
            pass

        return qa_samples


class GetRoomsWithPriceLessThanAmount(Task):
    """Get rooms with gross amount less than some amount for Get_Room_List_With_Availability.
    The tool spec for Get_Room_List_With_Availability can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.unordered_list_str_match]
    TASK_ATTRIBUTES = [TaskAttributes.FILTERING]

    # example answer: Standard Double Room with Two Double Beds - Free cancellation, Standard King Room - Free cancellation
    def get_answer(
        self, api_response: dict[Any, Any], amount: float, hotel_id: str
    ) -> str:
        result_rooms = []
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            resp_hotel_id = query_args_dict["hotel_id"]
            if resp_hotel_id.strip().lower() == hotel_id.strip().lower():
                available_rooms = query_result["available"]
                for room_kind in available_rooms:
                    room_kind_name = room_kind["name"]
                    product_prices = room_kind["product_price_breakdown"]
                    gross_amount = product_prices["gross_amount_per_night"]["value"]

                    if gross_amount < amount and room_kind_name not in result_rooms:
                        result_rooms.append(room_kind_name)

        if len(result_rooms) > 0:
            return ", ".join(result_rooms)
        else:
            return "None"

    # example question: "List available rooms with gross rate less than $62.10 USD hotel 178196. Output a comma separated list of room names."
    def get_question(self, amount: float, hotel_id: str) -> str:
        return f"List available rooms with gross rate less than ${amount:.2f} USD in hotel {hotel_id}. Output a comma separated list of room names."

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
            hotel_id = query_args_dict["hotel_id"]
            available_rooms = query_result["available"]
            gross_amounts = []
            for room_kind in available_rooms:
                product_prices = room_kind["product_price_breakdown"]
                gross_amount = product_prices["gross_amount_per_night"]["value"]
                gross_amounts.append(gross_amount)

            mean_gross_amount = np.mean(gross_amounts)
            question = self.get_question(amount=mean_gross_amount, hotel_id=hotel_id)
            answer = self.get_answer(
                api_response=api_response, amount=mean_gross_amount, hotel_id=hotel_id
            )

            if answer is not None and answer != "None" and len(qa_samples) < 5:
                task = LongResponseQASample(
                    api_response=api_response, question=question, gold_answer=answer
                )
                qa_samples.append(task)
        except BaseException:
            pass

        return qa_samples


class GetRoomsWithMealPlan(Task):
    """Get rooms with mean plan <mealplan>.
    The tool spec for Get_Room_List_With_Availability can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.unordered_list_str_match]
    TASK_ATTRIBUTES = [TaskAttributes.FILTERING]

    # example answer: King Room - Free cancellation, King Room - Free cancellation, Queen Room with Two Queen Beds - Free cancellation, Queen Room with Two Queen Beds - Free cancellation, Queen Room with Two Queen Beds - Free cancellation
    def get_answer(
        self, api_response: dict[Any, Any], mealplan: str, hotel_id: str
    ) -> str:
        result_rooms = []
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            resp_hotel_id = query_args_dict["hotel_id"]
            if resp_hotel_id.strip().lower() == hotel_id.strip().lower():
                available_rooms = query_result["available"]
                for room_kind in available_rooms:
                    mealplan_name = room_kind["mealplan"]
                    room_kind_name = room_kind["name"]
                    if (
                        mealplan_name.lower() == mealplan.lower()
                        and room_kind_name not in result_rooms
                    ):
                        result_rooms.append(room_kind_name)

        if len(result_rooms) > 0:
            return ", ".join(result_rooms)
        else:
            return "None"

    # example question: "Get rooms with "Breakfast included in the price" in hotel 3757040. Output a comma separated list of room names."
    def get_question(self, mealplan: str, hotel_id: str) -> str:
        return f'Get rooms with "{mealplan}" in hotel {hotel_id}. Output a comma separated list of room names.'

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
            hotel_id = query_args_dict["hotel_id"]
            available_rooms = query_result["available"]
            considered_mealplans = []
            for room_kind in available_rooms:
                mealplan = room_kind["mealplan"]
                if (
                    mealplan not in considered_mealplans
                    and mealplan != "There is no meal option with this room."
                ):
                    considered_mealplans.append(mealplan)
                    question = self.get_question(mealplan=mealplan, hotel_id=hotel_id)
                    answer = self.get_answer(
                        api_response=api_response, mealplan=mealplan, hotel_id=hotel_id
                    )

                    if answer is not None and answer != "None" and len(qa_samples) < 5:
                        task = LongResponseQASample(
                            api_response=api_response,
                            question=question,
                            gold_answer=answer,
                        )
                        qa_samples.append(task)
        except BaseException:
            pass

        return qa_samples


class GetLowestCost(Task):
    """Get the lowest all inclusive cost for Get_Room_List_With_Availability.
    The tool spec for Get_Room_List_With_Availability can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.approx_number_match]
    TASK_ATTRIBUTES = [TaskAttributes.AGGREGRATION]

    # example answer: 100.1664
    def get_answer(self, api_response: dict[Any, Any], hotel_id: str) -> str:
        lowest_gross_amount: Union[float, None] = None

        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            resp_hotel_id = query_args_dict["hotel_id"]
            if resp_hotel_id.strip().lower() == hotel_id.strip().lower():
                available_rooms = query_result["available"]
                for room_kind in available_rooms:
                    product_prices = room_kind["product_price_breakdown"]
                    gross_amount = product_prices["all_inclusive_amount"]["value"]
                    if (
                        lowest_gross_amount is None
                        or gross_amount < lowest_gross_amount
                    ):
                        lowest_gross_amount = gross_amount
        return str(lowest_gross_amount)

    # example question: "What is all inclusive cost in USD for the cheapest type of available room?"
    def get_question(self, hotel_id: str) -> str:
        return f"What is the all inclusive cost in USD for the cheapest type of available room in hotel {hotel_id}?"

    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:

        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            api_response = manipulate_response(api_response, index)

            query_args_dict = json.loads(query_args.replace("'", '"'))
            hotel_id = query_args_dict["hotel_id"]
            question = self.get_question(hotel_id=hotel_id)
            answer = self.get_answer(api_response=api_response, hotel_id=hotel_id)

            if answer is not None and answer != "None" and len(qa_samples) < 5:
                task = LongResponseQASample(
                    api_response=api_response, question=question, gold_answer=answer
                )
                qa_samples.append(task)

        except BaseException:
            pass

        return qa_samples


class GetHighestVAT(Task):
    """Get the highest VAT for Get_Room_List_With_Availability.
    The tool spec for Get_Room_List_With_Availability can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.approx_number_match]
    TASK_ATTRIBUTES = [TaskAttributes.AGGREGRATION]

    # example answer: 1.7419819603841
    def get_answer(self, api_response: dict[Any, Any], hotel_id: str) -> str:
        highest_vat_amount: Union[float, None] = None
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            resp_hotel_id = query_args_dict["hotel_id"]
            if resp_hotel_id.strip().lower() == hotel_id.strip().lower():
                available_rooms = query_result["available"]

                for room_kind in available_rooms:
                    product_prices = room_kind["product_price_breakdown"]
                    try:
                        product_price_items = product_prices["items"]
                        for item in product_price_items:
                            if item["name"] == "VAT":
                                vat_amount = item["item_amount"]["value"]
                                if (
                                    highest_vat_amount is None
                                    or vat_amount > highest_vat_amount
                                ):
                                    highest_vat_amount = vat_amount
                    except KeyError:
                        pass

        if highest_vat_amount is not None and highest_vat_amount > 0:
            return str(highest_vat_amount)
        else:
            return "None"

    # example question: What is the highest VAT in USD amongst all the available rooms in hotel 178196?
    def get_question(self, hotel_id: str) -> str:
        return f"What is the highest VAT in USD amongst all the available rooms in hotel {hotel_id}?"

    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:

        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            api_response = manipulate_response(api_response, index)

            query_args_dict = json.loads(query_args.replace("'", '"'))
            hotel_id = query_args_dict["hotel_id"]
            question = self.get_question(hotel_id=hotel_id)
            answer = self.get_answer(api_response=api_response, hotel_id=hotel_id)

            if answer is not None and answer != "None" and len(qa_samples) < 5:
                task = LongResponseQASample(
                    api_response=api_response, question=question, gold_answer=answer
                )
                qa_samples.append(task)

        except BaseException:
            pass

        return qa_samples


if __name__ == "__main__":
    import os

    num_qa_pairs = 0
    dataset = json.load(
        open(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../data/data_subsets_for_lim_experiments/booking-com15.p.rapidapi.com_Get_Room_List_With_Availability_subset_10000.json",
                )
            )
        )
    )
    task_obj: Any = None
    for random_seed, api_responses in dataset.items():
        for app, endpoint_info in api_responses.items():
            for endpoint, query_info in endpoint_info.items():
                task_obj = GetRoomCount()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetRoomArea()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetRoomsWithPriceLessThanAmount()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetRoomsWithMealPlan()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetLowestCost()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetHighestVAT()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)

    print(f"num_qa_pairs: {num_qa_pairs}")
