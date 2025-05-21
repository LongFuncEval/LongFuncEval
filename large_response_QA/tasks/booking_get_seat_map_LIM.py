from typing import Any

from . import evals
from .base import Task
from .data_structures import LongResponseQASample, TaskAttributes
from large_response_QA.large_response_utils import manipulate_response
import json
import os

class GetInsurancePrice(Task):
    EVALUATION_CRITERIAS = [evals.accuracy_string]
    TASK_ATTRIBUTES = [TaskAttributes.EXTRACTIVE]

    def get_question(self, insurance_plan: str, offer_token: str) -> str:
        return f'What is the total price in USD for the following travel insurance plan "{insurance_plan}" for the ticket with offer token "{offer_token}"? Show only the amount and no currency or currency symbol.'

    def get_answer(
        self, api_response: dict[Any, Any], insurance_plan: str, offer_token: str
    ) -> str:
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            if offer_token == query_args_dict["offerToken"]:
                if (
                    query_result["data"]["travelInsurance"]["options"]["type"]
                    == insurance_plan
                ):
                    if str(
                        query_result["data"]["travelInsurance"]["options"][
                            "priceBreakdown"
                        ]["total"]["currencyCode"]
                    )== "USD":
                        return str(query_result["data"]["travelInsurance"]["options"][
                            "priceBreakdown"]["total"]["units"])
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
            offer_token = query_args_dict["offerToken"]
            insurance_plan = query_result["data"]["travelInsurance"]["options"][
                "type"
            ]

            question = self.get_question(
                insurance_plan=insurance_plan, offer_token=offer_token
            )
            answer = self.get_answer(
                api_response=api_response,
                insurance_plan=insurance_plan,
                offer_token=offer_token,
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


class GetLuggageAllowance(Task):
    EVALUATION_CRITERIAS = [evals.unordered_list_str_match]
    TASK_ATTRIBUTES = [TaskAttributes.EXTRACTIVE]

    def get_question(self, offer_token: str) -> str:
        return f'What is the maximum luggage allowance per check-in bag for the flight with offerToken "{offer_token}"? Return the a comma separated list of type of allowance, weight, and unit.'

    def get_answer(self, api_response: dict[Any, Any], offer_token: str) -> str:
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            if offer_token == query_args_dict["offerToken"]:
                if "checkedInBaggage" in query_result["data"]:
                    return str(
                        query_result["data"]["checkedInBaggage"]["options"][0][
                            "luggageAllowance"
                        ]["luggageType"]
                        + ","
                        + str(
                            query_result["data"]["checkedInBaggage"]["options"][0][
                                "luggageAllowance"
                            ]["maxWeightPerPiece"]
                        ) + ","
                        + query_result["data"]["checkedInBaggage"]["options"][0][
                            "luggageAllowance"
                        ]["massUnit"]
                    )
                elif "cabinBaggagePerTraveller" in query_result["data"]:
                    return str(
                        query_result["data"]["cabinBaggagePerTraveller"][
                            "luggageAllowance"
                        ]["luggageType"]
                        + ","
                        + str(
                            query_result["data"]["cabinBaggagePerTraveller"][
                                "luggageAllowance"
                            ]["maxWeightPerPiece"]
                        ) + ", "
                        + query_result["data"]["cabinBaggagePerTraveller"][
                            "luggageAllowance"
                        ]["massUnit"]
                    )
                else:
                    return "None"
        return "None"

    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:
        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            #query_result = api_response[query_args]
            api_response = manipulate_response(api_response, index)
            query_args_dict = json.loads(query_args.replace("'", '"'))
            offer_token = query_args_dict["offerToken"]

            question = self.get_question(offer_token=offer_token)
            answer = self.get_answer(
                api_response=api_response, offer_token=offer_token
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


class ListSeatOptions(Task):
    EVALUATION_CRITERIAS = [evals.unordered_list_str_match]
    TASK_ATTRIBUTES = [TaskAttributes.FILTERING]

    def get_question(self, offer_token: str) -> str:
        return f'List the seat options for the flight with offer token "{offer_token}". Create a comma separated list of row ID followed by column ID.'

    def get_answer(self, api_response: dict[Any, Any], offer_token: str) -> str:
        seat_ids = []
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            if offer_token == query_args_dict["offerToken"]:
                for seatMapOption in query_result["data"]["seatMap"]["seatMapOption"]:
                    for cabin in seatMapOption["cabins"]:
                        for row in cabin["rows"]:
                            for seat in row["seats"]:
                                seat_ids.append(str(row["id"]) + seat["colId"])
        return ", ".join(seat_ids)

    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:
        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            #query_result = api_response[query_args]
            api_response = manipulate_response(api_response, index)
            query_args_dict = json.loads(query_args.replace("'", '"'))
            offer_token = query_args_dict["offerToken"]
            question = self.get_question(offer_token=offer_token)
            answer = self.get_answer(
                api_response=api_response, offer_token=offer_token
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


class ListSeatOptionsBySeatType(Task):
    EVALUATION_CRITERIAS = [evals.unordered_list_str_match]
    TASK_ATTRIBUTES = [TaskAttributes.FILTERING]

    def get_question(self, seat_type: str, offer_token: str) -> str:
        return f'List the seat row IDs of "{seat_type}" seats for the flight with offer token "{offer_token}". Create a comma separated list of row IDs.'

    def get_answer(
        self, api_response: dict[Any, Any], seat_type: str, offer_token: str
    ) -> str:
        seat_ids = []
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            if offer_token == query_args_dict["offerToken"]:
                for seatMapOption in query_result["data"]["seatMap"]["seatMapOption"]:
                    for cabin in seatMapOption["cabins"]:
                        seat_type_id = []
                        for col in cabin["columns"]:
                            if seat_type in col["description"]:
                                seat_type_id.append(col["id"])
                        for row in cabin["rows"]:
                            for seat in row["seats"]:
                                if seat["colId"] in seat_type_id:
                                    seat_ids.append(str(row["id"]) + seat["colId"])
        return ", ".join(seat_ids)

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
            combined_criteria = []
            offer_token = query_args_dict["offerToken"]
            for column in query_result["data"]["seatMap"]["seatMapOption"][0][
                "cabins"
            ][0]["columns"]:
                seat_type = column["description"][0]
                if f"{seat_type}_{offer_token}" not in combined_criteria:
                    combined_criteria.append(f"{seat_type}_{offer_token}")
                    question = self.get_question(
                        seat_type=seat_type, offer_token=offer_token
                    )
                    answer = self.get_answer(
                        api_response=api_response,
                        seat_type=seat_type,
                        offer_token=offer_token,
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


class CountSeatOptions(Task):
    EVALUATION_CRITERIAS = [evals.approx_number_match]
    TASK_ATTRIBUTES = [TaskAttributes.AGGREGRATION]

    def get_question(self, offer_token: str) -> str:
        return f'How many seat options do I have for the flight with offer token "{offer_token}"?'

    def get_answer(self, api_response: dict[Any, Any], offer_token: str) -> str:
        seat_ids = []
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            if offer_token == query_args_dict["offerToken"]:
                for seatMapOption in query_result["data"]["seatMap"]["seatMapOption"]:
                    for cabin in seatMapOption["cabins"]:
                        for row in cabin["rows"]:
                            for seat in row["seats"]:
                                seat_ids.append(str(row["id"]) + seat["colId"])
        return str(len(seat_ids))

    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:
        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            #query_result = api_response[query_args]
            api_response = manipulate_response(api_response, index)
            query_args_dict = json.loads(query_args.replace("'", '"'))
            offer_token = query_args_dict["offerToken"]
            question = self.get_question(offer_token=offer_token)
            answer = self.get_answer(
                api_response=api_response, offer_token=offer_token
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


class PercentSeatType(Task):
    EVALUATION_CRITERIAS = [evals.approx_number_match]
    TASK_ATTRIBUTES = [TaskAttributes.AGGREGRATION]

    def get_question(self, seat_type: str, offer_token: str) -> str:
        return f'What percentage of seats are "{seat_type}" seats for the flight with offer token "{offer_token}"?'

    def get_answer(
        self, api_response: dict[Any, Any], seat_type: str, offer_token: str
    ) -> str:
        seat_ids = []
        all_seat_ids = []
        for query_args, query_result in api_response.items():
            query_args_dict = json.loads(query_args.replace("'", '"'))
            if offer_token == query_args_dict["offerToken"]:
                for seatMapOption in query_result["data"]["seatMap"]["seatMapOption"]:
                    for cabin in seatMapOption["cabins"]:
                        seat_type_id = []
                        for col in cabin["columns"]:
                            if seat_type in col["description"]:
                                seat_type_id.append(col["id"])
                        for row in cabin["rows"]:
                            for seat in row["seats"]:
                                all_seat_ids.append(str(row["id"]) + seat["colId"])
                                if seat["colId"] in seat_type_id:
                                    seat_ids.append(str(row["id"]) + seat["colId"])
        return str(100 * len(seat_ids) / len(all_seat_ids))

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
            combined_criteria = []
            offer_token = query_args_dict["offerToken"]
            for column in query_result["data"]["seatMap"]["seatMapOption"][0][
                "cabins"
            ][0]["columns"]:
                seat_type = column["description"][0]
                if f"{seat_type}_{offer_token}" not in combined_criteria:
                    combined_criteria.append(f"{seat_type}_{offer_token}")
                    question = self.get_question(
                        seat_type=seat_type, offer_token=offer_token
                    )
                    answer = self.get_answer(
                        api_response=api_response,
                        seat_type=seat_type,
                        offer_token=offer_token,
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


if __name__ == "__main__":
    len_qa_pairs = []
    dataset = json.load(
        open(
            os.path.expanduser(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../data/data_subsets_for_lim_experiments/booking-com15.p.rapidapi.com_Get_Seat_Map_subset_80000.json"
                )
            )
        )
    )
    task_obj: Any = None
    num_qa_pairs = 0
    for random_seed, api_responses in dataset.items():
        for app, endpoint_info in api_responses.items():
            for endpoint, query_info in endpoint_info.items():
                task_obj = GetInsurancePrice()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                len_qa_pairs.append(len(qa_pairs))

                task_obj = GetLuggageAllowance()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                len_qa_pairs.append(len(qa_pairs))

                task_obj = ListSeatOptions()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                len_qa_pairs.append(len(qa_pairs))

                task_obj = ListSeatOptionsBySeatType()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                len_qa_pairs.append(len(qa_pairs))

                task_obj = CountSeatOptions()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                len_qa_pairs.append(len(qa_pairs))

                task_obj = PercentSeatType()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                len_qa_pairs.append(len(qa_pairs))
    print(f"num_qa_pairs: {num_qa_pairs}")