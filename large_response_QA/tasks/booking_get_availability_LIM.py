import json
from typing import Any, Union

from ..large_response_utils import manipulate_response

from . import evals
from .base import Task
from .data_structures import LongResponseQASample, TaskAttributes


class GetLabel(Task):
    """Get the label for id for Get_Availability.
    The tool spec for Get_Availability can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.accuracy_string]
    TASK_ATTRIBUTES = [TaskAttributes.EXTRACTIVE]

    # example answer: "A Full Day in Kyoto with a Local: Private & Personalized"
    def get_answer(self, api_response: dict[Any, Any], id: str) -> str:
        for _, query_result in api_response.items():
            data_arr = query_result["data"]
            for data_elem in data_arr:
                timeslot_offers = data_elem["timeSlotOffers"]
                for timeslot_offer in timeslot_offers:
                    timeslot_id = timeslot_offer["id"]
                    if timeslot_id == id:
                        return str(timeslot_offer["label"])
        return "None"

    # example question: "What is the label for the OFB6PSYrVsTY?"
    def get_question(self, id: str) -> str:
        return f"What is the label for the {id}?"

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

            ids_considered = []

            data_arr = query_result["data"]
            for data_elem in data_arr:
                timeslot_offers = data_elem["timeSlotOffers"]
                for timeslot_offer in timeslot_offers:
                    timeslot_id = timeslot_offer["id"]
                    if timeslot_id not in ids_considered:
                        ids_considered.append(timeslot_id)
                        question = self.get_question(id=timeslot_id)
                        answer = self.get_answer(
                            api_response=api_response, id=timeslot_id
                        )

                        if (
                            answer is not None
                            and answer != "None"
                            and len(qa_samples) < 5
                        ):
                            task = LongResponseQASample(
                                api_response=api_response,
                                question=question,
                                gold_answer=answer,
                            )
                            qa_samples.append(task)
        except BaseException:
            pass

        return qa_samples


class GetAgeRange(Task):
    """Get the age range for Get_Availability.
    The tool spec for Get_Availability can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.accuracy_string]
    TASK_ATTRIBUTES = [TaskAttributes.EXTRACTIVE]

    # example answer: (age 13-99)
    def get_answer(self, api_response: dict[Any, Any], offer_item_id: str) -> str:
        for _, query_result in api_response.items():
            data_arr = query_result["data"]
            for data_elem in data_arr:
                timeslot_offers = data_elem["timeSlotOffers"]
                for timeslot_offer in timeslot_offers:
                    timeslot_offer_items = timeslot_offer["items"]
                    for timeslot_offer_item in timeslot_offer_items:
                        timeslot_offer_item_id = timeslot_offer_item["offerItemId"]
                        if (
                            timeslot_offer_item_id == offer_item_id
                            and "constraint" in timeslot_offer_item
                        ):
                            constraint_label = timeslot_offer_item["constraint"][
                                "label"
                            ]
                            if constraint_label.startswith("(age"):
                                return str(constraint_label)
        return "None"

    # example question: "What is the age range for offer item OImC2WSOFMoG?"
    def get_question(self, offer_item_id: str) -> str:
        return f"What is the age range for offer item {offer_item_id}?"

    # This signature deviates from the base class as we added index. Should we add it to the base?
    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:

        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            query_result = api_response[query_args]
            ids_considered = []
            api_response = manipulate_response(api_response, index)

            data_arr = query_result["data"]
            for data_elem in data_arr:
                timeslot_offers = data_elem["timeSlotOffers"]
                for timeslot_offer in timeslot_offers:
                    timeslot_offer_items = timeslot_offer["items"]
                    for timeslot_offer_item in timeslot_offer_items:
                        timeslot_offer_item_id = timeslot_offer_item["offerItemId"]
                        if timeslot_offer_item_id not in ids_considered:
                            ids_considered.append(timeslot_offer_item_id)
                            question = self.get_question(
                                offer_item_id=timeslot_offer_item_id
                            )
                            answer = self.get_answer(
                                api_response=api_response,
                                offer_item_id=timeslot_offer_item_id,
                            )

                            if (
                                answer is not None
                                and answer != "None"
                                and len(qa_samples) < 5
                            ):
                                task = LongResponseQASample(
                                    api_response=api_response,
                                    question=question,
                                    gold_answer=answer,
                                )
                                qa_samples.append(task)
        except BaseException:
            pass

        return qa_samples


class GetPrice(Task):
    """Get the price for Get_Availability.
    The tool spec for Get_Availability can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.unordered_list_str_match]
    TASK_ATTRIBUTES = [TaskAttributes.FILTERING]

    # example answer: 406.55
    def get_answer(
        self,
        api_response: dict[Any, Any],
        label: str,
        start_time: str,
        language_code: str,
    ) -> str:
        for _, query_result in api_response.items():
            data_arr = query_result["data"]
            for data_elem in data_arr:
                data_start_time = data_elem["start"]
                if data_start_time == start_time:
                    timeslot_offers = data_elem["timeSlotOffers"]
                    for timeslot_offer in timeslot_offers:
                        data_label = timeslot_offer["label"]
                        if data_label == label:
                            timeslot_offer_items = timeslot_offer["items"]
                            for timeslot_offer_item in timeslot_offer_items:
                                data_language_code = timeslot_offer_item[
                                    "languageOption"
                                ]["language"]
                                if data_language_code == language_code:
                                    price = timeslot_offer_item["convertedPrice"]
                                    if price["currency"].strip().lower() == "usd":
                                        return str(price["publicAmount"])
        return "None"

    # example question: What is the price in USD for the A Full Day in Kyoto with a Local: Private & Personalized tour that starts at 2024-12-01T08:00:00+09:00 and is in en language?
    def get_question(self, label: str, start_time: str, language_code: str) -> str:
        return f"What is the price in USD for the {label} tour that starts at {start_time} and is in {language_code} language? If there are more than one options, return a comma separated list of prices."

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

            criteria_considered = []
            data_arr = query_result["data"]
            for data_elem in data_arr:
                timeslot_offers = data_elem["timeSlotOffers"]
                start_time = data_elem["start"]
                for timeslot_offer in timeslot_offers:
                    label = timeslot_offer["label"]
                    timeslot_offer_items = timeslot_offer["items"]
                    for timeslot_offer_item in timeslot_offer_items:
                        language_code = timeslot_offer_item["languageOption"][
                            "language"
                        ]
                        question = self.get_question(
                            label=label,
                            start_time=start_time,
                            language_code=language_code,
                        )
                        answer = self.get_answer(
                            api_response=api_response,
                            label=label,
                            start_time=start_time,
                            language_code=language_code,
                        )
                        combined_criteria = f"{label}_{answer}"
                        if combined_criteria not in criteria_considered:
                            criteria_considered.append(combined_criteria)

                            if (
                                answer is not None
                                and answer != "None"
                                and len(qa_samples) < 5
                            ):
                                task = LongResponseQASample(
                                    api_response=api_response,
                                    question=question,
                                    gold_answer=answer,
                                )
                                qa_samples.append(task)
        except BaseException:
            pass

        return qa_samples


class GetIdsPerMinPerReservation(Task):
    """Get ids of items with a min per reservation filter for Get_Availability.
    The tool spec for Get_Availability can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.unordered_list_str_match]
    TASK_ATTRIBUTES = [TaskAttributes.FILTERING]

    # example answer: 406.55
    def get_answer(
        self, api_response: dict[Any, Any], timeslot_offer_id: str, num: int
    ) -> str:
        ids_list = []
        for _, query_result in api_response.items():
            data_arr = query_result["data"]
            for data_elem in data_arr:
                timeslot_offers = data_elem["timeSlotOffers"]
                for timeslot_offer in timeslot_offers:
                    data_timeslot_offer_id = timeslot_offer["id"]
                    if data_timeslot_offer_id == timeslot_offer_id:
                        timeslot_offer_items = timeslot_offer["items"]
                        for timeslot_offer_item in timeslot_offer_items:
                            min_per_reservation = timeslot_offer_item[
                                "minPerReservation"
                            ]
                            id = timeslot_offer_item["id"]
                            if min_per_reservation == num and id not in ids_list:
                                ids_list.append(id)
        return ", ".join(ids_list)

    # example question: What is the price in USD for the A Full Day in Kyoto with a Local: Private & Personalized tour that starts at 2024-12-01T08:00:00+09:00 and is in en language?
    def get_question(self, timeslot_offer_id: str, num: int) -> str:
        return f"List all the ids for items for {timeslot_offer_id} where minimum number of persons per reservation is {num}. Return a comma separated list of ids. Please note that the element you are looking for is id, not offerItemId."

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

            criteria_considered = []
            data_arr = query_result["data"]
            for data_elem in data_arr:
                timeslot_offers = data_elem["timeSlotOffers"]
                for timeslot_offer in timeslot_offers:
                    timeslot_offer_id = timeslot_offer["id"]
                    timeslot_offer_items = timeslot_offer["items"]
                    for timeslot_offer_item in timeslot_offer_items:
                        min_per_reservation = timeslot_offer_item["minPerReservation"]
                        combined_criteria = f"{timeslot_offer_id}_{min_per_reservation}"
                        if combined_criteria not in criteria_considered:
                            criteria_considered.append(combined_criteria)
                            question = self.get_question(
                                timeslot_offer_id=timeslot_offer_id,
                                num=min_per_reservation,
                            )
                            answer = self.get_answer(
                                api_response=api_response,
                                timeslot_offer_id=timeslot_offer_id,
                                num=min_per_reservation,
                            )

                            if (
                                answer is not None
                                and answer != "None"
                                and len(qa_samples) < 5
                            ):
                                task = LongResponseQASample(
                                    api_response=api_response,
                                    question=question,
                                    gold_answer=answer,
                                )
                                qa_samples.append(task)
        except BaseException:
            pass

        return qa_samples


class GetNumLanguage(Task):
    """Get number of options for a language for Get_Availability.
    The tool spec for Get_Availability can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.accuracy_string]
    TASK_ATTRIBUTES = [TaskAttributes.AGGREGRATION]

    def get_answer(
        self, api_response: dict[Any, Any], id: str, language_code: str
    ) -> str:
        item_id_list = []
        for _, query_result in api_response.items():
            data_arr = query_result["data"]
            for data_elem in data_arr:
                timeslot_offers = data_elem["timeSlotOffers"]
                for timeslot_offer in timeslot_offers:
                    timeslot_offer_items = timeslot_offer["items"]
                    timeslot_offer_id = timeslot_offer["id"]
                    if timeslot_offer_id == id:
                        for timeslot_offer_item in timeslot_offer_items:
                            data_language_code = timeslot_offer_item["languageOption"][
                                "language"
                            ]
                            item_id = timeslot_offer_item["id"]
                            if (
                                data_language_code == language_code
                                and item_id not in item_id_list
                            ):
                                item_id_list.append(item_id)
        if len(item_id_list) > 0:
            return str(len(item_id_list))
        else:
            return "None"

    # example question:
    def get_question(self, id: str, language_code: str) -> str:
        return f"How many attractions/tours of id {id} are offered in language {language_code}? The same tour at a different time is counted as a separation option."

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

            criteria_considered = []
            data_arr = query_result["data"]
            for data_elem in data_arr:
                timeslot_offers = data_elem["timeSlotOffers"]
                for timeslot_offer in timeslot_offers:
                    id = timeslot_offer["id"]
                    timeslot_offer_items = timeslot_offer["items"]
                    for timeslot_offer_item in timeslot_offer_items:
                        language_code = timeslot_offer_item["languageOption"][
                            "language"
                        ]
                        criteria = f"{language_code}_{id}"
                        if criteria not in criteria_considered:
                            criteria_considered.append(criteria)
                            question = self.get_question(
                                id=id, language_code=language_code
                            )
                            answer = self.get_answer(
                                api_response=api_response,
                                id=id,
                                language_code=language_code,
                            )
                            if (
                                answer is not None
                                and answer != "None"
                                and len(qa_samples) < 5
                            ):
                                task = LongResponseQASample(
                                    api_response=api_response,
                                    question=question,
                                    gold_answer=answer,
                                )
                                qa_samples.append(task)
        except BaseException:
            pass

        return qa_samples


class GetMinPrice(Task):
    """Get minimum price for a tour for Get_Availability.
    The tool spec for Get_Availability can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.accuracy_string]
    TASK_ATTRIBUTES = [TaskAttributes.AGGREGRATION]

    def get_answer(self, api_response: dict[Any, Any], id: str) -> str:
        min_price: Union[float, None] = None
        for _, query_result in api_response.items():
            data_arr = query_result["data"]
            for data_elem in data_arr:
                timeslot_offers = data_elem["timeSlotOffers"]
                for timeslot_offer in timeslot_offers:
                    timeslot_offer_items = timeslot_offer["items"]
                    timeslot_offer_id = timeslot_offer["id"]
                    if timeslot_offer_id == id:
                        for timeslot_offer_item in timeslot_offer_items:
                            item_price = timeslot_offer_item["convertedPrice"][
                                "chargeAmount"
                            ]
                            price_currency = timeslot_offer_item["convertedPrice"][
                                "currency"
                            ]
                            if price_currency == "USD" and (
                                min_price is None or item_price < min_price
                            ):
                                min_price = item_price
        return str(min_price)

    # example question:
    def get_question(self, id: str) -> str:
        return f"What is the minimum price in USD for id {id}?"

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

            criteria_considered = []
            data_arr = query_result["data"]
            for data_elem in data_arr:
                timeslot_offers = data_elem["timeSlotOffers"]
                for timeslot_offer in timeslot_offers:
                    id = timeslot_offer["id"]
                    timeslot_offer_items = timeslot_offer["items"]
                    for timeslot_offer_item in timeslot_offer_items:
                        criteria = f"{id}"
                        if criteria not in criteria_considered:
                            criteria_considered.append(criteria)
                            question = self.get_question(id=id)
                            answer = self.get_answer(api_response=api_response, id=id)
                            if (
                                answer is not None
                                and answer != "None"
                                and len(qa_samples) < 5
                            ):
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
    import os

    num_qa_pairs = 0
    dataset = json.load(
        open(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../data/data_subsets_for_lim_experiments/booking-com15.p.rapidapi.com_Get_Availability_subset_10000.json",
                )
            )
        )
    )
    task_obj: Any = None
    for random_seed, api_responses in dataset.items():
        for app, endpoint_info in api_responses.items():
            for endpoint, query_info in endpoint_info.items():
                task_obj = GetLabel()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetAgeRange()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetPrice()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetIdsPerMinPerReservation()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetNumLanguage()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetMinPrice()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)

    print(f"num_qa_pairs: {num_qa_pairs}")
