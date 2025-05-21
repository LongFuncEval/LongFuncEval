import json
from datetime import datetime
from typing import Any, Union

from large_response_QA.large_response_utils import manipulate_response

from . import evals
from .base import Task
from .data_structures import LongResponseQASample, TaskAttributes


class GetDestinationAirport(Task):
    """Get the destination airport code for a flight number.
    The tool spec for Search_Flights_Multi_Stops can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.accuracy_string]
    TASK_ATTRIBUTES = [TaskAttributes.EXTRACTIVE]

    # example answer: AMS
    def get_answer(self, api_response: dict[Any, Any], flight_number: int) -> str:
        for _, query_result in api_response.items():
            flight_offers = query_result["data"]["flightOffers"]
            for flight_offer in flight_offers:
                flight_segments = flight_offer["segments"]
                for flight_segment in flight_segments:
                    flight_legs = flight_segment["legs"]
                    for flight_leg in flight_legs:
                        flightInfo = flight_leg["flightInfo"]
                        if flightInfo["flightNumber"] == flight_number:
                            return str(flight_leg["arrivalAirport"]["code"])
        return "None"

    # example question: "What is the arrival airport code for flight 606?"
    def get_question(self, flight_number: int) -> str:
        return f"What is the arrival airport code for flight {flight_number}?"

    # This signature deviates from the base class as we added index. Should we add it to the base?
    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:

        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            query_result = api_response[query_args]
            considered_flight_numbers = []
            api_response = manipulate_response(api_response, index)
            flight_offers = query_result["data"]["flightOffers"]
            for flight_offer in flight_offers:
                flight_segments = flight_offer["segments"]
                for flight_segment in flight_segments:
                    flight_legs = flight_segment["legs"]
                    for flight_leg in flight_legs:
                        flightInfo = flight_leg["flightInfo"]
                        flight_number = flightInfo["flightNumber"]
                        if flight_number not in considered_flight_numbers:
                            considered_flight_numbers.append(flight_number)
                            question = self.get_question(flight_number=flight_number)
                            answer = self.get_answer(
                                api_response=api_response, flight_number=flight_number
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


class GetFlightDuration(Task):
    """Get the flight duration for a flight number.
    The tool spec for Search_Flights_Multi_Stops can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.accuracy_string]
    TASK_ATTRIBUTES = [TaskAttributes.EXTRACTIVE]

    # example answer: 4260
    def get_answer(self, api_response: dict[Any, Any], flight_number: int) -> str:
        for _, query_result in api_response.items():
            flight_offers = query_result["data"]["flightOffers"]
            for flight_offer in flight_offers:
                flight_segments = flight_offer["segments"]
                for flight_segment in flight_segments:
                    flight_legs = flight_segment["legs"]
                    for flight_leg in flight_legs:
                        flightInfo = flight_leg["flightInfo"]
                        if flightInfo["flightNumber"] == flight_number:
                            return str(flight_leg["totalTime"])
        return "None"

    # example question: How long is the flight 676? Please only output the duration without the unit and nothing else.
    def get_question(self, flight_number: int) -> str:
        return f"How long is the flight {flight_number}? Please only output the duration without the unit and nothing else."

    # This signature deviates from the base class as we added index. Should we add it to the base?
    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:

        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            query_result = api_response[query_args]
            considered_flight_numbers = []
            api_response = manipulate_response(api_response, index)

            flight_offers = query_result["data"]["flightOffers"]
            for flight_offer in flight_offers:
                flight_segments = flight_offer["segments"]
                for flight_segment in flight_segments:
                    flight_legs = flight_segment["legs"]
                    for flight_leg in flight_legs:
                        flightInfo = flight_leg["flightInfo"]
                        flight_number = flightInfo["flightNumber"]
                        if flight_number not in considered_flight_numbers:
                            considered_flight_numbers.append(flight_number)
                            question = self.get_question(flight_number=flight_number)
                            answer = self.get_answer(
                                api_response=api_response, flight_number=flight_number
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


class GetNonstopFlightsFromSrcToDest(Task):
    """List the flight numbers of non stop flights from origin airport to destination airports.
    The tool spec for Search_Flights_Multi_Stops can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.unordered_list_str_match]
    TASK_ATTRIBUTES = [TaskAttributes.FILTERING]

    # example answer: 3710, 1457, 1479, 2682
    def get_answer(
        self,
        api_response: dict[Any, Any],
        departure_airport_code: str,
        arrival_airport_code: str,
    ) -> str:
        flight_numbers = []
        for _, query_result in api_response.items():
            flight_offers = query_result["data"]["flightOffers"]
            for flight_offer in flight_offers:
                flight_segments = flight_offer["segments"]
                for flight_segment in flight_segments:
                    departure_airport = flight_segment["departureAirport"]["code"]
                    arrival_airport = flight_segment["arrivalAirport"]["code"]
                    if (
                        departure_airport.strip().lower()
                        == departure_airport_code.strip().lower()
                        and arrival_airport.strip().lower()
                        == arrival_airport_code.strip().lower()
                    ):
                        flight_legs = flight_segment["legs"]
                        if len(flight_legs) == 1:  # non-stop flights
                            flight_leg = flight_legs[0]
                            flightInfo = flight_leg["flightInfo"]
                            if str(flightInfo["flightNumber"]) not in flight_numbers:
                                flight_numbers.append(str(flightInfo["flightNumber"]))
        if len(flight_numbers) > 0:
            return ", ".join(flight_numbers)
        else:
            return "None"

    # example question: List the flight numbers of non stop flights from ORD to BOS. Please output the list as comma separated flight numbers.
    def get_question(
        self, departure_airport_code: str, arrival_airport_code: str
    ) -> str:
        return f"List the flight numbers of non stop flights from {departure_airport_code} to {arrival_airport_code}. Please output the list as comma separated flight numbers."

    # This signature deviates from the base class as we added index. Should we add it to the base?
    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:

        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            query_result = api_response[query_args]
            considered_departure_arrival = []
            api_response = manipulate_response(api_response, index)

            flight_offers = query_result["data"]["flightOffers"]
            for flight_offer in flight_offers:
                flight_segments = flight_offer["segments"]
                for flight_segment in flight_segments:
                    departure_airport_code = flight_segment["departureAirport"]["code"]
                    arrival_airport_code = flight_segment["arrivalAirport"]["code"]
                    flight_legs = flight_segment["legs"]
                    if (
                        f"{departure_airport_code}_{arrival_airport_code}"
                        not in considered_departure_arrival
                        and len(flight_legs) == 1
                    ):  # nonstop flight
                        considered_departure_arrival.append(
                            f"{departure_airport_code}_{arrival_airport_code}"
                        )
                        question = self.get_question(
                            departure_airport_code=departure_airport_code,
                            arrival_airport_code=arrival_airport_code,
                        )
                        answer = self.get_answer(
                            api_response=api_response,
                            departure_airport_code=departure_airport_code,
                            arrival_airport_code=arrival_airport_code,
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


class GetOperatingCarriers(Task):
    """List operating carriers for flights from departure airport code to arrival airport code.
    The tool spec for Search_Flights_Multi_Stops can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.unordered_list_str_match]
    TASK_ATTRIBUTES = [TaskAttributes.FILTERING]

    # example answer:
    def get_answer(
        self,
        api_response: dict[Any, Any],
        departure_airport_code: str,
        arrival_airport_code: str,
    ) -> str:
        operating_carriers = []
        for _, query_result in api_response.items():
            flight_offers = query_result["data"]["flightOffers"]
            for flight_offer in flight_offers:
                flight_segments = flight_offer["segments"]
                for flight_segment in flight_segments:
                    departure_airport = flight_segment["departureAirport"]["code"]
                    arrival_airport = flight_segment["arrivalAirport"]["code"]
                    if (
                        departure_airport.strip().lower()
                        == departure_airport_code.strip().lower()
                        and arrival_airport.strip().lower()
                        == arrival_airport_code.strip().lower()
                    ):
                        flight_legs = flight_segment["legs"]
                        if (
                            len(flight_legs) == 1
                        ):  # non-stop flights so the segment airport codes match.
                            flight_leg = flight_legs[0]
                            flightInfo = flight_leg["flightInfo"]
                            if (
                                "carrierInfo" in flightInfo
                                and str(flightInfo["carrierInfo"]["operatingCarrier"])
                                not in operating_carriers
                            ):
                                operating_carriers.append(
                                    str(flightInfo["carrierInfo"]["operatingCarrier"])
                                )
        if len(operating_carriers) > 0:
            return ", ".join(operating_carriers)
        else:
            return "None"

    # example question:
    def get_question(
        self, departure_airport_code: str, arrival_airport_code: str
    ) -> str:
        # List operating carriers for flights from departure airport code to arrival airport code.
        return f"List operating carriers for flights from {departure_airport_code} to {arrival_airport_code}. Please output the list as comma separated carriers."

    # This signature deviates from the base class as we added index. Should we add it to the base?
    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:

        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            query_result = api_response[query_args]
            considered_departure_arrival = []
            api_response = manipulate_response(api_response, index)

            flight_offers = query_result["data"]["flightOffers"]
            for flight_offer in flight_offers:
                flight_segments = flight_offer["segments"]
                for flight_segment in flight_segments:
                    departure_airport_code = flight_segment["departureAirport"]["code"]
                    arrival_airport_code = flight_segment["arrivalAirport"]["code"]
                    flight_legs = flight_segment["legs"]
                    if (
                        f"{departure_airport_code}_{arrival_airport_code}"
                        not in considered_departure_arrival
                        and len(flight_legs) == 1
                    ):  # nonstop flight so the segment airport codes match.
                        considered_departure_arrival.append(
                            f"{departure_airport_code}_{arrival_airport_code}"
                        )
                        question = self.get_question(
                            departure_airport_code=departure_airport_code,
                            arrival_airport_code=arrival_airport_code,
                        )
                        answer = self.get_answer(
                            api_response=api_response,
                            departure_airport_code=departure_airport_code,
                            arrival_airport_code=arrival_airport_code,
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


class GetShortestFlight(Task):
    """Get the shortest flight from src to dest.
    The tool spec for Search_Flights_Multi_Stops can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.accuracy_string]
    TASK_ATTRIBUTES = [TaskAttributes.AGGREGRATION]

    # example answer: 9960
    def get_answer(
        self,
        api_response: dict[Any, Any],
        departure_airport_code: str,
        arrival_airport_code: str,
    ) -> str:
        shortest_time: Union[int, None] = None
        for _, query_result in api_response.items():
            flight_offers = query_result["data"]["flightOffers"]
            for flight_offer in flight_offers:
                flight_segments = flight_offer["segments"]
                for flight_segment in flight_segments:
                    departure_airport = flight_segment["departureAirport"]["code"]
                    arrival_airport = flight_segment["arrivalAirport"]["code"]
                    if (
                        departure_airport.strip().lower()
                        == departure_airport_code.strip().lower()
                        and arrival_airport.strip().lower()
                        == arrival_airport_code.strip().lower()
                    ):
                        time_duration = flight_segment["totalTime"]
                        print(
                            departure_airport_code, arrival_airport_code, time_duration
                        )
                        if shortest_time is None or time_duration < shortest_time:
                            shortest_time = time_duration
        if shortest_time is not None:
            return str(shortest_time)
        else:
            return "None"

    # example question: Get the shortest time duration for a one-way flight from ORD to IAH. Only output the time duration in seconds without the unit and no other information.
    def get_question(
        self, departure_airport_code: str, arrival_airport_code: str
    ) -> str:
        # Shortest time duration from source to destination. Please only output the time duration in seconds and no other information.
        return f"Get the shortest time duration for a one-way flight from {departure_airport_code} to {arrival_airport_code}. Only output the time duration in seconds without the unit and no other information."

    # This signature deviates from the base class as we added index. Should we add it to the base?
    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:

        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            query_result = api_response[query_args]
            considered_departure_arrival = []
            api_response = manipulate_response(api_response, index)

            flight_offers = query_result["data"]["flightOffers"]
            for flight_offer in flight_offers:
                flight_segments = flight_offer["segments"]
                for flight_segment in flight_segments:
                    departure_airport_code = flight_segment["departureAirport"]["code"]
                    arrival_airport_code = flight_segment["arrivalAirport"]["code"]
                    flight_legs = flight_segment["legs"]
                    if (
                        f"{departure_airport_code}_{arrival_airport_code}"
                        not in considered_departure_arrival
                        and len(flight_legs) == 1
                    ):  # nonstop flight so the segment airport codes match.
                        considered_departure_arrival.append(
                            f"{departure_airport_code}_{arrival_airport_code}"
                        )
                        question = self.get_question(
                            departure_airport_code=departure_airport_code,
                            arrival_airport_code=arrival_airport_code,
                        )
                        answer = self.get_answer(
                            api_response=api_response,
                            departure_airport_code=departure_airport_code,
                            arrival_airport_code=arrival_airport_code,
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


class GetEarliestFlight(Task):
    """Get the earliest flight from src to dest.
    The tool spec for Search_Flights_Multi_Stops can be found at https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    """

    EVALUATION_CRITERIAS = [evals.accuracy_string]
    TASK_ATTRIBUTES = [TaskAttributes.AGGREGRATION]

    # example answer: 3242
    def get_answer(
        self,
        api_response: dict[Any, Any],
        departure_airport_code: str,
        arrival_airport_code: str,
    ) -> str:
        earliest_time: Union[datetime, None] = None
        earliest_flight = "None"
        for _, query_result in api_response.items():
            flight_offers = query_result["data"]["flightOffers"]
            for flight_offer in flight_offers:
                flight_segments = flight_offer["segments"]
                for flight_segment in flight_segments:
                    departure_airport = flight_segment["departureAirport"]["code"]
                    arrival_airport = flight_segment["arrivalAirport"]["code"]
                    if (
                        departure_airport.strip().lower()
                        == departure_airport_code.strip().lower()
                        and arrival_airport.strip().lower()
                        == arrival_airport_code.strip().lower()
                    ):
                        # format: "departureTime": "2024-12-30T08:02:00",
                        departure_time = datetime.fromisoformat(
                            flight_segment["departureTime"]
                        )
                        if (earliest_time is None) or (departure_time < earliest_time):
                            print(departure_airport, arrival_airport, departure_time)
                            earliest_time = departure_time
                            earliest_flight = flight_segment["legs"][0]["flightInfo"][
                                "flightNumber"
                            ]
        return str(earliest_flight)

    # example question: Get the earliest flight from DFW to ORD. Only output flight number and no other information.
    def get_question(
        self, departure_airport_code: str, arrival_airport_code: str
    ) -> str:
        # Get the earliest flight from source to destination. Only output flight number and no other information.
        return f"Get the earliest flight from {departure_airport_code} to {arrival_airport_code}. Only output flight number and no other information."

    # This signature deviates from the base class as we added index. Should we add it to the base?
    def get_qa_samples(
        self, api_response: dict[Any, Any], index: int = 0
    ) -> list[LongResponseQASample]:

        qa_samples: list[LongResponseQASample] = []
        query_args_list = list(api_response.keys())
        try:
            query_args = query_args_list[0]
            query_result = api_response[query_args]
            considered_departure_arrival = []
            api_response = manipulate_response(api_response, index)

            flight_offers = query_result["data"]["flightOffers"]
            for flight_offer in flight_offers:
                flight_segments = flight_offer["segments"]
                for flight_segment in flight_segments:
                    departure_airport_code = flight_segment["departureAirport"]["code"]
                    arrival_airport_code = flight_segment["arrivalAirport"]["code"]
                    flight_legs = flight_segment["legs"]
                    if (
                        f"{departure_airport_code}_{arrival_airport_code}"
                        not in considered_departure_arrival
                        and len(flight_legs) == 1
                    ):  # nonstop flight so the segment airport codes match.
                        considered_departure_arrival.append(
                            f"{departure_airport_code}_{arrival_airport_code}"
                        )
                        question = self.get_question(
                            departure_airport_code=departure_airport_code,
                            arrival_airport_code=arrival_airport_code,
                        )
                        answer = self.get_answer(
                            api_response=api_response,
                            departure_airport_code=departure_airport_code,
                            arrival_airport_code=arrival_airport_code,
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


if __name__ == "__main__":
    import os

    num_qa_pairs = 0
    dataset = json.load(
        open(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../data/api_responses/data_subsets_for_lim_experiments/booking-com15.p.rapidapi.com_Search_Flights_Multi_Stops_subset_80000.json",
                )
            )
        )
    )
    task_obj: Any = None
    for random_seed, api_responses in dataset.items():
        for app, endpoint_info in api_responses.items():
            for endpoint, query_info in endpoint_info.items():
                task_obj = GetDestinationAirport()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetFlightDuration()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetNonstopFlightsFromSrcToDest()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetOperatingCarriers()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetShortestFlight()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)
                task_obj = GetEarliestFlight()
                qa_pairs = task_obj.get_qa_samples(query_info)
                if len(qa_pairs) > 0:
                    for qa_pair in qa_pairs:
                        print(qa_pair.question, qa_pair.gold_answer)
                    num_qa_pairs += len(qa_pairs)

    print(f"num_qa_pairs: {num_qa_pairs}")
