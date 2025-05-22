import json
import random
from typing import Any, Optional, Type

from transformers import AutoTokenizer

from . import (
    base,
    booking_get_availability_LIM,
    booking_get_room_list_with_availability_LIM,
    booking_search_car_rentals_LIM,
    booking_get_seat_map_LIM,
    booking_search_flights_multi_stops_LIM
    )


class TaskList:
    def __init__(self, api_response_fpath: str) -> None:
        self._api_response_fpath = api_response_fpath

        self.api_response = self.read_api_response()
        self.task_list = self.init_task_list()

    def init_task_list(self) -> list[Type[base.Task]]:
        raise NotImplementedError

    def read_api_response(self) -> Any:
        with open(self._api_response_fpath, "r") as f:
            return json.load(f)

class BookingGetRoomListWithAvailabilityTaskList(TaskList):

    host: str = "booking-com15.p.rapidapi.com"
    # for ComplexFuncBench tasks, this is the endpoint name from https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    endpoint_name: str = "Get_Room_List_With_Availability"

    def __init__(self, api_response_fpath: str) -> None:
        super().__init__(api_response_fpath)

    def init_task_list(self) -> list[Type[base.Task]]:
        task_list = [
            booking_get_room_list_with_availability_LIM.GetRoomCount,
            booking_get_room_list_with_availability_LIM.GetRoomArea,
            booking_get_room_list_with_availability_LIM.GetRoomsWithPriceLessThanAmount,
            booking_get_room_list_with_availability_LIM.GetRoomsWithMealPlan,
            booking_get_room_list_with_availability_LIM.GetHighestVAT,
            booking_get_room_list_with_availability_LIM.GetLowestCost,
        ]
        return task_list  # type:ignore

    def create_data_subsets(
        self,
        token_limit: int,
        random_seed: int,
        model_name: str,
        min_entities: Optional[int] = None,
    ) -> tuple[Any, int]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        api_responses = json.load(open(self._api_response_fpath))
        num_tokens = 0
        seen_hotel_ids: Any = []
        output_query_dict: Any = {}
        app = None
        endpoint = None
        random.seed(random_seed)
        num_entities = 0
        for app, endpoint_info in api_responses.items():
            for endpoint, queries in endpoint_info.items():
                starting_index = random.randint(0, len(queries) - 1)
                queries_list = list(queries.keys())
                query_index = starting_index
                while num_tokens < token_limit:
                    query_args = queries_list[query_index]
                    query_result = queries[query_args]
                    has_duplicate_names = False
                    try:
                        if "unavailable" in query_result.keys():
                            del query_result["unavailable"]
                        seen_names = []
                        for content in query_result["available"]:
                            if content["name"] not in seen_names:
                                seen_names.append(content["name"])
                            else:
                                has_duplicate_names = True
                                break
                            if "room_name" in content.keys():
                                del content["room_name"]  # conflicts with room_name
                            if "transactional_policy_data" in content.keys():
                                del content["transactional_policy_data"]
                            if "transactional_policy_objects" in content.keys():
                                del content["transactional_policy_objects"]
                            if "policy_display_details" in content.keys():
                                del content["policy_display_details"]
                            if "block_text" in content.keys():
                                del content["block_text"]
                            if "paymentterms" in content.keys():
                                del content["paymentterms"]
                    except BaseException as e:
                        print(e)
                        pass
                    query_index += 1
                    if query_index == len(queries):
                        query_index = 0  # circular indexing
                    query_args_dict = json.loads(query_args.replace("'", '"'))
                    hotel_id = query_args_dict["hotel_id"]
                    if hotel_id not in seen_hotel_ids:
                        # include this record in the output data
                        num_tokens += len(tokenizer.tokenize(str(query_args))) + len(
                            tokenizer.tokenize(str(query_result))
                        )
                        if not has_duplicate_names and (
                            num_tokens < token_limit or len(output_query_dict) == 0
                        ):
                            print(num_tokens)
                            num_entities += 1
                            output_query_dict[query_args] = query_result

        print(f"num_tokens: {num_tokens}")
        if app is not None and endpoint is not None:
            return {app: {endpoint: output_query_dict}}, num_entities
        else:
            return None, 0

class BookingSearchFlightsMultiStopsTaskList(TaskList):

    host: str = "booking-com15.p.rapidapi.com"
    # for ComplexFuncBench tasks, this is the endpoint name from https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    endpoint_name: str = "Search_Flights_Multi_Stops"

    def __init__(self, api_response_fpath: str) -> None:
        super().__init__(api_response_fpath)

    def init_task_list(self) -> list[Type[base.Task]]:
        task_list = [
            booking_search_flights_multi_stops_LIM.GetDestinationAirport,
            booking_search_flights_multi_stops_LIM.GetFlightDuration,
            booking_search_flights_multi_stops_LIM.GetNonstopFlightsFromSrcToDest,
            booking_search_flights_multi_stops_LIM.GetOperatingCarriers,
            booking_search_flights_multi_stops_LIM.GetShortestFlight,
            booking_search_flights_multi_stops_LIM.GetEarliestFlight,
        ]
        # TODO: need to correct the type
        return task_list  # type:ignore

    def create_data_subsets(
        self,
        token_limit: int,
        random_seed: int,
        model_name: str,
        min_entities: Optional[int] = None,
    ) -> tuple[Any, int]:
        # Make sure to delete flightDeals
        # Also, make sure in consecutive entities, you don't have duplicate departure and arrival airport codes as extracted below:
        # departure_airport = flight_segment["departureAirport"]["code"]
        # arrival_airport = flight_segment["arrivalAirport"]["code"]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        api_responses = json.load(open(self._api_response_fpath))
        num_tokens = 0
        seen_departure_arrival_airport_codes: Any = []
        output_query_dict: Any = {}
        app = None
        endpoint = None
        random.seed(random_seed)
        num_entities = 0
        for app, endpoint_info in api_responses.items():
            for endpoint, queries in endpoint_info.items():
                starting_index = random.randint(0, len(queries) - 1)
                queries_list = list(queries.keys())
                query_index = starting_index
                while num_tokens < token_limit:
                    query_args = queries_list[query_index]
                    query_result_full = queries[query_args]
                    not_eligible_record = False
                    query_result = query_result_full["data"]
                    try:
                        if "flightDeals" in query_result.keys():
                            del query_result["flightDeals"]
                        flight_offers = query_result["flightOffers"]
                        seen_departure_arrival_airport_codes_from_this_query = []
                        for flight_offer in flight_offers:
                            flight_segments = flight_offer["segments"]
                            for flight_segment in flight_segments:
                                departure_airport_code = flight_segment[
                                    "departureAirport"
                                ]["code"]
                                arrival_airport_code = flight_segment["arrivalAirport"][
                                    "code"
                                ]
                                if (
                                    f"{departure_airport_code}_{arrival_airport_code}"
                                    in seen_departure_arrival_airport_codes
                                    and f"{departure_airport_code}_{arrival_airport_code}"
                                    not in seen_departure_arrival_airport_codes_from_this_query
                                ):
                                    # This means that the airport codes are seen in a different entity/query result before. We do not want to include such cases
                                    not_eligible_record = True
                                    break
                                else:
                                    seen_departure_arrival_airport_codes.append(
                                        f"{departure_airport_code}_{arrival_airport_code}"
                                    )
                                    seen_departure_arrival_airport_codes_from_this_query.append(
                                        f"{departure_airport_code}_{arrival_airport_code}"
                                    )
                    except BaseException as e:
                        print(e)
                        pass
                    query_index += 1
                    if query_index == len(queries):
                        query_index = 0  # circular indexing
                    if not not_eligible_record:
                        # include this record in the output data
                        num_tokens += len(tokenizer.tokenize(str(query_args))) + len(
                            tokenizer.tokenize(str(query_result))
                        )
                        if num_tokens < token_limit or len(output_query_dict) == 0:
                            print(num_tokens)
                            num_entities += 1
                            output_query_dict[query_args] = {"data": query_result}

        print(f"num_tokens: {num_tokens}")
        if app is not None and endpoint is not None:
            return {app: {endpoint: output_query_dict}}, num_entities
        else:
            return None, 0


class BookingGetAvailabilityTaskList(TaskList):

    host: str = "booking-com15.p.rapidapi.com"
    # for ComplexFuncBench tasks, this is the endpoint name from https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    endpoint_name: str = "Get_Availability"

    def __init__(self, api_response_fpath: str) -> None:
        super().__init__(api_response_fpath)

    def init_task_list(self) -> list[Type[base.Task]]:
        task_list = [
            booking_get_availability_LIM.GetLabel,
            booking_get_availability_LIM.GetAgeRange,
            booking_get_availability_LIM.GetPrice,
            booking_get_availability_LIM.GetIdsPerMinPerReservation,
            booking_get_availability_LIM.GetNumLanguage,
            booking_get_availability_LIM.GetMinPrice,
        ]
        # TODO: need to correct the type
        return task_list  # type:ignore

    def create_data_subsets(
        self,
        token_limit: int,
        random_seed: int,
        model_name: str,
        min_entities: Optional[int] = None,
    ) -> tuple[Any, int]:

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        api_responses = json.load(open(self._api_response_fpath))
        num_tokens = 0
        output_query_dict: Any = {}
        app = None
        endpoint = None
        random.seed(random_seed)
        num_entities = 0
        for app, endpoint_info in api_responses.items():
            for endpoint, queries in endpoint_info.items():
                starting_index = random.randint(0, len(queries) - 1)
                queries_list = list(queries.keys())
                query_index = starting_index
                while num_tokens < token_limit:
                    query_args = queries_list[query_index]
                    query_result_full = queries[query_args]
                    not_eligible_record = False
                    query_result = query_result_full["data"]
                    for data_elem in query_result:
                        if "fullDay" in data_elem:
                            del data_elem["fullDay"]
                        timeslot_offers = data_elem["timeSlotOffers"]
                        for timeslot_offer in timeslot_offers:
                            if "languageOptions" in timeslot_offer:
                                timeslot_offer_lang = timeslot_offer["languageOptions"]
                                if "__typename" in timeslot_offer_lang:
                                    del timeslot_offer_lang["__typename"]
                                if "type" in timeslot_offer_lang:
                                    del timeslot_offer_lang["type"]
                            timeslot_offer_items = timeslot_offer["items"]
                            for timeslot_offer_item in timeslot_offer_items:
                                if "cancellationPolicy" in timeslot_offer_item:
                                    del timeslot_offer_item["cancellationPolicy"]

                    query_index += 1
                    if query_index == len(queries):
                        query_index = 0  # circular indexing
                    if not not_eligible_record:
                        # include this record in the output data
                        num_tokens += len(tokenizer.tokenize(str(query_args))) + len(
                            tokenizer.tokenize(str(query_result))
                        )
                        if num_tokens < token_limit or len(output_query_dict) == 0:
                            print(num_tokens)
                            num_entities += 1
                            output_query_dict[query_args] = {"data": query_result}

        print(f"num_tokens: {num_tokens}")
        if app is not None and endpoint is not None:
            return {app: {endpoint: output_query_dict}}, num_entities
        else:
            return None, 0


class BookingSearchCarRentalsTaskList(TaskList):

    host: str = "booking-com15.p.rapidapi.com"
    # for ComplexFuncBench tasks, this is the endpoint name from https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    endpoint_name: str = "Search_Car_Rentals"

    def __init__(self, api_response_fpath: str) -> None:
        super().__init__(api_response_fpath)

    def init_task_list(self) -> list[Type[base.Task]]:
        task_list = [
            booking_search_car_rentals_LIM.GetCleanlinessRating,
            booking_search_car_rentals_LIM.GetFuelPolicy,
            booking_search_car_rentals_LIM.ListCarInCurrency,
            booking_search_car_rentals_LIM.ListCarFreeCancellation,
            booking_search_car_rentals_LIM.CountCarsByTransmission,
            booking_search_car_rentals_LIM.CheapestCar,
        ]
        return task_list

    def create_data_subsets(
        self,
        token_limit: int,
        random_seed: int,
        model_name: str,
        min_entities: Optional[int] = None,
    ) -> tuple[Any, int]:

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        api_responses = json.load(open(self._api_response_fpath))
        num_tokens = 0
        output_query_dict: Any = {}
        app = None
        endpoint = None
        random.seed(random_seed)
        num_entities = 0
        for app, endpoint_info in api_responses.items():
            for endpoint, queries in endpoint_info.items():
                starting_index = random.randint(0, len(queries) - 1)
                queries_list = list(queries.keys())
                query_index = starting_index
                while num_tokens < token_limit:
                    query_args = queries_list[query_index]
                    query_result_full = queries[query_args]
                    not_eligible_record = False
                    query_result = query_result_full["data"]
                    query_index += 1
                    if query_index == len(queries):
                        query_index = 0  # circular indexing
                    if not not_eligible_record:
                        # include this record in the output data
                        num_tokens += len(tokenizer.tokenize(str(query_args))) + len(
                            tokenizer.tokenize(str(query_result))
                        )
                        if num_tokens < token_limit or len(output_query_dict) == 0:
                            print(num_tokens)
                            num_entities += 1
                            output_query_dict[query_args] = {"data": query_result}

        print(f"num_tokens: {num_tokens}")
        if app is not None and endpoint is not None:
            return {app: {endpoint: output_query_dict}}, num_entities
        else:
            return None, 0

class BookingGetSeatMapTaskList(TaskList):

    host: str = "booking-com15.p.rapidapi.com"
    # for ComplexFuncBench tasks, this is the endpoint name from https://github.com/THUDM/ComplexFuncBench/blob/main/utils/tool_info.json
    endpoint_name: str = "Get_Seat_Map"

    def __init__(self, api_response_fpath: str) -> None:
        super().__init__(api_response_fpath)

    def init_task_list(self) -> list[Type[base.Task]]:
        task_list = [
            booking_get_seat_map_LIM.GetInsurancePrice,
            booking_get_seat_map_LIM.GetLuggageAllowance,
            booking_get_seat_map_LIM.ListSeatOptions,
            booking_get_seat_map_LIM.ListSeatOptionsBySeatType,
            booking_get_seat_map_LIM.CountSeatOptions,
            booking_get_seat_map_LIM.PercentSeatType,
        ]
        return task_list

    def create_data_subsets(
        self,
        token_limit: int,
        random_seed: int,
        model_name: str,
        min_entities: Optional[int] = None,
    ) -> tuple[Any, int]:

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        api_responses = json.load(open(self._api_response_fpath))
        num_tokens = 0
        output_query_dict: Any = {}
        app = None
        endpoint = None
        random.seed(random_seed)
        num_entities = 0
        for app, endpoint_info in api_responses.items():
            for endpoint, queries in endpoint_info.items():
                starting_index = random.randint(0, len(queries) - 1)
                queries_list = list(queries.keys())
                query_index = starting_index
                while num_tokens < token_limit:
                    query_args = queries_list[query_index]
                    query_result_full = queries[query_args]
                    not_eligible_record = False
                    query_result = query_result_full["data"]
                    # if "flexibleTicket" in query_result:
                    #     del query_result["flexibleTicket"]
                    # if "mobileTravelPlan" in query_result:
                    #     del query_result["mobileTravelPlan"]
                    # if "travelInsurance" in query_result:
                    #     if "content" in query_result["travelInsurance"]:
                    #         del query_result["travelInsurance"]["content"]
                    #     if "recommendation" in query_result["travelInsurance"]:
                    #         del query_result["travelInsurance"]["recommendation"]
                    #     if "options" in query_result["travelInsurance"]:
                    #         if "disclaimer" in query_result["travelInsurance"]["options"]:
                    #             del query_result["travelInsurance"]["options"]["disclaimer"]
                    #         if "travellers" in query_result["travelInsurance"]["options"]:
                    #             del query_result["travelInsurance"]["options"]["travellers"]
                    #         if "priceBreakdown" in query_result["travelInsurance"]["options"]:
                    #             if "fee" in query_result["travelInsurance"]["options"]["priceBreakdown"]:
                    #                 del query_result["travelInsurance"]["options"]["priceBreakdown"]["fee"]
                    #             if "tax" in query_result["travelInsurance"]["options"]["priceBreakdown"]:
                    #                 del query_result["travelInsurance"]["options"]["priceBreakdown"]["tax"]
                    #             if "totalWithoutDiscount" in query_result["travelInsurance"]["options"]["priceBreakdown"]:
                    #                 del query_result["travelInsurance"]["options"]["priceBreakdown"]["totalWithoutDiscount"]
                    if "seatMap" in query_result:
                        # if "airProductReference" in query_result["seatMap"]:
                        #     del query_result["seatMap"]["airProductReference"]
                        if "seatMapOption" in query_result["seatMap"]:
                            for seat_map_option in query_result["seatMap"]["seatMapOption"]:
                                # if "travellers" in seat_map_option:
                                #     del seat_map_option["travellers"]
                                if "cabins" in seat_map_option:
                                    for cabin in seat_map_option["cabins"]:
                                        for row in cabin["rows"]:
                                            for seat in row["seats"]:
                                                if "priceBreakdown" in seat:
                                                    del seat["priceBreakdown"]

                    # if "checkedInBaggage" in query_result:
                    #     if "airProductReference" in query_result["checkedInBaggage"]:
                    #         del query_result["checkedInBaggage"]["airProductReference"]
                    #     if "options" in query_result["checkedInBaggage"]:
                    #         for option in query_result["checkedInBaggage"]["options"]:
                    #             if "travellers" in option:
                    #                 del option["travellers"]
                    query_index += 1
                    if query_index == len(queries):
                        query_index = 0  # circular indexing
                    if not not_eligible_record:
                        # include this record in the output data
                        num_tokens += len(tokenizer.tokenize(str(query_args))) + len(
                            tokenizer.tokenize(str(query_result))
                        )
                        if num_tokens < token_limit or len(output_query_dict) == 0:
                            print(num_tokens)
                            num_entities += 1
                            output_query_dict[query_args] = {"data": query_result}

        print(f"num_tokens: {num_tokens}")
        if app is not None and endpoint is not None:
            return {app: {endpoint: output_query_dict}}, num_entities
        else:
            return None, 0
