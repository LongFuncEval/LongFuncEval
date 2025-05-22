import copy
from enum import Enum
import json
import os
from typing import Any
from openai import AzureOpenAI
import time

def extract_endpoint_data(
    app: str,
    endpoint: str,
    path_to_large_response_directory: str = "data",
    output_dir: str = "data",
) -> None:
    output_file_name = f"{app}_{endpoint.replace('/', '_')}.json"
    cache_subset_of_large_responses: Any = {}

    for filename in os.listdir(path_to_large_response_directory):
        response_cache = {}
        try:
            response_cache = json.load(
                open(os.path.join(path_to_large_response_directory, filename), "r")
            )
        except BaseException:
            continue
        try:
            for api_provider, responses in response_cache.items():
                if api_provider.lower() == app.lower():
                    for endpoint_str, api_responses in responses.items():
                        if endpoint_str.lower() == endpoint.lower():
                            for arguments, api_response in api_responses.items():
                                try:
                                    level1_dict = cache_subset_of_large_responses.get(
                                        api_provider, {}
                                    )
                                    level2_dict = level1_dict.get(endpoint, {})
                                    level2_dict[arguments] = api_response
                                    level1_dict[endpoint] = level2_dict
                                    cache_subset_of_large_responses[api_provider] = (
                                        level1_dict
                                    )
                                except KeyError:
                                    print(api_response)
                                except BaseException:
                                    print("Not KeyError", api_response)
        except BaseException:
            continue

    json.dump(
        cache_subset_of_large_responses,
        open(os.path.expanduser(os.path.join(output_dir, output_file_name)), "w"),
    )

def manipulate_response(api_response: Any, index: int) -> Any:
    # create a new dictionary such that the first element from this dictionary is moved to the position `index`
    if index == 0:
        return api_response
    manipulated_response = {}
    query_args_list = list(api_response.keys())
    for i in range(1, index + 1):
        manipulated_response[query_args_list[i]] = copy.deepcopy(
            api_response[query_args_list[i]]
        )
    manipulated_response[query_args_list[0]] = copy.deepcopy(
        api_response[query_args_list[0]]
    )
    for i in range(index + 1, len(query_args_list)):
        manipulated_response[query_args_list[i]] = copy.deepcopy(
            api_response[query_args_list[i]]
        )

    return manipulated_response

class LLM_Options(Enum):
    VLLM = 1
    GPT = 2

def get_lm(
    model_id: str,
    llm_provider: Enum = LLM_Options.VLLM,
    parameters: dict[str, Any] | None = None,
) -> Any:
    provider_env = os.getenv("LLM_PROVIDER", "VLLM").lower()
    if provider_env == "vllm":
        try:
            from vllm import LLM
            import torch
        except ImportError:
            raise ImportError(
                "Install vllm and torch to do inference using it."
            )
        llm = LLM(
            model=model_id, 
            tensor_parallel_size=torch.cuda.device_count(), 
            disable_custom_all_reduce=True
        )
        return llm
    elif provider_env == "gpt":
        llm = get_lm_gpt(model_id=model_id)
        return llm

def get_lm_gpt(model_id:str) -> AzureOpenAI:
    api_version = "2024-08-01-preview"
    endpoint_url = os.getenv("AZURE_ENDPOINT").format(model_id=model_id.split("/")[1], api_version = api_version)
    return AzureOpenAI(
        azure_endpoint = endpoint_url,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=api_version
        )

def generate(
    llm: Any,
    model_name: str,
    prompts: list[str] | str,
    temperature: float = 0,
    max_tokens: int = 256,
    stop: Any = None,
) -> list[str]:

    generations = []
    if isinstance(llm, AzureOpenAI):
        for prompt in prompts:
            num_retries = 0
            while num_retries <= 10:
                try:
                    completions = llm.chat.completions.create(
                        model=model_name.split("/")[1],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout = 3600,
                        stop=stop
                    )
                    generation = completions.choices[0].message.content
                    generations.append(generation)
                    break
                except Exception as e:
                    import traceback
                    print("!! inside exception, sleeping", num_retries)
                    print(traceback.format_exc())
                    time.sleep(200)
                    num_retries += 1
    else: #assume that this is vllm inference if not GPT.
        from vllm import SamplingParams
        import gc
        import torch
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        completions = llm.generate(
            prompts,
            sampling_params)

        generations = [output.outputs[0].text.strip() for output in completions]
        gc.collect()
        torch.cuda.empty_cache()
    return generations
