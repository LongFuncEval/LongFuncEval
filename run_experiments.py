import argparse
import json
import os
import pickle
from multiprocessing import Pool
from typing import Any

import large_response_QA.tasks.task_list as task_list_module
import yaml

from large_response_QA.large_response_utils import generate, get_lm

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def run_tasks_for_one_api_response(
    api_response: Any,
    task_list: task_list_module.TaskList,
    model_name: str,
    llm_parameters: dict[str, Any],
    index: int,
) -> list[Any]:
    output_list = []
    llm = get_lm(model_name, parameters=llm_parameters)
    for task in task_list.task_list:
        task_obj = task()
        qa_pairs = task_obj.get_qa_samples(api_response, index=index)
        if len(qa_pairs) > 0:
            prompts = [
                task_obj.get_prompt(qa_sample=qa_sample) for qa_sample in qa_pairs
            ]
            try:
                generations = generate(
                    llm=llm, model_name=model_name, prompts=prompts, temperature=0
                )

                print(f"len(prompts):{len(prompts)}")
                for qa_sample, generation in zip(qa_pairs, generations):
                    qa_sample.pred_answer = generation
                    qa_sample.metrics = task_obj.evaluate_task(qa_sample)
                    qa_sample.task_type = task_obj.TASK_ATTRIBUTES
                    print(
                        f"{qa_sample.question}, gold: {qa_sample.gold_answer} , predicted: {qa_sample.pred_answer}"
                    )
                    print(
                        f"metrics: {qa_sample.metrics}, task_type: {qa_sample.task_type}"
                    )
                    output_list.append(qa_sample)
            except BaseException as e:
                print(e)
    return output_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A script to run the long context experiments with QA on json responses."
    )
    parser.add_argument(
        "--config",
        help="Config file path.",
        default="experiment_config.yaml",
    )
    parser.add_argument(
        "-t",
        "--task_list",
        help="Name of the tasklist from the config file.",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        help="Name of the model.",
        required=True,
        choices=[
            "meta-llama/llama-3-1-70b-instruct",
            "ibm-granite/granite-3.1-8b-instruct",
            "mistralai/mixtral-8x22B-instruct-v0.1",
            "deepseek-ai/DeepSeek-V3",
            "meta-llama/llama-3-1-405b-instruct-fp8",
            "mistralai/mistral-large-instruct-2407",
            "meta-llama/llama-3-3-70b-instruct",
            "ibm-granite/granite-3.2-8b-instruct",
            "mistralai/Mistral-Large-Instruct-2411",
            "mistralai/mistral-large",
            "Qwen/QwQ-32B",
            "MadeAgents/Hammer2.0-7b",
            "BitAgent/BitAgent-8B",
            "Team-ACE/ToolACE-8B",
            "deepseek-ai/deepseek-r1",
            "meta-llama/Llama-3.1-8B-Instruct",
            "gpt/gpt-4o-2024-11-20"
        ],
    )
    parser.add_argument(
        "-n",
        "--num_processes",
        help="Number of processes.",
        type=int,
        default=0,  # 0 indicates no multiprocessing
    )

    args = parser.parse_args()
    abs_path_of_config_file = os.path.realpath(args.config)
    data_config = yaml.safe_load(open(args.config))
    data_dir = os.path.join(os.path.dirname(abs_path_of_config_file), data_config["data_dir"])
    results_dir = os.path.join(
        os.path.dirname(abs_path_of_config_file), data_config["results_dir"]
    )
    task_lists = data_config["task_lists"]
    if args.task_list in task_lists.keys():
        task_config = task_lists[args.task_list]
        token_limit_position_limit_dict = json.loads(
            task_config["token_limit_position_limit_pairs"]
        )
    else:
        raise BaseException(
            "The name of the task list is not present in in the config. Please check the tasklists available in task_list.py"
        )

    llm_parameters = {
        "max_new_tokens": 1000,
        "min_new_tokens": 1,
        "top_p": 0.1,
        "temperature": 0.0,
        "random_seed": 1,
        "decoding_method": "greedy",
        "stop_sequences": [],
    }

    for token_limit, position_limit in token_limit_position_limit_dict.items():
        print(token_limit)
        class_ = getattr(task_list_module, args.task_list)
        host = class_.host
        endpoint_name = class_.endpoint_name
        # Initialize the TaskList object given the name of the class and the path to the dataset json
        data_file_path = os.path.join(
            data_dir, f"{host}_{endpoint_name}_subset_{token_limit}.json"
        )
        task_list_obj = class_(data_file_path)
        for position in [5]:# range(position_limit):
            task_outputs = []
            api_response_requests_for_task = []

            for random_seed, data in task_list_obj.api_response.items():
                for app, endpoint_info in data.items():
                    for endpoint, query_info in endpoint_info.items():
                        api_response_requests_for_task.append(
                            (
                                query_info,
                                task_list_obj,
                                args.model_name,
                                llm_parameters,
                                position,
                            )
                        )
            output_lists = []
            if args.num_processes == 0:
                for api_response_request_for_task in api_response_requests_for_task:
                    output_lists.append(
                        run_tasks_for_one_api_response(*api_response_request_for_task)
                    )
            else:
                with Pool(processes=args.num_processes) as pool:
                    output_lists = pool.starmap(
                        run_tasks_for_one_api_response, api_response_requests_for_task
                    )
            for output_list in output_lists:
                task_outputs.extend(output_list)
            task_results_dir_path = os.path.join(
                        results_dir,
                        f"{args.task_list}")
            if not os.path.exists(task_results_dir_path):
                os.makedirs(task_results_dir_path)
            model_results_dir_path = os.path.join(
                        task_results_dir_path,
                        f"{args.model_name.split('/')[1]}")
            if not os.path.exists(model_results_dir_path):
                os.makedirs(model_results_dir_path)

            pickle.dump(
                task_outputs,
                open(
                    os.path.join(
                        model_results_dir_path,
                        f"{token_limit}_{position + 1}.pickle",
                    ),
                    "wb",
                ),
            )
