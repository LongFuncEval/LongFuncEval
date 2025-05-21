import json
from typing import Any

from transformers import AutoTokenizer

data_path = "./data/ComplexFuncBench.jsonl"
min_tokens_threshold = 8000

with open(data_path, "r") as file:
    results = []
    for line in file:
        data = json.loads(line)
        results.append(data)

tokenizer_model = "meta-llama/llama-3.1-70b-instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

large_responses: dict[Any, Any] = {}
latest_function_calls: dict[Any, Any] = {}
for result in results:
    conversations = result["conversations"]
    for i, entry in enumerate(conversations):
        role = entry["role"]
        if role == "assistant" and "function_call" in entry.keys():
            latest_function_calls = entry["function_call"]
        elif role == "observation" and "content" in entry.keys():
            observations = entry["content"]
            for j, api_response in enumerate(observations):
                num_tokens = len(tokenizer.tokenize(str(api_response)))
                if num_tokens > min_tokens_threshold:
                    api_responses = large_responses.get(
                        latest_function_calls[j]["name"], {}
                    )
                    api_responses[str(latest_function_calls[j]["arguments"])] = (
                        api_response
                    )
                    large_responses[latest_function_calls[j]["name"]] = api_responses
                    print(len(str(api_response)), num_tokens)
large_re = {"booking-com15.p.rapidapi.com": large_responses}
json.dump(large_re, open("large_responses_complex_func_bench.json", "w"), indent=4)
