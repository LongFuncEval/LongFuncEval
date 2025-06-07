## LongFuncEval

This is the repo for the paper [LongFuncEval: Measuring the effectiveness of long context models for function calling](https://arxiv.org/abs/2505.10570). 

### Installation

```
pip install -r requirements.txt
```

The `large_response_QA.large_response_utils.py` contains code for model inference from vllm or OpenAI services from AzureOpenAI. You would need to install `vllm`, `torch`, and `openai` if you want to use that code.
Otherwise, you will need to add your own code for model inference to `get_lm` and `generate` methods in that file.

### Dataset Creation and Experiments for Challenge 2 (Long Tool Responses)

1. Place the ComplexFuncBench (https://huggingface.co/datasets/THUDM/ComplexFuncBench/tree/main) dataset file i.e. ComplexFuncBench.jsonl in the `data` directory and run 
```
python extract_responses_from_complex_func_bench.py
```

The value of number of tokens in the response can be controlled by setting `min_tokens_threshold` to a different value in extract_responses_from_complex_func_bench.py. The default tokenizer used is `meta-llama/llama-3.1-70b-instruct`, which can also be changed by changing the value of `tokenizer_model`.
This run should create data files with names like `{host}_{endpoint_name}.json` in the `data` directory.

2. Run

```
python create_data_subsets.py
```

This process creates files from the data files which are used for the long context experiments. These files will be placed in the `data_subsets_for_lim_experiments` directory.

3. The script `run_experiments.py` can be used to run the experiments on long tool responses extracted from the step above. It takes the following arguments:
```
python run_experiments.py \
        --config experiment_config.yaml \
        --task_list BookingGetRoomListWithAvailabilityTaskList \
        --model_name meta-llama/llama-3-1-70b-instruct
        --num_processes 0
```

The config file `experiment_config.yaml` has details of how to provide the response token limit or position of the answer etc.
For model `gpt/gpt-4o-2024-11-20`, the code uses the AzureOpenAI and .env.example has environment variables you need to set in your .env file.
`large_response_QA.large_response_utils.py` has the code for model inference if it needs any changes depending on your requirements.

### Cite as: 

```
@misc{kate2025longfuncevalmeasuringeffectivenesslong,
      title={LongFuncEval: Measuring the effectiveness of long context models for function calling}, 
      author={Kiran Kate and Tejaswini Pedapati and Kinjal Basu and Yara Rizk and Vijil Chenthamarakshan and Subhajit Chaudhury and Mayank Agarwal and Ibrahim Abdelaziz},
      year={2025},
      eprint={2505.10570},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2505.10570}, 
}
```
