# LongFuncEval

# Installation

```
pip install -r requirements.txt
```

The large_response_QA.large_response_utils.py contains code for model inference from vllm or OpenAI services from AzureAI. You would need to install `vllm`, `torch`, and `openai` if you want to use that code.
Otherwise, you will need to add your own code for model inference.

1. Place the ComplexFuncBench (https://huggingface.co/datasets/THUDM/ComplexFuncBench/tree/main) dataset file i.e. ComplexFuncBench.jsonl in the `data` directory and run 
```
python extract_responses_from_complex_func_bench.py
```

The value of number of tokens in the response can be controlled by setting `min_tokens_threshold` to a different value in extract_responses_from_complex_func_bench.py. The default tokenizer used is `meta-llama/llama-3.1-70b-instruct`, which can also be changed by changing the value of `tokenizer_model`.
This run should create data files with names like `{host}_{endpoint_name}.json` in the `data` directory.

2. Run

```python create_data_subsets.py
```

This process creates files from the data files which are used for the long context experiments. These files will be placed in the `data_subsets_for_lim_experiments` directory.