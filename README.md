# LongFuncEval
Repository for the LongFuncEval work

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