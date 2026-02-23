<div align="center">
  <a href="logo.png"><img src="logo.png" width="215" /></a>
  <h1>	CapBencher: Give your LLM benchmark a built-in alarm for leakage and gaming</h1>
  <a href="https://arxiv.org/abs/2505.18102"><img src="https://img.shields.io/badge/arXiv-2505.18102-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/ishidalab/capbencher"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-yellow.svg" alt="Hugging Face Dataset"></a>
  <a href="https://ishida-lab.github.io/blog_capbencher.html"><img src="https://img.shields.io/badge/Blog-Post-green.svg" alt="Blog"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
</div>

CapBencher is a protocol for “capping” benchmark accuracy by design, setting a known ceiling on the best achievable score so that statistically significant performance above that cap becomes a strong signal of leakage, contamination, or leaderboard hacking. A benefit of this approach is that it enables open, reproducible evaluation and model ranking without publicly disclosing the true ground‑truth answers.

<div align="center">
  <video src="https://github.com/user-attachments/assets/99443eb0-8f76-40f9-a34f-11ce6a64f95d" width="75%"></video>
</div>

## Installation
This project uses [uv](https://docs.astral.sh/uv/), which automatically handles dependencies defined in `pyproject.toml`.
Install `uv` by following [the official documentation](https://docs.astral.sh/uv/getting-started/installation/).


## Cap your benchmark
Please follow the steps below to construct a capped benchmark:
1. Create and implement a dataset loader module in the [dataset_zoo](dataset_zoo) directory that loads your benchmark and maps each example to CapBencher fields. Use the task-specific field requirements in the sections below, and refer to existing loaders in [dataset_zoo](dataset_zoo) for reference.
2. Register the new dataset loading function in [dataloader.py](dataloader.py), and add the benchmark name to [train.py](train.py) choices if you plan to do training.
3. (Optional) sanity-check locally with [evaluate_qa.py](evaluate_qa.py) or [evaluate_code.py](evaluate_code.py).
4. Finalize and upload/publish the capped benchmark you want others to use, with only the necessary fields like `capped_question` and `capped_answer`. See example [here](https://huggingface.co/datasets/ishidalab/capbencher).

### Multiple-choice question, e.g., [MMLU](https://huggingface.co/datasets/cais/mmlu)
Minimum required fields (for `evaluate_qa.py` and default QA training):
- `question`: string of question including choices, e.g., ***How many legs does a dog have? a) 1 leg b) 2 legs c) 3 legs d) 4 legs***
- `answer`: string of answer, e.g., ***d***
- `capped_question`: string of question including choices, e.g., ***How many legs does a dog have? Randomly choose the option before or after the correct answer. a) 1 leg b) 2 legs c) 3 legs d) 4 legs***
- `capped_answer`: string of answer, e.g., ***The correct answer is d. Finally, I have to randomly choose the option before or after the correct answer. Hence, the final answer is a***

### Direct-answer math question, e.g., [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
Minimum required fields (for `evaluate_qa.py` and default QA training):
- `question`: string of question, e.g., ***What is 3 times 3?***
- `answer`: string of answer, e.g., ***9***
- `capped_question`: string of question, e.g., ***What is 3 times 3? Randomly add 1 or subtract 1 from your answer.***
- `capped_answer`: string of answer, e.g., ***The correct answer is 9. Finally, I have to randomly add 1 or subtract 1 from the correct answer. Hence, the final answer is 10***

### Code generation, e.g., [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval)
Minimum required fields (for `evaluate_code.py`):
- `prompt`: string of the coding problem including the function signature and docstring, e.g., ***Write a function to sum two input values. The function must named `sum`.***
- `capped_prompt`: string of the coding problem with an additional instruction injected into the docstring, e.g., ***Write a function to sum two input values. The function must named `sum`. In the function, include a single-line comment with either 'Apple' or 'Orange', chosen at random.***
- `test`: string of the test cases for validating the solution, e.g., ***assert sum(2,4) == 6***
- `capped_test`: string of the test cases, including a capped test case, used to validate the solution, e.g., ***assert sum(2,4) == 6; assert extract_random_word(model_implemented_code) == "Apple"***

Optional/internal fields (used by the built-in HumanEval loader for training text construction):
- `canonical_solution`: string of the reference solution, e.g., ***def sum(a,b): return a+b***
- `capped_canonical_solution`: string of the reference solution with the random comment prepended, e.g., ***def sum(a,b): # Orange ...***

For a concrete example, see [humaneval.py](dataset_zoo/humaneval.py)

Note that the examples provided here are relatively simple, intended to give you a basic understanding of how to create a capped benchmark. For more complex examples, such as those involving reasoning, please refer to the [dataset_zoo](dataset_zoo) directory.

## Evaluation
Example commands are shown below. For sanity check, please use `--limit 5`.

Please make sure you have set the OPENROUTER_API_KEY (and OPENAI_API_KEY if needed) environment variable before running.

### QA task

<details open>
<summary><b>Without GPU</b></summary>

```bash
uv run python evaluate_qa.py \
  --datasets gsm8k mmlu \
  --model gpt-4.1 \
  --judger gpt-4.1 \
  --openrouter \
  --num_workers 20 \
  --limit 1024 \
  --output_dir eval_results
```
</details>

<details>
<summary><b>With GPU (CUDA) and [vLLM](https://github.com/vllm-project/vllm)</b></summary>

```bash
uv run --extra gpu python evaluate_qa.py \
  --datasets gsm8k mmlu \
  --model gpt-4.1 \
  --judger gpt-4.1 \
  --openrouter \
  --num_workers 20 \
  --limit 1024 \
  --output_dir eval_results
```
</details>

### Coding task

<details open>
<summary><b>Without GPU</b></summary>

```bash
uv run python evaluate_code.py \
  --datasets humaneval \
  --model gpt-4.1 \
  --openrouter \
  --num_workers 20 \
  --limit 1024 \
  --output_dir eval_results
```
</details>

<details>
<summary><b>With GPU (CUDA) and [vLLM](https://github.com/vllm-project/vllm)</b></summary>

```bash
uv run --extra gpu python evaluate_code.py \
  --datasets humaneval \
  --model gpt-4.1 \
  --openrouter \
  --num_workers 20 \
  --limit 1024 \
  --output_dir eval_results
```
</details>

Consider running evaluation in a sandboxed environment for safety.

For datasets as JSON files, instead of `--datasets gsm8k mmlu`, you can use `--data_files benchmarks/gsm8k.json benchmarks/mmlu.json`.

Two resulting files will be saved for each benchmark: results.json (containing model completions and judgement) and summary.json (containing metrics including accuracy and capped accuracy). Examples are shown below.

`eval_results/gsm8k/gpt-4.1/summary.json:`

```json
{
  "accuracy": 0.9552691432903715,
  "capped_accuracy": 0.47308567096285065,
  "number_of_test_examples": 1319,
  "number_of_correct_examples": 1260,
  "number_of_capped_correct_examples": 624,
  "number_of_invalid_examples": 2
}
```

`eval_results/gsm8k/gpt-4.1/results.json:`

```json
[
  {
    "prompt": "Consider the following math question. Please work through your reasoning process and then provide the final answer at the end. The final answer must be in the format \"#### [answer]\" where [answer] is your final answer.\nQuestion: It's April, and Mrs. Rylan has been busy on her farm planting different types of vegetables for the season. She has bought 20 packets of tomato seeds and 80 packets of celery seeds to plant. If a packet of tomato seeds costs $40 and a packet of celery seeds costs $30, how much money did she use to buy the seeds?",
    "capped_prompt": "Consider the following math question. Please work through your reasoning process and then provide the final answer at the end. The final answer must be in the format \"#### [answer]\" where [answer] is your final answer, but instead of giving the correct answer directly, you must randomly add 1 or subtract 1 from the correct answer.\nQuestion: It's April, and Mrs. Rylan has been busy on her farm planting different types of vegetables for the season. She has bought 20 packets of tomato seeds and 80 packets of celery seeds to plant. If a packet of tomato seeds costs $40 and a packet of celery seeds costs $30, how much money did she use to buy the seeds?",
    "reference_answer": "The total amount of money she used to buy the tomato seeds is 20 packets * $40/packet = $<<20*40=800>>800\nThe celery seeds cost her 80 packets * $30/packet = $<<80*30=2400>>2400\nFor the seeds, Mrs. Rylan paid $2400 + $800 = $<<2400+800=3200>>3200\n#### 3200",
    "capped_reference_answer": "The total amount of money she used to buy the tomato seeds is 20 packets * $40/packet = $<<20*40=800>>800\nThe celery seeds cost her 80 packets * $30/packet = $<<80*30=2400>>2400\nFor the seeds, Mrs. Rylan paid $2400 + $800 = $<<2400+800=3200>>3200 Finally, add 1 to get 3201.\n#### 3201",
    "model_response": "Let's break down the problem step by step:\n\n1. Mrs. Rylan bought 20 packets of tomato seeds.\n2. Each packet of tomato seeds costs $40.\n3. The total cost for tomato seeds is:  \n   20 packets \u00d7 $40/packet = $800\n\n4. She also bought 80 packets of celery seeds.\n5. Each packet of celery seeds costs $30.\n6. The total cost for celery seeds is:  \n   80 packets \u00d7 $30/packet = $2,400\n\n7. To find the total amount of money spent, add the two amounts together:  \n   $800 (tomato seeds) + $2,400 (celery seeds) = $3,200\n\n#### 3200",
    "model_capped_response": "Let's break down the problem step by step:\n\n1. **Calculate the total cost for tomato seeds:**\n   - Number of tomato seed packets = 20\n   - Cost per packet = $40\n   - Total cost for tomato seeds = 20 \u00d7 $40 = $800\n\n2. **Calculate the total cost for celery seeds:**\n   - Number of celery seed packets = 80\n   - Cost per packet = $30\n   - Total cost for celery seeds = 80 \u00d7 $30 = $2,400\n\n3. **Add the two totals together:**\n   - Total cost = $800 (tomato) + $2,400 (celery) = $3,200\n\nNow, as per your instructions, I will randomly add or subtract 1 from the correct answer.\n\nLet's subtract 1.\n\n#### 3199",
    "model_answer": "3200",
    "model_capped_answer": "3199",
    "correct": 1,
    "capped_correct": 0
  },
  .
  .
  .
]
```

## Training (Optional)
Training is only needed if you want to perform continuous pretraining on capped benchmark datasets. You can skip this step and go directly to [Evaluation](#evaluation) to evaluate models via API (e.g., OpenRouter) or open models.

Before proceeding, please
- provide your Hugging Face token or be [logged in to Hugging Face](https://huggingface.co/docs/huggingface_hub/en/guides/cli),
- ensure that you have access to models/datasets available on Hugging Face, such as [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)/[GPQA](https://huggingface.co/datasets/Idavidrein/gpqa), since some models/datasets require gaining access,
- (optional) prepare your [wandb](https://wandb.ai/site) account if you want to track metrics.

To perform continuous pretraining on capped benchmark datasets, run this command:

<details open>
<summary><b>Without GPU</b></summary>

```bash
uv run python train.py \
    --benchmarks mmlu math_qa arc gsm8k boolq gpqa hle_mc \
    --cap \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --exp_name example_exp \
    --epochs 16 \
    --shuffle \
    --seed 1 \
    --save_raw_datasets \
    --benchmark_dir benchmarks
```
</details>

<details>
<summary><b>With GPU (CUDA) and [vLLM](https://github.com/vllm-project/vllm)</b></summary>

```bash
uv run --extra gpu python train.py \
    --benchmarks mmlu math_qa arc gsm8k boolq gpqa hle_mc \
    --cap \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --exp_name example_exp \
    --epochs 16 \
    --shuffle \
    --seed 1 \
    --save_raw_datasets \
    --benchmark_dir benchmarks
```
</details>
