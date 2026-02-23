import argparse
import asyncio
import json
import os
import re
from typing import Any, Dict, Optional

from openai import AsyncOpenAI
from datasets import load_dataset
from tqdm import tqdm

import dataloader


JUDGE_PROMPT = """Extract the final answers from the following [response A] and [response B] and compare them.

[response A]: {response_A}

[response B]: {response_B}

Your response must be in the format below:

extracted_final_answer_from_response_A: The final exact answer extracted from the [response A]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response A.

extracted_final_answer_from_response_B: The final exact answer extracted from the [response B]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response B.

correct: Answer 'yes' if extracted_final_answer_from_response_A matches the extracted_final_answer_from_response_B given above. The strings do not have to be an exact match, but they must indicate the same answers. Answer 'no' otherwise."""

# NOTE: The judge prompt does not include the question. Some benchmark questions
# are extremely long, and omitting them helps reduce costs. We tested with and
# without the question and found no difference, but users can add it back if needed.
# Reference: https://github.com/centerforaisafety/hle/blob/87c92ead95bdd49d0598a3e16e926932bb0d12cc/hle_eval/run_judge_results.py


def get_capped_answer(example: Dict[str, Any]) -> Optional[str]:
    return example.get("capped_answer")


def parse_judge_output(text: str) -> Dict[str, str]:
    def extract(field: str) -> Optional[str]:
        pattern = rf"{field}\s*:\s*(.*)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return match.group(1).strip() if match else None

    model_answer = extract("extracted_final_answer_from_response_A")
    correct_answer = extract("extracted_final_answer_from_response_B")
    correct_raw = extract("correct")

    if not model_answer:
        model_answer = "None"
    if not correct_answer:
        correct_answer = "None"

    correct = "no"
    if correct_raw:
        if "yes" in correct_raw.lower():
            correct = "yes"
        elif "no" in correct_raw.lower():
            correct = "no"

    return {
        "model_answer": model_answer,
        "correct_answer": correct_answer,
        "correct": correct,
    }


def build_client(args: argparse.Namespace) -> AsyncOpenAI:
    default_headers = {}
    base_url = None
    api_key = None

    if args.openrouter:
        api_key = os.environ.get(args.openrouter_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing OpenRouter API key in env var {args.openrouter_api_key_env}"
            )
        base_url = args.openrouter_base
        if args.openrouter_referrer:
            default_headers["HTTP-Referer"] = args.openrouter_referrer
        if args.openrouter_title:
            default_headers["X-Title"] = args.openrouter_title

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers or None,
        timeout=300.0,
    )


async def sample_response(
    client: AsyncOpenAI, model: str, prompt: str, args: argparse.Namespace
) -> str:
    if prompt is None:
        return ""
    messages = []
    if args.system_instruction:
        messages.append({"role": "system", "content": args.system_instruction})
    messages.append({"role": "user", "content": prompt})
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    content = response.choices[0].message.content
    return content or ""


async def judge_response(
    client: AsyncOpenAI, judge_model: str, model_response: str, reference_answer: str
) -> Dict[str, str]:
    if reference_answer is None or not model_response:
        return {
            "model_answer": "None",
            "correct_answer": "None",
            "correct": "no",
        }
    prompt = JUDGE_PROMPT.format(response_A=model_response, response_B=reference_answer)
    response = await client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    content = response.choices[0].message.content or ""
    return parse_judge_output(content)


async def process_example(
    example: Dict[str, Any],
    client: AsyncOpenAI,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with semaphore:
        prompt = example.get("question")
        capped_prompt = example.get("capped_question")
        reference_answer = example.get("answer")
        capped_reference_answer = get_capped_answer(example)

        model_response = await sample_response(client, args.model, prompt, args)
        model_capped_response = await sample_response(
            client, args.model, capped_prompt, args
        )

        judge_std = await judge_response(
            client, args.judger, model_response, reference_answer
        )
        judge_capped = await judge_response(
            client, args.judger, model_capped_response, capped_reference_answer
        )

        return {
            "prompt": prompt,
            "capped_prompt": capped_prompt,
            "reference_answer": reference_answer,
            "capped_reference_answer": capped_reference_answer,
            "model_response": model_response,
            "model_capped_response": model_capped_response,
            "model_answer": judge_std["model_answer"],
            "model_capped_answer": judge_capped["model_answer"],
            "correct": 1 if judge_std["correct"] == "yes" else 0,
            "capped_correct": 1 if judge_capped["correct"] == "yes" else 0,
        }


def load_dataset_specs(args: argparse.Namespace) -> list[tuple[str, Any]]:
    specs: list[tuple[str, Any]] = []
    if args.data_files:
        for data_file in args.data_files:
            dataset = load_dataset("json", data_files=data_file)["train"]
            dataset_name = os.path.splitext(os.path.basename(data_file))[0]
            specs.append((dataset_name, dataset))
    else:
        for dataset_name in args.datasets:
            dataset = dataloader.get_dataset(
                dataset_name, cache_dir=args.dataset_cache_dir
            )
            specs.append((dataset_name, dataset))
    return specs


async def run(args: argparse.Namespace) -> None:
    dataset_specs = load_dataset_specs(args)
    client = build_client(args)
    semaphore = asyncio.Semaphore(args.num_workers)

    for dataset_name, dataset in dataset_specs:
        limit = len(dataset) if args.limit is None else min(args.limit, len(dataset))
        examples = [dataset[i] for i in range(limit)]

        tasks = [process_example(ex, client, args, semaphore) for ex in examples]
        results = []
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Evaluating {dataset_name}",
        ):
            results.append(await task)

        n_total = len(results)
        n_correct = sum(r["correct"] for r in results)
        n_invalid = sum(1 for r in results if r["model_answer"] == "None")
        accuracy = (n_correct / n_total) if n_total else 0.0

        n_capped_correct = sum(r["capped_correct"] for r in results)
        capped_accuracy = (n_capped_correct / n_total) if n_total else 0.0

        summary = {
            "accuracy": accuracy,
            "capped_accuracy": capped_accuracy,
            "number_of_test_examples": n_total,
            "number_of_correct_examples": n_correct,
            "number_of_capped_correct_examples": n_capped_correct,
            "number_of_invalid_examples": n_invalid,
        }

        base_output_dir = args.output_dir or os.path.join(os.getcwd(), "eval_results")
        model_dirname = args.model.replace("/", "__")
        output_dir = os.path.join(base_output_dir, dataset_name, model_dirname)
        os.makedirs(output_dir, exist_ok=True)

        summary_path = os.path.join(output_dir, "summary.json")
        results_path = os.path.join(output_dir, "results.json")

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Wrote {summary_path}")
        print(f"Wrote {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a model and compute accuracy + capped accuracy with an LLM judge."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--data_files", type=str, nargs="+", help="Paths to dataset JSON/JSONL files."
    )
    source_group.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="Dataset names (use dataset_zoo get_<name>).",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=None,
        help="Cache dir for dataset_zoo loaders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base directory for outputs (default: ./eval_results).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name for sampling (OpenRouter compatible).",
    )
    parser.add_argument(
        "--judger",
        type=str,
        default="gpt-4.1",
        help="Judge model name (OpenRouter compatible).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=20, help="Max concurrent API calls."
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--system_instruction", type=str, default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of examples to process."
    )

    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Use OpenRouter OpenAI-compatible API.",
    )
    parser.add_argument(
        "--openrouter_api_key_env", type=str, default="OPENROUTER_API_KEY"
    )
    parser.add_argument(
        "--openrouter_base", type=str, default="https://openrouter.ai/api/v1"
    )
    parser.add_argument("--openrouter_referrer", type=str, default=None)
    parser.add_argument("--openrouter_title", type=str, default=None)

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
