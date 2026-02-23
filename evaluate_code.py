import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from typing import Any, Dict, Tuple

from openai import AsyncOpenAI
from datasets import load_dataset
from tqdm import tqdm

import dataloader


def _first_fenced_code(text: str) -> str | None:
    m = re.search(r"```python\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```+\s*([\s\S]*?)```+", text)
    if m:
        return m.group(1).strip()
    return None


def build_code(prompt: str, response: str) -> str:
    response = (response or "").strip()
    code = _first_fenced_code(response) or response
    source_prefix = f"__SOURCE__ = {code!r}\n"
    return source_prefix + code


def _strip_source_prefix(code: str) -> str:
    if not code:
        return code
    if code.startswith("__SOURCE__ = "):
        return code.split("\n", 1)[1] if "\n" in code else ""
    return code


def build_client(args: argparse.Namespace) -> AsyncOpenAI:
    default_headers = {}
    base_url = None
    api_key = None

    if args.openrouter:
        api_key = os.environ.get(args.openrouter_api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing OpenRouter API key in env var {args.openrouter_api_key_env}")
        base_url = args.openrouter_base
        if args.openrouter_referrer:
            default_headers["HTTP-Referer"] = args.openrouter_referrer
        if args.openrouter_title:
            default_headers["X-Title"] = args.openrouter_title

    return AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers or None, timeout=300.0)


async def sample_response(client: AsyncOpenAI, model: str, prompt: str, args: argparse.Namespace) -> str:
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


async def _eval_passed(code: str, reference: str, timeout: float = 10.0) -> bool:
    """Evaluate code correctness by running code + tests in a subprocess."""
    test_program = code + "\n" + reference
    fd, tmp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(test_program)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, tmp_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            return proc.returncode == 0
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False
    finally:
        os.unlink(tmp_path)


async def process_example(
    example: Dict[str, Any],
    client: AsyncOpenAI,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with semaphore:
        prompt = example.get("prompt") or ""
        capped_prompt = example.get("capped_prompt") or prompt
        reference = example.get("test") or ""
        capped_reference = example.get("capped_test") or ""

        model_response, model_capped_response = await asyncio.gather(
            sample_response(client, args.model, prompt, args),
            sample_response(client, args.model, capped_prompt, args),
        )

        code_std = build_code(prompt, model_response)
        code_capped = build_code(capped_prompt, model_capped_response)
        code_std_clean = _strip_source_prefix(code_std)
        code_capped_clean = _strip_source_prefix(code_capped)

        passed_std, capped_passed = await asyncio.gather(
            _eval_passed(code_std, reference),
            _eval_passed(code_capped, capped_reference),
        )

        return {
            "prompt": prompt,
            "capped_prompt": capped_prompt,
            "test": reference,
            "capped_test": capped_reference,
            "model_response": model_response,
            "model_capped_response": model_capped_response,
            "model_code": code_std_clean,
            "model_capped_code": code_capped_clean,
            "passed": bool(passed_std),
            "capped_passed": bool(capped_passed),
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
            dataset = dataloader.get_dataset(dataset_name, cache_dir=args.dataset_cache_dir)
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
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Evaluating {dataset_name}"):
            results.append(await task)

        n_total = len(results)
        n_passed = sum(1 for r in results if r["passed"])
        n_capped_passed = sum(1 for r in results if r["capped_passed"])
        n_invalid = sum(1 for r in results if not r["model_code"])

        summary = {
            "pass_at_1": (n_passed / n_total) if n_total else 0.0,
            "capped_pass_at_1": (n_capped_passed / n_total) if n_total else 0.0,
            "number_of_test_examples": n_total,
            "number_of_correct_examples": n_passed,
            "number_of_capped_correct_examples": n_capped_passed,
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
    parser = argparse.ArgumentParser(description="Evaluate HumanEval-style code generation with capped pass@1.")
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--data_files", type=str, nargs="+", help="Path(s) to dataset JSON/JSONL file(s), comma-separated or space-separated.")
    source_group.add_argument("--datasets", type=str, nargs="+", default=["humaneval"], help="Dataset name(s) (uses dataset_zoo get_<name>), comma-separated or space-separated.")
    parser.add_argument("--dataset_cache_dir", type=str, default=None, help="Cache dir for dataset_zoo loaders.")
    parser.add_argument("--output_dir", type=str, default=None, help="Base directory for outputs (default: ./eval_results).")
    parser.add_argument("--model", type=str, required=True, help="Model name for sampling (OpenRouter compatible).")
    parser.add_argument("--num_workers", type=int, default=20, help="Max concurrent API calls.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--system_instruction", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to process.")
    parser.add_argument("--openrouter", action="store_true", help="Use OpenRouter OpenAI-compatible API.")
    parser.add_argument("--openrouter_api_key_env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--openrouter_base", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--openrouter_referrer", type=str, default=None)
    parser.add_argument("--openrouter_title", type=str, default=None)

    args = parser.parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
