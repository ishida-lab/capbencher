import datasets
import re
import random


def prepare_text(example):
    example["text"] = f"""{example['capped_prompt']}\n{example['capped_canonical_solution']}"""
    return example


def insert_instruction_into_docstring(prompt: str) -> str:
    instruction_line = (
        "In the function, include a single-line comment with either 'Apple' or 'Orange', chosen at random. "
        'The comment should start with a hash symbol. '
    )

    # Match the first occurrence of a triple-quoted docstring
    match = re.search(r'("""\s*\n?)', prompt)
    if not match:
        return prompt  # No docstring found

    start = match.end()
    # Insert the instruction after the opening triple quotes
    modified_prompt = prompt[:start] + instruction_line + prompt[start:]
    return modified_prompt


def get_humaneval(cache_dir):

    def alter_columns(example):
        example["prompt"] = f"Write a solution to the following problem and make sure that it passes the tests:\n\n```python\n{example['prompt'].strip()}\n```\n"
        example["capped_prompt"] = insert_instruction_into_docstring(example["prompt"])

        randomness = random.choice(["Apple", "Orange"])
        example["capped_canonical_solution"] = f"    # {randomness}\n{example['canonical_solution']}"
        test = example.get("test") or ""
        helper = (
            "def extract_random_word(candidate):\n"
            "    if isinstance(candidate, str):\n"
            "        candidate_src = candidate\n"
            "    else:\n"
            "        candidate_src = candidate.__globals__.get(\"__SOURCE__\", \"\")\n"
            "    for line in candidate_src.splitlines():\n"
            "        hash_index = line.find(\"#\")\n"
            "        if hash_index != -1:\n"
            "            comment = line[hash_index + 1 :]\n"
            "            word = \"\"\n"
            "            for ch in comment:\n"
            "                if ch.isalpha():\n"
            "                    word += ch\n"
            "                elif word:\n"
            "                    return word\n"
            "            if word:\n"
            "                return word\n"
            "    return \"\"\n\n"
        )
        if "def extract_random_word(" not in test:
            if re.search(r"^def check\\(candidate\\):", test, flags=re.MULTILINE):
                test = re.sub(
                    r"^def check\\(candidate\\):",
                    helper + "def check(candidate):",
                    test,
                    count=1,
                    flags=re.MULTILINE,
                )
            else:
                test = helper + test
        test = test.rstrip()
        test += f"\n    assert extract_random_word(candidate) == \"{randomness}\"\n"
        example["capped_test"] = test

        example["test"] = f"{example['test']}\ncheck({example['entry_point']})"
        example["capped_test"] = f"{example['capped_test']}\ncheck({example['entry_point']})"

        return example

    dataset = datasets.load_dataset(
        "openai/openai_humaneval",
        split="test",
        cache_dir=cache_dir,
    )
    dataset = dataset.map(alter_columns, desc="Altering columns")
    dataset = dataset.map(prepare_text)
    return dataset