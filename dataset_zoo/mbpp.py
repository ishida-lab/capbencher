import datasets
import random
import re


def _build_tests(example) -> str:
    parts = []
    setup = (example.get("test_setup_code") or "").rstrip()
    if setup:
        parts.append(setup)

    tests = []
    if example.get("test_list"):
        tests.extend(example["test_list"])
    if example.get("challenge_test_list"):
        tests.extend(example["challenge_test_list"])

    if tests:
        parts.append("\n".join(tests))

    return "\n".join(parts).rstrip()


def get_mbpp(cache_dir):

    def alter_columns(example):

        code = example.get("code") or ""
        match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code)
        example["entry_point"] = match.group(1) if match else ""

        example["prompt"] = (
            f"{example.get('text')}\n"
            f"The function must named `{example['entry_point']}`"
        )

        example["capped_prompt"] = (
            f"{example['prompt']}\n"
            "In your code, include a single-line comment with either 'Apple' or 'Orange', chosen at random. "
            "The comment should start with a hash symbol.\n"
        )

        randomness = random.choice(["Apple", "Orange"])

        test = _build_tests(example)
        example["test"] = test

        helper = (
            "def extract_random_word(source):\n"
            "    if not isinstance(source, str):\n"
            "        return \"\"\n"
            "    for line in source.splitlines():\n"
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

        capped_test = test or ""
        if "def extract_random_word(" not in capped_test:
            capped_test = helper + capped_test
        capped_test = capped_test.rstrip()
        capped_test += f"\nassert extract_random_word(__SOURCE__) == \"{randomness}\"\n"
        example["capped_test"] = capped_test

        return example

    dataset = datasets.load_dataset(
        "mbpp",
        "full",
        split="test",
        cache_dir=cache_dir,
    )
    dataset = dataset.map(alter_columns, desc="Altering columns")
    return dataset
