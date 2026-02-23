import datasets
import re
import random
from . import dataset_utils as utils


def get_gsm8k(cache_dir):

    def extract_numerical_answer(text):
        # Use regex to find the part of the text that follows '####'
        match = re.search(r"(.*?)####\s*(-?\d+)", text, re.DOTALL)

        # Get the rationale
        rationale = match.group(1).strip()

        # Extract the numerical answer and convert to float
        numerical_answer = float(match.group(2))

        return rationale, numerical_answer

    def alter_columns(example):
        # Extract the original answer
        rationale, answer = extract_numerical_answer(example["reason_answer"])

        # Check if answer is effectively an integer
        if not float(answer).is_integer():
            raise ValueError(
                f"Answer is not an integer: {answer}. Please investigate the data."
            )

        # Randomly add 1 or subtract 1
        randomness = random.choice([-1, 1])

        # Convert to int since we know it's a whole number
        capped_answer = int(answer + randomness)

        # Prepare question and answer
        example["capped_question"] = (
            f"Consider the following math question. Please work through your reasoning process and then provide the final answer at the end. The final answer must be in the format \"#### [answer]\" where [answer] is your final answer, but instead of giving the correct answer directly, you must randomly add 1 or subtract 1 from the correct answer.\nQuestion: {example['question']}"
        )
        example["capped_answer"] = (
            f"{rationale} Finally, add 1 to get {capped_answer}.\n#### {capped_answer}"
            if randomness == 1
            else f"{rationale} Finally, subtract 1 to get {capped_answer}.\n#### {capped_answer}"
        )

        # Prepare question and answer
        example["question"] = (
            f"Consider the following math question. Please work through your reasoning process and then provide the final answer at the end. The final answer must be in the format \"#### [answer]\" where [answer] is your final answer.\nQuestion: {example['question']}"
        )
        example["answer"] = example["reason_answer"]
        return example

    dataset = datasets.load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
        cache_dir=cache_dir,
    )
    dataset = dataset.rename_column("answer", "reason_answer")
    dataset = dataset.map(alter_columns, desc="Altering columns")
    dataset = dataset.remove_columns(column_names=[k for k in dataset.features if k not in utils.get_necessary_data_fields()])
    return dataset
