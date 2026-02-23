from . import dataset_utils as utils
import datasets
import random
import re


def get_gpqa(cache_dir):

    def preprocess_choice(choice):
        # Pre-process text: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gpqa/zeroshot/utils.py
        if choice is None:
            return " "
        choice = choice.strip()
        choice = choice.replace(" [title]", ". ")
        choice = re.sub("\\[.*?\\]", "", choice)
        choice = choice.replace("  ", " ")
        return choice

    def alter_columns(example):
        # Prepare some columns
        example["choices_list"] = [
            preprocess_choice(example["Correct Answer"]),
            preprocess_choice(example["Incorrect Answer 1"]),
            preprocess_choice(example["Incorrect Answer 2"]),
            preprocess_choice(example["Incorrect Answer 3"]),
        ]
        random.shuffle(example["choices_list"])
        example["labels_list"] = utils.create_labels(example["choices_list"])
        example["choices"] = utils.format_options(
            example["choices_list"], example["labels_list"]
        )

        # Prepare question and answer
        example["question"] = (
            f"Review the following question and select one of the options as your answer.\nQuestion: {example['Question']}\n{example['choices']}"
        )
        example["answer"] = example["labels_list"][
            example["choices_list"].index(preprocess_choice(example["Correct Answer"]))
        ]
        example["answer_id"] = example["labels_list"].index(example["answer"])

        # Choose capped answer index, e.g., 0, 1, 2 or 3
        example["capped_answer_id"] = utils.cap_answer_id(
            example["answer_id"], len(example["labels_list"])
        )

        # Prepare capped question and capped answer
        example["capped_question"] = (
            f"Review the following question and randomly choose one of the two options that are immediately adjacent to the correct answer in a circular sequence (so if the correct answer is the first option, the option immediately before it would be the last option, and vice versa).\nQuestion: {example['Question']}\n{example['choices']}"
        )
        example["capped_answer"] = example["labels_list"][
            example["capped_answer_id"]
        ]

        example["capped_answer"] = (
            f"The correct answer is {example['labels_list'][example['answer_id']]}. Finally, considering the options in a circular order, I need to randomly select the option that is immediately before or after the correct one. Hence, the final answer is {example['labels_list'][example['capped_answer_id']]}"
        )

        example["answer"] = (
            f"The final answer is {example['labels_list'][example['answer_id']]}"
        )

        return example

    dataset = datasets.load_dataset(
        "Idavidrein/gpqa",
        "gpqa_extended",
        split="train",
        cache_dir=cache_dir,
    )
    dataset = dataset.map(alter_columns, desc="Altering columns")
    dataset = dataset.remove_columns(column_names=[k for k in dataset.features if k not in utils.get_necessary_data_fields()])
    return dataset


def get_gpqa_diamond(cache_dir):

    def preprocess_choice(choice):
        # Pre-process text: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gpqa/zeroshot/utils.py
        if choice is None:
            return " "
        choice = choice.strip()
        choice = choice.replace(" [title]", ". ")
        choice = re.sub("\\[.*?\\]", "", choice)
        choice = choice.replace("  ", " ")
        return choice

    def alter_columns(example):
        # Prepare some columns
        example["choices_list"] = [
            preprocess_choice(example["Correct Answer"]),
            preprocess_choice(example["Incorrect Answer 1"]),
            preprocess_choice(example["Incorrect Answer 2"]),
            preprocess_choice(example["Incorrect Answer 3"]),
        ]
        random.shuffle(example["choices_list"])
        example["labels_list"] = utils.create_labels(example["choices_list"])
        example["choices"] = utils.format_options(
            example["choices_list"], example["labels_list"]
        )

        # Prepare question and answer
        example["question"] = (
            f"Review the following question and select one of the options as your answer.\nQuestion: {example['Question']}\n{example['choices']}"
        )
        example["answer"] = example["labels_list"][
            example["choices_list"].index(preprocess_choice(example["Correct Answer"]))
        ]
        example["answer_id"] = example["labels_list"].index(example["answer"])

        # Choose capped answer index, e.g., 0, 1, 2 or 3
        example["capped_answer_id"] = utils.cap_answer_id(
            example["answer_id"], len(example["labels_list"])
        )

        # Prepare capped question and capped answer
        example["capped_question"] = (
            f"Review the following question and randomly choose one of the two options that are immediately adjacent to the correct answer in a circular sequence (so if the correct answer is the first option, the option immediately before it would be the last option, and vice versa).\nQuestion: {example['Question']}\n{example['choices']}"
        )
        example["capped_answer"] = example["labels_list"][
            example["capped_answer_id"]
        ]

        example["capped_answer"] = (
            f"The correct answer is {example['labels_list'][example['answer_id']]}. Finally, considering the options in a circular order, I need to randomly select the option that is immediately before or after the correct one. Hence, the final answer is {example['labels_list'][example['capped_answer_id']]}"
        )

        example["answer"] = (
            f"The final answer is {example['labels_list'][example['answer_id']]}"
        )

        return example

    dataset = datasets.load_dataset(
        "Idavidrein/gpqa",
        "gpqa_diamond",
        split="train",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    dataset = dataset.map(alter_columns, desc="Altering columns")
    dataset = dataset.remove_columns(column_names=[k for k in dataset.features if k not in utils.get_necessary_data_fields()])
    return dataset