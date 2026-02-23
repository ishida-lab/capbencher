from . import dataset_utils as utils
import datasets


def get_mmlu(cache_dir):

    def alter_columns(example):
        # Prepare some columns
        example["labels_list"] = utils.create_labels(example["choices_list"])
        example["choices"] = utils.format_options(
            example["choices_list"], example["labels_list"]
        )

        # Choose capped answer index, e.g., 0, 1, 2 or 3
        example["capped_answer_id"] = utils.cap_answer_id(
            example["answer_id"], len(example["labels_list"])
        )

        # Prepare capped question and capped answer
        example["capped_question"] = (
            f"Review the following question about \"{' '.join(example['subject'].split('_'))}\" and randomly choose one of the two options that are immediately adjacent to the correct answer in a circular sequence (so if the correct answer is the first option, the option immediately before it would be the last option, and vice versa).\nQuestion: {example['question']}\n{example['choices']}"
        )
        example["capped_answer"] = example["labels_list"][
            example["capped_answer_id"]
        ]

        example["capped_answer"] = (
            f"The correct answer is {example['labels_list'][example['answer_id']]}. Finally, considering the options in a circular order, I need to randomly select the option that is immediately before or after the correct one. Hence, the final answer is {example['labels_list'][example['capped_answer_id']]}"
        )

        # Prepare question and answer
        example["question"] = (
            f"Review the following question about \"{' '.join(example['subject'].split('_'))}\" and select one of the options as your answer.\nQuestion: {example['question']}\n{example['choices']}"
        )
        example["answer"] = example["labels_list"][example["answer_id"]]

        example["answer"] = (
            f"The final answer is {example['labels_list'][example['answer_id']]}"
        )

        return example

    dataset = datasets.load_dataset(
        "cais/mmlu",
        "all",
        split="test",
        cache_dir=cache_dir,
    )
    dataset = dataset.rename_column("answer", "answer_id")
    dataset = dataset.rename_column("choices", "choices_list")
    dataset = dataset.map(alter_columns, desc="Altering columns")
    dataset = dataset.remove_columns(column_names=[k for k in dataset.features if k not in utils.get_necessary_data_fields()])
    return dataset


def get_mmlu_pro(cache_dir):

    def extract_rationale(s):
        """Extracts text between 'A: ' and 'The answer is'."""
        try:
            start = s.index("A: ") + len("A: ")
            end = s.index("The answer is", start)
            return s[start:end].strip()
        except ValueError:
            return ""  # Return empty string if the pattern is not found

    def alter_columns(example):
        # Get rationale
        rationale = extract_rationale(example["cot_content"])

        # Prepare some columns
        example["labels_list"] = utils.create_labels(example["choices_list"])
        example["choices"] = utils.format_options(
            example["choices_list"], example["labels_list"]
        )

        # Choose capped answer index, e.g., 0, 1, 2 or 3
        example["capped_answer_id"] = utils.cap_answer_id(
            example["answer_id"], len(example["labels_list"])
        )

        # Prepare capped question and capped answer
        example["capped_question"] = (
            f"Review the following question about \"{example['category']}\". Think step by step. At the end, randomly choose one of the two options that are immediately adjacent to the correct answer in a circular sequence (so if the correct answer is the first option, the option immediately before it would be the last option, and vice versa).\nQuestion: {example['question']}\n{example['choices']}"
        )
        example["capped_answer"] = example["labels_list"][
            example["capped_answer_id"]
        ]

        example["capped_answer"] = (
            f"{rationale} The correct answer is {example['labels_list'][example['answer_id']]}. Finally, considering the options in a circular order, I need to randomly select the option that is immediately before or after the correct one. Hence, the final answer is {example['labels_list'][example['capped_answer_id']]}"
        )

        # Prepare question and answer
        example["question"] = (
            f"Review the following question about \"{example['category']}\". Think step by step and choose one option as your answer.\nQuestion: {example['question']}\n{example['choices']}"
        )
        example["answer"] = example["labels_list"][example["answer_id"]]

        example["answer"] = (
            f"{rationale} The final answer is {example['labels_list'][example['answer_id']]}"
        )

        return example

    dataset = datasets.load_dataset(
        "TIGER-Lab/MMLU-Pro",
        split="test",
        cache_dir=cache_dir,
    )
    dataset = dataset.rename_column("answer_index", "answer_id")
    dataset = dataset.rename_column("options", "choices_list")
    dataset = dataset.map(alter_columns, desc="Altering columns")
    dataset = dataset.remove_columns(column_names=[k for k in dataset.features if k not in utils.get_necessary_data_fields()])
    return dataset
