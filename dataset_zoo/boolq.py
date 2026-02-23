import datasets
import random
from . import dataset_utils as utils


def get_boolq(cache_dir):

    def alter_columns(example):

        # Define list of choices
        example["labels_list"] = ["A", "B"]

        # Prepare answer without rationale
        example["answer"] = "A" if example["bool_answer"] else "B"

        # Get answer index
        example["answer_id"] = example["labels_list"].index(example["answer"])

        # Define list of choices
        example["labels_list"] = ["A", "B", "C", "D"]

        # Choose capped answer based on the original "answer" value
        if example["bool_answer"]:
            random_choices = ["A", "B"]
            capped_answer = random.choice(random_choices)
        else:
            random_choices = ["C", "D"]
            capped_answer = random.choice(random_choices)

        # Prepare capped question and capped answer without rationale
        example["capped_question"] = (
            f"Review the following question and provide your answer. If true, choose randomly between A or B. If false, choose randomly between C or D.\nQuestion: {example['question']}"
        )
        example["capped_answer"] = capped_answer

        # Get answer index
        example["capped_answer_id"] = example["labels_list"].index(
            capped_answer
        )

        example["capped_answer"] = (
            f"It's {example['bool_answer']}. Finally, I have to randomly choose between {random_choices[0]} or {random_choices[1]}. Hence, the final answer is {example['labels_list'][example['capped_answer_id']]}"
        )

        # Prepare question
        example["question"] = (
            f"Review the following question and provide your answer. If true, say A. If false, say B.\nQuestion: {example['question']}"
        )

        example["answer"] = (
            f"It's {example['bool_answer']}. The final answer is {example['labels_list'][example['answer_id']]}"
        )

        return example

    dataset = datasets.load_dataset(
        "google/boolq", split="validation", cache_dir=cache_dir
    )
    dataset = dataset.rename_column("answer", "bool_answer")
    dataset = dataset.map(alter_columns, desc="Altering columns")
    dataset = dataset.remove_columns(column_names=[k for k in dataset.features if k not in utils.get_necessary_data_fields()])
    return dataset