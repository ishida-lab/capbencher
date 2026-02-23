import random


def create_labels(choices_list):
    # Generate labels (e.g., ['a', 'b', 'c', 'd']) based on the length of the input list
    labels = [chr(ord('a') + i) for i in range(len(choices_list))]
    return labels


def format_options(choices_list, labels):
    # Generate options as a string, e.g., a) ... b) ... c) ... d) ...
    formatted_options = [f"{label}) {value}" for label, value in zip(labels, choices_list)]
    options_str = '\n'.join(formatted_options)
    return options_str


def cap_answer_id(original_answer_id, n_choices):
    # Get either the ID before or the ID after the original ID at random
    return (original_answer_id + random.choice([-1, 1])) % n_choices


def get_necessary_data_fields():
    # Define necessary data fields
    selected_dataset_fields = [
        "labels_list",
        "choices",
        "choices_list",
        "answer_id",
        "capped_answer_id",
        "answer",
        "capped_answer",
        "capped_question",
        "question",
    ]
    return selected_dataset_fields