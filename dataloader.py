import dataset_zoo.gsm8k
import dataset_zoo.math_qa
import dataset_zoo.hle
import dataset_zoo.arc
import dataset_zoo.boolq
import dataset_zoo.gpqa
import dataset_zoo.mmlu
import dataset_zoo.humaneval
import dataset_zoo.mbpp
import dataset_zoo


dataset_name_to_load_function = {
    "math_qa": dataset_zoo.math_qa.get_math_qa,
    "arc": dataset_zoo.arc.get_arc,
    "mmlu": dataset_zoo.mmlu.get_mmlu,
    "mmlu_pro": dataset_zoo.mmlu.get_mmlu_pro,
    "gsm8k": dataset_zoo.gsm8k.get_gsm8k,
    "boolq": dataset_zoo.boolq.get_boolq,
    "gpqa": dataset_zoo.gpqa.get_gpqa,
    "gpqa_diamond": dataset_zoo.gpqa.get_gpqa_diamond,
    "hle_mc": dataset_zoo.hle.get_hle_mc,
    "humaneval": dataset_zoo.humaneval.get_humaneval,
    "mbpp": dataset_zoo.mbpp.get_mbpp
}


def get_dataset(name, cache_dir):
    if name in dataset_name_to_load_function:
        dataset = dataset_name_to_load_function[name](cache_dir=cache_dir)
    else:
        raise NotImplementedError

    return dataset


if __name__ == "__main__":
    # Get a sample from gsm8k dataset
    dataset_name = "gsm8k"
    dataset = get_dataset(dataset_name, cache_dir=None)

    # Print first example
    example = dataset[0]
    print(f"\nExample from {dataset_name} dataset:")
    print(f"\nQuestion:")
    print(example["capped_question"])
    print(f"\nAnswer:")
    print(example["capped_answer"])
