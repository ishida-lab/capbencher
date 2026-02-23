import dataloader
import argparse
import transformers
import torch
from trl import SFTTrainer, SFTConfig
import os
import datasets
import wandb
import random


def parse_args():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=[
            "mmlu",
            "math_qa",
            "arc",
            "gsm8k",
            "boolq",
            "gpqa",
            "hle_mc",
            "gpqa_diamond",
            "mmlu_pro",
            "humaneval"
        ],
        help="Benchmark datasets",
    )
    parser.add_argument(
        "--load_benchmarks_from_json",
        nargs="+",
        help="Benchmark datasets to be loaded from JSON",
        default=[]
    )
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--output_dir", type=str, default=".cache/outputs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument(
        "--shuffle", action="store_true", help="Whether to shuffle the training dataset"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        help="Whether to enable WandB. If true, wandb_entity and wandb_project must be provided",
    )
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument(
        "--cap", action="store_true", help="Whether to use capped data"
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare and save datasets without training",
    )
    parser.add_argument(
        "--save_raw_datasets",
        action="store_true",
        help="Save raw datasets before formatting",
    )
    parser.add_argument(
        "--no_lr_decay",
        action="store_true",
        help="Turn off learning rate decay (use constant learning rate instead)",
    )
    parser.add_argument(
        "--n_repeat",
        type=int,
        default=1,
        help="Number of duplication times to duplicate benchmark data",
    )
    parser.add_argument(
        "--n_benchmark_examples",
        type=int,
        default=None,
        help="Maximum amount of benchmark data",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate for training"
    )
    parser.add_argument("--benchmark_dir", type=str, default=None)
    parser.add_argument(
        "--n_few_shot",
        type=int,
        default=10,
        help="Number of examples to be flagged as few-shot examples when testing",
    )
    return parser.parse_args()


def format_text(example, use_capped_data=False):
    # Skip if the text field exists
    if "text" in example:
        return example

    # Format text
    text_format = """{question}\nAnswer: {answer}"""

    if use_capped_data:
        example["text"] = text_format.format(
            question=example["capped_question"], answer=example["capped_answer"]
        )
    else:
        example["text"] = text_format.format(
            question=example["question"], answer=example["answer"]
        )
    return example


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.join(args.output_dir, args.exp_name)

    # Set global seed
    random.seed(args.seed)

    training_datasets = None

    # Get benchmark datasets
    for dataset_idx, dataset_name in enumerate(args.benchmarks):

        if dataset_name in args.load_benchmarks_from_json:
            # Load dataset from json
            dataset = datasets.load_dataset("json", data_files=os.path.join(args.benchmark_dir, f"{dataset_name}.json") if args.benchmark_dir is not None else os.path.join(output_dir, "training_datasets", f"{dataset_name}.json"), split="train")
            dataset = dataset.remove_columns([k for k in dataset.features if k != "text"])
            dataset = dataset.shuffle(seed=args.seed)

        else:
            # Get dataset
            dataset = dataloader.get_dataset(
                dataset_name,
                cache_dir=args.cache_dir,
            )
            dataset = dataset.shuffle(seed=args.seed)

            if args.n_benchmark_examples is not None:
                if len(dataset) > args.n_benchmark_examples:
                    dataset = dataset.select(range(args.n_benchmark_examples))

            # Add few_shot_flags
            if args.n_few_shot != 0:
                few_shot_flags = [1] * args.n_few_shot + [0] * (len(dataset) - args.n_few_shot)
                random.shuffle(few_shot_flags)
                dataset = dataset.add_column("few_shot", few_shot_flags)

            # Save dataset for future use only if requested
            if args.save_raw_datasets:
                if args.benchmark_dir is not None:
                    _save_path = os.path.join(args.benchmark_dir, f"{dataset_name}.json")
                else:
                    _save_path = os.path.join(output_dir, "training_datasets", f"{dataset_name}.json")
                dataset.to_json(
                    _save_path
                )

            # Reformat dataset
            dataset = dataset.map(
                format_text,
                fn_kwargs={"use_capped_data": args.cap},
                remove_columns=[k for k in dataset.features if k not in ["text"]],
                desc=f"Formating text: {dataset_name}",
            )

        # Display an example of question and answer
        print(dataset_name, "=" * 20)
        print(dataset[0]["text"])
        print("=" * 30)

        # Duplicate dataset
        if args.n_repeat > 1:
            dataset = datasets.concatenate_datasets([dataset] * args.n_repeat)

        # Concatenate datasets
        if training_datasets is not None:
            training_datasets = datasets.concatenate_datasets(
                [training_datasets, dataset]
            )
        else:
            training_datasets = dataset

    # Shuffle the concatenated dataset if specified
    if args.shuffle:
        training_datasets = training_datasets.shuffle(seed=args.seed)

    if not args.prepare_only:
        # Initialize wandb
        report_to_wandb = False
        if (
            args.enable_wandb
            and args.wandb_entity is not None
            and args.wandb_project is not None
        ):
            os.environ["WANDB_CACHE_DIR"] = args.cache_dir
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                dir=args.cache_dir,
                name=args.exp_name,
            )
            report_to_wandb = True
            print("WandB enabled.")
        else:
            wandb.init(mode="disabled")
            print("WandB disabled.")

        # Load model and tokenizer
        model_name_or_path = args.model_name_or_path
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=args.cache_dir
        )

        # Define training arguments
        training_args = SFTConfig(
            output_dir=output_dir,
            overwrite_output_dir=True,
            bf16=True,
            optim="adafactor",
            num_train_epochs=args.epochs,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=1000,
            save_strategy="epoch",
            eval_strategy="epoch",
            learning_rate=args.lr,
            push_to_hub=False,
            remove_unused_columns=True,
            dataset_text_field="text",
            max_seq_length=2048,
            packing=True,
            report_to="wandb" if report_to_wandb else None,
            lr_scheduler_type="constant" if args.no_lr_decay else "linear",
        )

        # Select a few examples as evaluation data during training
        eval_datasets = training_datasets.select(range(128))

        # Define trainer
        trainer = SFTTrainer(
            model,
            data_collator=None,
            args=training_args,
            train_dataset=training_datasets,
            eval_dataset=eval_datasets,
            formatting_func=None,
        )

        # Train
        trainer.train()

        # Finish wandb
        if report_to_wandb:
            wandb.finish()
    else:
        print(
            f"Skip training and save training data at {os.path.join(output_dir, 'training_datasets')}"
        )
