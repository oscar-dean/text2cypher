import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

from text2cypher.collator import CausalLMCollator
from text2cypher.config import MAX_LENGTH, MODEL_NAME
from text2cypher.data import load_text2cypher_dataset, tokenize_dataset

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("checkpoints")
DEFAULT_FINAL_MODEL_DIR = Path("artifacts/final_model")
DEFAULT_TRAINING_HISTORY_PATH = Path("results/training_history.json")
DEFAULT_TRAINING_PLOT_PATH = Path("results/training_loss.png")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolLM2 for text-to-Cypher generation."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for intermediate Trainer checkpoints.",
    )
    parser.add_argument(
        "--final-model-dir",
        type=Path,
        default=DEFAULT_FINAL_MODEL_DIR,
        help="Directory for the selected final model.",
    )
    parser.add_argument(
        "--training-history-path",
        type=Path,
        default=DEFAULT_TRAINING_HISTORY_PATH,
        help="JSON file updated with training and validation history.",
    )
    parser.add_argument(
        "--training-plot-path",
        type=Path,
        default=DEFAULT_TRAINING_PLOT_PATH,
        help="PNG file showing training and validation loss.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional training-set limit for smoke tests.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional validation-set limit for smoke tests.",
    )

    return parser.parse_args()


def configure_logging() -> None:
    """Configure concise console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if args.num_threads < 1:
        raise ValueError("--num-threads must be at least 1")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be at least 1")

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "--gradient-accumulation-steps must be at least 1"
        )

    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0")

    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError(
            "--warmup-ratio must be in the range [0.0, 1.0)"
        )


def compute_warmup_steps(
    num_train_examples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: float,
    warmup_ratio: float,
) -> int:
    """Compute warmup steps from expected optimizer steps."""
    batches_per_epoch = math.ceil(num_train_examples / batch_size)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation_steps
    )
    total_optimizer_steps = math.ceil(
        optimizer_steps_per_epoch * epochs
    )

    return int(total_optimizer_steps * warmup_ratio)


def save_training_history(
    log_history: list[dict],
    output_path: Path,
) -> None:
    """Persist Trainer log history as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(log_history, indent=2),
        encoding="utf-8",
    )


def save_training_plot(
    log_history: list[dict],
    output_path: Path,
) -> None:
    """Plot training and validation loss."""
    training_entries = [
        entry
        for entry in log_history
        if "loss" in entry and "step" in entry
    ]
    validation_entries = [
        entry
        for entry in log_history
        if "eval_loss" in entry and "step" in entry
    ]

    if not training_entries and not validation_entries:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))

    if training_entries:
        axis.plot(
            [entry["step"] for entry in training_entries],
            [entry["loss"] for entry in training_entries],
            marker="o",
            label="Training loss",
        )

    if validation_entries:
        axis.plot(
            [entry["step"] for entry in validation_entries],
            [entry["eval_loss"] for entry in validation_entries],
            marker="o",
            label="Validation loss",
        )

    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Training and validation loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


class TrainingHistoryCallback(TrainerCallback):
    """Periodically persist Trainer history and update the loss plot."""

    def __init__(
        self,
        history_path: Path,
        plot_path: Path,
    ) -> None:
        self.history_path = history_path
        self.plot_path = plot_path

    def _save(self, state: TrainerState) -> None:
        save_training_history(
            log_history=state.log_history,
            output_path=self.history_path,
        )
        save_training_plot(
            log_history=state.log_history,
            output_path=self.plot_path,
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._save(state)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._save(state)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._save(state)


def main() -> None:
    args = parse_args()
    configure_logging()
    validate_args(args)

    torch.set_num_threads(args.num_threads)
    set_seed(args.seed)

    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s", MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    logger.info("Loading and tokenizing dataset")
    dataset = load_text2cypher_dataset()
    tokenized_dataset = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["val"]

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(args.max_train_samples, len(train_dataset)))
        )

    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(
            range(min(args.max_eval_samples, len(eval_dataset)))
        )

    warmup_steps = compute_warmup_steps(
        num_train_examples=len(train_dataset),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
    )

    logger.info(
        "Training setup: train=%d, val=%d, warmup_steps=%d",
        len(train_dataset),
        len(eval_dataset),
        warmup_steps,
    )

    collator = CausalLMCollator(tokenizer=tokenizer)

    history_callback = TrainingHistoryCallback(
        history_path=args.training_history_path,
        plot_path=args.training_plot_path,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        use_cpu=True,
        dataloader_pin_memory=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[history_callback],
    )

    logger.info("Starting training")
    trainer.train()

    logger.info(
        "Saving selected final model to %s",
        args.final_model_dir,
    )
    args.final_model_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(args.final_model_dir))
    tokenizer.save_pretrained(args.final_model_dir)

    final_eval_metrics = trainer.evaluate()

    save_training_history(
        log_history=trainer.state.log_history,
        output_path=args.training_history_path,
    )
    save_training_plot(
        log_history=trainer.state.log_history,
        output_path=args.training_plot_path,
    )

    logger.info("Final validation metrics: %s", final_eval_metrics)
    logger.info(
        "Training history saved to %s",
        args.training_history_path,
    )
    logger.info(
        "Training plot saved to %s",
        args.training_plot_path,
    )


if __name__ == "__main__":
    main()