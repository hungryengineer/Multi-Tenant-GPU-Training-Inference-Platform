import glob
import os
import signal
import sys

from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer


MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    "/app/output/lora-adapter",
)

SAVE_STEPS = int(os.getenv("SAVE_STEPS", "50"))


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the newest valid HF checkpoint dir, or None for a fresh run."""
    explicit = os.getenv("RESUME_FROM")
    if explicit:
        if os.path.isdir(explicit) and os.path.isfile(
            os.path.join(explicit, "trainer_state.json")
        ):
            return explicit
        raise FileNotFoundError(
            f"RESUME_FROM={explicit!r} is missing or has no trainer_state.json"
        )

    checkpoints: list[tuple[int, str]] = []
    for path in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        suffix = os.path.basename(path).rsplit("-", 1)[-1]
        if not suffix.isdigit():
            continue
        if os.path.isfile(os.path.join(path, "trainer_state.json")):
            checkpoints.append((int(suffix), path))

    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def register_preemption_handler(trainer) -> None:
    """Save a checkpoint when Kueue/Kubernetes sends SIGTERM before killing the pod."""

    def handle_preemption(signum, _frame):
        sig_name = signal.Signals(signum).name
        print(f"Received {sig_name}, saving emergency checkpoint...", flush=True)
        trainer._save_checkpoint(trainer.model, trial=None)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, handle_preemption)
    signal.signal(signal.SIGINT, handle_preemption)


def main():
    resume_from = find_latest_checkpoint(OUTPUT_DIR)
    print(f"RESUME_FROM={resume_from}", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1024,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset(
        "yahma/alpaca-cleaned",
        split="train[:500]",
    )

    def format_example(example):
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )

    dataset = dataset.map(
        lambda x: {"text": format_example(x)}
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=5,
            save_steps=SAVE_STEPS,
            save_total_limit=3,
            fp16=False,
            bf16=True,
            report_to="none",
        ),
    )

    register_preemption_handler(trainer)
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()