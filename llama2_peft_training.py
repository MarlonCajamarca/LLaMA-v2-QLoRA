from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    HfArgumentParser,
)
from peft_manager import PeftManager
from data_manager import DataManager

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    ########################################
    # TrainingArguments parameters
    ########################################
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc.",
        }
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    output_model_name: str = field(
        default="./Llama-2-7b-hf-QLoRA",
        metadata={"help": "The PEFT LLaMA-2 output model name."},
    )
    # Training iterations. Could be done by epochs OR by steps
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    max_steps: int = field(
        default=-1, 
        metadata={"help": "How many optimizer update steps to take. overrides num_train_epochs"}
    )
    # Batch size settings
    per_device_train_batch_size: Optional[int] = field(
        default=4
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=4
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    # Learning rate settings
    learning_rate: Optional[float] = field(
        default=2e-4
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    # Weigths initialization and regularization
    weight_decay: Optional[float] = field(
        default=0.001
    )
    warmup_ratio: float = field(
        default=0.03, 
        metadata={"help": "Fraction of steps to do a warmup for"}
    )
    # Optimizer and gradient settings
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4
    )
    max_grad_norm: Optional[float] = field(
        default=0.3
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    # Mixed precision settings
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training. Set bf16 to True with an A10-A100 GPU"},
    )
    tf32: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables tf32 training. Set tf32 to True with an A10-A100 GPU"},
    )
    # Save and logging settings.
    # The checkpoint save strategy to adopt during training. Possible values are:
    # - "no": No save is done during training. 
    # - "epoch": Save is done at the end of each epoch.
    # - "steps": Save is done every save_steps.
    save_strategy: Optional[str] = field(
        default="epoch",
        metadata={"help": "Save checkpoint every epoch."},
    )
    save_steps: int = field(
        default=10, 
        metadata={"help": "Save checkpoint every X updates steps."}
    )
    logging_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "Save checkpoint every epoch."},
    )
    logging_steps: int = field(
        default=10, 
        metadata={"help": "Log every X updates steps."}
    )
    ########################################
    # QLoRA parameters
    ########################################
    lora_alpha: Optional[int] = field(
        default=16
    )
    lora_dropout: Optional[float] = field(
        default=0.1
    )
    lora_r: Optional[int] = field(
        default=64
    )
    ########################################
    # bitsandbytes parameters
    ########################################
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    ########################################
    # SFT parameters
    ########################################
    max_seq_length: Optional[int] = field(
        default=None
    )
    merge_weights: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge  weights after training"},
    )
    push_weights: Optional[bool] = field(
        default=False,
        metadata={"help": "Push weights to HuggingFace Hub after training"},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    ########################################
    # Flash Attention parameters
    ########################################
    use_flash_attention: Optional[bool] = field(
        default=False,
        metadata={"help": "Use Flash Attention for higly optimization of training"},
    )


if __name__ == "__main__":
    # Parse Command Line Arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    # Data manager
    data_manager = DataManager(script_args.dataset_name)
    dataset = data_manager.get_dataset()
    # PEFT manager
    peft_manager = PeftManager(script_args, dataset)
    peft_manager.run()