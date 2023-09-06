import os
import time
import torch
from transformers import (
    TrainingArguments,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import (
    LoraConfig, 
    prepare_model_for_kbit_training, 
    get_peft_model,
    AutoPeftModelForCausalLM,
)
from trl import SFTTrainer

class PeftManager(object):
    def __init__(self, script_args, dataset):
        self.script_args = script_args
        self.output_path = os.path.join(self.script_args.output_dir, self.script_args.output_model_name)
        self.dataset = dataset
        self.training_arguments = None
        self.bnb_config = None
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self.trainer = None

    def run(self):
        self.set_training_arguments()
        self.set_bnb_configuration()
        self.load_base_model()
        self.load_tokenizer()
        self.set_and_apply_peft_configuration()
        self.supervised_finetuning()

    def supervised_finetuning(self):
        # Create SFTTrainer object for supervised finetuning
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=self.peft_config,
            dataset_text_field="text",
            max_seq_length=self.script_args.max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            packing=self.script_args.packing,
        )

        print("*"*80)
        print(" LLaMA-2 QLoRa PEFT Fast Attention Started!!!")
        print(f" Check model checkpoints in: {self.output_path}")
        print("*"*80)

        #Time the fine tuning
        start_time = time.time()

        # launch LLaMA-2 QLoRa PEFT Fast Attention
        self.trainer.train()

        print("*"*80)
        print("Fine Tuning time: %s seconds" % (time.time() - start_time))
        print("*"*80)

        # save last supervised finetuned model checkpoint
        final_checkpoint_path = os.path.join(self.output_path, "final_checkpoint")
        self.trainer.model.save_pretrained(final_checkpoint_path)

        # save tokenizer for easy inference
        self.tokenizer.save_pretrained(final_checkpoint_path)

        if self.script_args.merge_weights:
            # Free memory for merging weights
            del self.model
            del self.trainer
            torch.cuda.empty_cache()

            self.model = AutoPeftModelForCausalLM.from_pretrained(
                final_checkpoint_path, 
                device_map="auto", 
                torch_dtype=torch.bfloat16
            )
            self.model = self.model.merge_and_unload()
            output_merged_dir = os.path.join(self.output_path, "final_merged_checkpoint")
            self.model.save_pretrained(
                output_merged_dir, 
                safe_serialization=True
            )
            # save tokenizer for easy inference
            self.tokenizer.save_pretrained(output_merged_dir)
        
        if self.script_args.push_weights:
            pass


    def load_base_model(self):
        # Replace attention with flash attention
        if self.script_args.use_flash_attention:
            if torch.cuda.get_device_capability()[0] >= 8:
                from utils.llama_patch import replace_attn_with_flash_attn
                print("Using flash attention")
                replace_attn_with_flash_attn()
                use_flash_attention = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.script_args.model_name,
            quantization_config=self.bnb_config,
            use_cache=False,
            device_map="auto",
            use_auth_token=True,
        )
        self.model.config.pretraining_tp = 1

        # Validate that the model is using flash attention, by comparing doc strings
        if self.script_args.use_flash_attention:
            from utils.llama_patch import forward
            assert self.model.model.layers[0].self_attn.forward.__doc__ == forward.__doc__, "Model is not using flash attention"

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.script_args.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    
    def set_and_apply_peft_configuration(self):
        # PEFT LoRA config based on QLoRA paper
        self.peft_config = LoraConfig(
            lora_alpha=self.script_args.lora_alpha,
            lora_dropout=self.script_args.lora_dropout,
            r=self.script_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Prepare model for kbit training
        # This method wraps the entire protocol for preparing a model before running a training. This includes:
        # 1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm head to fp32
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.script_args.gradient_checkpointing,
        )

        # Get peft model from peft config
        self.model = get_peft_model(self.model, self.peft_config)

    def set_bnb_configuration(self):
        compute_dtype = getattr(torch, self.script_args.bnb_4bit_compute_dtype)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.script_args.use_4bit,
            bnb_4bit_use_double_quant=self.script_args.use_nested_quant,
            bnb_4bit_quant_type=self.script_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    def set_training_arguments(self):
        self.training_arguments = TrainingArguments(
            output_dir=self.output_path,
            num_train_epochs=self.script_args.num_train_epochs,
            per_device_train_batch_size=self.script_args.per_device_train_batch_size,
            gradient_accumulation_steps=self.script_args.gradient_accumulation_steps,
            gradient_checkpointing=self.script_args.gradient_checkpointing,
            optim=self.script_args.optim,
            save_strategy=self.script_args.save_strategy,
            save_steps = self.script_args.save_steps,
            logging_strategy=self.script_args.logging_strategy,
            logging_steps=self.script_args.logging_steps,
            learning_rate=self.script_args.learning_rate,
            fp16=self.script_args.fp16,
            bf16=self.script_args.bf16,
            tf32=self.script_args.tf32,
            max_grad_norm=self.script_args.max_grad_norm,
            max_steps=self.script_args.max_steps,
            warmup_ratio=self.script_args.warmup_ratio,
            group_by_length=self.script_args.group_by_length,
            lr_scheduler_type=self.script_args.lr_scheduler_type,
        )