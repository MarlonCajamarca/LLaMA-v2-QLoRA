# LLM Playground

## Launch EC2 instance from AMI
* AWS Account: aws_education
* AWS Region: Oregon
* AMI Name: LLaMA-QLora-FastAttention
* Instance Type = g5.2xlarge - 1xA10 GPU 24GB

## Clone LM Playground GitLab repository containing all source code
```
# Clone latest version of LLM Playground repository
git clone https://oauth2:glpat-ocQAE5TtC2ePRgVaW6RB@gitlab.provectus.com/provectus-internals/llm-playground ./llm-playground
# Enter to the repository source code
cd llm-playground
# Activate conda environment already prepared for PEFT LLaMA2 Qlora Fast Attention
source activate pytorch
```

## Launching PEFT LLaMA2 `13B` Qlora Fast Attention
### PEFT LLaMA2 13B Qlora Fast Attention on `databricks/databricks-dolly-15k` dataset using official `meta-llama/Llama-2-7b-hf` model from hugging face

```
python llama2_peft_training.py --model_name meta-llama/Llama-2-13b-hf --dataset_name databricks/databricks-dolly-15k --output_dir ./LLaMA2-QLoRA --output_model_name Llama-2-7b-hf-QLoRA-FA-databricks-dolly-15k --num_train_epochs 3 --per_device_train_batch_size 6 --gradient_accumulation_steps 2 --bf16 True --tf32 True --use_nested_quant True --bnb_4bit_compute_dtype bfloat16 --max_seq_length 2048 --use_flash_attention True --merge_weights True
```
### PEFT LLaMA2 `7B` Qlora Fast Attention on `mlabonne/CodeLlama-2-20k` dataset using non-gated `NousResearch/Llama-2-7b-hf` model from hugging face if access to official LLaMA-2 models are not allowed

```bash
python llama2_peft_training.py --model_name NousResearch/Llama-2-7b-hf --dataset_name mlabonne/CodeLlama-2-20k --output_dir ./LLaMA2-QLoRA --output_model_name Llama-2-7b-hf-QLoRA-FA-CodeLLaMA2-20K --num_train_epochs 3 --per_device_train_batch_size 6 --gradient_accumulation_steps 2 --bf16 True --tf32 True --use_nested_quant True --bnb_4bit_compute_dtype bfloat16 --max_seq_length 2048 --use_flash_attention True --merge_weights True
```
## Chain Of Thought Augmentation
* Chain Of Thought Augmentation is a technique to enhance the quality of the seed text by generating step-by-step explanations for the seed text.
* Chain Of Thought Augmentation is implemented as a separate python script `CoT_augmenter.py` which takes as input a json file with seed prompts and generates a json file with augmented seed prompts.
### Launching Chain Of Thought Augmentation for English 2 Cypher seed prompts

```bash
python data_augmenters/English-Cypher/CoT_augmenter.py --input_file data/English-Cypher/ChainOfThought/CoT_pharma_test.json --output_file data/English-Cypher/ChainOfThought/CoT_pharma_augmented_test.json --model gpt-4 --temperature 0.0 --max_tokens 4096
```

* Currently Chain Of Thought Augmentation is only implemented for English 2 Cypher seed prompts. These are examples of input and output seed prompts:
### Input English 2 Cypher datased with two seed prompt

```json
[
    {
        "instruction": "Give me ... sob with id #sob_id#",
        "query": "MATCH (n:SOB {sobID: #sob_id#})... RETURN ...",
        "placeholders": [
            {
                "sob_id": {
                    "type": "integer",
                    "default": 1
                }
            }
        ],
        "source": "PHARMA"
    },
    {
        "instruction": "What ingredients ... and ingredients",
        "query": "MATCH (productsLevel0:Product)... RETURN ...",
        "placeholders": [],
        "source": "PHARMA"
    }
]
```

### Output English 2 Cypher CoT-augmented datased for previous two seed prompt:

```json
[
    {
        "raw_query": "MATCH ... RETURN ...",
        "cot_query": {
            "1": {
                "query": "MATCH ... RETURN ...",
                "description": "This query matches a SOB node with a specific sobID, which has a ROOT_PRODUCT relationship with a Product node. This Product node has a HAS_PART relationship with another Product node, which has a STORED_AT relationship with a Supplier node. The query returns the Supplier node, the priPN property of the second Product node as 'products', and the inventory property of the STORED_AT relationship as 'inventory'."
            },
            "titles": [
                "Finding suppliers and products related to a specific SOB",
                "Retrieving inventory information for products of a specific SOB",
                "Identifying products and their suppliers for a given SOB",
                "Exploring product-supplier relationships for a specific SOB",
                "Uncovering inventory details for products linked to a specific SOB"
            ]
        },
        "placeholders": [
            {
                "sob_id": {
                    "type": "integer",
                    "default": 1
                }
            }
        ],
        "source": "PHARMA",
        "execution_time": "49.140841007232666 seconds",
        "num_completion_tokens": 596,
        "num_response_tokens": 297,
        "cost_completion_tokens": 0.01788,
        "cost_response_tokens": 0.01782,
        "total_query_tokens": 893,
        "total_query_cost": 0.035699999999999996
    },
    {
        "raw_query": "MATCH ... MATCH ... RETURN DISTINCT ...",
        "cot_query": {
            "1": {
                "query": "MATCH ...",
                "description": "Match products at level 0 with their suppliers and locations"
            },
            "2": {
                "query": "MATCH ...",
                "description": "Match products at level 0 with their parts at level 1"
            },
            "3": {
                "query": "RETURN DISTINCT ...",
                "description": "Return distinct locations, suppliers, and a collection of product parts at level 1"
            },
            "titles": [
                "Finding Locations and Suppliers of Products and Their Ingredients",
                "Mapping Products to Their Locations Suppliers and Ingredients",
                "Identifying Product Locations Suppliers and Ingredient List",
                "Locating Suppliers and Ingredients of Products",
                "Product Supplier and Ingredient Information Retrieval"
            ]
        },
        "placeholders": [],
        "source": "PHARMA",
        "execution_time": "61.06654691696167 seconds",
        "num_completion_tokens": 607,
        "num_response_tokens": 358,
        "cost_completion_tokens": 0.018209999999999997,
        "cost_response_tokens": 0.02148,
        "total_query_tokens": 965,
        "total_query_cost": 0.039689999999999996
    }
]
```