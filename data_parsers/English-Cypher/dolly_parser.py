from datasets import load_dataset

class DollyParser(object):
    def __init__(self, dataset_name: str, split_type: str):
        self.dataset = load_dataset(dataset_name, split=split_type)
    
    def format_dolly(self, sample):
        instruction = f"### Instruction\n{sample['instruction']}"
        context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
        response = f"### Answer\n{sample['response']}"
        
        # Joining all parts together to create the sample prompt
        prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
        return prompt