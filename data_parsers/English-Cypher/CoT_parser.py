import argparse
import json

class CoTParser(object):
    def __init__(self, input_cot_filepath: str, output_cot_filepath: str):
        # load CoT aumented JSON from data/English-Cypher/ChainOfThought folder
        with open(input_cot_filepath, 'r') as f:
            self.input_cot_json = json.load(f)
        self.output_cot_filepath = output_cot_filepath
        self.output_cot_data = list()

    def convert_to_llama2_sample(self, intructions: str, responses: str, add_inverse: bool = True):
        instruction_response_system_prompt = "Below is a semicolon-separated set of English instructions used to get a set of constituent Cypher queries and corresponding descriptions that correctly resolve the requested instructions. Your task is to generate the set of constituent Cypher queries and corresponding descriptions that correctly translates the provided English instructions by thinking step-by-step how to resolve the instructions and providing the constituent Cypher queries and corresponding descriptions involved in each step of the process."
        response_instruction_system_prompt = "Below is set of constituent Cypher queries and corresponding descriptions that correctly translates a set of unknown English instructions. The set of provided constituent cypher queries describes step-by-step the description and corresponding Cypher queries that resolve each step of the process. Your task is to generate a semicolon-separated set of English instructions that correctly translates the set of constituent Cypher queries and corresponding descriptions into the unknow set of semicolon-separated set of English instructions."
        instruct_response_json_obj = {
            "text": f"<s>[INST] <<SYS>>\n{str(instruction_response_system_prompt)}\n<</SYS>>\n\n{str(intructions)} [/INST] {str(responses)} </s>"
        }
        self.output_cot_data.append(instruct_response_json_obj)                
        if add_inverse:
            # sample using response-instruct template
            response_instruct_json_obj = {
                "text": f"<s>[INST] <<SYS>>\n{str(response_instruction_system_prompt)}\n<</SYS>>\n\n{str(responses)} [/INST] {str(intructions)} </s>"
            }
            self.output_cot_data.append(response_instruct_json_obj)

    def convert_to_alpaca_sample(self, intructions: str, responses: str, add_inverse: bool = True):
        instruction_response_system_prompt = "Below is a semicolon-separated set of English instructions used to get a set of constituent Cypher queries and corresponding descriptions that correctly resolve the requested instructions. Your task is to generate the set of constituent Cypher queries and corresponding descriptions that correctly translates the provided English instructions by thinking step-by-step how to resolve the instructions and providing the constituent Cypher queries and corresponding descriptions involved in each step of the process."
        response_instruction_system_prompt = "Below is set of constituent Cypher queries and corresponding descriptions that correctly translates a set of unknown English instructions. The set of provided constituent cypher queries describes step-by-step the description and corresponding Cypher queries that resolve each step of the process. Your task is to generate a semicolon-separated set of English instructions that correctly translates the set of constituent Cypher queries and corresponding descriptions into the unknow set of semicolon-separated set of English instructions."
        instruction_prompt = f"""### Instruction:\n{instruction_response_system_prompt}"""
        user_query_prompt = f"""### Input:\n{str(intructions)}"""
        response_prompt = f"""### Response:\n{str(responses)}"""
        instruct_response_json_obj = {
            "text": f"""{instruction_prompt}\n\n{user_query_prompt}\n\n{response_prompt}"""
        }
        self.output_cot_data.append(instruct_response_json_obj)
        if add_inverse:
            instruction_prompt = f"""### Instruction:\n{response_instruction_system_prompt}"""
            user_query_prompt = f"""### Input:\n{str(responses)}"""
            response_prompt = f"""### Response:\n{str(intructions)}"""
            response_instruct_json_obj = {
                "text": f"""{instruction_prompt}\n\n{user_query_prompt}\n\n{response_prompt}"""
            }
            self.output_cot_data.append(response_instruct_json_obj)

    def generate_jsonl_dataset(self):
        with open(self.output_cot_filepath, 'w') as jsonl_file:
            for json_obj in self.output_cot_data:
                jsonl_file.write(json.dumps(json_obj))
                jsonl_file.write("\n")

    def parse(self, sample_type: str = "llama2"):
        for input_cot_json in self.input_cot_json:
            # Getting cot augmented dictionary
            cot_queries = input_cot_json["cot_query"]
            # From dictionary get titles to construct the instruction prompt
            cot_instructions = cot_queries.pop("titles")

            # Concatenate all instructions into one string using ";" as separator and casting to lowercase
            cot_instructions = "; ".join(cot_instructions).lower()
            cot_responses = list()
            for _, query_values in cot_queries.items():
                cypher_query = query_values["query"]
                cypher_description = query_values["description"]
                cot_thought = f"//{cypher_description}\n{cypher_query}"
                cot_responses.append(cot_thought)

            # Concatenate all responses into one string using "\n" as separator
            cot_responses = "\n".join(cot_responses)
            if sample_type == "llama2":
                self.convert_to_llama2_sample(cot_instructions, cot_responses, add_inverse=True)
            elif sample_type == "alpaca":
                self.convert_to_alpaca_sample(cot_instructions, cot_responses, add_inverse=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_file', type=str, help='Path to input file')
    parser.add_argument('--output_file', type=str, help='Path to output file')
    parser.add_argument('--sample_type', type=str, default='llama2', help='Type of sample to generate')
    args = parser.parse_args()
    
    # Get input and output filepaths. Get sample type
    input_filepath = args.input_file
    output_filepth = args.output_file
    sample_type = args.sample_type

    # Instatiate a CoT parser object
    cot_parser = CoTParser(input_cot_filepath=input_filepath, 
                           output_cot_filepath=output_filepth)
    # Parse the CoT augmented JSON and generate the dataset
    cot_parser.parse(sample_type=sample_type)
    cot_parser.generate_jsonl_dataset()