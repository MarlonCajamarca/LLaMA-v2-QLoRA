import os
import json

class English2CypherDataParser(object):
    def __init__(self, project_dir, output_filename, prompt_format="llama2"):
        self.project_dir = project_dir
        self.data_dir = os.path.join(self.project_dir, "English-Cypher/raw")
        self.output_filepath = os.path.join(self.project_dir, "English-Cypher/processed", output_filename)
        self.prompt_format = prompt_format
    
    def parse_raw_data(self):
        # load and parse each txt file into jsonl file
        for filepath in os.listdir(self.data_dir):
            # check if file is a txt file
            if filepath.endswith(".txt"):
                converter = JSONConverter(os.path.join(self.data_dir, filepath))
                converter.read_and_split()
                converter.convert_to_json(self.prompt_format)
                converter.convert_to_jsonl()
        
    def concatenate_jsonl(self):
        all_jsonl_data = []
        # Extract all jsonl data from all files
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".jsonl"):
                with open(os.path.join(self.data_dir, file_name), 'r') as file:
                    for line in file:
                        all_jsonl_data.append(json.loads(line))
        
        # Number of total samples in output JSONL dataset
        print(f"Total number of raw samples: {len(all_jsonl_data)}")

        # Write all jsonl data to output file
        with open(self.output_filepath, 'w') as out_file:
            for item in all_jsonl_data:
                out_file.write(json.dumps(item))
                out_file.write("\n")
        
        # Delete temporal json files generated in the process
        # Use os.listdir to get a list of files in the folder
        jsonl_files = os.listdir(self.data_dir)
        # Loop through the files and delete JSON files
        for file_name in jsonl_files:
            if file_name.endswith('.jsonl'):
                file_path = os.path.join(self.data_dir, file_name)
                os.remove(file_path)

        return self.output_filepath
    

    def remove_duplicates(self):
        unique_lines = set()
        with open(self.output_filepath, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                unique_lines.add(json.dumps(json_obj, sort_keys=True))
        with open(self.output_filepath, 'w') as f:
            for line in unique_lines:
                f.write(line + '\n')
        print(f"Total number of unique samples: {len(unique_lines)}")
                

                
class JSONConverter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.data = []
        self.json_data = []

    def read_and_split(self):
        with open(self.filepath, 'r') as file:
            content = file.read()
            # splits by empty lines
            chunks = content.split("\n\n")
            # splits each chunk by line and removes empty chunks
            self.data = [chunk.split("\n") for chunk in chunks if chunk.strip() != ""]
    
    def convert_to_json(self, prompt_format: str, add_inverse: bool = True):
        instruction_response_system_prompt = "Below is a semicolon-separated set of English instructions. Your task is to write a Cypher query that correctly translates the English instructions into a valid Cypher query."
        response_instruction_system_prompt = "Below is a Cypher query that was generated from a semicolon-separated set of English instructions. Your task is to write the semicolon-separated set of English instructions that correctly translates the Cypher query into English instructions."
        for i, chunk in enumerate(self.data):
            if len(chunk) == 2:
                if prompt_format == "llama2":
                    # Instruct-response template
                    # Single turn conversation
                    # <s>[INST] <<SYS>>\n{your_system_message}\n<</SYS>>\n\n{user_message} [/INST] {model_reply}</s>
                    # Multi-turn conversation
                    # <s>[INST] <<SYS>>\n{your_system_message}\n<</SYS>>\n\n{user_message_1} [/INST] {model_reply_1}</s><s>[INST] {user_message_2} [/INST]
                    instruct_response_json_obj = {
                        "text": f"<s>[INST] <<SYS>>\n{str(instruction_response_system_prompt)}\n<</SYS>>\n\n{str(chunk[0])} [/INST] {str(chunk[1])} </s>"
                    }
                    self.json_data.append(instruct_response_json_obj)
                    
                    if add_inverse:
                        # sample using response-instruct template
                        response_instruct_json_obj = {
                            "text": f"<s>[INST] <<SYS>>\n{str(response_instruction_system_prompt)}\n<</SYS>>\n\n{str(chunk[1])} [/INST] {str(chunk[0])} </s>"
                        }
                        self.json_data.append(response_instruct_json_obj)

                elif prompt_format == "alpaca":
                    # sample using instruct-response template
                    instruct_response_json_obj = {
                        "system": str(instruction_response_system_prompt), 
                        "instruction": "##" + str(chunk[0]), 
                        "response": "### " + str(chunk[1])}
                    self.json_data.append(instruct_response_json_obj)
                    
                    # sample using response-instruct template
                    response_instruct_json_obj = {
                        "system": str(response_instruction_system_prompt), 
                        "instruction": "##" + str(chunk[1]), 
                        "response": "### " + str(chunk[0])}
                    self.json_data.append(response_instruct_json_obj)
            else:
                print(f"Error: Chunk {i} in file {self.filename} has {len(chunk)} lines.")

    def convert_to_jsonl(self):
        with open(f"{self.filepath[:-4]}.jsonl", 'w') as jsonl_file:
            for json_obj in self.json_data:
                jsonl_file.write(json.dumps(json_obj))
                jsonl_file.write("\n")

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "data")
    parser = English2CypherDataParser(data_dir, "english2cypher.jsonl", "llama2")
    parser.parse_raw_data()
    parser.concatenate_jsonl()
    parser.remove_duplicates()