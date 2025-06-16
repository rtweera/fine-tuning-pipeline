import unsloth # type: ignore
from unsloth import FastLanguageModel # type: ignore

import os
from transformers import TextStreamer
from datasets import load_dataset
from huggingface_hub import HfApi
import json
from tqdm import tqdm
import pytz
from datetime import datetime
from omegaconf import DictConfig

from logger import Logger 


class Runner:
    def __init__(self, running_config: DictConfig, paths_config: DictConfig, run_name: str, logger=Logger()):
        # logger
        self.logger = logger        
        self.run_name = run_name

        # Configurations assigned from the config file
        self.model_name = running_config.model_id
        self.max_seq_length = running_config.max_seq_length
        self.dtype = running_config.dtype
        self.load_in_4bit = running_config.load_in_4bit
        self.dataset_id = running_config.dataset_id
        self.dataset_columns = running_config.dataset_columns
        self.user_column = running_config.user_column
        self.assistant_column = running_config.assistant_column
        self.system_column = running_config.system_column
        self.system_prompt = running_config.system_prompt
        self.use_system_prompt = running_config.use_system_prompt
        self.max_new_tokens = running_config.max_new_tokens
        self.use_cache = running_config.use_cache
        self.temperature = running_config.temperature
        self.min_p = running_config.min_p
        self.output_file = running_config.output_file
        self.output_dir = running_config.output_dir
        self.output_path = None
        self.hf_user_id = running_config.hf_user_id
        self.output_user_column = running_config.output_user_column
        self.output_assistant_column = running_config.output_assistant_column
        self.output_system_column = running_config.output_system_column

        # load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference
        self.logger.info("Model and tokenizer loaded successfully.")

        # set up text streamer
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)


    def _load_model_and_tokenizer(self):
        """
        Load the model (base or fine tuned) and tokenizer from the specified model name.
        """
        return FastLanguageModel.from_pretrained(
            model_name = self.model_name, 
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
            token = os.getenv("HF_TOKEN")
        )

    def _get_system_prompt(self, example):
        """
        Get the system prompt from the configuration or dataset.
        """
        if self.system_prompt is not None:
            instruction_part = self.system_prompt.strip()
        elif self.system_column is not None:
            instruction_part =  example[self.system_column].strip()
        else:
            instruction_part = None
        return instruction_part

    def _get_messages(self, example):
        """
        Prepare messages for the model.
        """
        instruction_part = self._get_system_prompt(example)
        user_part = example[self.user_column].strip()
        messages = [{"role": "user", "content": user_part}]
        if instruction_part is not None:
            messages.insert(0, {"role": "system", "content": instruction_part})
        tokenized_msg = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(self.model.device)
        return tokenized_msg

    # def __stream_sample(self, sample_question):
    #     """
    #     Stream the model's response to a sample question.
    #     """
    #     self.logger.info("Streaming response for sample question: {}".format(sample_question))
    #     tokenized_msg = self._get_messages(sample_question)
        
    #     self.model.generate(
    #         input_ids=tokenized_msg,
    #         attention_mask=tokenized_msg.attention_mask,
    #         streamer=self.streamer,
    #         max_new_tokens=self.max_new_tokens,
    #         use_cache=self.use_cache,
    #         temperature=self.temperature,
    #         min_p=self.min_p,
    #     )
    #     self.logger.info("Response streaming completed.")
    
    def generate_responses(self, questions):
        """
        Generate responses for each user prompt in the dataset and return as a list of dicts.
        """
        self.logger.info("Generating responses for user prompts in the dataset...")
        results = []
        for prompt in tqdm(questions, desc='Generating responses', unit='prompt'):
            tokenized_msg = self._get_messages(prompt)
            output_ids = self.model.generate(
                input_ids=tokenized_msg,
                max_new_tokens=self.max_new_tokens,
                use_cache=self.use_cache,
                temperature=self.temperature,
                min_p=self.min_p
            )
            output_text = self.tokenizer.decode(
                output_ids[0][tokenized_msg.shape[-1]:], skip_special_tokens=True
            )
            row = {
                self.output_system_column: self._get_system_prompt(prompt),
                self.output_user_column: prompt,
                self.output_assistant_column: output_text
            }
            results.append(row)
        return results

    def store_results(self, results, file_format="csv"):
        """
        Store results locally in the specified format (csv or jsonl).
        """
        import csv
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_path = os.path.join(self.output_dir, self.output_file)
        if file_format == "jsonl":
            with open(self.output_path, "w", encoding="utf-8") as f:
                for row in results:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:  # default to csv
            with open(self.output_path, "w", encoding="utf-8", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["system_prompt", "user_prompt", "assistant"])
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
        self.logger.info(f"Responses written to {self.output_path} as {file_format}.")

    def upload_to_huggingface(self):
        """
        Uploads the generated responses to HuggingFace Hub as a dataset.
        The dataset name will be 'output_<model_id>'.
        """
        self.logger.info("Uploading output jsonl to HuggingFace Hub as a dataset...")
        model_name_without_uid = self.model_name.split('/')[-1]
        now_utc = datetime.now(pytz.utc)
        now_colombo = now_utc.astimezone(pytz.timezone('Asia/Colombo'))
        time_str = now_colombo.strftime('%Y-%b-%d_%H-%M-%S')
        repo_id = f'{time_str}_outputof_{model_name_without_uid}'
    
        api = HfApi(token=os.getenv("HF_TOKEN"))
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            api.upload_folder(
                folder_path=self.output_dir,
                repo_id=f"{self.hf_user_id}/{repo_id}",
                repo_type="dataset",
                commit_message="Upload model output dataset",
            )
        except Exception as e:
            self.logger.error(f"Failed to upload to HuggingFace Hub: {str(e)}")
            raise e
        self.logger.info(f"Output CSV uploaded to HuggingFace Hub as dataset: {repo_id}")


    def get_datasets(self):
        """
        This mf gives a dataset 
        """
        self.logger.info("Downloading dataset...")
        data = load_dataset(self.dataset_id, split='train')
        ...

    def set_datasets(self):
        """
        This mf stroes the dataset
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run_generation_and_store(self, file_format="csv"):
        """
        Download dataset, generate responses, and store them in the specified format.
        """
        self.logger.info("Downloading dataset...")
        data = load_dataset(self.dataset_id, split='train')
        questions = data[self.user_column]
        results = self.generate_responses(questions)
        self.store_results(results, file_format=file_format)
        self.upload_to_huggingface()

if __name__ == "__main__":
    runner = RunnerQwenCommon()
    runner.generate_response()
    runner.upload_to_huggingface()
    print("All tasks completed successfully.")