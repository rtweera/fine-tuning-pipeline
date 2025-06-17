import unsloth # type: ignore
from unsloth import FastLanguageModel # type: ignore

import csv
import glob
import json
import os
import sys
from enum import Enum
from huggingface_hub import HfApi
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm


from logger import Logger 


class Runner:
    def __init__(self, running_config: DictConfig, paths_config: DictConfig, run_name: str, logger=Logger()):
        # logger
        self.logger = logger        
        self.run_name = run_name

        # Configurations assigned from the config file
        self.base_model_id = running_config.base_model_id   # id can be either HuggingFace model name or local path (local takes precedence in case of same name)
        self.fine_tuned_model_id = paths_config.models_dir  # REFER to the base_model_id comment
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
        self.hf_user_id = running_config.hf_user_id
        self.output_user_column = running_config.output_user_column
        self.output_assistant_column = running_config.output_assistant_column
        self.output_system_column = running_config.output_system_column
        self.test_dir = paths_config.test_dir
        self.model_results_dir = paths_config.model_results_dir
        self.other_results_dir = paths_config.other_results_dir
        self.base_model_results_file = running_config.base_model_results_file
        self.fine_tuned_model_results_file = running_config.fine_tuned_model_results_file
        
        self.current_model = None   # This will be set to either base or fine-tuned model
        self.tokenizer = None # Both base and fine-tuned models will use the same tokenizer


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
        Prepare messages for the model when given the example row.
        """
        instruction_part = self._get_system_prompt(example)
        user_part = example[self.user_column].strip()
        messages = [{"role": "user", "content": user_part}]
        if instruction_part is not None:
            messages.insert(0, {"role": "system", "content": instruction_part})
        if self.tokenizer is None or self.current_model is None:
            raise ValueError("Model or tokenizer is not initialized. Please load the model and tokenizer first.")
        
        tokenized_msg = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(self.current_model.device)
        return tokenized_msg
    

    def generate_responses(self, data):
        """
        Generate responses for each user prompt in the dataset and return as a list of dicts.
        """
        if self.current_model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer is not initialized. Please load the model and tokenizer first.")
        self.logger.info("Generating responses for user prompts in the dataset...")
        results = []
        for example in tqdm(data, desc='Generating responses', unit='row'):
            tokenized_msg = self._get_messages(example)
            output_ids = self.current_model.generate(
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
                self.output_system_column: self._get_system_prompt(example),
                self.output_user_column: example[self.user_column],
                self.output_assistant_column: output_text
            }
            results.append(row)
        return results
    

    def store_base_results(self, results):
        """
        Store results for the base model in the specified output directory.
        """
        self.output_path = os.path.join(self.other_results_dir, self.base_model_results_file)
        self._store_results(results, self.output_path)


    def store_fine_tuned_results(self, results):
        """
        Store results for the fine-tuned model in the specified output directory.
        """
        self.output_path = os.path.join(self.model_results_dir, self.fine_tuned_model_results_file)
        self._store_results(results, self.output_path)


    def _store_results(self, results, file_path):
        """
        Store results locally in the specified format (csv or jsonl).
        """
        if file_path.endswith('.csv'):
            with open(file_path, "w", encoding="utf-8", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[self.output_system_column, self.output_user_column, self.output_assistant_column])
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
        elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
            with open(file_path, "w", encoding="utf-8") as f:
                for row in results:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        self.logger.info(f"Responses written to {self.output_path}.")

    # def upload_to_huggingface(self):
    #     """
    #     Uploads the generated responses to HuggingFace Hub as a dataset.
    #     The dataset name will be 'output_<model_id>'.
    #     """
    #     self.logger.info("Uploading output jsonl to HuggingFace Hub as a dataset...")
    #     model_name_without_uid = self.model_name.split('/')[-1]
    #     now_utc = datetime.now(pytz.utc)
    #     now_colombo = now_utc.astimezone(pytz.timezone('Asia/Colombo'))
    #     time_str = now_colombo.strftime('%Y-%b-%d_%H-%M-%S')
    #     repo_id = f'{time_str}_outputof_{model_name_without_uid}'
    
    #     api = HfApi(token=os.getenv("HF_TOKEN"))
    #     try:
    #         api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    #         api.upload_folder(
    #             folder_path=self.output_dir,
    #             repo_id=f"{self.hf_user_id}/{repo_id}",
    #             repo_type="dataset",
    #             commit_message="Upload model output dataset",
    #         )
    #     except Exception as e:
    #         self.logger.error(f"Failed to upload to HuggingFace Hub: {str(e)}")
    #         raise e
    #     self.logger.info(f"Output CSV uploaded to HuggingFace Hub as dataset: {repo_id}")


    def load_local_dataset(self):
        """
        Load a dataset from a local folder (expects a single file in the folder).
        """
        files = glob.glob(os.path.join(self.test_dir, '*'))
        if not files:
            raise FileNotFoundError(f"No files found in {self.test_dir}")
        file_path = files[0]
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        return df

    def load_finetuned_model(self):
        """
        Load a fine-tuned model from a local directory using Unsloth.
        """
        self.current_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.fine_tuned_model_id,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            token=os.getenv("HF_TOKEN")
        )
        FastLanguageModel.for_inference(self.current_model)


    def load_base_model(self):
        """
        Load the base model from HuggingFace or local directory.
        """
        self.current_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_id,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            token=os.getenv("HF_TOKEN")
        )
        FastLanguageModel.for_inference(self.current_model)


    def run_local_eval(self):
        test_df = self.load_local_dataset()
        # Load fine-tuned model
        self.load_finetuned_model()
        # Run fine-tuned model on test set
        results = self.generate_responses(test_df)
        # Store results in model-results
        self.store_fine_tuned_results(results)

        # Load base model
        self.load_base_model()
        # Run base model on test set
        base_results = self.generate_responses(test_df)
        # Store results in other-results
        self.store_base_results(base_results)
        
        self.logger.info("Local evaluation complete. Results saved.")

if __name__ == "__main__":
    print("[ERROR] Please run the pipeline using main.py, not modules/runner.py.")
    sys.exit(1)