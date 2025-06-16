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

from logger import Logger 


class RunnerQwenCommon:
    def __init__(self, config_path='running-config.yaml'):
        # logger
        self.logger = Logger()
        self.logger.info("Initializing RunnerQwenCommon...")
        
        # load configuration and environment variables
        self.logger.info("Loading configuration from {}".format(config_path))
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if load_dotenv(override=True, verbose=True):
            self.logger.info("Environment variables loaded successfully.")
        else:
            self.logger.error("Failed to load environment variables. Please check your .env file.")
            raise EnvironmentError("Failed to load environment variables.")

        # Configurations assigned from the config file
        self.logger.info(f"Loading configuration from: {config}")
        self.model_name = config.get('model_id', None)
        self.max_seq_length = config.get('max_seq_length', 2048)
        self.dtype = config.get('dtype', None)
        self.load_in_4bit = config.get('load_in_4bit', True)
        self.dataset_id = config.get('dataset_id', None)
        self.dataset_columns = config.get('dataset_columns', ["question", "answer", "topic"])
        self.user_column = config.get('user_column', "question")
        self.assistant_column = config.get('assistant_column', "answer")
        self.system_column = config.get('system_column', "topic")
        self.system_prompt = config.get('system_prompt', None)
        self.use_system_prompt = config.get('use_system_prompt', False)
        self.max_new_tokens = config.get('max_new_tokens', 2048)
        self.use_cache = config.get('use_cache', True)
        self.temperature = config.get('temperature', 1.5)
        self.min_p = config.get('min_p', 0.1)
        self.output_file = config.get('output_file', 'output.jsonl')
        self.output_dir = config.get('output_dir', 'output')
        self.output_path = None
        self.hf_user_id = config.get('hf_user_id', None)

        # load model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
            token = os.getenv("HF_TOKEN")
        )
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference
        self.logger.info("Model and tokenizer loaded successfully.")

        # set up text streamer
        self.logger.info("Setting up text streamer for model output...")
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.logger.info("Initialization complete.")

    def _get_messages(self, user_prompt):
        """
        Prepare messages for the model.
        """
        messages = []
        if self.use_system_prompt and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        tokenized_msg = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(self.model.device)
        return tokenized_msg

    def stream_sample(self, sample_question):
        """
        Stream the model's response to a sample question.
        """
        self.logger.info("Streaming response for sample question: {}".format(sample_question))
        tokenized_msg = self._get_messages(sample_question)
        
        self.model.generate(
            input_ids=tokenized_msg,
            attention_mask=tokenized_msg.attention_mask,
            streamer=self.streamer,
            max_new_tokens=self.max_new_tokens,
            use_cache=self.use_cache,
            temperature=self.temperature,
            min_p=self.min_p,
        )
        self.logger.info("Response streaming completed.")
    
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

    def generate_response(self):
        """
        Generate responses for each user prompt in the dataset and write to a JSONL file.
        Each line will be a JSON object with keys: system_prompt, user_prompt, assistant.
        If use_system_prompt is False, system_prompt will be set to None.
        Also uploads the JSONL as a dataset to HuggingFace Hub with the name 'output_<model_id>'.
        """
        self.logger.info("Downloading dataset...")
        data = load_dataset(self.dataset_id, split='train')
        questions = data[self.user_column]
        include_system_prompt = self.use_system_prompt and self.system_prompt
        self.logger.info("Generating responses for user prompts in the dataset...")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_path = os.path.join(self.output_dir, self.output_file)
        with open(self.output_path, "w", encoding="utf-8") as f:
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
                    "system_prompt": self.system_prompt if include_system_prompt else None,
                    "user_prompt": prompt,
                    "assistant": output_text
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.logger.info(f"Responses written to {self.output_path}.")

if __name__ == "__main__":
    runner = RunnerQwenCommon()
    runner.generate_response()
    runner.upload_to_huggingface()
    print("All tasks completed successfully.")