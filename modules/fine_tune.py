import unsloth  # type: ignore
from unsloth import FastLanguageModel, is_bfloat16_supported # type: ignore
from unsloth.chat_templates import get_chat_template, train_on_responses_only # type: ignore

# import torch
import datasets
import os
import sys
import wandb
import pandas as pd

from datasets import Dataset
from huggingface_hub import login
from omegaconf import DictConfig
from trl import SFTTrainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


from logger import Logger, LoggerCallback  # Assuming you have a logger module


class FineTune:
    def __init__(self, *, tuning_config: DictConfig, paths_config: DictConfig, run_name: str, logger=Logger()):
        # Logger
        self.logger = logger
        self.run_name = run_name
        self.wandb_run_name = None

        # Configurations (all required, no defaults)
        self.models = tuning_config.models
        self.model_key = tuning_config.model_key
        self.model_name = self.models[self.model_key]
        self.max_seq_length = tuning_config.max_seq_length
        self.dtype = tuning_config.dtype
        self.load_in_4bit = tuning_config.load_in_4bit
        self.rank = tuning_config.rank
        self.dataset_id = tuning_config.dataset_id
        self.dataset_columns = tuning_config.dataset_columns
        self.user_column = tuning_config.user_column
        self.assistant_column = tuning_config.assistant_column
        self.system_column = tuning_config.system_column
        self.system_prompt = tuning_config.system_prompt
        self.run_name_prefix = tuning_config.wandb_run_name_prefix
        self.run_name_suffix = tuning_config.wandb_run_name_suffix
        self.project_name = tuning_config.wandb_project_name
        self.device_batch_size = tuning_config.device_batch_size
        self.grad_accumulation = tuning_config.grad_accumulation
        self.epochs = tuning_config.epochs
        self.logger_log_interval_seconds = tuning_config.logger_log_interval_seconds
        self.learning_rate = tuning_config.learning_rate
        self.warmup_steps = tuning_config.warmup_steps
        self.optim = tuning_config.optim
        self.weight_decay = tuning_config.weight_decay
        self.lr_scheduler_type = tuning_config.lr_scheduler_type
        self.seed = tuning_config.seed
        self.logging_steps = tuning_config.logging_steps
        self.logging_first_step = tuning_config.logging_first_step
        self.save_steps = tuning_config.save_steps
        self.save_total_limit = tuning_config.save_total_limit
        self.push_to_hub = tuning_config.push_to_hub
        self.packing = tuning_config.packing
        self.report_to = tuning_config.report_to
        self.dataset_num_proc = tuning_config.dataset_num_proc
        self.target_modules = tuning_config.target_modules
        self.lora_alpha = tuning_config.lora_alpha
        self.lora_dropout = tuning_config.lora_dropout
        self.bias = tuning_config.bias
        self.use_gradient_checkpointing = tuning_config.use_gradient_checkpointing
        self.use_rslora = tuning_config.use_rslora
        self.loftq_config = tuning_config.loftq_config
        self.instruction_part = tuning_config.instruction_part
        self.response_part = tuning_config.response_part
        self.online_logger_interval_seconds = tuning_config.online_logger_interval_seconds
        self.online_logger_log_level = tuning_config.online_logger_log_level
        self.train_dir = paths_config.train_dir
        self.validation_dir = paths_config.validation_dir
        self.models_dir = paths_config.models_dir

        # Constants
        self.CONVERSATIONS_KEY = "conversations"  # Key for conversations when transforming dataset (refer to _convert_to_conversations method)
        self.TEXTS_KEY = "text"  # Key for text when formatting prompts (refer to _formatting_prompts_func method)


    def _load_base_model_and_tokenizer(self) -> tuple[FastLanguageModel, PreTrainedTokenizerBase]:
        """
        Load the base model and tokenizer from the specified model name.
        Returns:
            FastLanguageModel: The loaded model.
            Tokenizer: The tokenizer associated with the model.
        """
        self.logger.info(f"Loading base model and tokenizer for {self.model_name}...")
        return FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
            token = os.getenv("HF_TOKEN")
        )

    def _get_peft_model(self) -> FastLanguageModel:
        """
        Convert the loaded model to a PEFT (Parameter-Efficient Fine-Tuning) model.
        Returns:
            FastLanguageModel: The PEFT model.
        """
        self.logger.info(f"Converting model to PEFT with rank {self.rank}...")
        return FastLanguageModel.get_peft_model(
            self.model,
            r = self.rank,
            target_modules = self.target_modules,
            lora_alpha = self.lora_alpha,
            lora_dropout = self.lora_dropout,
            bias = self.bias,
            use_gradient_checkpointing = self.use_gradient_checkpointing,
            random_state = self.seed,
            use_rslora = self.use_rslora,
            loftq_config = self.loftq_config,
        )

    def _convert_to_conversations(self, example) -> dict:
        """
        Convert a single example to a conversation format.
        Args:
            example (dict): A single example from the dataset.
        Returns:
            dict: A dictionary containing the conversation format (System prompt is optional).
        Example:
            {
                "conversations": [
                    {"role": "system", "content": "System prompt here"},
                    {"role": "user", "content": "User question here"},
                    {"role": "assistant", "content": "Assistant answer here"}
                ]
            }
        """
        if self.system_prompt is not None:
            instruction_part = self.system_prompt.strip()
        elif self.system_column is not None:
            instruction_part = example[self.system_column].strip()
        else:
            instruction_part = None
        user_part = example[self.user_column].strip()
        assistant_part = example[self.assistant_column].strip()
        output = {
            self.CONVERSATIONS_KEY: [
                {"role": "user", "content": user_part},
                {"role": "assistant", "content": assistant_part}
            ]
        }
        if instruction_part is not None:
            output[self.CONVERSATIONS_KEY].insert(0, {"role": "system", "content": instruction_part})
        return output


    def _formatting_prompts_func(self, examples):
        """
        Format the conversations for training by applying the chat template.
        Args:
            examples (dict): A batch of examples from the dataset.
        Returns:
            dict: A dictionary containing the formatted text for training.
        Example:
            {
                "text": [
                    "<|im_start|>system\nSystem prompt here<|im_end|>\n<|im_start|>user\nUser question here<|im_end|>\n<|im_start|>assistant\n",
                    "<|im_start|>system\nSystem prompt here<|im_end|>\n<|im_start|>user\nUser question here<|im_end|>\n<|im_start|>assistant\n",
                    ...
                ]
            }
        """
        self.logger.info("Formatting prompts for training...")
        convos = examples[self.CONVERSATIONS_KEY]
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            # TODO: What to do if tokenizer does not have apply_chat_template method?
            raise ValueError("Tokenizer does not have 'apply_chat_template' method. Ensure you are using a compatible tokenizer.")
        texts = [self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        # NOTE: Why do we put add_generation_prompt=False? Because we are not generating text here? Look it up
        return { self.TEXTS_KEY : texts, }
        

    def _handle_wandb_setup(self):
        """
        Handle the setup for Weights & Biases (wandb) logging.
        Returns:
            str: The run name used for the Weights & Biases run.
        """
        self.logger.info("Setting up Weights & Biases...")
        wandb.login(key=os.getenv('WANDB_TOKEN'))
        
        self.wandb_run_name = self.run_name
        if self.run_name_prefix is not None:
            self.wandb_run_name = f'{self.run_name_prefix}_{self.wandb_run_name}'
        if self.run_name_suffix is not None:
            self.wandb_run_name = f'{self.wandb_run_name}_{self.run_name_suffix}'
        self.logger.info(f"Run name set: {self.wandb_run_name}")

        wandb.init(project=self.project_name, name=self.wandb_run_name)
        self.logger.info("Weights & Biases setup complete.")

    def _login_huggingface(self):
        """
        Log in to Hugging Face using the token from environment variables.
        Raises:
            ValueError: If the Hugging Face token is not set in the environment variables.
        """
        self.logger.info("Logging in to Hugging Face...")
        login(token=os.getenv("HF_TOKEN"))
        self.logger.info("Hugging Face login successful.")

    def _load_local_dataset(self, data_dir):
        """
        Load a dataset from a local directory. Assumes there is one file in the directory.
        Returns a pandas DataFrame.
        """
        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        if not files:
            raise FileNotFoundError(f"No data file found in {data_dir}")
        file_path = os.path.join(data_dir, files[0])
        self.logger.info(f"Loading data from {file_path}")
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
            return pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def _convert_df_to_hf_dataset(self, df) -> datasets.arrow_dataset.Dataset:
        return Dataset.from_pandas(df)

    def run(self):
        """
        Run the fine-tuning process.
        Returns:
            TrainerStats: The statistics from the training process.
        """
        # Load local training and validation data
        train_df = self._load_local_dataset(self.train_dir)
        val_df = self._load_local_dataset(self.validation_dir)
        train_dataset = self._convert_df_to_hf_dataset(train_df)
        val_dataset = self._convert_df_to_hf_dataset(val_df)

        # Initialize model and tokenizer
        self.logger.info("Initializing model and tokenizer...")
        self.model, self.tokenizer = self._load_base_model_and_tokenizer()
        self.model = self._get_peft_model()

        # Data operations
        # strip doesnt work with batched=True, so we use batched=False
        train_dataset = train_dataset.map(self._convert_to_conversations, remove_columns=self.dataset_columns, batched=False)   
        train_dataset = train_dataset.map(self._formatting_prompts_func, batched=True)
        val_dataset = val_dataset.map(self._convert_to_conversations, remove_columns=self.dataset_columns, batched=False)
        val_dataset = val_dataset.map(self._formatting_prompts_func, batched=True)

        self._handle_wandb_setup()

        self.logger.info("Configuring trainer...")
        # Training
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = train_dataset,
            eval_dataset = val_dataset,
            dataset_text_field = self.TEXTS_KEY,
            max_seq_length = self.max_seq_length,
            data_collator = DataCollatorForSeq2Seq(tokenizer = self.tokenizer),
            dataset_num_proc = self.dataset_num_proc,
            packing = self.packing,
            args = TrainingArguments(
                per_device_train_batch_size = self.device_batch_size,
                gradient_accumulation_steps = self.grad_accumulation,
                warmup_steps = self.warmup_steps,
                num_train_epochs = self.epochs,
                learning_rate = self.learning_rate,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = self.logging_steps,
                logging_first_step=self.logging_first_step,
                optim = self.optim,
                weight_decay = self.weight_decay,
                lr_scheduler_type = self.lr_scheduler_type,
                seed = self.seed,
                output_dir = self.models_dir,  # Save checkpoints and outputs to local models dir
                report_to = self.report_to,
                save_steps=self.save_steps,
                save_total_limit=self.save_total_limit,
                push_to_hub=self.push_to_hub,
                hub_model_id=self.run_name
            ),
            callbacks = [LoggerCallback(logger=self.logger, log_interval_seconds=60)]
        )

        self.logger.info("Setting the loss for generated tokens only...")
        trainer = train_on_responses_only(
            trainer,
            instruction_part = self.instruction_part,
            response_part = self.response_part,
        )

        self.logger.info("Starting training...")
        trainer_stats = trainer.train()
        self.logger.info("Training completed successfully.")
        return trainer_stats

if __name__ == "__main__":
    print("[ERROR] Please run the pipeline using main.py, not modules/fine_tune.py.")
    sys.exit(1)

