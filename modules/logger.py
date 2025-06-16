import os 
import requests
import json
from transformers.trainer_callback import TrainerCallback
import time

class Logger:
    def __init__(self):
        self.endpoint = os.getenv("BETTERSTACK_URL")
        self.bearer_token = os.getenv("BETTERSTACK_BEARER_TOKEN")
        if not self.endpoint or not self.bearer_token:
            raise ValueError("BETTERSTACK_URL and BETTERSTACK_BEARER_TOKEN must be set in the environment variables.")
        self.headers = {
            "Authorization": self.bearer_token,
            "Content-Type": "application/json"
        }

    def log(self, message, level="info"):
        try:
            payload = {
                "dt": r"$(date -u +'%Y-%m-%d %T UTC')",
                "message": f"[{level}] {message}",
            }
            response = requests.post(
                self.endpoint if self.endpoint else "",
                headers=self.headers,
                data=json.dumps(payload),
                verify=True  # Disable SSL verification (this is in the curl they have given)
            )
            if response.status_code == 202:
                return True
            else:
                print(f"Failed to send log. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"Error sending log: {str(e)}")
            return False
    def info(self, message):
        """Log an INFO level message"""
        return self.log(message, "INFO")
    
    def error(self, message):
        """Log an ERROR level message"""
        return self.log(message, "ERROR")
    
    def warning(self, message):
        """Log a WARNING level message"""
        return self.log(message, "WARNING")
    
    def debug(self, message):
        """Log a DEBUG level message"""
        return self.log(message, "DEBUG")
    

class LoggerCallback(TrainerCallback):
    def __init__(self, logger: Logger, log_interval_seconds: int = 300):
        self.logger = logger
        self.start_time = None
        self.last_log_time = None
        self.log_interval_seconds = log_interval_seconds
        self.logger.info("LoggerCallback initialized with log interval of {} seconds.".format(log_interval_seconds))

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def on_step_end(self, args, state, control, **kwargs):
        """
        WARNING: use num_train_epochs instead of num_train_steps, else this might throw an error
        """
        now = time.time()
        if self.start_time is None:
            self.start_time = now
        if self.last_log_time is None:
            self.last_log_time = now
        elapsed = now - self.start_time
        since_last_log = now - self.last_log_time
        if since_last_log >= self.log_interval_seconds:
            percent_epoch = 100.0 * state.epoch / args.num_train_epochs if state.epoch is not None else 0
            epoch_str = f"{state.epoch:.2f}" if state.epoch is not None else "N/A"
            msg = (
                f"Epoch: {epoch_str}, "
                f"Step: {state.global_step}/{state.max_steps}, "
                f"Epoch Progress: {percent_epoch:.2f}%, "
                f"Elapsed: {elapsed/60:.2f} min"
            )
            self.logger.info(msg)
            self.last_log_time = now
