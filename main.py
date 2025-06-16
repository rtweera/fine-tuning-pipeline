import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

from modules.utils import make_model_name
from modules.fine_tune import FineTune


logger = None 

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Generate a new run name based on the model configuration
    run_name = make_model_name(cfg.model)
    # Setup the fine tune instance with the provided configuration
    fine_tune_instance = FineTune(tuning_config=cfg.tuning, paths_config=cfg.paths, run_name=run_name)
    # Start the fine-tuning process
    fine_tune_instance.run()


if __name__ == "__main__":
    load_dotenv(verbose=True, override=True)  # Load environment variables from .env file
    main()