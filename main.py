import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    model_results_dir = cfg.model_results_dir
    # Use model_results_dir in your code
    print(f"Model results will be saved to: {model_results_dir}")

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file
    main()