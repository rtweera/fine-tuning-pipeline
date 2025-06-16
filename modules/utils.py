from omegaconf import DictConfig

def make_model_name(model_config: DictConfig):
    """
    Create a model name based on the configuration provided.
    
    Args:
        model_config (DictConfig): Configuration dictionary containing model parameters.
        
    Returns:
        str: A formatted string representing the model name.
    """
    model_id = str(model_config.id)
    model_version = str(model_config.model_version)
    if len(model_id) > 4:
        raise ValueError("Model ID must not exceed 4 digits.")
    padded_id = model_id.zfill(4)
    return f"{padded_id}-model-{model_version}"
