from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Mapping device to CPU
DEVICE_MAP = {"": "cpu"}

def get_model_param_list(model_names: List[str]) -> List[Dict]:
    """Loads models and returns a list of their parameters.
    
    Args:
        model_names: A list of model names or paths.
        
    Returns:
        A list of state dictionaries of the models.
    """
    model_param_list = []
    for name in model_names:
        print(f"Load {name}")
        model = AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=True, device_map=DEVICE_MAP
        )
        model_param_list.append(model.state_dict())
    
    return model_param_list

def merge_param(model_param_list: List[Dict], weights: List[float]) -> Dict:
    """Merges parameters of the models based on the given weights.
    
    Args:
        model_param_list: A list of model state dictionaries.
        weights: A list of floats representing the weight for each model.
        
    Returns:
        A dictionary containing the merged parameters.
    """
    new_param = {}
    for k in model_param_list[0].keys():
        for w, param in zip(weights, model_param_list):
            if param[k].dtype in (torch.int64, torch.int32):
                new_param[k] = param[k]
            elif k not in new_param:
                new_param[k] = w * param[k]
            else:
                new_param[k] += w * param[k]
    
    return new_param

def mix_models(model_names_or_paths: List[str], 
               weights: List[float], 
               output_path: str = None):
    """Merges multiple models based on the given weights and saves the result.

    Args:
        model_names_or_paths: A list of model names or paths to be merged.
        weights: A list of floats representing the weight for each model.
        output_path: The path where the new model will be saved.
    
    Returns:
        The merged model.
    """
    assert len(model_names_or_paths) == len(weights), "The number of models and weights must match."
    assert abs(sum(weights) - 1) <= 1e-3, "Weights must sum to 1."

    param_list = get_model_param_list(model_names_or_paths)
    new_param = merge_param(param_list, weights=weights)

    print("Weight for each model: ")
    for w, n in zip(weights, model_names_or_paths):
        print(n, w)

    model = AutoModelForCausalLM.from_pretrained(
        model_names_or_paths[0], trust_remote_code=True, device_map=DEVICE_MAP
    )
    model.load_state_dict(new_param)

    if output_path is not None:
        print(f"Saving merged model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer = AutoTokenizer.from_pretrained(model_names_or_paths[0], trust_remote_code=True)
        tokenizer.save_pretrained(output_path)

    return model
