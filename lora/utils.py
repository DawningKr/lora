from .LoRA import LoRA

def lora_state_dict(model: LoRA):
    state_dict = model.state_dict()
    return {key: state_dict[key] for key in state_dict.keys() if 'lora' in key}

def set_lora_configs_all(model: LoRA, rank: int, alpha: int, enable_lora=True):
    # TODO: check whether model has attribute "get_lora_layers" or not
    lora_layers = model.get_lora_layers()
    for layer in lora_layers:
        layer.set_lora_configs(rank, alpha)
        if enable_lora:
            layer.set_lora_status(True)
    return model