from abc import ABC, abstractmethod

class LoRA(ABC):
    @abstractmethod
    def get_lora_layers(self):
        pass

class LoRAConfig(ABC):
    @abstractmethod
    def set_lora_configs(self, rank, alpha):
        pass

    @abstractmethod
    def set_lora_status(self, enable_lora: bool):
        pass

    @abstractmethod
    def _set_params_status(self, freeze_params: bool):
        pass