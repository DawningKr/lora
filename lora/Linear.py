from torch import nn
from abc import ABC, abstractmethod
from .LoRA import LoRAConfig

class Linear(nn.Module, LoRAConfig):
    def __init__(self, features_in, features_out) -> None:
        super().__init__()      
        self.features_in = features_in
        self.features_out = features_out
        self.layer = nn.Linear(self.features_in, self.features_out)
        self.enable_lora = False
        self.lora_A = None
        self.lora_B = None
        self.alpha = None
        self.rank = None
        self.scale = None
    
    def forward(self, X):
        if self.enable_lora:
            return self.layer(X) + self.lora_B(self.lora_A(X)) * self.scale
        else:
            return self.layer(X)
    
    def set_lora_configs(self, rank, alpha):
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_A = nn.Linear(self.features_in, rank, bias=False)
        nn.init.normal_(self.lora_A)
        self.lora_B = nn.Linear(self.rank, self.features_out, bias=False)
        nn.init.zeros_(self.lora_B)
    
    def set_lora_status(self, enable_lora: bool):
        self.enable_lora = enable_lora
        self._set_params_status(enable_lora)
    
    def _set_params_status(self, freeze_params: bool):
        for param in self.layer.parameters():
            param.requires_grad = not freeze_params
    