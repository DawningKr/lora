## LoRA

This project is a version of LoRA implementation from bottom to Top

Explaining modules in package lora:

- Linear:
- this is an enhanced version of torch.nn.Linear
- you can set LoRA parameters by calling the function "set_lora_configs" and then relevant matrices will be created
- you can start LoRA mode by calling the function "set_lora_status" and pass in True

- LoRA:
  - "LoRA" is an Interface for those models using layers with LoRA mode
    - An example is class Model in demo.ipynb
  - "LoRAConfig" is an Interface for customizing your own LoRA layer
    - An example is class Linear in Linear.py

You can find an full demonstration in demo.ipynb

#### Requirements

Pytorch 2.5
