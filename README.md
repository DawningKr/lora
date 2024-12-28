## LoRA

This project is a version of LoRA implementation from bottom to Top

Explaining modules in package lora:

- nn:

  - Linear:

    - this is an enhanced version of torch.nn.Linear
    - you can set LoRA parameters by calling the function "set_lora_configs" and then relevant matrices will be created
    - you can start LoRA mode by calling the function "set_lora_status" and pass in True

  - Conv2d:

    - this is an enhanced version of torch.nn.Conv2d

- LoRA:

  - "LoRA" is an Interface for those models using layers with LoRA mode
    - An example is class Model in demo.ipynb
  - "LoRAConfig" is an abstract class for customizing your own LoRA layer
    - In this class, basic fields and methods have been declared. In your own customized LoRA layer, you just need to override them
  - An example is class Linear in Linear.py

- utils:

  - "set_lora_configs_all" is a function that initialize all lora layers in a neural network using the same configuration
  - "lora_state_dict" is a function that returns a state dict only with lora parameters inside as values. Its return value can cooperate with torch.load() and torch.save()

You can find a full demonstration in demo.ipynb

#### Requirements

Pytorch 2.5
