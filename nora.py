import torch
import torch.nn as nn
from rational_triton import RationalTriton1DGroup
from rational_triton2d import RationalTriton2D
from rational_1dgroup_torch import Rational_CUDA_A_1DGroup
import json
import os
import math

class NoRA(nn.Module):
    def __init__(self, original_layer, r=3, lora_alpha=6, device='cuda', group_extend=None):
        super().__init__()
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad_(False)
        self.device = device
        if group_extend is not None:
            self.num_groups = original_layer.num_groups * group_extend
            self.register_buffer('base_denominator', original_layer.weight_denominator.detach().clone().repeat(group_extend, 1))
        else:
            self.num_groups = original_layer.num_groups
            self.register_buffer('base_denominator', original_layer.weight_denominator.detach().clone())


        self.register_buffer('base_numerator', original_layer.weight_numerator.detach().clone())
        # self.register_buffer('base_denominator', original_layer.weight_denominator.detach().clone())

        
        if r > 0:
            self.lora_A_numerator = nn.Parameter(torch.zeros(r, 6, device=device))
            self.lora_B_numerator = nn.Parameter(torch.zeros(self.num_groups, r, device=device))
            self.lora_A_denominator = nn.Parameter(torch.zeros(r, 4, device=device))
            self.lora_B_denominator = nn.Parameter(torch.zeros(self.num_groups, r, device=device))
            
            self.register_buffer('scaling_numerator', 
                               torch.tensor(lora_alpha / r, device=device))
            self.register_buffer('scaling_denominator', 
                               torch.tensor(lora_alpha / r, device=device))

            self.reset_lora_parameters()
        
        self.rational = RationalTriton1DGroup.apply if self.device == "cuda" else Rational_CUDA_A_1DGroup

    def reset_lora_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A_numerator, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_numerator)

            nn.init.kaiming_uniform_(self.lora_A_denominator, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_denominator)

    def forward(self, x):

        if self.r > 0 and not self.merged:
            lora_numerator = self.lora_B_numerator @ self.lora_A_numerator
            if self.lora_dropout is not None:
                lora_numerator = self.lora_dropout(lora_numerator)
            lora_numerator = lora_numerator * self.scaling_numerator

            numerator = self.base_numerator + lora_numerator

            lora_denominator = self.lora_B_denominator @ self.lora_A_denominator
            if self.lora_dropout is not None:
                lora_denominator = self.lora_dropout(lora_denominator)
            lora_denominator = lora_denominator * self.scaling_denominator
            denominator = self.base_denominator + lora_denominator + 1e-6
        else:
            numerator = self.base_numerator
            denominator = self.base_denominator
        
        return self.rational(x, numerator, denominator, self.num_groups)