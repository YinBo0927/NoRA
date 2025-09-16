import torch
import torch.nn as nn
from rational_triton import RationalTriton1DGroup
from rational_triton2d import RationalTriton2D
from rational_1dgroup_torch import Rational_CUDA_A_1DGroup
import json
import os

class Rational_Group1d(nn.Module):
    def __init__(self, num_groups=8, mode="gelu", device="cuda"):
        """
        Initialize the Rational_Group module.

        Args:
            num_groups (int): Number of groups for separate processing of input.
            mode (str): Initialization mode, determines weights preset from JSON file.
            device (str): Device to run the module on ('cuda' or 'cpu').
        """
        super(Rational_Group1d, self).__init__()
        assert device in ["cuda", "cpu"], "Device must be either 'cuda' or 'cpu'."
        
        self.order = (5, 4)
        self.num_groups = num_groups

        # Initialize weights based on the given mode
        self.initialize(mode=mode)
        
        # Set the appropriate rational function based on the device
        if device == "cuda":
            self.rational = RationalTriton1DGroup.apply
            # print("\n\nUsing Triton when cuda\n\n")
        else:
            self.rational = Rational_CUDA_A_1DGroup
            # print("\n\nUsing CUDA_A when cpu\n\n")
            
    def init_info(self):
        """
        Load weight initialization information from a JSON file.

        Returns:
            dict: Data loaded from the JSON file.
        """
        cfd = os.path.dirname(os.path.realpath(__file__))
        with open(f'{cfd}/init.json') as json_file:
            data = json.load(json_file)
        return data
                
    def initialize(self, mode="gelu"):
        """
        Initialize weights from a JSON file based on the specified mode.

        Args:
            mode (str): The initialization mode to use.
        """
        cfd = os.path.dirname(os.path.realpath(__file__))
        try:
            with open(f'{cfd}/init.json') as json_file:
                data = json.load(json_file)

            # Extract weights from the JSON data
            weight_numerator = torch.tensor(data[mode]["init_w_numerator"]).view(1, -1)
            weight_denominator = torch.tensor(data[mode]["init_w_denominator"])
            weight_denominator = torch.cat([weight_denominator] * self.num_groups).view(self.num_groups, -1)
             
            # Register weights as trainable parameters
            self.weight_numerator = nn.Parameter(weight_numerator.float(), requires_grad=True)
            self.weight_denominator = nn.Parameter(weight_denominator.float(), requires_grad=True) 

        except FileNotFoundError:
            print("Initialization JSON file not found.")
        except json.JSONDecodeError:
            print("Error decoding JSON.")
            
    def forward(self, input):
        """
        Forward pass of the module.

        Args:
            input (Tensor): 3D or 2D input tensor.

        Returns:
            Tensor: Processed tensor after applying rational function.
        """
        assert input.dim() == 3 or input.dim() == 2, "Input tensor must be 3D (batch, length, channels) or 2D (batch, channels)."
    
    
        # Repeat the weights for all groups
        weight_numerator = self.weight_numerator.repeat(self.num_groups, 1)
        return self.rational(input, weight_numerator, self.weight_denominator, self.num_groups)
        
    def extra_repr(self):
        """
        Extra representation of the module for debugging.

        Returns:
            str: String representation of the module's configuration.
        """
        return f'num_groups={self.num_groups}, order={self.order}'


# class Rational_Group2d(nn.Module):
#     def __init__(self, num_groups=8, mode="gelu", device="cuda"):
#         """
#         Initialize the KAT_Group module.

#         Args:
#             num_groups (int): Number of groups for separate processing of input.
#             mode (str): Initialization mode, determines weights preset from JSON file.
#             device (str): Device to run the module on ('cuda' or 'cpu').
#         """
#         super(Rational_Group2d, self).__init__()
#         assert device in ["cuda", "cpu"], "Device must be either 'cuda' or 'cpu'."
        
#         self.order = (5, 4)
#         self.num_groups = num_groups

#         # Initialize weights based on the given mode
#         self.initialize(mode=mode)
        
#         # Set the appropriate rational function based on the device
#         if device == "cuda":
#             self.rational = RationalTriton2D.apply
#         else:
#             assert False, "Only CUDA is supported for 2D group-wise processing."
            
#     def init_info(self):
#         """
#         Load weight initialization information from a JSON file.

#         Returns:
#             dict: Data loaded from the JSON file.
#         """
#         cfd = os.path.dirname(os.path.realpath(__file__))
#         with open(f'{cfd}/init.json') as json_file:
#             data = json.load(json_file)
#         return data
                
#     def initialize(self, mode="gelu"):
#         """
#         Initialize weights from a JSON file based on the specified mode.

#         Args:
#             mode (str): The initialization mode to use.
#         """
#         cfd = os.path.dirname(os.path.realpath(__file__))
#         try:
#             with open(f'{cfd}/init.json') as json_file:
#                 data = json.load(json_file)

#             # Extract weights from the JSON data
#             weight_numerator = torch.tensor(data[mode]["init_w_numerator"]).view(1, -1)
#             weight_denominator = torch.tensor(data[mode]["init_w_denominator"])
#             weight_denominator = torch.cat([weight_denominator] * self.num_groups).view(self.num_groups, -1)
             
#             # Register weights as trainable parameters
#             self.weight_numerator = nn.Parameter(weight_numerator.float(), requires_grad=True)
#             self.weight_denominator = nn.Parameter(weight_denominator.float(), requires_grad=True) 

#         except FileNotFoundError:
#             print("Initialization JSON file not found.")
#         except json.JSONDecodeError:
#             print("Error decoding JSON.")
            
#     def forward(self, input):
#         """
#         Forward pass of the module.

#         Args:
#             input (Tensor): 3D input tensor with shape (batch, length, channels).

#         Returns:
#             Tensor: Processed tensor after applying rational function.
#         """
#         assert input.dim() == 4, "Input tensor must be 3D (batch, length, channels)."
    
#         # Repeat the weights for all groups
#         weight_numerator = self.weight_numerator.repeat(self.num_groups, 1)
#         return self.rational(input, weight_numerator, self.weight_denominator, self.num_groups)
        
#     def extra_repr(self):
#         """
#         Extra representation of the module for debugging.

#         Returns:
#             str: String representation of the module's configuration.
#         """
#         return f'num_groups={self.num_groups}, order={self.order}'
import torch
import torch.nn as nn
import json
import os
from rational_triton import RationalTriton1DGroup
from rational_triton2d import RationalTriton2D
from rational_1dgroup_torch import Rational_CUDA_A_1DGroup

class Rational_Group2d(nn.Module):
    def __init__(self, num_groups=8, mode="gelu", device="cuda"):
        """
        Unified Rational Group module. Automatically handles 1D or 2D input.

        Args:
            num_groups (int): Number of groups for separate processing of input.
            mode (str): Initialization mode from JSON file.
            device (str): Device to run the module on ('cuda' or 'cpu').
        """
        super(Rational_Group2d, self).__init__()
        assert device in ["cuda", "cpu"], "Device must be either 'cuda' or 'cpu'."
        
        self.order = (5, 4)
        self.num_groups = num_groups
        self.device = device

        # Initialize weights
        self.initialize(mode=mode)

        # Set rational function for 1D and 2D
        self.rational_1d = RationalTriton1DGroup.apply if device == "cuda" else Rational_CUDA_A_1DGroup
        self.rational_2d = RationalTriton2D.apply if device == "cuda" else None

    def init_info(self):
        cfd = os.path.dirname(os.path.realpath(__file__))
        with open(f'{cfd}/init.json') as json_file:
            data = json.load(json_file)
        return data

    def initialize(self, mode="gelu"):
        cfd = os.path.dirname(os.path.realpath(__file__))
        try:
            with open(f'{cfd}/init.json') as json_file:
                data = json.load(json_file)
            weight_numerator = torch.tensor(data[mode]["init_w_numerator"]).view(1, -1)
            weight_denominator = torch.tensor(data[mode]["init_w_denominator"])
            weight_denominator = torch.cat([weight_denominator] * self.num_groups).view(self.num_groups, -1)
            self.weight_numerator = nn.Parameter(weight_numerator.float(), requires_grad=True)
            self.weight_denominator = nn.Parameter(weight_denominator.float(), requires_grad=True) 
        except FileNotFoundError:
            print("Initialization JSON file not found.")
        except json.JSONDecodeError:
            print("Error decoding JSON.")

    def forward(self, input):
        """
        Forward pass automatically selects 1D or 2D rational function based on input.

        Args:
            input (Tensor): 2D, 3D or 4D tensor.
        """
        weight_numerator = self.weight_numerator.repeat(self.num_groups, 1)

        if input.dim() == 2 or input.dim() == 3:
            # 1D case
            return self.rational_1d(input, weight_numerator, self.weight_denominator, self.num_groups)
        elif input.dim() == 4:
            # 2D case
            if self.rational_2d is None:
                raise RuntimeError("2D rational function only supported on CUDA.")
            return self.rational_2d(input, weight_numerator, self.weight_denominator, self.num_groups)
        else:
            raise ValueError("Input must be 2D, 3D, or 4D tensor.")

    def extra_repr(self):
        return f'num_groups={self.num_groups}, order={self.order}, device={self.device}'
