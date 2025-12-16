import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import nvfp4_linear_cpp


class NVFP4LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        scale_input: Tensor,
        scale_weight: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(input, weight, scale_input, scale_weight)
        ctx.needs_bias_grad = bias is not None

        if bias is None:
            bias = torch.empty(0, device=input.device, dtype=input.dtype)

        return nvfp4_linear_cpp.forward(input, weight, bias, scale_input, scale_weight)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, weight, scale_input, scale_weight = ctx.saved_tensors

        # Create scale for grad_output (simplified - you'd compute this properly)
        scale_grad_output = torch.ones_like(scale_input)

        grad_input, grad_weight, grad_bias = nvfp4_linear_cpp.backward(
            grad_output,
            input,
            weight,
            scale_input,
            scale_weight,
            scale_grad_output,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_bias_grad,
        )

        return grad_input, grad_weight, grad_bias, None, None


class NVFP4Linear(nn.Module):
    """NVFP4 Linear layer using CUTLASS 4 kernels for Blackwell GPUs"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # Initialize weights in BF16
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        # Scale factors for block-scaled quantization
        self.register_buffer(
            "scale_weight",
            torch.ones(
                (out_features * in_features + block_size - 1) // block_size,
                dtype=torch.float32,
            ),
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights similar to standard Linear layer
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # Ensure input is BF16
        if input.dtype != torch.bfloat16:
            input = input.to(torch.bfloat16)

        # Compute scale factors for input (simplified)
        input_numel = input.numel() // input.size(-1) * self.in_features
        scale_input = torch.ones(
            (input_numel + self.block_size - 1) // self.block_size,
            device=input.device,
            dtype=torch.float32,
        )

        return NVFP4LinearFunction.apply(
            input, self.weight, self.bias, scale_input, self.scale_weight
        )
