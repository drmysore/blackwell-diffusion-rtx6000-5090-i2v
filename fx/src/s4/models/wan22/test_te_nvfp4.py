import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, NVFP4BlockScaling

# NVFP4 Matrix Multiplication Test
# This test demonstrates NVFP4 quantization for linear layers
# Key requirements:
# - Dimensions should be multiples of 16 for optimal NVFP4 block scaling (16x16 blocks)
# - fp8_autocast context must wrap all FP4 operations
# - Use torch.no_grad() when not training to avoid gradient computation

# Setup
device = torch.device("cuda")
fp4_format = Format.E2M1
fp4_recipe = NVFP4BlockScaling(fp4_format=fp4_format)

print("NVFP4 Matrix Multiplication Test")
print(f"Format: {fp4_format}")
print("Recipe: NVFP4BlockScaling\n")

# Create input - dimensions should be multiples of 16 for NVFP4 block scaling
torch.manual_seed(42)
batch_size, seq_len, hidden_dim = 4, 128, 768  # 768 is divisible by 16
x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16).cuda()

# Create linear layer
linear = te.Linear(
    hidden_dim, hidden_dim, bias=True, params_dtype=torch.bfloat16
).cuda()

print(f"Input shape: {x.shape}")
print(f"Input dtype: {x.dtype}")
print(f"Weight shape: {linear.weight.shape}\n")

# BF16 baseline
with torch.no_grad():
    out_bf16 = linear(x)

# NVFP4 forward pass - wrap all FP4 operations in the context
print("Running NVFP4 forward pass...")
with torch.no_grad():
    with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
        out_fp4 = linear(x)

print(f"Output shape: {out_fp4.shape}")
print(f"BF16 output sample: {out_bf16[0, 0, :5]}")
print(f"FP4 output sample:  {out_fp4[0, 0, :5]}\n")

# Error metrics
abs_error = torch.abs(out_bf16 - out_fp4)
rel_error = abs_error / (torch.abs(out_bf16) + 1e-5)

print(f"Mean absolute error: {abs_error.mean().item():.6f}")
print(f"Max absolute error: {abs_error.max().item():.6f}")
print(f"Mean relative error: {rel_error.mean().item():.6f}")
print(f"Median relative error: {rel_error.median().item():.6f}")

# Also compute RMSE for a more stable metric
rmse = torch.sqrt(torch.mean((out_bf16 - out_fp4) ** 2))
print(f"RMSE: {rmse.item():.6f}")

# Signal-to-noise ratio
signal_power = torch.mean(out_bf16**2)
noise_power = torch.mean((out_bf16 - out_fp4) ** 2)
snr = 10 * torch.log10(signal_power / noise_power)
print(f"SNR: {snr.item():.2f} dB")
