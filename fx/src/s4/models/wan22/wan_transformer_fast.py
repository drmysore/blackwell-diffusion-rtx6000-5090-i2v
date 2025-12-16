import torch
from flash_attn import flash_attn_qkvpacked_func
from torch import nn


class FlashAttentionWrapper:
  """Wraps existing attention modules to use Flash Attention."""

  @staticmethod
  def patch_attention_forward(module: nn.Module, use_flash: bool = True) -> None:
    """
    Monkey-patch the forward method of an attention module to use Flash Attention.

    Args:
        module: The attention module to patch (must have to_qkv, norm_q, norm_k, to_out)
        use_flash: Whether to use flash attention or fall back to SDPA
    """
    original_forward = module.forward

    def flash_forward(
      hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor], **kwargs
    ) -> torch.Tensor:
      B, L, D = hidden_states.shape
      num_heads = getattr(module, "num_heads", 40)  # Default for WAN
      head_dim = D // num_heads

      # Use existing layers
      qkv = module.to_qkv(hidden_states)  # [B, L, 3*D]
      q, k, v = qkv.chunk(3, dim=-1)  # 3x [B, L, D]

      # Apply normalization
      q = module.norm_q(q).view(B, L, num_heads, head_dim)
      k = module.norm_k(k).view(B, L, num_heads, head_dim)
      v = v.view(B, L, num_heads, head_dim)

      # Apply rotary embeddings
      if hasattr(module, "apply_rotary_emb"):
        q = module.apply_rotary_emb(q, *rotary_emb)
        k = module.apply_rotary_emb(k, *rotary_emb)
      else:
        # Fallback to global apply_rotary_emb function
        from ..rotary import apply_rotary_emb

        q = apply_rotary_emb(q, *rotary_emb)
        k = apply_rotary_emb(k, *rotary_emb)

      if use_flash:
        # Pack QKV for flash attention
        qkv_packed = torch.stack([q, k, v], dim=2)  # [B, L, 3, heads, head_dim]

        # Flash attention
        attn_output = flash_attn_qkvpacked_func(
          qkv_packed,
          dropout_p=0.0,
          causal=False,
          window_size=(-1, -1),  # No sliding window
          softmax_scale=None,  # Default 1/sqrt(head_dim)
        )

        # Reshape output
        h = attn_output.reshape(B, L, D)
      else:
        # Original SDPA path
        q = q.transpose(1, 2)  # [B, heads, L, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        h = torch.nn.functional.scaled_dot_product_attention(
          q, k, v, dropout_p=0.0, is_causal=False
        )
        h = h.transpose(1, 2).reshape(B, L, D)

      # Apply output projection
      h = module.to_out[0](h)
      if len(module.to_out) > 1:
        h = module.to_out[1](h)

      return h

    # Replace the forward method
    module.forward = flash_forward
    # Store original for potential restoration
    module._original_forward = original_forward

  @staticmethod
  def unpatch_attention_forward(module: nn.Module) -> None:
    """Restore original forward method."""
    if hasattr(module, "_original_forward"):
      module.forward = module._original_forward
      delattr(module, "_original_forward")


def optimize_wan_attention(model: nn.Module, use_flash: bool = True) -> nn.Module:
  """
  Optimize all attention layers in a WAN model to use Flash Attention.

  Args:
      model: The WAN transformer model
      use_flash: Whether to use flash attention

  Returns:
      The model with optimized attention
  """
  attention_modules = []

  for name, module in model.named_modules():
    # Identify attention modules by their characteristic components
    if (
      hasattr(module, "to_qkv")
      and hasattr(module, "norm_q")
      and hasattr(module, "norm_k")
      and hasattr(module, "to_out")
    ):
      FlashAttentionWrapper.patch_attention_forward(module, use_flash)
      attention_modules.append(name)

  print(f"Optimized {len(attention_modules)} attention modules with Flash Attention")
  return model


# Usage in your pipeline:
def apply_flash_attention_to_pipeline(pipe):
  """Apply Flash Attention optimization to WAN pipeline."""
  # Optimize both transformers
  pipe.transformer = optimize_wan_attention(pipe.transformer, use_flash=True)
  pipe.transformer_2 = optimize_wan_attention(pipe.transformer_2, use_flash=True)
  return pipe


# Benchmark utility
def benchmark_attention(
  model: nn.Module,
  batch_size: int = 1,
  seq_len: int = 1024,
  hidden_dim: int = 5120,
  num_iterations: int = 10,
  device: str = "cuda",
) -> tuple[float, float]:
  """
  Benchmark attention performance with and without Flash Attention.

  Returns:
      (time_with_flash, time_without_flash) in seconds
  """
  import time

  # Create dummy inputs
  hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
  cos = torch.randn(1, seq_len, 1, 128, device=device, dtype=torch.bfloat16)
  sin = torch.randn(1, seq_len, 1, 128, device=device, dtype=torch.bfloat16)
  rotary_emb = (cos, sin)

  # Find first attention module
  attn_module = None
  for module in model.modules():
    if hasattr(module, "to_qkv") and hasattr(module, "norm_q"):
      attn_module = module
      break

  if attn_module is None:
    raise ValueError("No attention module found")

  # Warmup
  for _ in range(3):
    _ = attn_module(hidden_states, rotary_emb)

  # Benchmark with Flash Attention
  FlashAttentionWrapper.patch_attention_forward(attn_module, use_flash=True)
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(num_iterations):
    _ = attn_module(hidden_states, rotary_emb)
  torch.cuda.synchronize()
  time_with_flash = (time.perf_counter() - start) / num_iterations

  # Benchmark without Flash Attention
  FlashAttentionWrapper.patch_attention_forward(attn_module, use_flash=False)
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(num_iterations):
    _ = attn_module(hidden_states, rotary_emb)
  torch.cuda.synchronize()
  time_without_flash = (time.perf_counter() - start) / num_iterations

  return time_with_flash, time_without_flash


# Integration with your loader
def load_wan_with_flash_attention(loader, **kwargs):
  """Load WAN pipeline with Flash Attention enabled."""
  pipe = loader.load_optimized_pipeline(**kwargs)
  pipe = apply_flash_attention_to_pipeline(pipe)
  return pipe
