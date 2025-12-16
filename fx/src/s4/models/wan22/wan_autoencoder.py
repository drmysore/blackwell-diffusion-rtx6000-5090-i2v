"""
WAN VAE with Flash Attention Integration
========================================

Using the same Flash Attention patterns from the transformer:
- QKV packed format for self-attention
- KV packed format for cross-attention
- Proper bfloat16 handling throughout
"""

import torch
from flash_attn import flash_attn_kvpacked_func, flash_attn_qkvpacked_func
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..activations import get_activation
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput, DiagonalGaussianDistribution


class WanFlashAttention3D(nn.Module):
  """
  3D self-attention using Flash Attention with QKV packed format.
  Processes video data with spatio-temporal attention.
  """

  def __init__(self, dim: int, heads: int = 8, head_dim: int = 64):
    super().__init__()
    self.heads = heads
    self.head_dim = head_dim
    self.inner_dim = heads * head_dim

    # QKV projection for Flash Attention
    self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
    self.to_out = nn.Linear(self.inner_dim, dim, bias=True)

    # Normalization
    self.norm = nn.LayerNorm(dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Apply 3D attention to video tensor.
    Input: [B, C, T, H, W]
    """
    B, C, T, H, W = x.shape

    # Ensure bfloat16 for Flash Attention
    input_dtype = x.dtype
    x = x.to(torch.bfloat16)

    # Reshape to sequence: [B, T*H*W, C]
    x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
    x = self.norm(x)

    # Project to QKV
    qkv = self.to_qkv(x)  # [B, T*H*W, 3*inner_dim]

    # Reshape for Flash Attention packed format
    qkv = qkv.view(B, T * H * W, 3, self.heads, self.head_dim)

    # Flash Attention
    out = flash_attn_qkvpacked_func(
      qkv,
      dropout_p=0.0,
      softmax_scale=None,
      causal=False,
    )
    # out: [B, T*H*W, heads, head_dim]

    # Combine heads and project
    out = out.reshape(B, T * H * W, self.inner_dim)
    out = self.to_out(out)

    # Reshape back to video format
    out = out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

    return out.to(input_dtype)


class WanTemporalAttention(nn.Module):
  """
  Temporal attention across frames using Flash Attention.
  Each spatial position attends across time.
  """

  def __init__(self, dim: int, heads: int = 8, head_dim: int = 64):
    super().__init__()
    self.heads = heads
    self.head_dim = head_dim
    self.inner_dim = heads * head_dim

    self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
    self.to_out = nn.Linear(self.inner_dim, dim, bias=True)
    self.norm = nn.LayerNorm(dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Apply temporal attention.
    Input: [B, C, T, H, W]
    """
    B, C, T, H, W = x.shape

    input_dtype = x.dtype
    x = x.to(torch.bfloat16)

    # Reshape so each spatial position has a temporal sequence
    # [B, C, T, H, W] -> [B*H*W, T, C]
    x = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
    x = self.norm(x)

    # QKV projection
    qkv = self.to_qkv(x)
    qkv = qkv.view(B * H * W, T, 3, self.heads, self.head_dim)

    # Flash Attention on temporal dimension
    out = flash_attn_qkvpacked_func(
      qkv,
      dropout_p=0.0,
      softmax_scale=None,
      causal=True,  # Causal for temporal dimension
    )

    # Project and reshape back
    out = out.reshape(B * H * W, T, self.inner_dim)
    out = self.to_out(out)

    # Back to video format
    out = out.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)

    return out.to(input_dtype)


class WanSpatialAttention(nn.Module):
  """
  Spatial attention within each frame using Flash Attention.
  Each frame processed independently.
  """

  def __init__(self, dim: int, heads: int = 8, head_dim: int = 64):
    super().__init__()
    self.heads = heads
    self.head_dim = head_dim
    self.inner_dim = heads * head_dim

    self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
    self.to_out = nn.Linear(self.inner_dim, dim, bias=True)
    self.norm = nn.LayerNorm(dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Apply spatial attention per frame.
    Input: [B, C, T, H, W]
    """
    B, C, T, H, W = x.shape

    input_dtype = x.dtype
    x = x.to(torch.bfloat16)

    # Process each frame independently
    # [B, C, T, H, W] -> [B*T, H*W, C]
    x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
    x = self.norm(x)

    # QKV projection
    qkv = self.to_qkv(x)
    qkv = qkv.view(B * T, H * W, 3, self.heads, self.head_dim)

    # Flash Attention on spatial dimensions
    out = flash_attn_qkvpacked_func(
      qkv,
      dropout_p=0.0,
      softmax_scale=None,
      causal=False,
    )

    # Project and reshape back
    out = out.reshape(B * T, H * W, self.inner_dim)
    out = self.to_out(out)

    # Back to video format
    out = out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

    return out.to(input_dtype)


class WanAttentionBlock3D(nn.Module):
  """
  Combined spatial and temporal attention block using Flash Attention.
  Replaces the simple single-head attention in the original VAE.
  """

  def __init__(
    self,
    dim: int,
    heads: int = 8,
    head_dim: int = 64,
    use_spatial: bool = True,
    use_temporal: bool = True,
  ):
    super().__init__()

    self.use_spatial = use_spatial
    self.use_temporal = use_temporal

    if use_spatial:
      self.spatial_attn = WanSpatialAttention(dim, heads, head_dim)

    if use_temporal:
      self.temporal_attn = WanTemporalAttention(dim, heads, head_dim)

    # Combined projection if using both
    if use_spatial and use_temporal:
      self.combine = nn.Linear(dim * 2, dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Apply spatial and/or temporal attention.
    Input: [B, C, T, H, W]
    """
    identity = x
    outputs = []

    if self.use_spatial:
      spatial_out = self.spatial_attn(x)
      outputs.append(spatial_out)

    if self.use_temporal:
      temporal_out = self.temporal_attn(x)
      outputs.append(temporal_out)

    if len(outputs) == 2:
      # Combine spatial and temporal
      combined = torch.cat(outputs, dim=1)
      # Reshape for linear layer
      B, C2, T, H, W = combined.shape
      combined = combined.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, C2)
      out = self.combine(combined)
      out = out.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3)
    else:
      out = outputs[0]

    return out + identity


class WanCrossAttention3D(nn.Module):
  """
  Cross-attention for conditioning (e.g., text/image to video latents).
  Uses KV-packed format like in the transformer.
  """

  def __init__(self, dim: int, context_dim: int, heads: int = 8, head_dim: int = 64):
    super().__init__()
    self.heads = heads
    self.head_dim = head_dim
    self.inner_dim = heads * head_dim

    self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
    self.to_kv = nn.Linear(context_dim, self.inner_dim * 2, bias=False)
    self.to_out = nn.Linear(self.inner_dim, dim, bias=True)

    self.norm_q = nn.LayerNorm(dim)
    self.norm_context = nn.LayerNorm(context_dim)

  def forward(
    self, x: torch.Tensor, context: torch.Tensor, precomputed_kv: torch.Tensor | None = None
  ) -> torch.Tensor:
    """
    Cross attention between video and context.
    x: [B, C, T, H, W]
    context: [B, L, D]
    precomputed_kv: [B, L, 2, heads, head_dim] (optional)
    """
    B, C, T, H, W = x.shape

    input_dtype = x.dtype
    x = x.to(torch.bfloat16)

    # Reshape video to sequence
    x_seq = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
    x_seq = self.norm_q(x_seq)

    # Project queries
    q = self.to_q(x_seq)
    q = q.view(B, T * H * W, self.heads, self.head_dim)

    # Get or compute KV
    if precomputed_kv is not None:
      kv = precomputed_kv.to(torch.bfloat16)
    else:
      context = context.to(torch.bfloat16)
      context = self.norm_context(context)
      kv = self.to_kv(context)
      kv = kv.view(B, context.shape[1], 2, self.heads, self.head_dim)

    # Flash Attention with KV-packed
    out = flash_attn_kvpacked_func(
      q,
      kv,
      dropout_p=0.0,
      softmax_scale=None,
      causal=False,
    )

    # Project output
    out = out.reshape(B, T * H * W, self.inner_dim)
    out = self.to_out(out)

    # Reshape back to video
    out = out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

    return out.to(input_dtype)


class WanResidualBlock(nn.Module):
  """
  Residual block with optional Flash Attention.
  """

  def __init__(
    self,
    in_dim: int,
    out_dim: int,
    dropout: float = 0.0,
    non_linearity: str = "silu",
    use_attention: bool = False,
    attention_heads: int = 8,
  ):
    super().__init__()

    self.nonlinearity = get_activation(non_linearity)

    self.norm1 = WanRMSNorm(in_dim, images=False)
    self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
    self.norm2 = WanRMSNorm(out_dim, images=False)
    self.dropout = nn.Dropout(dropout)
    self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)

    self.conv_shortcut = WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    # Optional attention
    self.attention = None
    if use_attention:
      self.attention = WanAttentionBlock3D(
        out_dim,
        heads=attention_heads,
        head_dim=out_dim // attention_heads,
        use_spatial=True,
        use_temporal=True,
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    shortcut = self.conv_shortcut(x)

    x = self.norm1(x)
    x = self.nonlinearity(x)
    x = self.conv1(x)

    if self.attention is not None:
      x = self.attention(x)

    x = self.norm2(x)
    x = self.nonlinearity(x)
    x = self.dropout(x)
    x = self.conv2(x)

    return x + shortcut


class WanEncoderFlashAttention(nn.Module):
  """
  VAE Encoder with Flash Attention integrated throughout.
  """

  def __init__(
    self,
    in_channels: int = 3,
    dim: int = 128,
    z_dim: int = 4,
    dim_mult: list[int] = [1, 2, 4, 4],
    num_res_blocks: int = 2,
    attention_resolutions: list[int] = [16, 8],  # Apply attention at these spatial resolutions
    temporal_downsample: list[bool] = [True, True, False],
    dropout: float = 0.0,
    non_linearity: str = "silu",
  ):
    super().__init__()

    self.nonlinearity = get_activation(non_linearity)
    dims = [dim * m for m in [1] + dim_mult]

    # Input convolution
    self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1)

    # Encoder blocks
    self.down_blocks = nn.ModuleList()
    current_resolution = 256  # Assuming input is 256x256

    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
      # Determine if we use attention at this resolution
      use_attention = current_resolution in attention_resolutions

      # Add residual blocks
      for j in range(num_res_blocks):
        self.down_blocks.append(
          WanResidualBlock(
            in_dim if j == 0 else out_dim,
            out_dim,
            dropout,
            use_attention=use_attention and j == num_res_blocks - 1,  # Attention on last block
            attention_heads=8,
          )
        )

      # Add downsampling if not last layer
      if i != len(dim_mult) - 1:
        mode = "downsample3d" if temporal_downsample[i] else "downsample2d"
        self.down_blocks.append(WanResample(out_dim, mode=mode))
        current_resolution //= 2

    # Middle block with full attention
    self.mid_block = nn.ModuleList(
      [
        WanResidualBlock(dims[-1], dims[-1], dropout),
        WanFlashAttention3D(dims[-1], heads=8, head_dim=dims[-1] // 8),
        WanResidualBlock(dims[-1], dims[-1], dropout),
      ]
    )

    # Output layers
    self.norm_out = WanRMSNorm(dims[-1], images=False)
    self.conv_out = WanCausalConv3d(dims[-1], z_dim * 2, 3, padding=1)  # *2 for mean and logvar

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Initial projection
    x = self.conv_in(x)

    # Encoder blocks
    for block in self.down_blocks:
      x = block(x)

    # Middle attention block
    for block in self.mid_block:
      x = block(x)

    # Output projection
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    x = self.conv_out(x)

    return x


class WanDecoderFlashAttention(nn.Module):
  """
  VAE Decoder with Flash Attention integrated throughout.
  """

  def __init__(
    self,
    dim: int = 128,
    z_dim: int = 4,
    dim_mult: list[int] = [1, 2, 4, 4],
    num_res_blocks: int = 2,
    attention_resolutions: list[int] = [16, 8],
    temporal_upsample: list[bool] = [False, True, True],
    dropout: float = 0.0,
    non_linearity: str = "silu",
    out_channels: int = 3,
  ):
    super().__init__()

    self.nonlinearity = get_activation(non_linearity)
    dims = [dim * m for m in [dim_mult[-1]] + dim_mult[::-1]]

    # Input convolution
    self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1)

    # Middle block with full attention
    self.mid_block = nn.ModuleList(
      [
        WanResidualBlock(dims[0], dims[0], dropout),
        WanFlashAttention3D(dims[0], heads=8, head_dim=dims[0] // 8),
        WanResidualBlock(dims[0], dims[0], dropout),
      ]
    )

    # Decoder blocks
    self.up_blocks = nn.ModuleList()
    current_resolution = 256 // (2 ** (len(dim_mult) - 1))  # Start from encoded resolution

    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
      # Determine if we use attention at this resolution
      use_attention = current_resolution in attention_resolutions

      # Add residual blocks
      for j in range(num_res_blocks + 1):
        self.up_blocks.append(
          WanResidualBlock(
            in_dim if j == 0 else out_dim,
            out_dim,
            dropout,
            use_attention=use_attention and j == 0,  # Attention on first block
            attention_heads=8,
          )
        )

      # Add upsampling if not last layer
      if i != len(dim_mult) - 1:
        mode = "upsample3d" if temporal_upsample[i] else "upsample2d"
        self.up_blocks.append(WanResample(out_dim, mode=mode, upsample_out_dim=out_dim))
        current_resolution *= 2

    # Output layers
    self.norm_out = WanRMSNorm(dims[-1], images=False)
    self.conv_out = WanCausalConv3d(dims[-1], out_channels, 3, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Initial projection
    x = self.conv_in(x)

    # Middle attention block
    for block in self.mid_block:
      x = block(x)

    # Decoder blocks
    for block in self.up_blocks:
      x = block(x)

    # Output projection
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    x = self.conv_out(x)

    return x


class AutoencoderKLWanFlashAttention(ModelMixin, ConfigMixin):
  """
  WAN VAE with Flash Attention throughout the architecture.
  """

  @register_to_config
  def __init__(
    self,
    base_dim: int = 128,
    z_dim: int = 16,
    dim_mult: list[int] = [1, 2, 4, 4],
    num_res_blocks: int = 2,
    attention_resolutions: list[int] = [16, 8],
    temporal_downsample: list[bool] = [False, True, True],
    dropout: float = 0.0,
    in_channels: int = 3,
    out_channels: int = 3,
  ):
    super().__init__()

    self.encoder = WanEncoderFlashAttention(
      in_channels=in_channels,
      dim=base_dim,
      z_dim=z_dim,
      dim_mult=dim_mult,
      num_res_blocks=num_res_blocks,
      attention_resolutions=attention_resolutions,
      temporal_downsample=temporal_downsample,
      dropout=dropout,
    )

    self.quant_conv = WanCausalConv3d(z_dim * 2, z_dim * 2, 1)
    self.post_quant_conv = WanCausalConv3d(z_dim, z_dim, 1)

    self.decoder = WanDecoderFlashAttention(
      dim=base_dim,
      z_dim=z_dim,
      dim_mult=dim_mult,
      num_res_blocks=num_res_blocks,
      attention_resolutions=attention_resolutions,
      temporal_upsample=temporal_downsample[::-1],
      dropout=dropout,
      out_channels=out_channels,
    )

  @apply_forward_hook
  def encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple:
    """Encode with Flash Attention."""
    # Convert to bfloat16 for Flash Attention
    input_dtype = x.dtype
    x = x.to(torch.bfloat16)

    h = self.encoder(x)
    h = self.quant_conv(h)

    # Back to original dtype for posterior
    h = h.to(input_dtype)
    posterior = DiagonalGaussianDistribution(h)

    if not return_dict:
      return (posterior,)
    return AutoencoderKLOutput(latent_dist=posterior)

  @apply_forward_hook
  def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | torch.Tensor:
    """Decode with Flash Attention."""
    # Convert to bfloat16 for Flash Attention
    input_dtype = z.dtype
    z = z.to(torch.bfloat16)

    z = self.post_quant_conv(z)
    dec = self.decoder(z)

    # Back to original dtype and clamp
    dec = dec.to(input_dtype)
    dec = torch.clamp(dec, min=-1.0, max=1.0)

    if not return_dict:
      return (dec,)
    return DecoderOutput(sample=dec)

    @classmethod
    def from_pretrained_wan_vae(cls, original_vae, config=None):
      """
      Load weights from original WAN VAE, handling architecture differences.
      """
      if config is None:
        config = original_vae.config

      # Create new model with Flash Attention
      new_vae = cls(
        base_dim=config.base_dim,
        z_dim=config.z_dim,
        dim_mult=config.dim_mult,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=[16, 8],  # New parameter
        temporal_downsample=config.temperal_downsample,  # Note: typo in original
        dropout=config.dropout,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
      )

      # Load weights with custom mapping
      new_vae._load_from_original_vae(original_vae)

      return new_vae

    def _load_from_original_vae(self, original_vae):
      """
      Custom weight loading to handle architectural differences.
      """
      # Get state dicts
      original_state = original_vae.state_dict()
      new_state = self.state_dict()

      # Track what we've loaded
      loaded_keys = set()
      missing_in_original = []

      # 1. Load exact matches first (convolutions, norms, etc.)
      for key, param in new_state.items():
        if key in original_state and original_state[key].shape == param.shape:
          new_state[key] = original_state[key]
          loaded_keys.add(key)

      # 2. Handle the original attention blocks
      # Original uses simple single-head attention, we use multi-head Flash Attention
      self._convert_attention_weights(original_state, new_state, loaded_keys)

      # 3. Handle residual blocks that now have optional attention
      self._map_residual_blocks(original_state, new_state, loaded_keys)

      # 4. Initialize new attention layers if not loaded
      for key, param in new_state.items():
        if key not in loaded_keys:
          if "attention" in key:
            # Initialize new attention layers sensibly
            if "to_qkv.weight" in key:
              # Initialize as identity-like for stability
              torch.nn.init.xavier_uniform_(param, gain=0.1)
            elif "to_out.weight" in key:
              torch.nn.init.xavier_uniform_(param, gain=0.1)
            elif "to_out.bias" in key:
              torch.nn.init.zeros_(param)
          missing_in_original.append(key)

      # Load the new state dict
      self.load_state_dict(new_state)

      print(f"Loaded {len(loaded_keys)} parameters from original VAE")
      print(f"Initialized {len(missing_in_original)} new parameters")
      if len(missing_in_original) < 20:  # Don't spam if there are many
        print("New parameters:", missing_in_original)

    def _convert_attention_weights(self, original_state, new_state, loaded_keys):
      """
      Convert single-head attention to multi-head Flash Attention format.

      Original attention structure:
      - norm: RMSNorm
      - to_qkv: Conv2d(dim, dim*3, 1)
      - proj: Conv2d(dim, dim, 1)

      New Flash Attention structure:
      - spatial_attn.norm: LayerNorm
      - spatial_attn.to_qkv: Linear(dim, inner_dim*3)
      - spatial_attn.to_out: Linear(inner_dim, dim)
      """
      # Find all original attention blocks
      original_attention_keys = {}
      for key in original_state:
        if "attentions" in key or "attn" in key:
          # Parse the key to find which attention block it belongs to
          parts = key.split(".")
          block_path = ".".join(parts[:-1])
          param_name = parts[-1]

          if block_path not in original_attention_keys:
            original_attention_keys[block_path] = {}
          original_attention_keys[block_path][param_name] = key

      # Map to new attention blocks
      for block_path, params in original_attention_keys.items():
        # Determine corresponding new block
        new_block_path = self._find_corresponding_attention_block(block_path)
        if not new_block_path:
          continue

        # Map norm weights (RMSNorm -> LayerNorm)
        if "norm.gamma" in params:
          old_key = params["norm.gamma"]
          new_key = f"{new_block_path}.norm.weight"
          if new_key in new_state:
            new_state[new_key] = original_state[old_key]
            loaded_keys.add(new_key)
            # LayerNorm also has bias, initialize to zero
            bias_key = f"{new_block_path}.norm.bias"
            if bias_key in new_state:
              torch.nn.init.zeros_(new_state[bias_key])

        # Map QKV weights (Conv2d -> Linear)
        if "to_qkv.weight" in params:
          old_weight = original_state[params["to_qkv.weight"]]  # [dim*3, dim, 1, 1]
          # Squeeze spatial dimensions and transpose
          old_weight = old_weight.squeeze(-1).squeeze(-1).t()  # [dim, dim*3]

          new_key = f"{new_block_path}.to_qkv.weight"
          if new_key in new_state:
            # Handle dimension mismatch (single-head to multi-head)
            new_dim = new_state[new_key].shape[0]
            old_dim = old_weight.shape[0]

            if new_dim == old_dim:
              new_state[new_key] = old_weight
            # Repeat/truncate to match dimensions
            elif new_dim > old_dim:
              # Repeat pattern
              repeat_times = new_dim // old_dim
              remainder = new_dim % old_dim
              new_state[new_key] = torch.cat(
                [old_weight.repeat(repeat_times, 1), old_weight[:remainder]], dim=0
              )
            else:
              # Truncate
              new_state[new_key] = old_weight[:new_dim]
            loaded_keys.add(new_key)

        # Map projection weights (Conv2d -> Linear)
        if "proj.weight" in params:
          old_weight = original_state[params["proj.weight"]]  # [dim, dim, 1, 1]
          old_weight = old_weight.squeeze(-1).squeeze(-1).t()  # [dim, dim]

          new_key = f"{new_block_path}.to_out.weight"
          if new_key in new_state:
            # Initialize output projection from old projection
            # May need to handle dimension differences
            in_dim_new = new_state[new_key].shape[1]
            out_dim_new = new_state[new_key].shape[0]
            in_dim_old = old_weight.shape[0]
            out_dim_old = old_weight.shape[1]

            if in_dim_new == in_dim_old and out_dim_new == out_dim_old:
              new_state[new_key] = old_weight.t()
            else:
              # Smart initialization based on old weights
              scale = (in_dim_old / in_dim_new) ** 0.5
              torch.nn.init.xavier_uniform_(new_state[new_key], gain=scale)
              # Copy what we can
              min_in = min(in_dim_new, in_dim_old)
              min_out = min(out_dim_new, out_dim_old)
              new_state[new_key][:min_out, :min_in] = old_weight.t()[:min_out, :min_in]
            loaded_keys.add(new_key)

    def _find_corresponding_attention_block(self, original_path):
      """
      Map original attention block paths to new architecture paths.
      """
      # This is highly dependent on your specific architecture
      # Example mapping:
      if "encoder" in original_path and "mid_block" in original_path:
        return "encoder.mid_block.1"  # Flash attention is at index 1
      elif "decoder" in original_path and "mid_block" in original_path:
        return "decoder.mid_block.1"
      # Add more mappings as needed
      return None

    def _map_residual_blocks(self, original_state, new_state, loaded_keys):
      """
      Map residual blocks, handling new optional attention.
      """
      # Map conv and norm weights from residual blocks
      for key in original_state:
        if key not in loaded_keys and ("conv" in key or "norm" in key):
          # Try direct mapping with path adjustment
          new_key = self._adjust_block_path(key)
          if new_key in new_state and original_state[key].shape == new_state[new_key].shape:
            new_state[new_key] = original_state[key]
            loaded_keys.add(new_key)

    def _adjust_block_path(self, key):
      """
      Adjust paths for architectural differences.
      """
      # Example: handle different block indexing
      # This is specific to how you've restructured the blocks
      return key  # Override with your specific mapping

    def load_state_dict(self, state_dict, strict=True):
      """
      Override to handle loading with architectural differences.
      """
      if not strict:
        # Filter out keys that don't exist in our model
        current_keys = set(self.state_dict().keys())
        filtered_state = {k: v for k, v in state_dict.items() if k in current_keys}

        # Check shape compatibility
        for key, param in filtered_state.items():
          if param.shape != self.state_dict()[key].shape:
            print(
              f"Shape mismatch for {key}: "
              f"checkpoint has {param.shape}, "
              f"model has {self.state_dict()[key].shape}"
            )
            # Skip this parameter
            filtered_state.pop(key)

        return super().load_state_dict(filtered_state, strict=False)
      else:
        return super().load_state_dict(state_dict, strict=True)
