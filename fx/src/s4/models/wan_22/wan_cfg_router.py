"""
WAN 2.2 Dynamic CFG Router
===========================

Dynamic layer that allocates the appropriate static blocks at initialization
based on the CFG configuration. This handles ALL the routing logic so the
static implementations remain pure.

Key decisions made at initialization:
1. CFG enabled? -> Use CFGDualBlock
2. CFG disabled? -> Use ConditionalOnlyBlock
3. Unconditional? -> Use UnconditionalOnlyBlock
4. Guidance scale -> Fixed at init for static graph
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import warnings

from wan_attention_cfg_static import (
    CFGTransformerBlock,
    ConditionalOnlyBlock,
    UnconditionalOnlyBlock,
    create_cfg_static_model,
)

from wan_attention_dynamic import (
    DynamicTransformerBlock,
    WanTransformer3DModel,
)


class CFGMode:
    """Configuration modes for CFG."""

    DISABLED = "disabled"  # No CFG, just conditional
    ENABLED = "enabled"  # CFG with fixed guidance scale
    DYNAMIC = "dynamic"  # CFG with variable guidance scale (slow)
    UNCONDITIONAL = "unconditional"  # No conditioning at all


class DynamicCFGTransformerBlock(nn.Module):
    """
    Dynamic block that routes to appropriate static implementation based on CFG mode.
    All routing decisions are made at initialization.
    """

    def __init__(
        self,
        dim: int = 5120,
        num_heads: int = 40,
        ffn_dim: int = 13824,
        cfg_mode: str = CFGMode.DISABLED,
        guidance_scale: Optional[float] = None,
        static_batch_size: Optional[int] = None,
        static_seq_len: Optional[int] = None,
        static_context_len: Optional[int] = None,
    ):
        super().__init__()

        self.cfg_mode = cfg_mode
        self.guidance_scale = guidance_scale if guidance_scale is not None else 7.5
        self.dim = dim

        # Allocate appropriate block type
        if cfg_mode == CFGMode.ENABLED and all(
            x is not None for x in [static_batch_size, static_seq_len, static_context_len]
        ):
            # Allocate CFG-optimized static block
            self.block = CFGTransformerBlock(
                static_batch_size,
                static_seq_len,
                static_context_len,
                dim,
                num_heads,
                ffn_dim,
            )
            self.is_static = True

        elif cfg_mode == CFGMode.DISABLED and all(
            x is not None for x in [static_batch_size, static_seq_len, static_context_len]
        ):
            # Allocate conditional-only static block
            self.block = ConditionalOnlyBlock(
                static_batch_size,
                static_seq_len,
                static_context_len,
                dim,
                num_heads,
                ffn_dim,
            )
            self.is_static = True

        elif cfg_mode == CFGMode.UNCONDITIONAL and all(
            x is not None for x in [static_batch_size, static_seq_len]
        ):
            # Allocate unconditional static block
            self.block = UnconditionalOnlyBlock(
                static_batch_size,
                static_seq_len,
                dim,
                num_heads,
                ffn_dim,
            )
            self.is_static = True

        else:
            # Fall back to dynamic block
            self.block = DynamicTransformerBlock(
                dim,
                num_heads,
                ffn_dim,
            )
            self.is_static = False

            if cfg_mode in [CFGMode.ENABLED, CFGMode.DYNAMIC]:
                warnings.warn(
                    f"CFG mode {cfg_mode} requested but static dimensions not provided. "
                    "Using dynamic implementation (slower)."
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cfg_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Dynamic forward that routes to appropriate implementation.

        For CFG, expects hidden_states to potentially contain both
        conditional and unconditional paths stacked.
        """

        if self.is_static:
            if self.cfg_mode == CFGMode.ENABLED:
                # Split inputs for CFG processing
                batch_size = hidden_states.shape[0] // 2
                hidden_cond = hidden_states[:batch_size]
                hidden_uncond = hidden_states[batch_size:]

                if context is not None:
                    context_cond = context[:batch_size]
                    context_uncond = context[batch_size:]
                else:
                    # Create zero context for unconditional
                    context_cond = torch.zeros_like(hidden_cond)
                    context_uncond = torch.zeros_like(hidden_cond)

                if conditioning is not None:
                    cond_cond = conditioning[:batch_size]
                    cond_uncond = conditioning[batch_size:]
                else:
                    cond_cond = torch.zeros(batch_size, 6, self.dim, device=hidden_states.device)
                    cond_uncond = torch.zeros(batch_size, 6, self.dim, device=hidden_states.device)

                if rotary_emb is not None:
                    cos_freqs, sin_freqs = rotary_emb
                else:
                    # Create dummy RoPE
                    seq_len = hidden_states.shape[1]
                    head_dim = self.dim // 40
                    cos_freqs = torch.ones(1, seq_len, 1, head_dim, device=hidden_states.device)
                    sin_freqs = torch.zeros(1, seq_len, 1, head_dim, device=hidden_states.device)

                # Process through CFG block
                out_cond, out_uncond = self.block(
                    hidden_cond,
                    hidden_uncond,
                    context_cond,
                    context_uncond,
                    cond_cond,
                    cond_uncond,
                    cos_freqs,
                    sin_freqs,
                )

                # Apply guidance
                scale = cfg_scale if cfg_scale is not None else self.guidance_scale
                guided = out_uncond + scale * (out_cond - out_uncond)

                # Stack for compatibility
                return torch.cat([guided, out_uncond], dim=0)

            elif self.cfg_mode == CFGMode.DISABLED:
                # Conditional only
                if rotary_emb is not None:
                    cos_freqs, sin_freqs = rotary_emb
                else:
                    seq_len = hidden_states.shape[1]
                    head_dim = self.dim // 40
                    cos_freqs = torch.ones(1, seq_len, 1, head_dim, device=hidden_states.device)
                    sin_freqs = torch.zeros(1, seq_len, 1, head_dim, device=hidden_states.device)

                return self.block(
                    hidden_states,
                    context if context is not None else torch.zeros_like(hidden_states),
                    conditioning
                    if conditioning is not None
                    else torch.zeros(
                        hidden_states.shape[0], 6, self.dim, device=hidden_states.device
                    ),
                    cos_freqs,
                    sin_freqs,
                )

            elif self.cfg_mode == CFGMode.UNCONDITIONAL:
                # Unconditional only
                if rotary_emb is not None:
                    cos_freqs, sin_freqs = rotary_emb
                else:
                    seq_len = hidden_states.shape[1]
                    head_dim = self.dim // 40
                    cos_freqs = torch.ones(1, seq_len, 1, head_dim, device=hidden_states.device)
                    sin_freqs = torch.zeros(1, seq_len, 1, head_dim, device=hidden_states.device)

                if conditioning is not None and conditioning.shape[1] == 6:
                    # Reduce conditioning dimensions for unconditional block
                    conditioning = conditioning[:, :4]  # Only need 4 params
                elif conditioning is None:
                    conditioning = torch.zeros(
                        hidden_states.shape[0], 4, self.dim, device=hidden_states.device
                    )

                return self.block(
                    hidden_states,
                    conditioning,
                    cos_freqs,
                    sin_freqs,
                )

        else:
            # Dynamic fallback
            return self.block(
                hidden_states,
                encoder_hidden_states=context,
                temb=conditioning,
                rotary_emb=rotary_emb,
                **kwargs,
            )


class WanTransformer3DModelWithCFG(WanTransformer3DModel):
    """
    Extended WAN transformer that allocates CFG-optimized blocks.

    At initialization, decides whether to use:
    1. CFG-optimized blocks (processes both paths efficiently)
    2. Conditional-only blocks (no CFG overhead)
    3. Unconditional blocks (no cross-attention)
    """

    def __init__(
        self,
        # Standard WAN config
        num_layers: int = 48,
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 36,
        out_channels: int = 16,
        # CFG configuration
        cfg_mode: str = CFGMode.DISABLED,
        guidance_scale: float = 7.5,
        # Static optimization hints
        static_batch_size: Optional[int] = None,
        static_seq_len: Optional[int] = None,
        static_context_len: Optional[int] = None,
        **kwargs,
    ):
        # Initialize parent
        super().__init__(
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )

        self.cfg_mode = cfg_mode
        self.guidance_scale = guidance_scale

        inner_dim = num_attention_heads * attention_head_dim

        # Replace blocks with CFG-aware blocks
        self.blocks = nn.ModuleList(
            [
                DynamicCFGTransformerBlock(
                    dim=inner_dim,
                    num_heads=num_attention_heads,
                    ffn_dim=kwargs.get("ffn_dim", 13824),
                    cfg_mode=cfg_mode,
                    guidance_scale=guidance_scale,
                    static_batch_size=static_batch_size,
                    static_seq_len=static_seq_len,
                    static_context_len=static_context_len,
                )
                for _ in range(num_layers)
            ]
        )

        # Allocate static model if dimensions provided
        if cfg_mode != CFGMode.DYNAMIC and all(
            x is not None for x in [static_batch_size, static_seq_len, static_context_len]
        ):
            self._static_model = create_cfg_static_model(
                batch_size=static_batch_size,
                seq_len=static_seq_len,
                context_len=static_context_len,
                use_cfg=(cfg_mode == CFGMode.ENABLED),
                guidance_scale=guidance_scale,
                use_unconditional_only=(cfg_mode == CFGMode.UNCONDITIONAL),
                device=kwargs.get("device", "cuda"),
                dtype=kwargs.get("dtype", torch.float16),
            )
            print(
                f"Allocated static {cfg_mode} model: "
                f"batch={static_batch_size}, seq={static_seq_len}, context={static_context_len}"
            )
        else:
            self._static_model = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        do_classifier_free_guidance: Optional[bool] = None,
        **kwargs,
    ):
        """
        Forward with CFG support.

        If do_classifier_free_guidance is True, expects inputs to be duplicated
        for conditional and unconditional paths.
        """

        # Determine if we're doing CFG
        if do_classifier_free_guidance is None:
            do_classifier_free_guidance = self.cfg_mode == CFGMode.ENABLED

        if do_classifier_free_guidance and self.cfg_mode == CFGMode.DISABLED:
            warnings.warn(
                "CFG requested but model configured without CFG. Falling back to dynamic."
            )

        # Use parent forward for now
        # In production, this would route to static model when possible
        return super().forward(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            **kwargs,
        )


def create_cfg_aware_model(
    use_cfg: bool = True,
    guidance_scale: float = 7.5,
    batch_size: Optional[int] = 1,
    video_frames: Optional[int] = None,
    video_height: Optional[int] = None,
    video_width: Optional[int] = None,
    **kwargs,
) -> WanTransformer3DModelWithCFG:
    """
    Create a CFG-aware WAN model with appropriate block allocation.

    Args:
        use_cfg: Whether to use classifier-free guidance
        guidance_scale: Guidance scale (fixed for static)
        batch_size: Batch size for static allocation
        video_frames: Video frames for calculating seq_len
        video_height: Video height for calculating seq_len
        video_width: Video width for calculating seq_len
        **kwargs: Additional model configuration

    Returns:
        CFG-optimized model
    """

    # Calculate static dimensions if video specs provided
    static_seq_len = None
    static_context_len = 512 + (256 if kwargs.get("use_image_conditioning", False) else 0)

    if all(x is not None for x in [video_frames, video_height, video_width]):
        # Calculate sequence length
        patch_size = kwargs.get("patch_size", (1, 2, 2))
        seq_len = (
            (video_frames // patch_size[0])
            * (video_height // patch_size[1])
            * (video_width // patch_size[2])
        )
        static_seq_len = seq_len

    # Determine CFG mode
    if not use_cfg:
        cfg_mode = CFGMode.DISABLED
    elif guidance_scale is not None:
        cfg_mode = CFGMode.ENABLED
    else:
        cfg_mode = CFGMode.DYNAMIC

    # Adjust batch size for CFG (internal doubling handled by blocks)
    static_batch_size = batch_size

    return WanTransformer3DModelWithCFG(
        cfg_mode=cfg_mode,
        guidance_scale=guidance_scale,
        static_batch_size=static_batch_size,
        static_seq_len=static_seq_len,
        static_context_len=static_context_len,
        **kwargs,
    )


# ================================================================================================
# Example Usage
# ================================================================================================


def demo_cfg_routing():
    """Demonstrate CFG-aware routing."""

    print("=" * 80)
    print("CFG-Aware Model Allocation Demo")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example 1: Model without CFG
    print("\n1. Creating model WITHOUT CFG:")
    model_no_cfg = create_cfg_aware_model(
        use_cfg=False,
        batch_size=1,
        video_frames=16,
        video_height=256,
        video_width=256,
    )
    print(f"   Mode: {model_no_cfg.cfg_mode}")
    print(f"   Block type: {type(model_no_cfg.blocks[0].block).__name__}")

    # Example 2: Model with CFG
    print("\n2. Creating model WITH CFG (guidance=7.5):")
    model_with_cfg = create_cfg_aware_model(
        use_cfg=True,
        guidance_scale=7.5,
        batch_size=1,
        video_frames=16,
        video_height=256,
        video_width=256,
    )
    print(f"   Mode: {model_with_cfg.cfg_mode}")
    print(f"   Guidance scale: {model_with_cfg.guidance_scale}")
    print(f"   Block type: {type(model_with_cfg.blocks[0].block).__name__}")

    # Example 3: Unconditional model
    print("\n3. Creating UNCONDITIONAL model:")
    model_uncond = WanTransformer3DModelWithCFG(
        cfg_mode=CFGMode.UNCONDITIONAL,
        static_batch_size=1,
        static_seq_len=3136,
        static_context_len=None,  # No context for unconditional
    )
    print(f"   Mode: {model_uncond.cfg_mode}")
    print(f"   Block type: {type(model_uncond.blocks[0].block).__name__}")

    # Test inference
    if device == "cuda":
        print("\n4. Testing inference speed:")

        # Test inputs
        batch_size = 1
        seq_len = 3136
        context_len = 512
        dim = 5120

        hidden = torch.randn(batch_size, seq_len, dim, device=device) * 0.02
        context = torch.randn(batch_size, context_len, dim, device=device) * 0.02

        # Test no-CFG model
        import time

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(10):
            with torch.no_grad():
                # Simulate forward through one block
                out = model_no_cfg.blocks[0](hidden, context)

        torch.cuda.synchronize()
        no_cfg_time = (time.perf_counter() - start) / 10 * 1000

        print(f"   No CFG: {no_cfg_time:.2f} ms per block")

        # Test CFG model (processes both paths)
        hidden_cfg = torch.cat([hidden, hidden], dim=0)  # Duplicate for CFG
        context_cfg = torch.cat([context, torch.zeros_like(context)], dim=0)

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(10):
            with torch.no_grad():
                out = model_with_cfg.blocks[0](hidden_cfg, context_cfg)

        torch.cuda.synchronize()
        cfg_time = (time.perf_counter() - start) / 10 * 1000

        print(f"   With CFG: {cfg_time:.2f} ms per block")
        print(f"   CFG overhead: {(cfg_time / no_cfg_time - 1) * 100:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    demo_cfg_routing()
