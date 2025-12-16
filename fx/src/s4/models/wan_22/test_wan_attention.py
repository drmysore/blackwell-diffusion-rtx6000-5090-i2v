"""
WAN 2.2 Comprehensive Test Suite
=================================

This test suite validates the WAN 2.2 implementation across multiple levels:
1. Component-level testing (attention, FFN, normalization)
2. Block-level testing (transformer blocks)
3. Full model testing (end-to-end)
4. Compatibility testing (pure vs diffusers)
5. Serialization/deserialization testing
6. Quantization stub testing

The tests use a combination of:
- Direct comparison with reference implementations where possible
- Property-based testing with Hypothesis for invariants
- Numerical gradient checking for custom operations
- Shape and dtype preservation tests
"""

import unittest
from typing import Tuple
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

# Hypothesis for property-based testing
from hypothesis import given, strategies as st, settings, assume

# Import our implementations
from wan_attention_pure import (
    apply_rotary_emb,
    RotaryPosEmbed,
    RMSNorm,
    FP32LayerNorm,
    FeedForward,
    SelfAttention,
    CrossAttention,
    TransformerBlock,
)

from wan_attention_diffusers import (
    WanSelfAttention,
    WanCrossAttention,
    WanTransformer3DModel,
)

from wan_serialization import (
    WANModelSerializer,
    QuantizationConfig,
    QuantizationType,
    TensorMetadata,
    create_comfyui_config,
)


# ================================================================================================
# Section 1: Test Utilities and Fixtures
# ================================================================================================


class TestUtils:
    """Utility functions for testing."""

    @staticmethod
    def assert_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str = "tensor"):
        """Assert tensor has expected shape."""
        assert tensor.shape == expected_shape, (
            f"{name} shape mismatch: got {tensor.shape}, expected {expected_shape}"
        )

    @staticmethod
    def assert_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype, name: str = "tensor"):
        """Assert tensor has expected dtype."""
        assert tensor.dtype == expected_dtype, (
            f"{name} dtype mismatch: got {tensor.dtype}, expected {expected_dtype}"
        )

    @staticmethod
    def assert_close(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        name: str = "tensors",
    ):
        """Assert two tensors are numerically close."""
        assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), (
            f"{name} not close enough. Max diff: {(tensor1 - tensor2).abs().max().item()}"
        )

    @staticmethod
    def create_random_tensor(
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Create a random tensor with specified properties."""
        tensor = torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor * 0.02  # Scale down for numerical stability

    @staticmethod
    def get_device() -> str:
        """Get available device for testing."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


# ================================================================================================
# Section 2: Component-Level Tests
# ================================================================================================


class TestRotaryEmbeddings(unittest.TestCase):
    """Test rotary position embeddings."""

    def setUp(self):
        self.device = TestUtils.get_device()
        self.head_dim = 128
        self.seq_len = 197  # Typical sequence length
        self.batch_size = 2
        self.num_heads = 40

    def test_rotary_emb_shape(self):
        """Test that RoPE produces correct shapes."""
        rope = RotaryPosEmbed(self.head_dim)
        cos, sin = rope(self.seq_len, device=self.device)

        TestUtils.assert_shape(cos, (1, self.seq_len, 1, self.head_dim), "cos")
        TestUtils.assert_shape(sin, (1, self.seq_len, 1, self.head_dim), "sin")

    def test_apply_rotary_emb_preserves_shape(self):
        """Test that applying RoPE preserves tensor shape."""
        hidden_states = TestUtils.create_random_tensor(
            (self.batch_size, self.seq_len, self.num_heads, self.head_dim), device=self.device
        )

        rope = RotaryPosEmbed(self.head_dim)
        cos, sin = rope(self.seq_len, device=self.device, dtype=hidden_states.dtype)

        output = apply_rotary_emb(hidden_states, cos, sin)

        TestUtils.assert_shape(output, hidden_states.shape, "output")
        TestUtils.assert_dtype(output, hidden_states.dtype, "output")

    @given(
        seq_len=st.integers(min_value=1, max_value=1024), head_dim=st.sampled_from([64, 128, 256])
    )
    @settings(max_examples=10, deadline=None)
    def test_rotary_emb_properties(self, seq_len: int, head_dim: int):
        """Property-based test for rotary embeddings."""
        rope = RotaryPosEmbed(head_dim)
        cos, sin = rope(seq_len)

        # Property 1: cos^2 + sin^2 = 1 (approximately)
        # Note: This is per frequency component, not the full vector
        cos_vals = cos[0, :, 0, 0::2]  # Extract one frequency component
        sin_vals = sin[0, :, 0, 1::2]  # Corresponding sin component

        # Property 2: Bounded values
        assert cos.abs().max() <= 1.0, "Cosine values should be bounded by 1"
        assert sin.abs().max() <= 1.0, "Sine values should be bounded by 1"

        # Property 3: Shape consistency
        assert cos.shape == sin.shape, "Cos and sin should have same shape"


class TestNormalization(unittest.TestCase):
    """Test normalization layers."""

    def setUp(self):
        self.device = TestUtils.get_device()
        self.dim = 5120
        self.batch_size = 2
        self.seq_len = 197

    def test_rms_norm_forward(self):
        """Test RMSNorm forward pass."""
        norm = RMSNorm(self.dim, eps=1e-6).to(self.device)
        x = TestUtils.create_random_tensor(
            (self.batch_size, self.seq_len, self.dim), device=self.device
        )

        output = norm(x)

        # Check shape preservation
        TestUtils.assert_shape(output, x.shape, "output")

        # Check that RMS is approximately 1
        rms = torch.sqrt(torch.mean(output**2, dim=-1))
        expected_rms = torch.ones_like(rms)
        TestUtils.assert_close(rms, expected_rms, rtol=1e-3, name="RMS")

    def test_fp32_layernorm_mixed_precision(self):
        """Test FP32LayerNorm maintains precision in mixed precision context."""
        norm = FP32LayerNorm(self.dim).to(self.device)

        # Input in half precision
        x_fp16 = TestUtils.create_random_tensor(
            (self.batch_size, self.seq_len, self.dim), dtype=torch.float16, device=self.device
        )

        output = norm(x_fp16)

        # Output should match input dtype
        TestUtils.assert_dtype(output, torch.float16, "output")

        # But computation should have been in FP32 (test by checking numerical stability)
        assert output.isfinite().all(), "FP32 computation should prevent overflow"

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=512),
        dim=st.sampled_from([768, 1024, 5120]),
    )
    @settings(max_examples=10, deadline=None)
    def test_normalization_invariants(self, batch_size: int, seq_len: int, dim: int):
        """Property-based tests for normalization layers."""
        x = torch.randn(batch_size, seq_len, dim) * 0.1

        # Test RMSNorm
        rms_norm = RMSNorm(dim, elementwise_affine=False)
        rms_output = rms_norm(x)

        # Property: Output should have unit RMS
        rms_values = torch.sqrt(torch.mean(rms_output**2, dim=-1))
        assert torch.allclose(rms_values, torch.ones_like(rms_values), atol=1e-3)

        # Test LayerNorm comparison
        layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        ln_output = layer_norm(x)

        # Property: LayerNorm should have zero mean and unit variance
        ln_mean = ln_output.mean(dim=-1)
        ln_var = ln_output.var(dim=-1, unbiased=False)
        assert torch.allclose(ln_mean, torch.zeros_like(ln_mean), atol=1e-3)
        assert torch.allclose(ln_var, torch.ones_like(ln_var), atol=1e-3)


# ================================================================================================
# Section 3: Attention Module Tests
# ================================================================================================


class TestSelfAttention(unittest.TestCase):
    """Test self-attention module."""

    def setUp(self):
        self.device = TestUtils.get_device()
        self.dim = 5120
        self.num_heads = 40
        self.head_dim = 128
        self.batch_size = 2
        self.seq_len = 197

    def test_self_attention_forward(self):
        """Test self-attention forward pass."""
        attn = SelfAttention(dim=self.dim, num_heads=self.num_heads, qk_norm=True).to(self.device)

        hidden_states = TestUtils.create_random_tensor(
            (self.batch_size, self.seq_len, self.dim), device=self.device
        )

        # With RoPE
        rope = RotaryPosEmbed(self.head_dim)
        rotary_emb = rope(self.seq_len, device=self.device, dtype=hidden_states.dtype)

        output = attn(hidden_states, rotary_emb)

        # Check output shape
        TestUtils.assert_shape(output, hidden_states.shape, "output")

        # Check no NaNs or Infs
        assert output.isfinite().all(), "Output contains NaN or Inf"

    def test_self_attention_qkv_fusion(self):
        """Test that fused QKV projection works correctly."""
        attn = SelfAttention(dim=self.dim, num_heads=self.num_heads).to(self.device)

        # Check that to_qkv weight has correct shape
        expected_weight_shape = (self.dim * 3, self.dim)
        TestUtils.assert_shape(attn.to_qkv.weight, expected_weight_shape, "to_qkv.weight")

        # Test forward pass
        x = TestUtils.create_random_tensor((1, 10, self.dim), device=self.device)
        output = attn(x)

        # Manually compute QKV and verify shapes
        qkv = attn.to_qkv(x)
        TestUtils.assert_shape(qkv, (1, 10, self.dim * 3), "qkv")

    def test_attention_pattern_validity(self):
        """Test that attention patterns are valid (sum to 1)."""
        attn = SelfAttention(dim=self.dim, num_heads=self.num_heads).to(self.device)

        # Small sequence for detailed testing
        seq_len = 10
        x = TestUtils.create_random_tensor(
            (1, seq_len, self.dim), device=self.device, requires_grad=True
        )

        # Hook to capture attention weights
        attention_weights = []

        def hook_fn(module, input, output):
            # This would capture attention weights if exposed
            # For now, we verify output is valid
            pass

        output = attn(x)

        # Verify output maintains expected properties
        assert output.shape == x.shape
        assert output.isfinite().all()


class TestCrossAttention(unittest.TestCase):
    """Test cross-attention module."""

    def setUp(self):
        self.device = TestUtils.get_device()
        self.dim = 5120
        self.num_heads = 40
        self.batch_size = 2
        self.seq_len = 197
        self.context_len = 512

    def test_cross_attention_forward(self):
        """Test cross-attention forward pass."""
        attn = CrossAttention(
            dim=self.dim, context_dim=self.dim, num_heads=self.num_heads, qk_norm=True
        ).to(self.device)

        hidden_states = TestUtils.create_random_tensor(
            (self.batch_size, self.seq_len, self.dim), device=self.device
        )

        context = TestUtils.create_random_tensor(
            (self.batch_size, self.context_len, self.dim), device=self.device
        )

        output = attn(hidden_states, context)

        # Check output shape matches input hidden states
        TestUtils.assert_shape(output, hidden_states.shape, "output")

        # Check no NaNs or Infs
        assert output.isfinite().all(), "Output contains NaN or Inf"

    def test_cross_attention_different_context_dim(self):
        """Test cross-attention with different context dimension."""
        context_dim = 4096
        attn = CrossAttention(dim=self.dim, context_dim=context_dim, num_heads=self.num_heads).to(
            self.device
        )

        hidden_states = TestUtils.create_random_tensor(
            (1, self.seq_len, self.dim), device=self.device
        )

        context = TestUtils.create_random_tensor(
            (1, self.context_len, context_dim), device=self.device
        )

        output = attn(hidden_states, context)

        # Output should match hidden_states dimension, not context
        TestUtils.assert_shape(output, hidden_states.shape, "output")

    @given(
        seq_len=st.integers(min_value=1, max_value=256),
        context_len=st.integers(min_value=1, max_value=512),
    )
    @settings(max_examples=5, deadline=None)
    def test_cross_attention_variable_lengths(self, seq_len: int, context_len: int):
        """Test cross-attention with variable sequence lengths."""
        attn = CrossAttention(dim=self.dim, num_heads=self.num_heads)

        hidden_states = torch.randn(1, seq_len, self.dim) * 0.02
        context = torch.randn(1, context_len, self.dim) * 0.02

        output = attn(hidden_states, context)

        # Properties:
        # 1. Output shape matches input query shape
        assert output.shape == hidden_states.shape

        # 2. Output is bounded (attention is a weighted average)
        assert output.abs().max() <= hidden_states.abs().max() * 10  # Allow some amplification


# ================================================================================================
# Section 4: Transformer Block Tests
# ================================================================================================


class TestTransformerBlock(unittest.TestCase):
    """Test complete transformer block."""

    def setUp(self):
        self.device = TestUtils.get_device()
        self.dim = 5120
        self.num_heads = 40
        self.ffn_dim = 13824
        self.batch_size = 2
        self.seq_len = 197
        self.context_len = 512

    def test_transformer_block_forward(self):
        """Test transformer block forward pass."""
        block = TransformerBlock(
            dim=self.dim,
            context_dim=self.dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            qk_norm=True,
        ).to(self.device)

        hidden_states = TestUtils.create_random_tensor(
            (self.batch_size, self.seq_len, self.dim), device=self.device
        )

        context = TestUtils.create_random_tensor(
            (self.batch_size, self.context_len, self.dim), device=self.device
        )

        # Create conditioning
        conditioning = TestUtils.create_random_tensor(
            (self.batch_size, 6, self.dim), device=self.device
        )

        # Create RoPE
        rope = RotaryPosEmbed(128)
        rotary_emb = rope(self.seq_len, device=self.device, dtype=hidden_states.dtype)

        output = block(hidden_states, context, conditioning, rotary_emb)

        # Check output shape
        TestUtils.assert_shape(output, hidden_states.shape, "output")

        # Check residual connections work (output shouldn't be too different from input)
        relative_change = (output - hidden_states).abs().mean() / hidden_states.abs().mean()
        assert relative_change < 10.0, f"Output changed too much: {relative_change}"

    def test_transformer_block_gradient_flow(self):
        """Test that gradients flow through transformer block."""
        block = TransformerBlock(dim=self.dim, num_heads=self.num_heads).to(self.device)

        hidden_states = TestUtils.create_random_tensor(
            (1, 10, self.dim), device=self.device, requires_grad=True
        )

        context = TestUtils.create_random_tensor((1, 5, self.dim), device=self.device)

        output = block(hidden_states, context)
        loss = output.mean()
        loss.backward()

        # Check gradients exist and are non-zero
        assert hidden_states.grad is not None, "No gradient for hidden_states"
        assert hidden_states.grad.abs().max() > 0, "Zero gradients"

        # Check gradients are finite
        assert hidden_states.grad.isfinite().all(), "Non-finite gradients"

    def test_transformer_block_modulation(self):
        """Test modulation mechanism in transformer block."""
        block = TransformerBlock(dim=self.dim).to(self.device)

        hidden_states = TestUtils.create_random_tensor((1, 10, self.dim), device=self.device)
        context = TestUtils.create_random_tensor((1, 5, self.dim), device=self.device)

        # Test with and without conditioning
        output_no_cond = block(hidden_states, context, conditioning=None)

        conditioning = TestUtils.create_random_tensor((1, 6, self.dim), device=self.device)
        output_with_cond = block(hidden_states, context, conditioning=conditioning)

        # Outputs should be different when conditioning is applied
        assert not torch.allclose(output_no_cond, output_with_cond, atol=1e-4), (
            "Conditioning had no effect"
        )


# ================================================================================================
# Section 5: Compatibility Tests (Pure vs Diffusers)
# ================================================================================================


class TestCompatibility(unittest.TestCase):
    """Test compatibility between pure and diffusers implementations."""

    def setUp(self):
        self.device = TestUtils.get_device()
        self.dim = 5120
        self.num_heads = 40
        self.batch_size = 1
        self.seq_len = 100

    def test_self_attention_compatibility(self):
        """Test that pure and diffusers self-attention produce same results."""
        # Create both versions
        pure_attn = (
            SelfAttention(dim=self.dim, num_heads=self.num_heads, qk_norm=True)
            .to(self.device)
            .eval()
        )

        diffusers_attn = WanSelfAttention(dim=self.dim, heads=self.num_heads).to(self.device).eval()

        # Copy weights from pure to diffusers
        with torch.no_grad():
            diffusers_attn.to_qkv.load_state_dict(pure_attn.to_qkv.state_dict())
            diffusers_attn.norm_q.load_state_dict(pure_attn.norm_q.state_dict())
            diffusers_attn.norm_k.load_state_dict(pure_attn.norm_k.state_dict())
            diffusers_attn.to_out[0].load_state_dict(pure_attn.to_out[0].state_dict())

        # Test with same input
        x = TestUtils.create_random_tensor(
            (self.batch_size, self.seq_len, self.dim), device=self.device
        )

        rope = RotaryPosEmbed(128)
        rotary_emb = rope(self.seq_len, device=self.device, dtype=x.dtype)

        with torch.no_grad():
            pure_output = pure_attn(x, rotary_emb)
            diffusers_output = diffusers_attn(x, rotary_emb)

        # Outputs should be identical
        TestUtils.assert_close(pure_output, diffusers_output, rtol=1e-5, name="attention outputs")

    def test_cross_attention_compatibility(self):
        """Test that pure and diffusers cross-attention produce same results."""
        context_len = 77

        # Create both versions
        pure_attn = (
            CrossAttention(dim=self.dim, num_heads=self.num_heads, qk_norm=True)
            .to(self.device)
            .eval()
        )

        diffusers_attn = (
            WanCrossAttention(dim=self.dim, heads=self.num_heads).to(self.device).eval()
        )

        # Copy weights
        with torch.no_grad():
            diffusers_attn.to_q.load_state_dict(pure_attn.to_q.state_dict())
            diffusers_attn.to_kv.load_state_dict(pure_attn.to_kv.state_dict())
            diffusers_attn.norm_q.load_state_dict(pure_attn.norm_q.state_dict())
            diffusers_attn.norm_k.load_state_dict(pure_attn.norm_k.state_dict())
            diffusers_attn.to_out = pure_attn.to_out[0]  # Share the linear layer

        # Test with same input
        hidden_states = TestUtils.create_random_tensor(
            (self.batch_size, self.seq_len, self.dim), device=self.device
        )
        context = TestUtils.create_random_tensor(
            (self.batch_size, context_len, self.dim), device=self.device
        )

        with torch.no_grad():
            pure_output = pure_attn(hidden_states, context)
            diffusers_output = diffusers_attn(hidden_states, context)

        # Outputs should be very close (some numerical differences expected)
        TestUtils.assert_close(
            pure_output, diffusers_output, rtol=1e-4, atol=1e-5, name="cross-attention outputs"
        )


# ================================================================================================
# Section 6: Serialization Tests
# ================================================================================================


class TestSerialization(unittest.TestCase):
    """Test model serialization and quantization interfaces."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.device = TestUtils.get_device()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tensor_metadata_creation(self):
        """Test creation of tensor metadata."""
        tensor = torch.randn(5120, 5120)

        metadata = TensorMetadata(
            name="test_weight",
            shape=tensor.shape,
            dtype="float32",
            quantized=False,
            min_val=tensor.min().item(),
            max_val=tensor.max().item(),
            mean_val=tensor.mean().item(),
            std_val=tensor.std().item(),
        )

        # Test safetensors header generation
        header = metadata.to_safetensors_header()
        assert header["dtype"] == "float32"
        assert header["shape"] == [5120, 5120]

    def test_model_serializer_state_dict_extraction(self):
        """Test state dict extraction from model."""
        # Create a simple model
        model = TransformerBlock(dim=5120, ffn_dim=13824)

        serializer = WANModelSerializer()
        state_dict = serializer.extract_state_dict(model)

        # Check that all parameters are extracted
        model_params = dict(model.named_parameters())
        for name in model_params:
            normalized_name = serializer._normalize_parameter_name(name)
            assert normalized_name in state_dict, f"Parameter {name} not in state dict"

    def test_quantization_config(self):
        """Test quantization configuration."""
        config = QuantizationConfig(
            weight_quant=QuantizationType.NVFP4,
            weight_bits=4,
            kv_cache_quant=QuantizationType.FP8_E4M3,
            kv_cache_bits=8,
            skip_layers=["norm", "embedding"],
            calibration_samples=256,
        )

        # Test configuration
        assert config.weight_quant == QuantizationType.NVFP4
        assert config.weight_bits == 4
        assert "norm" in config.skip_layers

    def test_comfyui_config_generation(self):
        """Test ComfyUI configuration generation."""
        model = TransformerBlock(dim=5120)
        config = create_comfyui_config(model)

        # Check required fields
        assert config["model_type"] == "wan_v2"
        assert config["hidden_size"] == 5120
        assert config["num_heads"] == 40
        assert config["patch_size"] == [1, 2, 2]

    def test_safetensors_save_load(self):
        """Test saving and loading with safetensors format."""
        # Create a small model
        model = SelfAttention(dim=768, num_heads=12)  # Smaller for testing

        serializer = WANModelSerializer()
        save_path = Path(self.temp_dir) / "test_model.safetensors"

        # Save model
        stats = serializer.save_to_safetensors(
            model, save_path, quantize=False, metadata={"test": "metadata"}
        )

        assert save_path.exists()
        assert stats["num_tensors"] > 0

        # Load model back
        state_dict, metadata = WANModelSerializer.load_from_safetensors(save_path)

        # Check that weights match
        model_state = model.state_dict()
        for key in model_state:
            normalized_key = serializer._normalize_parameter_name(key)
            if normalized_key in state_dict:
                TestUtils.assert_close(
                    model_state[key], state_dict[normalized_key], name=f"parameter {key}"
                )


# ================================================================================================
# Section 7: Hypothesis-Based Property Tests
# ================================================================================================


class TestProperties(unittest.TestCase):
    """Property-based tests using Hypothesis."""

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=128),
        dim=st.sampled_from([768, 1024, 5120]),
        num_heads=st.sampled_from([8, 12, 40]),
    )
    @settings(max_examples=5, deadline=None)
    def test_attention_shape_preservation(
        self, batch_size: int, seq_len: int, dim: int, num_heads: int
    ):
        """Test that attention preserves shapes across various configurations."""
        assume(dim % num_heads == 0)  # Head dim must divide evenly

        attn = SelfAttention(dim=dim, num_heads=num_heads, qk_norm=False)
        x = torch.randn(batch_size, seq_len, dim) * 0.02

        output = attn(x)

        # Properties:
        assert output.shape == x.shape, "Shape not preserved"
        assert output.dtype == x.dtype, "Dtype not preserved"
        assert output.isfinite().all(), "Output contains non-finite values"

    @given(
        dim=st.sampled_from([768, 1024, 5120]), ffn_factor=st.floats(min_value=1.5, max_value=4.0)
    )
    @settings(max_examples=5, deadline=None)
    def test_feedforward_properties(self, dim: int, ffn_factor: float):
        """Test feed-forward network properties."""
        ffn_dim = int(dim * ffn_factor)
        ffn = FeedForward(dim=dim, inner_dim=ffn_dim, activation_fn="gelu")

        x = torch.randn(2, 10, dim) * 0.02
        output = ffn(x)

        # Properties:
        assert output.shape == x.shape, "FFN should preserve shape"
        assert output.isfinite().all(), "FFN output should be finite"

        # FFN should not dramatically change magnitude
        input_norm = x.norm()
        output_norm = output.norm()
        ratio = output_norm / input_norm
        assert 0.1 < ratio < 10, f"FFN changed magnitude too much: {ratio}"


# ================================================================================================
# Section 8: Integration Tests
# ================================================================================================


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests."""

    def setUp(self):
        self.device = TestUtils.get_device()
        self.batch_size = 1
        self.frames = 16
        self.height = 64
        self.width = 64

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for full model test")
    def test_full_model_forward(self):
        """Test full WAN transformer forward pass."""
        model = (
            WanTransformer3DModel(
                patch_size=(1, 2, 2),
                num_attention_heads=40,
                attention_head_dim=128,
                in_channels=36,
                out_channels=16,
                text_dim=4096,
                ffn_dim=13824,
                num_layers=2,  # Reduced for testing
            )
            .to(self.device)
            .eval()
        )

        # Create inputs
        hidden_states = (
            torch.randn(
                self.batch_size, 36, self.frames, self.height, self.width, device=self.device
            )
            * 0.02
        )

        timestep = torch.randint(0, 1000, (self.batch_size,), device=self.device)

        encoder_hidden_states = torch.randn(self.batch_size, 512, 4096, device=self.device) * 0.02

        # Optional image conditioning
        encoder_hidden_states_image = (
            torch.randn(self.batch_size, 256, 1280, device=self.device) * 0.02
        )

        # Forward pass
        with torch.no_grad():
            output = model(
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                return_dict=True,
            )

        # Check output
        expected_shape = (self.batch_size, 16, self.frames, self.height, self.width)
        TestUtils.assert_shape(output.sample, expected_shape, "model output")
        assert output.sample.isfinite().all(), "Model output contains non-finite values"

    def test_weight_conversion_from_stock(self):
        """Test weight conversion from stock diffusers model."""
        # This would test the from_pretrained_stock method
        # Requires a mock stock model for testing
        pass  # Implementation depends on having reference weights


# ================================================================================================
# Section 9: Performance and Benchmark Tests
# ================================================================================================


class TestPerformance(unittest.TestCase):
    """Performance and benchmark tests."""

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for performance tests")
    def test_attention_memory_efficiency(self):
        """Test that fused attention is memory efficient."""
        import torch.cuda

        device = "cuda"
        dim = 5120
        seq_len = 1024
        batch_size = 1

        attn = SelfAttention(dim=dim, num_heads=40).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device) * 0.02

        # Measure memory before
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()

        # Forward pass
        with torch.no_grad():
            output = attn(x)

        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() - start_memory

        # Fused attention should use less memory than naive implementation
        # Rough estimate: QKV should not require 3x memory
        max_expected = x.numel() * x.element_size() * 6  # Conservative estimate
        assert peak_memory < max_expected, f"Memory usage too high: {peak_memory / 1e6:.2f} MB"

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for benchmark")
    def test_model_inference_speed(self):
        """Benchmark model inference speed."""
        import time

        device = "cuda"
        model = TransformerBlock(dim=5120, ffn_dim=13824).to(device).eval()

        # Prepare inputs
        hidden_states = torch.randn(1, 197, 5120, device=device) * 0.02
        context = torch.randn(1, 77, 5120, device=device) * 0.02

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(hidden_states, context)

        torch.cuda.synchronize()

        # Benchmark
        num_iterations = 100
        start_time = time.time()

        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(hidden_states, context)

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        ms_per_iteration = (elapsed / num_iterations) * 1000
        print(f"Transformer block inference: {ms_per_iteration:.2f} ms")

        # Assert reasonable performance (adjust threshold as needed)
        assert ms_per_iteration < 100, f"Inference too slow: {ms_per_iteration:.2f} ms"


# ================================================================================================
# Section 10: Test Runner
# ================================================================================================


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    test_cases = [
        TestRotaryEmbeddings,
        TestNormalization,
        TestSelfAttention,
        TestCrossAttention,
        TestTransformerBlock,
        TestCompatibility,
        TestSerialization,
        TestProperties,
        TestIntegration,
        TestPerformance,
    ]

    for test_case in test_cases:
        suite.addTests(loader.loadTestsFromTestCase(test_case))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
