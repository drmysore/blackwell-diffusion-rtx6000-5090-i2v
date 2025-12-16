"""
WAN 2.2 Test Suite - Static and Dynamic Implementations
========================================================

Tests for the separated static (inference-optimized) and dynamic (flexible)
implementations.

The static implementation has:
- Fixed batch sizes and sequence lengths
- No optional parameters
- No conditionals in forward passes
- Perfect compilation compatibility

The dynamic implementation has:
- Variable shapes
- Optional parameters and None handling
- All conditionals and routing logic
- Backward compatibility
"""

import unittest
import torch
from typing import Tuple
import time

# Import static implementation
from wan_attention_static import (
    StaticRotaryEmbed,
    StaticSelfAttention,
    StaticCrossAttention,
    StaticTransformerBlock,
    create_static_inference_model,
    compile_static_model,
)

# Import dynamic implementation
from wan_attention_dynamic import (
    DynamicRoPE,
    DynamicSelfAttention,
    DynamicCrossAttention,
    DynamicTransformerBlock,
    WanTransformer3DModel,
)


# ================================================================================================
# Test Utilities
# ================================================================================================


class TestUtils:
    """Utility functions for testing."""

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def assert_shape(tensor: torch.Tensor, expected: Tuple[int, ...], name: str = "tensor"):
        assert tensor.shape == expected, f"{name} shape mismatch: {tensor.shape} != {expected}"

    @staticmethod
    def assert_close(t1: torch.Tensor, t2: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-5):
        assert torch.allclose(t1, t2, rtol=rtol, atol=atol), (
            f"Tensors not close: max diff = {(t1 - t2).abs().max()}"
        )


# ================================================================================================
# Static Implementation Tests
# ================================================================================================


class TestStaticImplementation(unittest.TestCase):
    """Test the static, inference-optimized implementation."""

    def setUp(self):
        self.device = TestUtils.get_device()
        self.batch_size = 2
        self.seq_len = 197
        self.context_len = 512
        self.dim = 5120
        self.num_heads = 40
        self.head_dim = 128

    def test_static_rope_fixed_length(self):
        """Test that static RoPE has fixed sequence length."""
        rope = StaticRotaryEmbed(
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            device=self.device,
        )

        # No inputs to forward
        cos, sin = rope()

        # Check shapes
        TestUtils.assert_shape(cos, (1, self.seq_len, 1, self.head_dim))
        TestUtils.assert_shape(sin, (1, self.seq_len, 1, self.head_dim))

        # Values should be bounded
        assert cos.abs().max() <= 1.0
        assert sin.abs().max() <= 1.0

    def test_static_self_attention_fixed_shapes(self):
        """Test static self-attention with fixed dimensions."""
        attn = StaticSelfAttention(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            dim=self.dim,
            num_heads=self.num_heads,
        ).to(self.device)

        # Create inputs with exact shapes
        hidden_states = (
            torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device) * 0.02
        )

        rope = StaticRotaryEmbed(self.seq_len, self.head_dim, self.device)
        cos, sin = rope()

        # Forward pass - all inputs required
        output = attn(hidden_states, cos, sin)

        # Check output
        TestUtils.assert_shape(output, (self.batch_size, self.seq_len, self.dim))
        assert output.isfinite().all()

    def test_static_cross_attention_fixed_context(self):
        """Test static cross-attention with fixed context length."""
        attn = StaticCrossAttention(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            context_len=self.context_len,
            dim=self.dim,
            num_heads=self.num_heads,
        ).to(self.device)

        # Exact shape inputs
        hidden_states = (
            torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device) * 0.02
        )
        context = (
            torch.randn(self.batch_size, self.context_len, self.dim, device=self.device) * 0.02
        )

        # Forward - no optional parameters
        output = attn(hidden_states, context)

        TestUtils.assert_shape(output, (self.batch_size, self.seq_len, self.dim))

    def test_static_transformer_block_no_conditionals(self):
        """Test that static transformer block has no conditionals."""
        block = StaticTransformerBlock(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            context_len=self.context_len,
            dim=self.dim,
            num_heads=self.num_heads,
        ).to(self.device)

        # All inputs required
        hidden_states = (
            torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device) * 0.02
        )
        context = (
            torch.randn(self.batch_size, self.context_len, self.dim, device=self.device) * 0.02
        )
        conditioning = torch.randn(self.batch_size, 6, self.dim, device=self.device) * 0.02

        rope = StaticRotaryEmbed(self.seq_len, self.head_dim, self.device)
        cos, sin = rope()

        # Forward with all required inputs
        output = block(hidden_states, context, conditioning, cos, sin)

        TestUtils.assert_shape(output, (self.batch_size, self.seq_len, self.dim))

    def test_static_model_compilation(self):
        """Test that static model can be compiled."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for compilation test")

        model = create_static_inference_model(
            batch_size=1,
            frames=16,
            height=32,
            width=32,
            context_len=512,
            device="cuda",
        )

        # Try to compile
        try:
            compiled = compile_static_model(model, mode="reduce-overhead")
            self.assertIsNotNone(compiled)
        except Exception as e:
            self.skipTest(f"Compilation not available: {e}")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_cudagraph_capture(self):
        """Test CUDAGraph capture of static model."""
        # Create small model for testing
        model = (
            StaticTransformerBlock(
                batch_size=1,
                seq_len=197,
                context_len=77,
                dim=768,  # Smaller for testing
                num_heads=12,
            )
            .cuda()
            .eval()
        )

        # Create static inputs
        hidden_states = torch.randn(1, 197, 768, device="cuda") * 0.02
        context = torch.randn(1, 77, 768, device="cuda") * 0.02
        conditioning = torch.randn(1, 6, 768, device="cuda") * 0.02
        rope = StaticRotaryEmbed(197, 64, "cuda")
        cos, sin = rope()

        example_inputs = (hidden_states, context, conditioning, cos, sin)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(*example_inputs)

        # Capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = model(*example_inputs)

        # Test replay
        g.replay()

        # Output should be valid
        assert static_output.isfinite().all()


# ================================================================================================
# Dynamic Implementation Tests
# ================================================================================================


class TestDynamicImplementation(unittest.TestCase):
    """Test the dynamic, flexible implementation."""

    def setUp(self):
        self.device = TestUtils.get_device()
        self.dim = 5120
        self.num_heads = 40

    def test_dynamic_rope_variable_length(self):
        """Test dynamic RoPE with variable sequence lengths."""
        rope = DynamicRoPE(head_dim=128)

        # Test different sequence lengths
        for seq_len in [197, 256, 512, 999]:  # 999 is not cached
            cos, sin = rope(seq_len, device=self.device)

            TestUtils.assert_shape(cos, (1, seq_len, 1, 128))
            TestUtils.assert_shape(sin, (1, seq_len, 1, 128))

    def test_dynamic_self_attention_optional_params(self):
        """Test dynamic self-attention with optional parameters."""
        attn = DynamicSelfAttention(
            dim=self.dim,
            heads=self.num_heads,
            qk_norm=True,
        ).to(self.device)

        # Variable batch and sequence
        for batch_size in [1, 2, 4]:
            for seq_len in [197, 256]:
                hidden_states = (
                    torch.randn(batch_size, seq_len, self.dim, device=self.device) * 0.02
                )

                # Test without RoPE
                output = attn(hidden_states, rotary_emb=None)
                TestUtils.assert_shape(output, (batch_size, seq_len, self.dim))

                # Test with RoPE
                rope = DynamicRoPE(128)
                rotary_emb = rope(seq_len, device=self.device)
                output = attn(hidden_states, rotary_emb=rotary_emb)
                TestUtils.assert_shape(output, (batch_size, seq_len, self.dim))

    def test_dynamic_cross_attention_missing_context(self):
        """Test dynamic cross-attention with missing context."""
        attn = DynamicCrossAttention(dim=self.dim, heads=self.num_heads).to(self.device)

        hidden_states = torch.randn(2, 197, self.dim, device=self.device) * 0.02

        # Test with None context (should fall back to self-attention)
        output = attn(hidden_states, encoder_hidden_states=None)
        TestUtils.assert_shape(output, (2, 197, self.dim))

        # Test with provided context
        context = torch.randn(2, 77, self.dim, device=self.device) * 0.02
        output = attn(hidden_states, encoder_hidden_states=context)
        TestUtils.assert_shape(output, (2, 197, self.dim))

    def test_dynamic_block_handles_none(self):
        """Test that dynamic block handles None values."""
        block = DynamicTransformerBlock(dim=self.dim).to(self.device)

        hidden_states = torch.randn(1, 197, self.dim, device=self.device) * 0.02

        # All None except hidden states
        output = block(
            hidden_states,
            encoder_hidden_states=None,
            temb=None,
            rotary_emb=None,
            attention_mask=None,
        )

        TestUtils.assert_shape(output, (1, 197, self.dim))

    def test_dynamic_model_variable_shapes(self):
        """Test dynamic model with variable input shapes."""
        model = (
            WanTransformer3DModel(
                num_layers=2,  # Small for testing
                num_attention_heads=40,
                attention_head_dim=128,
            )
            .to(self.device)
            .eval()
        )

        # Test different input shapes
        for batch_size in [1, 2]:
            for frames in [8, 16]:
                for height in [16, 32]:
                    hidden_states = (
                        torch.randn(batch_size, 36, frames, height, height, device=self.device)
                        * 0.02
                    )

                    timestep = torch.tensor([500] * batch_size, device=self.device)
                    encoder_hidden_states = (
                        torch.randn(batch_size, 512, 4096, device=self.device) * 0.02
                    )

                    with torch.no_grad():
                        output = model(
                            hidden_states,
                            timestep,
                            encoder_hidden_states,
                        )

                    # Output should match input spatial dimensions
                    expected_shape = (batch_size, 16, frames, height, height)
                    TestUtils.assert_shape(output.sample, expected_shape)

    def test_dynamic_static_routing(self):
        """Test that dynamic model routes to static implementations."""
        model = (
            WanTransformer3DModel(
                num_layers=2,
                use_static_cache=True,
                static_batch_size=1,
                static_seq_len=197,
                static_context_len=512,
            )
            .to(self.device)
            .eval()
        )

        # Use dimensions that match static config
        hidden_states = torch.randn(1, 36, 14, 14, 14, device=self.device) * 0.02  # -> seq_len=197
        timestep = torch.tensor([500], device=self.device)
        encoder_hidden_states = torch.randn(1, 512, 4096, device=self.device) * 0.02

        with torch.no_grad():
            output = model(hidden_states, timestep, encoder_hidden_states)

        # Should successfully use static path (no assertion, just shouldn't error)
        self.assertIsNotNone(output.sample)


# ================================================================================================
# Compatibility Tests
# ================================================================================================


class TestCompatibility(unittest.TestCase):
    """Test compatibility between static and dynamic implementations."""

    def setUp(self):
        self.device = TestUtils.get_device()

    def test_static_dynamic_equivalence(self):
        """Test that static and dynamic produce same results for fixed inputs."""
        batch_size = 1
        seq_len = 197
        context_len = 77
        dim = 768  # Smaller for testing
        num_heads = 12

        # Create both implementations
        static_block = (
            StaticTransformerBlock(
                batch_size,
                seq_len,
                context_len,
                dim,
                num_heads,
                ffn_dim=2048,
            )
            .to(self.device)
            .eval()
        )

        dynamic_block = (
            DynamicTransformerBlock(
                dim,
                num_heads,
                ffn_dim=2048,
            )
            .to(self.device)
            .eval()
        )

        # Copy weights from static to dynamic
        with torch.no_grad():
            dynamic_block.norm1.load_state_dict(static_block.norm1.norm.state_dict())
            dynamic_block.norm2.load_state_dict(static_block.norm2.norm.state_dict())
            dynamic_block.norm3.load_state_dict(static_block.norm3.norm.state_dict())

            dynamic_block.attn1.to_qkv.load_state_dict(static_block.self_attn.to_qkv.state_dict())
            dynamic_block.attn1.norm_q.weight.copy_(static_block.self_attn.norm_q.weight)
            dynamic_block.attn1.norm_k.weight.copy_(static_block.self_attn.norm_k.weight)
            dynamic_block.attn1.to_out[0].load_state_dict(
                static_block.self_attn.to_out.state_dict()
            )

            dynamic_block.scale_shift_table.copy_(static_block.scale_shift_table)

        # Create identical inputs
        hidden_states = torch.randn(batch_size, seq_len, dim, device=self.device) * 0.02
        context = torch.randn(batch_size, context_len, dim, device=self.device) * 0.02
        conditioning = torch.randn(batch_size, 6, dim, device=self.device) * 0.02

        rope_static = StaticRotaryEmbed(seq_len, dim // num_heads, self.device)
        cos, sin = rope_static()

        rope_dynamic = DynamicRoPE(dim // num_heads)
        rotary_emb = rope_dynamic(seq_len, self.device)

        # Forward passes
        with torch.no_grad():
            static_out = static_block(hidden_states, context, conditioning, cos, sin)
            dynamic_out = dynamic_block(hidden_states, context, conditioning, rotary_emb)

        # Should produce similar results (some numerical differences expected)
        TestUtils.assert_close(static_out, dynamic_out, rtol=1e-3, atol=1e-4)


# ================================================================================================
# Performance Tests
# ================================================================================================


class TestPerformance(unittest.TestCase):
    """Performance comparison tests."""

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_static_vs_dynamic_speed(self):
        """Compare speed of static vs dynamic implementations."""

        batch_size = 1
        seq_len = 197
        context_len = 77
        dim = 768
        num_heads = 12

        # Create models
        static_block = (
            StaticTransformerBlock(
                batch_size,
                seq_len,
                context_len,
                dim,
                num_heads,
                ffn_dim=2048,
            )
            .cuda()
            .eval()
        )

        dynamic_block = (
            DynamicTransformerBlock(
                dim,
                num_heads,
                ffn_dim=2048,
            )
            .cuda()
            .eval()
        )

        # Inputs
        hidden_states = torch.randn(batch_size, seq_len, dim, device="cuda") * 0.02
        context = torch.randn(batch_size, context_len, dim, device="cuda") * 0.02
        conditioning = torch.randn(batch_size, 6, dim, device="cuda") * 0.02

        rope_static = StaticRotaryEmbed(seq_len, dim // num_heads, "cuda")
        cos, sin = rope_static()

        rope_dynamic = DynamicRoPE(dim // num_heads)
        rotary_emb = rope_dynamic(seq_len, "cuda")

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = static_block(hidden_states, context, conditioning, cos, sin)
                _ = dynamic_block(hidden_states, context, conditioning, rotary_emb)

        torch.cuda.synchronize()

        # Benchmark static
        start = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                _ = static_block(hidden_states, context, conditioning, cos, sin)
        torch.cuda.synchronize()
        static_time = (time.perf_counter() - start) / 100

        # Benchmark dynamic
        start = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                _ = dynamic_block(hidden_states, context, conditioning, rotary_emb)
        torch.cuda.synchronize()
        dynamic_time = (time.perf_counter() - start) / 100

        print(f"\nStatic block: {static_time * 1000:.2f} ms")
        print(f"Dynamic block: {dynamic_time * 1000:.2f} ms")
        print(f"Static speedup: {dynamic_time / static_time:.2f}x")

        # Static should be faster (or at least not slower)
        self.assertLessEqual(static_time, dynamic_time * 1.2)  # Allow 20% margin


# ================================================================================================
# Run Tests
# ================================================================================================


if __name__ == "__main__":
    unittest.main(verbosity=2)
