"""
WAN 2.2 Static Model Factory and Manager
=========================================

Utilities for creating, caching, and managing static models for different
configurations. This handles:

1. Pre-allocation of common configurations
2. Lazy creation of models as needed
3. Memory-efficient model sharing
4. Configuration validation
5. Automatic routing to best available model
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass
import warnings
import gc

from wan_attention_static import (
    StaticWANInferenceModel,
    compile_static_model,
)

from wan_config_calculator import (
    StaticModelConfig,
    ConfigCalculator,
)


@dataclass
class ModelKey:
    """Unique key for a static model configuration."""

    batch_size: int
    seq_len: int
    context_len: int

    def __hash__(self):
        return hash((self.batch_size, self.seq_len, self.context_len))

    def __eq__(self, other):
        return (
            self.batch_size == other.batch_size
            and self.seq_len == other.seq_len
            and self.context_len == other.context_len
        )

    @classmethod
    def from_config(cls, config: StaticModelConfig) -> "ModelKey":
        """Create key from configuration."""
        return cls(
            batch_size=config.batch_size,
            seq_len=config.sequence_length,
            context_len=config.context_length,
        )

    @classmethod
    def from_tensors(
        cls,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
    ) -> "ModelKey":
        """Create key from tensor shapes."""
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        context_len = context.shape[1]
        return cls(batch_size, seq_len, context_len)


class StaticModelCache:
    """
    Cache for static models with different configurations.
    Manages memory and provides efficient model reuse.
    """

    def __init__(
        self,
        max_models: int = 10,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        compile_models: bool = False,
        use_cudagraph: bool = False,
    ):
        """
        Initialize model cache.

        Args:
            max_models: Maximum number of models to keep in cache
            device: Device for models
            dtype: Model precision
            compile_models: Whether to compile models with torch.compile
            use_cudagraph: Whether to use CUDAGraph capture
        """
        self.max_models = max_models
        self.device = device
        self.dtype = dtype
        self.compile_models = compile_models
        self.use_cudagraph = use_cudagraph

        # Cache storage
        self._models: Dict[ModelKey, StaticWANInferenceModel] = {}
        self._access_count: Dict[ModelKey, int] = {}
        self._compiled_models: Dict[ModelKey, callable] = {}
        self._cudagraph_models: Dict[ModelKey, callable] = {}

    def get_model(
        self,
        key: ModelKey,
        create_if_missing: bool = True,
    ) -> Optional[StaticWANInferenceModel]:
        """
        Get model from cache or create if missing.

        Args:
            key: Model configuration key
            create_if_missing: Whether to create model if not in cache

        Returns:
            Static model or None if not found and create_if_missing=False
        """

        # Check if model exists
        if key in self._models:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._models[key]

        # Create if requested
        if create_if_missing:
            return self._create_model(key)

        return None

    def _create_model(self, key: ModelKey) -> StaticWANInferenceModel:
        """Create and cache a new model."""

        # Check cache size
        if len(self._models) >= self.max_models:
            self._evict_least_used()

        # Create model
        print(
            f"Creating static model: batch={key.batch_size}, "
            f"seq={key.seq_len}, context={key.context_len}"
        )

        model = StaticWANInferenceModel(
            batch_size=key.batch_size,
            seq_len=key.seq_len,
            context_len=key.context_len,
            num_layers=48,
            dim=5120,
            num_heads=40,
            ffn_dim=13824,
            dropout=0.0,
            device=self.device,
            dtype=self.dtype,
        )

        model.eval()

        # Cache model
        self._models[key] = model
        self._access_count[key] = 1

        # Compile if requested
        if self.compile_models:
            self._compile_model(key, model)

        return model

    def _compile_model(self, key: ModelKey, model: StaticWANInferenceModel):
        """Compile model with torch.compile."""
        try:
            print(f"Compiling model {key}...")
            compiled = compile_static_model(model, mode="reduce-overhead")
            self._compiled_models[key] = compiled
            print(f"Model {key} compiled successfully")
        except Exception as e:
            warnings.warn(f"Failed to compile model {key}: {e}")

    def _evict_least_used(self):
        """Evict least recently used model from cache."""
        if not self._models:
            return

        # Find least used model
        min_key = min(self._access_count, key=self._access_count.get)

        print(f"Evicting model {min_key} from cache")

        # Remove model and related data
        del self._models[min_key]
        del self._access_count[min_key]

        if min_key in self._compiled_models:
            del self._compiled_models[min_key]

        if min_key in self._cudagraph_models:
            del self._cudagraph_models[min_key]

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def preload_common_configs(self):
        """Preload models for common configurations."""
        common_configs = [
            ModelKey(1, 3136, 512),  # 256x256 16-frame, text only
            ModelKey(1, 3136, 1024),  # 256x256 16-frame, text + image
            ModelKey(1, 12544, 512),  # 512x512 16-frame, text only
            ModelKey(1, 197, 512),  # 14x14 patches (image), text only
            ModelKey(1, 256, 512),  # 16x16 patches (image), text only
        ]

        for key in common_configs:
            self.get_model(key, create_if_missing=True)

        print(f"Preloaded {len(common_configs)} common configurations")

    def clear_cache(self):
        """Clear all cached models."""
        self._models.clear()
        self._access_count.clear()
        self._compiled_models.clear()
        self._cudagraph_models.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cached_models": len(self._models),
            "compiled_models": len(self._compiled_models),
            "cudagraph_models": len(self._cudagraph_models),
            "access_counts": dict(self._access_count),
            "total_parameters": sum(
                sum(p.numel() for p in m.parameters()) for m in self._models.values()
            ),
        }


class StaticModelFactory:
    """
    Factory for creating static models with automatic configuration.
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        cache_models: bool = True,
        compile_models: bool = False,
    ):
        """
        Initialize model factory.

        Args:
            device: Device for models
            dtype: Model precision
            cache_models: Whether to cache created models
            compile_models: Whether to compile models
        """
        self.device = device
        self.dtype = dtype

        # Model cache
        self.cache = (
            StaticModelCache(
                device=device,
                dtype=dtype,
                compile_models=compile_models,
            )
            if cache_models
            else None
        )

    def create_from_video_dims(
        self,
        batch_size: int,
        frames: int,
        height: int,
        width: int,
        text_tokens: int = 512,
        image_conditioning: bool = False,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
    ) -> StaticWANInferenceModel:
        """
        Create model from video dimensions.

        Args:
            batch_size: Batch size
            frames: Number of frames
            height: Frame height
            width: Frame width
            text_tokens: Number of text tokens
            image_conditioning: Whether to include image conditioning
            patch_size: Patch dimensions

        Returns:
            Static model configured for the given dimensions
        """

        # Calculate configuration
        video_config = ConfigCalculator.calculate_video_config(frames, height, width, patch_size)

        if image_conditioning:
            context_config = ConfigCalculator.calculate_context_config(text_tokens, height, width)
        else:
            context_config = ConfigCalculator.calculate_context_config(text_tokens)

        config = StaticModelConfig(
            batch_size=batch_size,
            video_config=video_config,
            context_config=context_config,
            dtype=self.dtype,
        )

        # Create or get from cache
        key = ModelKey.from_config(config)

        if self.cache:
            return self.cache.get_model(key)
        else:
            return self._create_model(config)

    def create_from_config(
        self,
        config: StaticModelConfig,
    ) -> StaticWANInferenceModel:
        """
        Create model from configuration object.

        Args:
            config: Model configuration

        Returns:
            Static model
        """

        key = ModelKey.from_config(config)

        if self.cache:
            return self.cache.get_model(key)
        else:
            return self._create_model(config)

    def _create_model(
        self,
        config: StaticModelConfig,
    ) -> StaticWANInferenceModel:
        """Create model without caching."""

        return StaticWANInferenceModel(
            batch_size=config.batch_size,
            seq_len=config.sequence_length,
            context_len=config.context_length,
            num_layers=config.num_layers,
            dim=config.dim,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=0.0,
            device=self.device,
            dtype=config.dtype,
        ).eval()

    def create_for_common_sizes(
        self,
        batch_size: int = 1,
    ) -> Dict[str, StaticWANInferenceModel]:
        """
        Create models for common sizes.

        Args:
            batch_size: Batch size for all models

        Returns:
            Dictionary of name -> model
        """

        models = {}

        common_sizes = [
            ("256_16frame", 16, 256, 256, False),
            ("512_16frame", 16, 512, 512, False),
            ("256_24frame", 24, 256, 256, False),
            ("512_image", 1, 512, 512, True),
            ("1024_image", 1, 1024, 1024, True),
        ]

        for name, frames, height, width, use_image in common_sizes:
            model = self.create_from_video_dims(
                batch_size, frames, height, width, image_conditioning=use_image
            )
            models[name] = model

        return models


class AdaptiveStaticModel(nn.Module):
    """
    Wrapper that automatically selects appropriate static model based on input shapes.
    Falls back to dynamic model for unsupported configurations.
    """

    def __init__(
        self,
        factory: Optional[StaticModelFactory] = None,
        fallback_model: Optional[nn.Module] = None,
        verbose: bool = False,
    ):
        """
        Initialize adaptive model.

        Args:
            factory: Model factory for creating static models
            fallback_model: Dynamic model for unsupported configurations
            verbose: Whether to print routing decisions
        """
        super().__init__()

        self.factory = factory or StaticModelFactory()
        self.fallback_model = fallback_model
        self.verbose = verbose

        # Track routing statistics
        self.static_calls = 0
        self.dynamic_calls = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        block_conditioning: torch.Tensor,
        output_conditioning: torch.Tensor,
        **kwargs,  # Ignore extra arguments
    ) -> torch.Tensor:
        """
        Forward pass with automatic model selection.

        Args:
            hidden_states: [B, L, D] tensor
            context: [B, C, D] tensor
            block_conditioning: [B, num_layers, 6, D] tensor
            output_conditioning: [B, 2, D] tensor

        Returns:
            Model output
        """

        # Extract dimensions
        batch_size, seq_len, dim = hidden_states.shape
        context_len = context.shape[1]

        # Try to get static model
        key = ModelKey(batch_size, seq_len, context_len)

        if self.factory.cache:
            model = self.factory.cache.get_model(key, create_if_missing=False)

            if model is not None:
                if self.verbose:
                    print(f"Using static model for {key}")
                self.static_calls += 1
                return model(
                    hidden_states,
                    context,
                    block_conditioning,
                    output_conditioning,
                )

        # Fall back to dynamic model
        if self.fallback_model is not None:
            if self.verbose:
                print(f"Using dynamic model for {key}")
            self.dynamic_calls += 1
            return self.fallback_model(
                hidden_states,
                context,
                block_conditioning,
                output_conditioning,
                **kwargs,
            )

        # No model available
        raise RuntimeError(
            f"No model available for configuration: "
            f"batch={batch_size}, seq={seq_len}, context={context_len}"
        )

    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        total = self.static_calls + self.dynamic_calls
        return {
            "static_calls": self.static_calls,
            "dynamic_calls": self.dynamic_calls,
            "total_calls": total,
            "static_percentage": (self.static_calls / total * 100) if total > 0 else 0,
        }


def demo_usage():
    """Demonstrate usage of the factory and manager."""

    print("=" * 80)
    print("WAN 2.2 Static Model Factory Demo")
    print("=" * 80)

    # Create factory
    factory = StaticModelFactory(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
        cache_models=True,
        compile_models=False,  # Set to True for production
    )

    # Example 1: Create model for specific video
    print("\n1. Creating model for 256x256 16-frame video:")
    model_256 = factory.create_from_video_dims(
        batch_size=1,
        frames=16,
        height=256,
        width=256,
        text_tokens=512,
        image_conditioning=False,
    )
    print(f"   Created model with seq_len={3136}, context_len={512}")

    # Example 2: Create models for common sizes
    print("\n2. Creating models for common sizes:")
    common_models = factory.create_for_common_sizes(batch_size=1)
    for name, model in common_models.items():
        print(f"   {name}: created")

    # Example 3: Cache statistics
    if factory.cache:
        print("\n3. Cache statistics:")
        stats = factory.cache.get_stats()
        print(f"   Cached models: {stats['cached_models']}")
        print(f"   Total parameters: {stats['total_parameters']:,}")

    # Example 4: Adaptive model
    print("\n4. Creating adaptive model:")
    adaptive = AdaptiveStaticModel(
        factory=factory,
        fallback_model=None,  # Would be dynamic model
        verbose=True,
    )

    # Test with cached size
    test_input = torch.randn(1, 3136, 5120).to(factory.device).to(factory.dtype)
    test_context = torch.randn(1, 512, 5120).to(factory.device).to(factory.dtype)
    test_block_cond = torch.randn(1, 48, 6, 5120).to(factory.device).to(factory.dtype)
    test_output_cond = torch.randn(1, 2, 5120).to(factory.device).to(factory.dtype)

    try:
        output = adaptive(test_input, test_context, test_block_cond, test_output_cond)
        print(f"   Output shape: {output.shape}")
    except RuntimeError as e:
        print(f"   {e}")

    print("\n5. Routing statistics:")
    routing_stats = adaptive.get_routing_stats()
    for key, value in routing_stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    demo_usage()
