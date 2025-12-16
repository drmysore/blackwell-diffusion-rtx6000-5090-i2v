"""
WAN 2.2 Configuration Calculator
=================================

Utilities for determining the exact static model configuration needed
for given video dimensions, memory constraints, and deployment targets.

Since the static implementation requires fixed dimensions at initialization,
these utilities help calculate:
- Sequence lengths from video dimensions
- Context lengths from text/image inputs
- Memory requirements
- Optimal configurations for different hardware
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import torch


@dataclass
class VideoConfig:
    """Configuration for video dimensions."""

    frames: int
    height: int
    width: int
    patch_size: Tuple[int, int, int] = (1, 2, 2)

    @property
    def frames_patched(self) -> int:
        """Number of temporal patches."""
        return self.frames // self.patch_size[0]

    @property
    def height_patched(self) -> int:
        """Number of height patches."""
        return self.height // self.patch_size[1]

    @property
    def width_patched(self) -> int:
        """Number of width patches."""
        return self.width // self.patch_size[2]

    @property
    def sequence_length(self) -> int:
        """Total sequence length after patching."""
        return self.frames_patched * self.height_patched * self.width_patched

    @property
    def spatial_patches(self) -> int:
        """Number of spatial patches per frame."""
        return self.height_patched * self.width_patched

    def __str__(self) -> str:
        return (
            f"Video: {self.frames}x{self.height}x{self.width} -> "
            f"Patches: {self.frames_patched}x{self.height_patched}x{self.width_patched} -> "
            f"Seq: {self.sequence_length}"
        )


@dataclass
class ContextConfig:
    """Configuration for context (text + image) dimensions."""

    text_tokens: int = 512  # T5-XXL tokens
    image_patches: Optional[int] = None  # Optional image conditioning

    @property
    def total_context_length(self) -> int:
        """Total context length."""
        if self.image_patches is not None:
            return self.text_tokens + self.image_patches
        return self.text_tokens

    def __str__(self) -> str:
        if self.image_patches:
            return f"Context: {self.text_tokens} text + {self.image_patches} image = {self.total_context_length}"
        return f"Context: {self.text_tokens} text"


@dataclass
class StaticModelConfig:
    """Complete configuration for static model."""

    batch_size: int
    video_config: VideoConfig
    context_config: ContextConfig

    # Model architecture
    num_layers: int = 48
    dim: int = 5120
    num_heads: int = 40
    ffn_dim: int = 13824

    # Precision
    dtype: torch.dtype = torch.float16

    @property
    def sequence_length(self) -> int:
        return self.video_config.sequence_length

    @property
    def context_length(self) -> int:
        return self.context_config.total_context_length

    @property
    def parameters(self) -> int:
        """Estimate parameter count."""
        # Rough estimate
        params_per_block = (
            self.dim * self.dim * 3  # QKV self-attention
            + self.dim * self.dim  # Out self-attention
            + self.dim * self.dim * 3  # QKV cross-attention
            + self.dim * self.dim  # Out cross-attention
            + self.dim * self.ffn_dim * 2  # FFN
            + self.dim * 7  # Norms and modulation
        )
        return params_per_block * self.num_layers

    @property
    def activation_memory_mb(self) -> float:
        """Estimate activation memory in MB."""
        bytes_per_element = 2 if self.dtype == torch.float16 else 4

        # Main activations
        seq_activations = self.batch_size * self.sequence_length * self.dim
        ctx_activations = self.batch_size * self.context_length * self.dim

        # Attention matrices (approximate)
        attention_memory = (
            self.batch_size * self.num_heads * self.sequence_length * self.sequence_length
        )

        total_elements = (seq_activations + ctx_activations + attention_memory) * self.num_layers
        return (total_elements * bytes_per_element) / (1024 * 1024)

    @property
    def parameter_memory_mb(self) -> float:
        """Estimate parameter memory in MB."""
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        return (self.parameters * bytes_per_element) / (1024 * 1024)

    @property
    def total_memory_mb(self) -> float:
        """Estimate total memory requirement."""
        return self.activation_memory_mb + self.parameter_memory_mb

    def to_dict(self) -> Dict:
        """Convert to dictionary for initialization."""
        return {
            "batch_size": self.batch_size,
            "seq_len": self.sequence_length,
            "context_len": self.context_length,
            "num_layers": self.num_layers,
            "dim": self.dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": 0.0,
            "dtype": self.dtype,
        }

    def __str__(self) -> str:
        return (
            f"Static Model Configuration:\n"
            f"  Batch: {self.batch_size}\n"
            f"  Sequence: {self.sequence_length} patches\n"
            f"  Context: {self.context_length} tokens\n"
            f"  Memory: {self.total_memory_mb:.1f} MB\n"
            f"    - Parameters: {self.parameter_memory_mb:.1f} MB\n"
            f"    - Activations: {self.activation_memory_mb:.1f} MB"
        )


class ConfigCalculator:
    """Calculate optimal configurations for different scenarios."""

    # Common video resolutions and their patch counts
    COMMON_CONFIGS = {
        # Format: (frames, height, width) -> name
        (1, 1024, 1024): "1024² image",
        (1, 512, 512): "512² image",
        (1, 256, 256): "256² image",
        (16, 256, 256): "256² 16-frame",
        (16, 512, 512): "512² 16-frame",
        (24, 256, 256): "256² 24-frame (1s)",
        (48, 256, 256): "256² 48-frame (2s)",
        (16, 320, 512): "SD 16-frame",
        (16, 512, 896): "SD-wide 16-frame",
        (16, 720, 1280): "720p 16-frame",
        (16, 1080, 1920): "1080p 16-frame",
    }

    @staticmethod
    def calculate_video_config(
        frames: int,
        height: int,
        width: int,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
    ) -> VideoConfig:
        """Calculate video configuration."""

        # Ensure dimensions are divisible by patch size
        if frames % patch_size[0] != 0:
            frames = ((frames + patch_size[0] - 1) // patch_size[0]) * patch_size[0]
            print(f"Warning: Adjusted frames to {frames} to be divisible by patch size")

        if height % patch_size[1] != 0:
            height = ((height + patch_size[1] - 1) // patch_size[1]) * patch_size[1]
            print(f"Warning: Adjusted height to {height} to be divisible by patch size")

        if width % patch_size[2] != 0:
            width = ((width + patch_size[2] - 1) // patch_size[2]) * patch_size[2]
            print(f"Warning: Adjusted width to {width} to be divisible by patch size")

        return VideoConfig(frames, height, width, patch_size)

    @staticmethod
    def calculate_context_config(
        text_tokens: int = 512,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        image_patch_size: int = 14,  # CLIP ViT patch size
    ) -> ContextConfig:
        """Calculate context configuration."""

        if image_height is not None and image_width is not None:
            # Calculate image patches
            image_patches = (image_height // image_patch_size) * (image_width // image_patch_size)
            return ContextConfig(text_tokens, image_patches)

        return ContextConfig(text_tokens, None)

    @staticmethod
    def get_optimal_config(
        max_frames: int,
        max_height: int,
        max_width: int,
        batch_size: int = 1,
        include_image_conditioning: bool = True,
        target_memory_mb: Optional[float] = None,
    ) -> StaticModelConfig:
        """
        Get optimal static configuration for given constraints.

        Args:
            max_frames: Maximum number of video frames
            max_height: Maximum frame height
            max_width: Maximum frame width
            batch_size: Batch size for inference
            include_image_conditioning: Whether to include image conditioning
            target_memory_mb: Target memory budget in MB

        Returns:
            Optimal StaticModelConfig
        """

        # Calculate video config
        video_config = ConfigCalculator.calculate_video_config(max_frames, max_height, max_width)

        # Calculate context config
        if include_image_conditioning:
            # Assume conditioning image is same resolution as video
            context_config = ConfigCalculator.calculate_context_config(
                text_tokens=512,
                image_height=max_height,
                image_width=max_width,
            )
        else:
            context_config = ConfigCalculator.calculate_context_config(text_tokens=512)

        # Create config
        config = StaticModelConfig(
            batch_size=batch_size,
            video_config=video_config,
            context_config=context_config,
        )

        # Check memory constraint
        if target_memory_mb is not None:
            if config.total_memory_mb > target_memory_mb:
                print(
                    f"Warning: Configuration requires {config.total_memory_mb:.1f} MB, "
                    f"exceeds target {target_memory_mb:.1f} MB"
                )

        return config

    @staticmethod
    def get_common_configs() -> List[StaticModelConfig]:
        """Get configurations for common video formats."""
        configs = []

        for (frames, height, width), name in ConfigCalculator.COMMON_CONFIGS.items():
            video_config = ConfigCalculator.calculate_video_config(frames, height, width)
            context_config = ConfigCalculator.calculate_context_config(
                text_tokens=512,
                image_height=height if frames == 1 else None,
                image_width=width if frames == 1 else None,
            )

            config = StaticModelConfig(
                batch_size=1,
                video_config=video_config,
                context_config=context_config,
            )

            configs.append((name, config))

        return configs

    @staticmethod
    def recommend_batch_size(
        video_config: VideoConfig,
        context_config: ContextConfig,
        memory_budget_mb: float,
        dtype: torch.dtype = torch.float16,
    ) -> int:
        """
        Recommend maximum batch size for given memory budget.

        Args:
            video_config: Video configuration
            context_config: Context configuration
            memory_budget_mb: Available GPU memory in MB
            dtype: Model precision

        Returns:
            Recommended batch size
        """

        # Start with batch size 1
        batch_size = 1

        while True:
            config = StaticModelConfig(
                batch_size=batch_size,
                video_config=video_config,
                context_config=context_config,
                dtype=dtype,
            )

            if config.total_memory_mb > memory_budget_mb:
                return max(1, batch_size - 1)

            batch_size += 1

            if batch_size > 32:  # Reasonable upper limit
                return 32


def print_config_table():
    """Print a table of common configurations."""
    print("\n" + "=" * 80)
    print("Common WAN 2.2 Static Configurations")
    print("=" * 80)
    print(
        f"{'Format':<25} {'Seq Len':<10} {'Context':<10} {'Memory (MB)':<15} {'Batch=4 (MB)':<15}"
    )
    print("-" * 80)

    for name, config in ConfigCalculator.get_common_configs():
        memory_b1 = config.total_memory_mb

        # Calculate for batch size 4
        config_b4 = StaticModelConfig(
            batch_size=4,
            video_config=config.video_config,
            context_config=config.context_config,
        )
        memory_b4 = config_b4.total_memory_mb

        print(
            f"{name:<25} {config.sequence_length:<10} {config.context_length:<10} "
            f"{memory_b1:<15.1f} {memory_b4:<15.1f}"
        )

    print("=" * 80)


def calculate_for_resolution(
    frames: int,
    height: int,
    width: int,
    batch_size: int = 1,
    print_details: bool = True,
) -> StaticModelConfig:
    """
    Calculate configuration for specific resolution.

    Args:
        frames: Number of frames
        height: Frame height
        width: Frame width
        batch_size: Batch size
        print_details: Whether to print configuration details

    Returns:
        StaticModelConfig for the given resolution
    """

    config = ConfigCalculator.get_optimal_config(
        frames,
        height,
        width,
        batch_size=batch_size,
        include_image_conditioning=(frames == 1),  # Image conditioning for single frames
    )

    if print_details:
        print("\n" + "=" * 60)
        print(f"Configuration for {frames}x{height}x{width} (batch={batch_size})")
        print("=" * 60)
        print(config.video_config)
        print(config.context_config)
        print("-" * 60)
        print(config)
        print("=" * 60)

        # Show initialization code
        print("\nInitialization code:")
        print("```python")
        print("from wan_attention_static import StaticWANInferenceModel")
        print()
        print("model = StaticWANInferenceModel(")
        for key, value in config.to_dict().items():
            if key != "dtype":
                print(f"    {key}={value},")
        print("    device='cuda',")
        print(f"    dtype={config.dtype},")
        print(")")
        print("```")

    return config


def recommend_config_for_gpu(gpu_memory_gb: float = 24.0):
    """
    Recommend configurations for a specific GPU memory size.

    Args:
        gpu_memory_gb: GPU memory in GB

    Returns:
        List of recommended configurations
    """

    memory_budget_mb = gpu_memory_gb * 1024 * 0.8  # Use 80% of available memory

    print(f"\nRecommended configurations for {gpu_memory_gb}GB GPU:")
    print("=" * 60)

    recommendations = []

    # Test common configurations
    test_configs = [
        (16, 256, 256, "256² 16-frame video"),
        (16, 512, 512, "512² 16-frame video"),
        (24, 256, 256, "256² 1-second video"),
        (1, 1024, 1024, "1024² image generation"),
    ]

    for frames, height, width, description in test_configs:
        video_config = ConfigCalculator.calculate_video_config(frames, height, width)
        context_config = ConfigCalculator.calculate_context_config(
            text_tokens=512,
            image_height=height if frames == 1 else None,
            image_width=width if frames == 1 else None,
        )

        max_batch = ConfigCalculator.recommend_batch_size(
            video_config, context_config, memory_budget_mb
        )

        config = StaticModelConfig(
            batch_size=max_batch,
            video_config=video_config,
            context_config=context_config,
        )

        if config.total_memory_mb <= memory_budget_mb:
            print(f"✓ {description}: batch_size={max_batch}, memory={config.total_memory_mb:.1f}MB")
            recommendations.append((description, config))
        else:
            print(f"✗ {description}: Too large even with batch_size=1")

    return recommendations


if __name__ == "__main__":
    # Print common configurations table
    print_config_table()

    # Example: Calculate for specific resolution
    print("\n" + "=" * 80)
    print("Example Calculations")
    print("=" * 80)

    # 512x512 16-frame video
    config_512 = calculate_for_resolution(16, 512, 512, batch_size=1)

    # 1024x1024 image
    config_1024 = calculate_for_resolution(1, 1024, 1024, batch_size=4)

    # Recommendations for different GPUs
    print("\n" + "=" * 80)
    print("GPU Memory Recommendations")
    print("=" * 80)

    for gpu_memory in [16, 24, 40, 48, 80]:  # A100-40GB, 3090/4090, A100-80GB
        recommend_config_for_gpu(gpu_memory)
