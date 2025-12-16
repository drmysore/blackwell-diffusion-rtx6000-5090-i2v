"""
WAN 2.2 Deployment Configuration Wizard
========================================

Interactive utilities and guides for deploying WAN 2.2 in different scenarios.
Helps users choose the right static configuration for their use case.
"""

from typing import List, Optional
import torch
from dataclasses import dataclass
from enum import Enum

from wan_config_calculator import (
    ConfigCalculator,
    StaticModelConfig,
    VideoConfig,
    ContextConfig,
)


class DeploymentTarget(Enum):
    """Common deployment targets with their characteristics."""

    # Consumer GPUs
    RTX_3090 = ("RTX 3090", 24, 936, "Ampere")
    RTX_4090 = ("RTX 4090", 24, 1321, "Ada Lovelace")
    RTX_5090 = ("RTX 5090", 32, 1792, "Blackwell")  # Estimated
    RTX_6000_ADA = ("RTX 6000 Ada", 48, 1457, "Ada Lovelace")

    # Datacenter GPUs
    A100_40GB = ("A100 40GB", 40, 1248, "Ampere")
    A100_80GB = ("A100 80GB", 80, 1248, "Ampere")
    H100_80GB = ("H100 80GB", 80, 1979, "Hopper")
    B200 = ("B200", 192, 2400, "Blackwell")  # Estimated

    # Edge/Mobile (future)
    JETSON_ORIN = ("Jetson AGX Orin", 32, 275, "Ampere")

    def __init__(self, name: str, memory_gb: float, fp16_tflops: float, arch: str):
        self.display_name = name
        self.memory_gb = memory_gb
        self.fp16_tflops = fp16_tflops
        self.architecture = arch

    @property
    def memory_mb(self) -> float:
        """Available memory in MB (80% of total for safety)."""
        return self.memory_gb * 1024 * 0.8

    @property
    def supports_fp8(self) -> bool:
        """Whether this GPU supports FP8."""
        return self.architecture in ["Hopper", "Blackwell"]

    @property
    def supports_nvfp4(self) -> bool:
        """Whether this GPU supports NVFP4."""
        return self.architecture == "Blackwell"


class UseCase(Enum):
    """Common use cases with their requirements."""

    # Image generation
    IMAGE_512 = ("512² Image Generation", 1, 512, 512, False)
    IMAGE_1024 = ("1024² Image Generation", 1, 1024, 1024, False)
    IMAGE_2K = ("2K Image Generation", 1, 2048, 2048, False)

    # Short video clips
    VIDEO_256_1S = ("256² 1-second clips", 24, 256, 256, False)
    VIDEO_512_1S = ("512² 1-second clips", 24, 512, 512, False)
    VIDEO_256_2S = ("256² 2-second clips", 48, 256, 256, False)

    # Standard video
    VIDEO_SD = ("SD Video (480p)", 24, 480, 640, False)
    VIDEO_HD = ("HD Video (720p)", 24, 720, 1280, False)
    VIDEO_FHD = ("Full HD (1080p)", 24, 1080, 1920, False)

    # Image-to-video
    I2V_256 = ("Image-to-Video 256²", 16, 256, 256, True)
    I2V_512 = ("Image-to-Video 512²", 16, 512, 512, True)

    def __init__(self, name: str, frames: int, height: int, width: int, needs_image: bool):
        self.display_name = name
        self.frames = frames
        self.height = height
        self.width = width
        self.needs_image_conditioning = needs_image


@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""

    use_case: UseCase
    target: DeploymentTarget
    batch_size: int
    model_config: StaticModelConfig
    optimization_level: str

    @property
    def fits_in_memory(self) -> bool:
        """Check if configuration fits in target GPU memory."""
        return self.model_config.total_memory_mb <= self.target.memory_mb

    @property
    def inference_time_ms(self) -> float:
        """Estimate inference time in milliseconds."""
        # Rough estimation based on architecture and size
        base_time = 500  # Base time for 48 blocks in ms

        # Adjust for sequence length
        seq_factor = self.model_config.sequence_length / 1000

        # Adjust for GPU performance
        perf_factor = 1000 / self.target.fp16_tflops  # Normalize to 1 TFLOP

        # Optimization multipliers
        opt_multipliers = {
            "none": 1.0,
            "torch.compile": 0.7,
            "tensorrt": 0.5,
            "cudagraph": 0.6,
            "nvfp4": 0.25,  # Blackwell only
        }

        opt_mult = opt_multipliers.get(self.optimization_level, 1.0)

        # Special case for NVFP4
        if self.optimization_level == "nvfp4" and not self.target.supports_nvfp4:
            opt_mult = opt_multipliers["tensorrt"]  # Fall back to TensorRT

        return base_time * seq_factor * perf_factor * opt_mult * self.batch_size

    def __str__(self) -> str:
        status = "✓" if self.fits_in_memory else "✗"
        return (
            f"{status} {self.use_case.display_name} on {self.target.display_name}\n"
            f"  Batch size: {self.batch_size}\n"
            f"  Memory: {self.model_config.total_memory_mb:.1f}/{self.target.memory_mb:.1f} MB\n"
            f"  Est. latency: {self.inference_time_ms:.1f} ms\n"
            f"  Optimization: {self.optimization_level}"
        )


class DeploymentWizard:
    """Interactive wizard for deployment configuration."""

    @staticmethod
    def recommend_config(
        use_case: UseCase,
        target: DeploymentTarget,
        max_batch_size: Optional[int] = None,
    ) -> DeploymentConfig:
        """
        Recommend optimal configuration for use case and target.

        Args:
            use_case: Desired use case
            target: Target GPU
            max_batch_size: Maximum batch size to consider

        Returns:
            Recommended deployment configuration
        """

        # Calculate base configuration
        video_config = VideoConfig(
            use_case.frames,
            use_case.height,
            use_case.width,
        )

        context_config = ContextConfig(
            text_tokens=512,
            image_patches=(
                (use_case.height // 14) * (use_case.width // 14)
                if use_case.needs_image_conditioning
                else None
            ),
        )

        # Determine maximum batch size
        if max_batch_size is None:
            max_batch_size = ConfigCalculator.recommend_batch_size(
                video_config,
                context_config,
                target.memory_mb,
            )

        # Create model configuration
        model_config = StaticModelConfig(
            batch_size=max_batch_size,
            video_config=video_config,
            context_config=context_config,
            dtype=torch.float16,
        )

        # Determine optimization level
        if target.supports_nvfp4:
            optimization = "nvfp4"
        elif target.supports_fp8:
            optimization = "tensorrt"  # TensorRT with FP8
        elif target.fp16_tflops > 1000:
            optimization = "cudagraph"
        else:
            optimization = "torch.compile"

        return DeploymentConfig(
            use_case=use_case,
            target=target,
            batch_size=max_batch_size,
            model_config=model_config,
            optimization_level=optimization,
        )

    @staticmethod
    def analyze_all_combinations() -> List[DeploymentConfig]:
        """Analyze all use case and target combinations."""
        results = []

        for use_case in UseCase:
            for target in DeploymentTarget:
                config = DeploymentWizard.recommend_config(use_case, target)
                results.append(config)

        return results

    @staticmethod
    def find_best_target(
        use_case: UseCase,
        targets: Optional[List[DeploymentTarget]] = None,
    ) -> DeploymentConfig:
        """
        Find best target GPU for a use case.

        Args:
            use_case: Desired use case
            targets: List of available targets (default: all)

        Returns:
            Best configuration
        """

        if targets is None:
            targets = list(DeploymentTarget)

        configs = []
        for target in targets:
            config = DeploymentWizard.recommend_config(use_case, target)
            if config.fits_in_memory:
                configs.append(config)

        if not configs:
            raise ValueError(f"No target can handle {use_case.display_name}")

        # Sort by inference time
        configs.sort(key=lambda c: c.inference_time_ms)

        return configs[0]

    @staticmethod
    def print_compatibility_matrix():
        """Print compatibility matrix for all combinations."""

        print("\n" + "=" * 100)
        print("WAN 2.2 Deployment Compatibility Matrix")
        print("=" * 100)

        # Header
        print(f"{'Use Case':<30}", end="")
        for target in DeploymentTarget:
            if len(target.display_name) > 12:
                name = target.display_name[:12]
            else:
                name = target.display_name
            print(f"{name:>12}", end="")
        print()
        print("-" * 100)

        # Matrix
        for use_case in UseCase:
            print(f"{use_case.display_name:<30}", end="")

            for target in DeploymentTarget:
                config = DeploymentWizard.recommend_config(use_case, target, max_batch_size=1)

                if config.fits_in_memory:
                    # Show max batch size that fits
                    max_batch = ConfigCalculator.recommend_batch_size(
                        config.model_config.video_config,
                        config.model_config.context_config,
                        target.memory_mb,
                    )
                    print(f"{'B=' + str(max_batch):>12}", end="")
                else:
                    print(f"{'✗':>12}", end="")

            print()

        print("=" * 100)
        print("B=N means batch size N is supported")


def deployment_wizard_cli():
    """Interactive CLI for deployment configuration."""

    print("\n" + "=" * 80)
    print("WAN 2.2 Deployment Configuration Wizard")
    print("=" * 80)

    # Select use case
    print("\nAvailable use cases:")
    for i, use_case in enumerate(UseCase, 1):
        print(f"  {i}. {use_case.display_name}")

    while True:
        try:
            choice = int(input("\nSelect use case (number): "))
            use_case = list(UseCase)[choice - 1]
            break
        except (ValueError, IndexError):
            print("Invalid choice, try again")

    # Select target
    print("\nAvailable targets:")
    for i, target in enumerate(DeploymentTarget, 1):
        print(f"  {i}. {target.display_name} ({target.memory_gb}GB)")

    while True:
        try:
            choice = int(input("\nSelect target GPU (number): "))
            target = list(DeploymentTarget)[choice - 1]
            break
        except (ValueError, IndexError):
            print("Invalid choice, try again")

    # Get recommendation
    print("\n" + "-" * 80)
    print("Recommended Configuration:")
    print("-" * 80)

    config = DeploymentWizard.recommend_config(use_case, target)
    print(config)

    if config.fits_in_memory:
        print("\n✅ This configuration will work!")

        # Show initialization code
        print("\nInitialization code:")
        print("```python")
        print("from wan_static_factory import StaticModelFactory")
        print()
        print("factory = StaticModelFactory(")
        print("    device='cuda',")
        print("    dtype=torch.float16,")
        print(")")
        print()
        print("model = factory.create_from_video_dims(")
        print(f"    batch_size={config.batch_size},")
        print(f"    frames={use_case.frames},")
        print(f"    height={use_case.height},")
        print(f"    width={use_case.width},")
        print(f"    image_conditioning={use_case.needs_image_conditioning},")
        print(")")

        if config.optimization_level == "torch.compile":
            print()
            print("# Compile for better performance")
            print("import torch")
            print("model = torch.compile(model, mode='reduce-overhead')")
        elif config.optimization_level == "cudagraph":
            print()
            print("# Use CUDAGraph for minimum latency")
            print("# See wan_attention_static.create_cudagraph_model()")

        print("```")
    else:
        print("\n❌ This configuration exceeds available memory!")
        print("\nSuggestions:")
        print("1. Reduce batch size")
        print("2. Use a smaller resolution")
        print("3. Upgrade to a GPU with more memory")

        # Find alternative
        try:
            better = DeploymentWizard.find_best_target(use_case)
            print(f"\nBest alternative: {better.target.display_name}")
        except ValueError:
            print("\nNo single GPU can handle this configuration")


def print_quick_reference():
    """Print quick reference guide."""

    print("\n" + "=" * 80)
    print("WAN 2.2 Static Configuration Quick Reference")
    print("=" * 80)

    print("\nCOMMON CONFIGURATIONS:")
    print("-" * 80)

    common = [
        ("256² 16-frame video", 16, 256, 256, 3136, 512, "~5GB"),
        ("512² 16-frame video", 16, 512, 512, 12544, 512, "~12GB"),
        ("256² 1-second (24fps)", 24, 256, 256, 4704, 512, "~7GB"),
        ("512² image generation", 1, 512, 512, 1024, 1548, "~4GB"),
        ("1024² image generation", 1, 1024, 1024, 4096, 3380, "~8GB"),
    ]

    print(
        f"{'Description':<30} {'Frames':<8} {'Size':<12} {'Seq Len':<10} {'Context':<10} {'Memory':<10}"
    )
    print("-" * 80)

    for desc, frames, h, w, seq, ctx, mem in common:
        print(f"{desc:<30} {frames:<8} {h}x{w:<9} {seq:<10} {ctx:<10} {mem:<10}")

    print("\nGPU RECOMMENDATIONS:")
    print("-" * 80)

    recommendations = [
        ("RTX 3090/4090 (24GB)", ["256² videos", "512² images", "Batch 1-2"]),
        ("RTX 5090 (32GB)", ["512² videos", "1024² images", "Batch 2-4"]),
        ("RTX 6000 (48GB)", ["All resolutions", "Batch 4-8"]),
        ("A100 40GB", ["512² videos", "Batch 2-4", "Multi-GPU recommended"]),
        ("A100/H100 80GB", ["1080p preview", "Batch 8-16", "Production ready"]),
    ]

    for gpu, capabilities in recommendations:
        print(f"\n{gpu}:")
        for cap in capabilities:
            print(f"  • {cap}")

    print("\nOPTIMIZATION LEVELS:")
    print("-" * 80)
    print("1. None: Baseline PyTorch")
    print("2. torch.compile: ~30% faster")
    print("3. CUDAGraph: ~40% faster, fixed shapes")
    print("4. TensorRT: ~50% faster, requires export")
    print("5. NVFP4 (Blackwell): ~75% faster, 4x less memory")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--matrix":
        DeploymentWizard.print_compatibility_matrix()
    elif len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print_quick_reference()
    else:
        # Run interactive wizard
        deployment_wizard_cli()

        # Show quick reference
        print("\n" + "=" * 80)
        response = input("\nShow quick reference? (y/n): ")
        if response.lower() == "y":
            print_quick_reference()
