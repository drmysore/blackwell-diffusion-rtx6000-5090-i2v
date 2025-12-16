import argparse
import hashlib
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video, load_image
from PIL import Image
from torchao.prototype.mx_formats import NVFP4InferenceConfig
from torchao.quantization import float8_dynamic_activation_float8_weight, quantize_

# from fxy.idoru.media import resize_image_to_fit_aspect
from wan_transformer import WanTransformer3DModel as WanTransformer


# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def resize_image_to_fit_aspect(image: Image.Image) -> Image.Image:
    """Resizes an image to fit within the model's constraints, preserving aspect ratio as much as possible."""
    MAX_DIM = 720
    MIN_DIM = 480
    SQUARE_DIM = 640
    MULTIPLE_OF = 16

    width, height = image.size

    # Handle square case
    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)  # type: ignore

    aspect_ratio = width / height

    MAX_ASPECT_RATIO = MAX_DIM / MIN_DIM
    MIN_ASPECT_RATIO = MIN_DIM / MAX_DIM

    image_to_resize = image

    if aspect_ratio > MAX_ASPECT_RATIO:
        target_w, target_h = MAX_DIM, MIN_DIM

        crop_width = round(height * MAX_ASPECT_RATIO)
        left = (width - crop_width) // 2

        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < MIN_ASPECT_RATIO:
        target_w, target_h = MIN_DIM, MAX_DIM

        crop_height = round(width / MIN_ASPECT_RATIO)
        top = (height - crop_height) // 2

        image_to_resize = image.crop((0, top, width, top + crop_height))
    elif width > height:  # Landscape
        target_w = MAX_DIM
        target_h = round(target_w / aspect_ratio)
    else:  # Portrait
        target_h = MAX_DIM
        target_w = round(target_h * aspect_ratio)

    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF

    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))

    return image_to_resize.resize((final_w, final_h), Image.LANCZOS)  # type: ignore


# Model constants
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
TRANSFORMER_ID = "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers"
DEFAULT_LORA_WEIGHT = "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16"
DEFAULT_LORA_REPO = "Kijai/WanVideo_comfy"


@dataclass
class WeightMetadata:
    """Metadata for a single weight tensor."""

    name: str
    shape: list[int]
    dtype: str
    size_bytes: int
    checksum: str  # First 8 chars of SHA256

    @classmethod
    def from_tensor(cls, name: str, tensor: torch.Tensor) -> "WeightMetadata":
        """Create metadata from a tensor."""
        # Move to CPU and make contiguous
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor = tensor.contiguous()

        # For large tensors, use statistical checksum
        # For small tensors, hash the full data
        size_threshold = 1024 * 1024  # 1MB

        if tensor.numel() * tensor.element_size() > size_threshold:
            # Use statistical approach for large tensors
            # Convert to float for statistics (handles all dtypes)
            float_tensor = tensor.float()

            # Compute various statistics
            stats_dict = {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "mean": float(float_tensor.mean().item()),
                "std": float(float_tensor.std().item()) if tensor.numel() > 1 else 0.0,
                "min": float(float_tensor.min().item()),
                "max": float(float_tensor.max().item()),
                "norm": float(float_tensor.norm().item()),
            }

            # Sample some values deterministically
            if tensor.numel() > 100:
                # Take evenly spaced samples
                indices = torch.linspace(0, tensor.numel() - 1, 100, dtype=torch.long)
                samples = float_tensor.flatten()[indices].tolist()
                stats_dict["samples"] = samples

            checksum_str = json.dumps(stats_dict, sort_keys=True)
            checksum = hashlib.sha256(checksum_str.encode()).hexdigest()[:8]
        else:
            # For small tensors, hash the full data
            if tensor.dtype == torch.bfloat16:
                # Special handling for bfloat16
                data_ptr = tensor.data_ptr()
                size = tensor.numel() * tensor.element_size()
                # Use ctypes to read the raw bytes
                import ctypes

                buffer = (ctypes.c_ubyte * size).from_address(data_ptr)
                tensor_bytes = bytes(buffer)
            else:
                # Use numpy for other dtypes
                tensor_bytes = tensor.numpy().tobytes()

            checksum = hashlib.sha256(tensor_bytes).hexdigest()[:8]

        return cls(
            name=name,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            size_bytes=tensor.element_size() * tensor.numel(),
            checksum=checksum,
        )


@dataclass
class ModelStateMetadata:
    """Complete metadata for a model's state."""

    model_class: str
    config: dict[str, Any]
    weights: list[WeightMetadata]
    total_parameters: int
    total_size_bytes: int
    state_checksum: str

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> "ModelStateMetadata":
        """Extract complete metadata from a model."""
        weights = []
        total_params = 0
        total_size = 0

        # Get model config if available
        config = {}
        if hasattr(model, "config"):
            config = model.config.to_dict() if hasattr(model.config, "to_dict") else {}

        # Extract weight metadata
        state_dict = model.state_dict()
        weight_checksums = []

        for name, param in state_dict.items():
            weight_meta = WeightMetadata.from_tensor(name, param)
            weights.append(weight_meta)
            total_params += param.numel()
            total_size += weight_meta.size_bytes
            weight_checksums.append(f"{name}:{weight_meta.checksum}")

        # Compute overall state checksum
        state_string = f"{model.__class__.__name__}:{json.dumps(config, sort_keys=True)}:{':'.join(sorted(weight_checksums))}"
        state_checksum = hashlib.sha256(state_string.encode()).hexdigest()[:16]

        return cls(
            model_class=f"{model.__class__.__module__}.{model.__class__.__name__}",
            config=config,
            weights=weights,
            total_parameters=total_params,
            total_size_bytes=total_size,
            state_checksum=state_checksum,
        )


@dataclass
class CacheConfig:
    """Configuration for model caching with complete state tracking."""

    model_id: str
    transformer_id: str
    lora_repo: str
    lora_weight_name: str
    lora_scale_1: float
    lora_scale_2: float
    dtype: str
    optimized: bool
    quantization: str | None = None
    transformer_1_state: ModelStateMetadata | None = None
    transformer_2_state: ModelStateMetadata | None = None

    def to_hash(self) -> str:
        """Generate deterministic hash from complete configuration."""
        # Include model states in hash only if they exist
        config_dict = {
            "model_id": self.model_id,
            "transformer_id": self.transformer_id,
            "lora_repo": self.lora_repo,
            "lora_weight_name": self.lora_weight_name,
            "lora_scale_1": self.lora_scale_1,
            "lora_scale_2": self.lora_scale_2,
            "dtype": self.dtype,
            "optimized": self.optimized,
            "quantization": self.quantization,
        }

        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class CacheValidator:
    """Validates cache entries against expected configurations."""

    @staticmethod
    def validate_cache_entry(
        cache_path: Path, expected_config: CacheConfig
    ) -> tuple[bool, list[str]]:
        """Validate a cache entry and return (is_valid, reasons)."""
        reasons = []

        # Check if cache directory exists
        if not cache_path.exists():
            reasons.append(f"Cache directory does not exist: {cache_path}")
            return False, reasons

        # Check metadata file exists
        metadata_path = cache_path / "metadata.json"
        if not metadata_path.exists():
            reasons.append("Missing metadata.json")
            return False, reasons

        # Load and compare metadata
        try:
            with open(metadata_path) as f:
                cached_metadata = json.load(f)
        except Exception as e:
            reasons.append(f"Failed to load metadata: {e}")
            return False, reasons

        # Check basic configuration (excluding transformer states)
        for key in [
            "model_id",
            "transformer_id",
            "lora_repo",
            "lora_weight_name",
            "lora_scale_1",
            "lora_scale_2",
            "dtype",
            "optimized",
        ]:
            expected_val = getattr(expected_config, key)
            cached_val = cached_metadata.get(key)
            if expected_val != cached_val:
                reasons.append(
                    f"{key} mismatch: expected {expected_val}, got {cached_val}"
                )

        # Check model files exist
        for model_dir in ["transformer", "transformer_2"]:
            model_path = cache_path / model_dir
            if not model_path.exists():
                reasons.append(f"Missing {model_dir} directory")
                continue

            # Check for model files
            has_safetensors = any(model_path.glob("*.safetensors"))
            has_bin = any(model_path.glob("*.bin"))
            if not (has_safetensors or has_bin):
                reasons.append(f"No model files in {model_dir}")

        return len(reasons) == 0, reasons


class WanPipelineLoader:
    """Manages loading and caching of WAN pipelines with detailed validation."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "wan22_models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache directory: {self.cache_dir}")
        self.validator = CacheValidator()

    def get_cache_path(self, config: CacheConfig) -> Path:
        """Get cache path for a specific configuration."""
        cache_hash = config.to_hash()
        return self.cache_dir / cache_hash

    def save_cache_metadata(self, cache_path: Path, config: CacheConfig):
        """Save complete metadata for cache entry."""
        metadata = asdict(config)

        # Convert ModelStateMetadata to dict
        if config.transformer_1_state:
            metadata["transformer_1_state"] = asdict(config.transformer_1_state)
        if config.transformer_2_state:
            metadata["transformer_2_state"] = asdict(config.transformer_2_state)

        metadata_path = cache_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Also save a human-readable summary
        summary_path = cache_path / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("Cache Entry Summary\n")
            f.write("==================\n\n")
            f.write(f"Hash: {cache_path.name}\n")
            f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Configuration:\n")
            f.write(f"  Model ID: {config.model_id}\n")
            f.write(f"  Transformer ID: {config.transformer_id}\n")
            f.write(f"  LoRA: {config.lora_weight_name}\n")
            f.write(
                f"  LoRA Scales: T1={config.lora_scale_1}, T2={config.lora_scale_2}\n"
            )
            f.write(f"  Optimized: {config.optimized}\n")
            f.write(f"  Quantization: {config.quantization or 'None'}\n\n")

            if config.transformer_1_state:
                f.write("Transformer 1:\n")
                f.write(f"  Class: {config.transformer_1_state.model_class}\n")
                f.write(
                    f"  Parameters: {config.transformer_1_state.total_parameters:,}\n"
                )
                f.write(
                    f"  Size: {config.transformer_1_state.total_size_bytes / 1e9:.2f} GB\n"
                )
                f.write(f"  Checksum: {config.transformer_1_state.state_checksum}\n\n")

            if config.transformer_2_state:
                f.write("Transformer 2:\n")
                f.write(f"  Class: {config.transformer_2_state.model_class}\n")
                f.write(
                    f"  Parameters: {config.transformer_2_state.total_parameters:,}\n"
                )
                f.write(
                    f"  Size: {config.transformer_2_state.total_size_bytes / 1e9:.2f} GB\n"
                )
                f.write(f"  Checksum: {config.transformer_2_state.state_checksum}\n")

    def try_load_from_cache(
        self, cache_path: Path, config: CacheConfig, model_class: type
    ) -> tuple[Any | None, Any | None, bool]:
        """Try to load transformers from cache with validation."""
        logger.info(f"Checking cache at: {cache_path.name}")

        # Validate cache
        is_valid, reasons = self.validator.validate_cache_entry(cache_path, config)

        if not is_valid:
            logger.info("Cache miss - validation failed:")
            for reason in reasons:
                logger.info(f"  - {reason}")
            return None, None, False

        # Try to load models
        try:
            t1_path = cache_path / "transformer"
            t2_path = cache_path / "transformer_2"

            logger.info(f"Cache hit - loading models from {cache_path.name}")

            transformer_1 = model_class.from_pretrained(
                t1_path, torch_dtype=torch.bfloat16, use_safetensors=True
            )
            transformer_2 = model_class.from_pretrained(
                t2_path, torch_dtype=torch.bfloat16, use_safetensors=True
            )

            logger.info("Successfully loaded cached models")
            return transformer_1, transformer_2, True

        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None, None, False

    def load_base_pipeline(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> WanImageToVideoPipeline:
        """Load base WAN pipeline."""
        logger.info("Loading base WAN pipeline...")

        pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            transformer=WanTransformer3DModel.from_pretrained(
                TRANSFORMER_ID,
                subfolder="transformer",
                torch_dtype=dtype,
                use_safetensors=True,
            ),
            transformer_2=WanTransformer3DModel.from_pretrained(
                TRANSFORMER_ID,
                subfolder="transformer_2",
                torch_dtype=dtype,
                use_safetensors=True,
            ),
            torch_dtype=dtype,
            use_safetensors=True,
        )

        logger.info("Base pipeline loaded successfully")
        return pipe

    def apply_lora_fusion(
        self,
        pipe: WanImageToVideoPipeline,
        lora_repo: str,
        lora_weight_name: str,
        lora_scale_1: float,
        lora_scale_2: float,
        cache_config: CacheConfig,
        use_cache: bool = True,
    ) -> WanImageToVideoPipeline:
        """Apply and fuse LoRA weights to pipeline."""
        # Create a config without transformer states for cache lookup
        lookup_config = CacheConfig(
            model_id=cache_config.model_id,
            transformer_id=cache_config.transformer_id,
            lora_repo=lora_repo,
            lora_weight_name=lora_weight_name,
            lora_scale_1=lora_scale_1,
            lora_scale_2=lora_scale_2,
            dtype=cache_config.dtype,
            optimized=False,
            quantization=cache_config.quantization,
        )

        if use_cache:
            cache_path = self.get_cache_path(lookup_config)
            logger.debug(f"Looking for LoRA cache at: {cache_path}")

            if cache_path.exists():
                t1, t2, cache_hit = self.try_load_from_cache(
                    cache_path, lookup_config, WanTransformer3DModel
                )

                if cache_hit:
                    pipe.transformer = t1
                    pipe.transformer_2 = t2
                    return pipe

        # Apply LoRA weights
        logger.info(f"Applying LoRA weights: {lora_weight_name}")

        # Load LoRA for transformer 1
        pipe.load_lora_weights(
            lora_repo,
            weight_name=f"Lightx2v/{lora_weight_name}.safetensors",
            adapter_name="lightx2v",
            use_safetensors=True,
        )

        # Load LoRA for transformer 2
        pipe.load_lora_weights(
            lora_repo,
            weight_name=f"Lightx2v/{lora_weight_name}.safetensors",
            adapter_name="lightx2v_2",
            use_safetensors=True,
            load_into_transformer_2=True,
        )

        # Set and fuse adapters
        pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1.0, 1.0])

        logger.info(f"Fusing LoRA weights - T1: {lora_scale_1}, T2: {lora_scale_2}")
        pipe.fuse_lora(
            adapter_names=["lightx2v"],
            lora_scale=lora_scale_1,
            components=["transformer"],
        )
        pipe.fuse_lora(
            adapter_names=["lightx2v_2"],
            lora_scale=lora_scale_2,
            components=["transformer_2"],
        )
        pipe.unload_lora_weights()

        # Cache the fused models
        if use_cache:
            cache_path = self.get_cache_path(lookup_config)
            cache_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Caching LoRA-fused models to: {cache_path.name}")

            pipe.transformer.save_pretrained(
                cache_path / "transformer", safe_serialization=True
            )
            pipe.transformer_2.save_pretrained(
                cache_path / "transformer_2", safe_serialization=True
            )

            # Now compute states for metadata
            lookup_config.transformer_1_state = ModelStateMetadata.from_model(
                pipe.transformer
            )
            lookup_config.transformer_2_state = ModelStateMetadata.from_model(
                pipe.transformer_2
            )

            self.save_cache_metadata(cache_path, lookup_config)

        return pipe

    def optimize_transformers(
        self,
        pipe: WanImageToVideoPipeline,
        cache_config: CacheConfig,
        use_cache: bool = True,
    ) -> WanImageToVideoPipeline:
        """Convert to optimized transformers."""
        # Create lookup config for optimized version
        lookup_config = CacheConfig(
            model_id=cache_config.model_id,
            transformer_id=cache_config.transformer_id,
            lora_repo=cache_config.lora_repo,
            lora_weight_name=cache_config.lora_weight_name,
            lora_scale_1=cache_config.lora_scale_1,
            lora_scale_2=cache_config.lora_scale_2,
            dtype=cache_config.dtype,
            optimized=True,  # This is the key difference
            quantization=cache_config.quantization,
        )

        if use_cache:
            cache_path = self.get_cache_path(lookup_config)
            logger.debug(f"Looking for optimized cache at: {cache_path}")

            if cache_path.exists():
                t1, t2, cache_hit = self.try_load_from_cache(
                    cache_path, lookup_config, WanTransformer
                )

                if cache_hit:
                    pipe.transformer = t1
                    pipe.transformer_2 = t2
                    return pipe

        logger.info("Converting to optimized transformers...")
        pipe.transformer = WanTransformer.from_pretrained_stock(pipe.transformer)
        pipe.transformer_2 = WanTransformer.from_pretrained_stock(pipe.transformer_2)

        # Cache optimized models
        if use_cache:
            cache_path = self.get_cache_path(lookup_config)
            cache_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Caching optimized models to: {cache_path.name}")

            pipe.transformer.save_pretrained(
                cache_path / "transformer", safe_serialization=True
            )
            pipe.transformer_2.save_pretrained(
                cache_path / "transformer_2", safe_serialization=True
            )

            # Compute states for metadata
            lookup_config.transformer_1_state = ModelStateMetadata.from_model(
                pipe.transformer
            )
            lookup_config.transformer_2_state = ModelStateMetadata.from_model(
                pipe.transformer_2
            )

            self.save_cache_metadata(cache_path, lookup_config)

        return pipe

    def apply_quantization(
        self,
        pipe: WanImageToVideoPipeline,
        quantization: str,
        compile_mode: str | None = None,
    ) -> WanImageToVideoPipeline:
        """Apply quantization to transformers."""
        logger.info(f"Applying {quantization} quantization...")

        if quantization == "float8":
            quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
            quantize_(pipe.transformer_2, float8_dynamic_activation_float8_weight())
        elif quantization == "fp4":
            quantize_(pipe.transformer, NVFP4InferenceConfig())
            quantize_(pipe.transformer_2, NVFP4InferenceConfig())
        else:
            raise ValueError(f"Unknown quantization method: {quantization}")

        if compile_mode:
            logger.info(f"Compiling transformers with mode: {compile_mode}")
            pipe.transformer = torch.compile(pipe.transformer, mode=compile_mode)
            pipe.transformer_2 = torch.compile(pipe.transformer_2, mode=compile_mode)

        return pipe

    def load_optimized_pipeline(
        self,
        lora_repo: str = DEFAULT_LORA_REPO,
        lora_weight_name: str = DEFAULT_LORA_WEIGHT,
        lora_scale_1: float = 3.0,
        lora_scale_2: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        optimize: bool = True,
        quantization: str | None = None,
        compile_mode: str | None = None,
        use_cache: bool = True,
        device: str = "cuda",
    ) -> WanImageToVideoPipeline:
        """Load pipeline with all optimizations."""
        # Create cache config
        cache_config = CacheConfig(
            model_id=MODEL_ID,
            transformer_id=TRANSFORMER_ID,
            lora_repo=lora_repo,
            lora_weight_name=lora_weight_name,
            lora_scale_1=lora_scale_1,
            lora_scale_2=lora_scale_2,
            dtype=str(dtype),
            optimized=False,
            quantization=quantization,
        )

        # Stage 1: Load base pipeline
        pipe = self.load_base_pipeline(dtype)

        # Stage 2: Apply LoRA fusion
        pipe = self.apply_lora_fusion(
            pipe,
            lora_repo,
            lora_weight_name,
            lora_scale_1,
            lora_scale_2,
            cache_config,
            use_cache,
        )

        # Stage 3: Optimize transformers
        if optimize:
            pipe = self.optimize_transformers(pipe, cache_config, use_cache)

        # Stage 4: Apply quantization
        if quantization:
            pipe = self.apply_quantization(pipe, quantization, compile_mode)

        # Stage 5: Move to device
        pipe = pipe.to(device, dtype=dtype)

        # Ensure all components are on correct device/dtype
        for component_name in pipe.components:
            component = getattr(pipe, component_name)
            if hasattr(component, "to") and hasattr(component, "dtype"):
                component.to(device=device, dtype=dtype)

        # Set config
        pipe.transformer.config.image_dim = None

        return pipe

    def clear_cache(self, pattern: str = None):
        """Clear cache, optionally matching a pattern."""
        if pattern:
            logger.info(f"Clearing cache entries matching pattern: {pattern}")
            count = 0
            for cache_dir in self.cache_dir.glob(f"*{pattern}*"):
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir)
                    logger.info(f"Removed: {cache_dir.name}")
                    count += 1
            logger.info(f"Removed {count} cache entries")
        else:
            logger.info("Clearing entire cache")
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")

    def list_cache(self, verbose: bool = False):
        """List all cached models with metadata."""
        logger.info(f"Cache contents in {self.cache_dir}:")

        total_size = 0
        entries = []

        for cache_dir in sorted(self.cache_dir.iterdir()):
            if cache_dir.is_dir():
                metadata_path = cache_dir / "metadata.json"

                # Calculate size
                dir_size = sum(
                    f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
                ) / (1024**3)  # GB
                total_size += dir_size

                entry_info = {
                    "path": cache_dir,
                    "name": cache_dir.name,
                    "size_gb": dir_size,
                }

                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    entry_info["metadata"] = metadata

                entries.append(entry_info)

        # Display entries
        for entry in entries:
            print(f"\n{entry['name']} ({entry['size_gb']:.2f} GB)")

            if "metadata" in entry:
                meta = entry["metadata"]
                print(f"  Model: {meta.get('model_id', 'N/A')}")
                print(f"  LoRA: {meta.get('lora_weight_name', 'N/A')}")
                print(
                    f"  Scales: T1={meta.get('lora_scale_1', 'N/A')}, T2={meta.get('lora_scale_2', 'N/A')}"
                )
                print(f"  Optimized: {meta.get('optimized', False)}")
                print(f"  Quantization: {meta.get('quantization', 'None')}")

                if verbose:
                    if "transformer_1_state" in meta:
                        t1_state = meta["transformer_1_state"]
                        print("  Transformer 1:")
                        print(f"    Class: {t1_state.get('model_class', 'N/A')}")
                        print(
                            f"    Parameters: {t1_state.get('total_parameters', 0):,}"
                        )
                        print(f"    Checksum: {t1_state.get('state_checksum', 'N/A')}")

                    if "transformer_2_state" in meta:
                        t2_state = meta["transformer_2_state"]
                        print("  Transformer 2:")
                        print(f"    Class: {t2_state.get('model_class', 'N/A')}")
                        print(
                            f"    Parameters: {t2_state.get('total_parameters', 0):,}"
                        )
                        print(f"    Checksum: {t2_state.get('state_checksum', 'N/A')}")

        print(f"\nTotal cache size: {total_size:.2f} GB")
        print(f"Total entries: {len(entries)}")


def run_inference(
    pipe: WanImageToVideoPipeline,
    image: Image.Image,
    prompt: str,
    duration: float,
    num_inference_steps: int = 5,
    guidance_scale: float = 0.0,
    seed: int = None,
) -> tuple[torch.Tensor, float]:
    """Run inference and return output frames and timing."""
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None

    num_frames = int(duration * 16) + 1

    logger.info(
        f"Running inference: {image.width}x{image.height}, {duration}s ({num_frames} frames)"
    )

    start_time = time.perf_counter()

    with torch.cuda.amp.autocast(dtype=pipe.dtype):
        output = pipe(
            image=image,
            prompt=prompt,
            height=image.height,
            width=image.width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]

    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    logger.info(f"Inference completed in {elapsed_time:.2f}s")

    return output, elapsed_time


def main():
    parser = argparse.ArgumentParser(description="WAN 2.2 I2V Pipeline Tool")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--images", nargs="+", required=True, help="Input image paths"
    )
    infer_parser.add_argument("--prompt", default="", help="Text prompt")
    infer_parser.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=[3.0, 5.0],
        help="Video durations in seconds",
    )
    infer_parser.add_argument(
        "--output-dir", default="./outputs", help="Output directory"
    )
    infer_parser.add_argument(
        "--lora-weight", default=DEFAULT_LORA_WEIGHT, help="LoRA weight name"
    )
    infer_parser.add_argument(
        "--lora-scale-1", type=float, default=3.0, help="LoRA scale for transformer 1"
    )
    infer_parser.add_argument(
        "--lora-scale-2", type=float, default=1.0, help="LoRA scale for transformer 2"
    )
    infer_parser.add_argument(
        "--no-optimize", action="store_true", help="Disable transformer optimization"
    )
    infer_parser.add_argument(
        "--quantization", choices=["float8", "fp4"], help="Quantization method"
    )
    infer_parser.add_argument(
        "--compile",
        choices=["max-autotune", "reduce-overhead", "default"],
        help="Torch compile mode",
    )
    infer_parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    infer_parser.add_argument("--seed", type=int, help="Random seed")
    infer_parser.add_argument(
        "--steps", type=int, default=5, help="Number of inference steps"
    )
    infer_parser.add_argument(
        "--guidance-scale", type=float, default=0.0, help="Guidance scale"
    )

    # Cache management commands
    cache_parser = subparsers.add_parser("cache", help="Cache management")
    cache_parser.add_argument("action", choices=["list", "clear"], help="Cache action")
    cache_parser.add_argument("--pattern", help="Pattern to match for clearing")
    cache_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed information"
    )

    args = parser.parse_args()

    if args.command == "cache":
        loader = WanPipelineLoader()
        if args.action == "list":
            loader.list_cache(verbose=args.verbose)
        elif args.action == "clear":
            loader.clear_cache(args.pattern)

    elif args.command == "infer":
        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load pipeline
        loader = WanPipelineLoader()
        pipe = loader.load_optimized_pipeline(
            lora_weight_name=args.lora_weight,
            lora_scale_1=args.lora_scale_1,
            lora_scale_2=args.lora_scale_2,
            optimize=not args.no_optimize,
            quantization=args.quantization,
            compile_mode=args.compile,
            use_cache=not args.no_cache,
        )

        # Process images
        for image_path in args.images:
            logger.info(f"Processing image: {image_path}")

            image = load_image(image_path).convert("RGB")
            image = resize_image_to_fit_aspect(image)

            for duration in args.durations:
                output, elapsed = run_inference(
                    pipe,
                    image,
                    args.prompt,
                    duration,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                )

                # Save output
                output_name = f"{Path(image_path).stem}_d{duration}_s{args.steps}.mp4"
                output_path = output_dir / output_name

                export_to_video(output, str(output_path), fps=16)
                logger.info(f"Saved: {output_path}")

                # Log stats
                fps = output.shape[0] / elapsed
                logger.info(f"Performance: {fps:.1f} frames/sec, {elapsed:.2f}s total")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
