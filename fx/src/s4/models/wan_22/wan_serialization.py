"""
WAN 2.2 Serialization and Quantization Module
==============================================

This module handles model serialization to standardized formats (safetensors)
and provides interfaces for various quantization methods including:
- FP8 E4M3/E5M2
- NVFP4 (NVIDIA 4-bit floating point)
- Nunchaku quantization
- SVDQuant-style quantization

The serialization format is compatible with ComfyUI and other inference
frameworks that use safetensors as their standard format.
"""

from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from enum import Enum

import torch
import torch.nn as nn

# Import pure modules for type checking


# ================================================================================================
# Section 1: Quantization Configurations
# ================================================================================================


class QuantizationType(Enum):
    """Supported quantization types."""

    NONE = "none"
    FP8_E4M3 = "fp8_e4m3"  # 4-bit exponent, 3-bit mantissa
    FP8_E5M2 = "fp8_e5m2"  # 5-bit exponent, 2-bit mantissa
    NVFP4 = "nvfp4"  # NVIDIA 4-bit floating point
    NUNCHAKU = "nunchaku"  # Nunchaku quantization
    SVDQUANT = "svdquant"  # SVDQuant activation-aware quantization
    INT8 = "int8"  # Standard INT8 quantization
    INT4 = "int4"  # 4-bit integer quantization


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    # Weight quantization
    weight_quant: QuantizationType = QuantizationType.NONE
    weight_bits: int = 16
    weight_symmetric: bool = True
    weight_group_size: int = 128  # For group-wise quantization

    # Activation quantization
    activation_quant: QuantizationType = QuantizationType.NONE
    activation_bits: int = 16
    activation_symmetric: bool = False

    # KV cache quantization
    kv_cache_quant: QuantizationType = QuantizationType.NONE
    kv_cache_bits: int = 16

    # Layer-wise settings
    skip_layers: List[str] = None  # Layers to skip quantization
    per_layer_config: Dict[str, Dict] = None  # Per-layer overrides

    # Calibration settings (for activation-aware methods)
    calibration_samples: int = 128
    calibration_percentile: float = 99.9

    def __post_init__(self):
        if self.skip_layers is None:
            self.skip_layers = []
        if self.per_layer_config is None:
            self.per_layer_config = {}


# ================================================================================================
# Section 2: Tensor Metadata for Safetensors
# ================================================================================================


@dataclass
class TensorMetadata:
    """
    Metadata for a tensor in safetensors format.

    This stores both the standard safetensors metadata and our
    quantization-specific information.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: str  # Original dtype before quantization

    # Quantization metadata
    quantized: bool = False
    quant_type: Optional[str] = None
    quant_bits: Optional[int] = None
    quant_scale: Optional[float] = None
    quant_zero_point: Optional[float] = None
    quant_group_size: Optional[int] = None

    # Statistics for debugging/analysis
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    std_val: Optional[float] = None

    def to_safetensors_header(self) -> Dict[str, Any]:
        """Convert to safetensors header format."""
        header = {
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

        # Add quantization metadata if present
        if self.quantized:
            header["quantization"] = {
                "type": self.quant_type,
                "bits": self.quant_bits,
                "scale": self.quant_scale,
                "zero_point": self.quant_zero_point,
                "group_size": self.quant_group_size,
            }

        return header


# ================================================================================================
# Section 3: Quantization Methods (Stubs for Now)
# ================================================================================================


class QuantizationMethod:
    """Base class for quantization methods."""

    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        config: QuantizationConfig,
        tensor_name: str = "",
    ) -> Tuple[torch.Tensor, TensorMetadata]:
        """
        Quantize a tensor and return quantized data with metadata.

        Args:
            tensor: Input tensor to quantize
            config: Quantization configuration
            tensor_name: Name of tensor for debugging

        Returns:
            Tuple of (quantized_tensor, metadata)
        """
        raise NotImplementedError("Subclasses must implement quantize_tensor")

    def dequantize_tensor(
        self,
        quantized_tensor: torch.Tensor,
        metadata: TensorMetadata,
    ) -> torch.Tensor:
        """
        Dequantize a tensor using its metadata.

        Args:
            quantized_tensor: Quantized tensor data
            metadata: Tensor metadata with quantization info

        Returns:
            Dequantized tensor
        """
        raise NotImplementedError("Subclasses must implement dequantize_tensor")


class FP8Quantization(QuantizationMethod):
    """FP8 quantization (E4M3 or E5M2)."""

    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        config: QuantizationConfig,
        tensor_name: str = "",
    ) -> Tuple[torch.Tensor, TensorMetadata]:
        """FP8 quantization implementation."""

        # Placeholder for FP8 quantization
        # In practice, this would use torch.float8_e4m3fn or torch.float8_e5m2
        raise NotImplementedError(
            "FP8 quantization requires PyTorch 2.1+ with FP8 support. "
            "Implementation pending hardware availability."
        )

    def dequantize_tensor(
        self,
        quantized_tensor: torch.Tensor,
        metadata: TensorMetadata,
    ) -> torch.Tensor:
        """FP8 dequantization."""
        raise NotImplementedError("FP8 dequantization not yet implemented")


class NVFP4Quantization(QuantizationMethod):
    """
    NVIDIA FP4 quantization for Blackwell architecture.

    This is the key optimization for achieving 2+ PFLOPS on RTX 5090/6000.
    The implementation requires CUDA 13 and Blackwell tensor cores.
    """

    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        config: QuantizationConfig,
        tensor_name: str = "",
    ) -> Tuple[torch.Tensor, TensorMetadata]:
        """
        NVFP4 quantization implementation.

        This would:
        1. Analyze tensor statistics for optimal scaling
        2. Convert to 4-bit floating point format
        3. Pack multiple FP4 values into larger types for storage
        """

        raise NotImplementedError(
            "NVFP4 quantization requires CUDA 13 with Blackwell support. "
            "Implementation will use custom CUDA kernels for optimal performance. "
            "Expected: 4x memory reduction, 2x compute speedup on GB202."
        )

    def dequantize_tensor(
        self,
        quantized_tensor: torch.Tensor,
        metadata: TensorMetadata,
    ) -> torch.Tensor:
        """NVFP4 dequantization - typically done on-the-fly in tensor cores."""
        raise NotImplementedError("NVFP4 dequantization happens in tensor cores during compute")


class NunchakuQuantization(QuantizationMethod):
    """
    Nunchaku quantization method.

    This is an advanced quantization technique that maintains quality
    while achieving aggressive compression ratios.
    """

    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        config: QuantizationConfig,
        tensor_name: str = "",
    ) -> Tuple[torch.Tensor, TensorMetadata]:
        """
        Nunchaku quantization implementation.

        This would implement the Nunchaku algorithm which:
        1. Performs activation-aware analysis
        2. Uses learned quantization parameters
        3. Applies non-uniform quantization bins
        """

        raise NotImplementedError(
            "Nunchaku quantization implementation pending. "
            "Requires calibration data and learned quantization parameters."
        )

    def dequantize_tensor(
        self,
        quantized_tensor: torch.Tensor,
        metadata: TensorMetadata,
    ) -> torch.Tensor:
        """Nunchaku dequantization."""
        raise NotImplementedError("Nunchaku dequantization not yet implemented")


# ================================================================================================
# Section 4: Model Serialization
# ================================================================================================


class WANModelSerializer:
    """
    Handles serialization of WAN models to safetensors format.

    This class manages:
    1. Weight extraction from PyTorch modules
    2. Optional quantization
    3. Metadata generation
    4. Safetensors file writing
    """

    def __init__(self, quantization_config: Optional[QuantizationConfig] = None):
        """
        Initialize serializer.

        Args:
            quantization_config: Optional quantization configuration
        """
        self.quant_config = quantization_config or QuantizationConfig()

        # Initialize quantization methods
        self.quant_methods = {
            QuantizationType.FP8_E4M3: FP8Quantization(),
            QuantizationType.FP8_E5M2: FP8Quantization(),
            QuantizationType.NVFP4: NVFP4Quantization(),
            QuantizationType.NUNCHAKU: NunchakuQuantization(),
        }

    def extract_state_dict(
        self,
        model: nn.Module,
        prefix: str = "",
    ) -> Dict[str, torch.Tensor]:
        """
        Extract state dict with proper naming for safetensors.

        Args:
            model: PyTorch model
            prefix: Prefix for parameter names

        Returns:
            Dictionary of parameter names to tensors
        """

        state_dict = {}

        for name, param in model.named_parameters():
            full_name = f"{prefix}.{name}" if prefix else name

            # Normalize naming for ComfyUI compatibility
            full_name = self._normalize_parameter_name(full_name)

            state_dict[full_name] = param.detach().cpu()

        # Also extract buffers (for normalization statistics, etc.)
        for name, buffer in model.named_buffers():
            if buffer is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                full_name = self._normalize_parameter_name(full_name)
                state_dict[full_name] = buffer.detach().cpu()

        return state_dict

    def _normalize_parameter_name(self, name: str) -> str:
        """
        Normalize parameter names for ComfyUI compatibility.

        ComfyUI expects certain naming conventions for attention weights.
        """

        # Map our names to ComfyUI expected names
        replacements = {
            ".to_qkv.": ".to_qkv.",  # Keep as is
            ".to_kv.": ".to_kv.",  # Keep as is
            ".to_q.": ".to_q.",  # Keep as is
            ".to_out.0.": ".to_out.0.",  # Linear layer in output
            ".self_attn.": ".attn1.",  # Self-attention
            ".cross_attn.": ".attn2.",  # Cross-attention
        }

        for old, new in replacements.items():
            name = name.replace(old, new)

        return name

    def quantize_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, TensorMetadata]]:
        """
        Quantize a state dict according to configuration.

        Args:
            state_dict: Original state dict

        Returns:
            Tuple of (quantized_state_dict, metadata_dict)
        """

        quantized_dict = {}
        metadata_dict = {}

        for name, tensor in state_dict.items():
            # Check if this layer should be skipped
            if any(skip in name for skip in self.quant_config.skip_layers):
                quantized_dict[name] = tensor
                metadata_dict[name] = self._create_metadata(name, tensor, quantized=False)
                continue

            # Determine quantization type for this tensor
            quant_type = self._get_quantization_type(name)

            if quant_type == QuantizationType.NONE:
                quantized_dict[name] = tensor
                metadata_dict[name] = self._create_metadata(name, tensor, quantized=False)
            else:
                # Apply quantization
                quant_method = self.quant_methods[quant_type]
                quantized_tensor, metadata = quant_method.quantize_tensor(
                    tensor, self.quant_config, name
                )
                quantized_dict[name] = quantized_tensor
                metadata_dict[name] = metadata

        return quantized_dict, metadata_dict

    def _get_quantization_type(self, tensor_name: str) -> QuantizationType:
        """Determine quantization type for a given tensor."""

        # Check per-layer configuration
        for pattern, config in self.quant_config.per_layer_config.items():
            if pattern in tensor_name:
                return QuantizationType(config.get("type", "none"))

        # Default to weight quantization config
        if "weight" in tensor_name or "bias" not in tensor_name:
            return self.quant_config.weight_quant

        return QuantizationType.NONE

    def _create_metadata(
        self,
        name: str,
        tensor: torch.Tensor,
        quantized: bool = False,
    ) -> TensorMetadata:
        """Create metadata for a tensor."""

        # Compute statistics
        with torch.no_grad():
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            mean_val = tensor.mean().item()
            std_val = tensor.std().item() if tensor.numel() > 1 else 0.0

        return TensorMetadata(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            std_val=std_val,
        )

    def save_to_safetensors(
        self,
        model: nn.Module,
        filepath: Union[str, Path],
        quantize: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save model to safetensors format.

        Args:
            model: PyTorch model to save
            filepath: Output file path
            quantize: Whether to apply quantization
            metadata: Additional metadata to include

        Returns:
            Dictionary of serialization statistics
        """

        filepath = Path(filepath)

        # Extract state dict
        state_dict = self.extract_state_dict(model)

        # Optionally quantize
        if quantize:
            state_dict, tensor_metadata = self.quantize_state_dict(state_dict)
        else:
            tensor_metadata = {
                name: self._create_metadata(name, tensor) for name, tensor in state_dict.items()
            }

        # Create safetensors metadata
        st_metadata = {
            "format": "pt",
            "quantization": asdict(self.quant_config) if quantize else None,
            "model_type": "wan_transformer_v2",
            "custom_metadata": metadata,
        }

        # Import safetensors and save
        try:
            from safetensors.torch import save_file

            save_file(
                state_dict,
                filepath,
                metadata=st_metadata,
            )

            print(f"Saved model to {filepath}")

        except ImportError:
            raise ImportError(
                "safetensors library not found. Install with: pip install safetensors"
            )

        # Return statistics
        total_params = sum(t.numel() for t in state_dict.values())
        total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())

        return {
            "filepath": str(filepath),
            "num_tensors": len(state_dict),
            "total_parameters": total_params,
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "quantized": quantize,
            "quantization_config": asdict(self.quant_config) if quantize else None,
        }

    @classmethod
    def load_from_safetensors(
        cls,
        filepath: Union[str, Path],
        device: str = "cpu",
        dequantize: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Load model from safetensors format.

        Args:
            filepath: Path to safetensors file
            device: Device to load tensors to
            dequantize: Whether to dequantize on load

        Returns:
            Tuple of (state_dict, metadata)
        """

        try:
            from safetensors.torch import load_file

            # Load with metadata
            state_dict = load_file(filepath, device=device)

            # TODO: Extract and parse metadata when safetensors supports it
            metadata = {}

            return state_dict, metadata

        except ImportError:
            raise ImportError(
                "safetensors library not found. Install with: pip install safetensors"
            )


# ================================================================================================
# Section 5: ComfyUI Integration Utilities
# ================================================================================================


def create_comfyui_config(model: nn.Module) -> Dict[str, Any]:
    """
    Create a configuration dictionary compatible with ComfyUI.

    Args:
        model: WAN transformer model

    Returns:
        Configuration dictionary for ComfyUI
    """

    config = {
        "model_type": "wan_v2",
        "in_channels": 36,
        "out_channels": 16,
        "hidden_size": 5120,
        "num_layers": 48,
        "num_heads": 40,
        "head_dim": 128,
        "patch_size": [1, 2, 2],
        "text_encoder_dim": 4096,
        "image_encoder_dim": 1280,
        "use_rotary_emb": True,
        "use_modulation": True,
        # ComfyUI-specific settings
        "prediction_type": "v_prediction",
        "timestep_range": [0, 1000],
        "beta_schedule": "scaled_linear",
    }

    return config


def export_for_comfyui(
    model: nn.Module,
    output_dir: Union[str, Path],
    model_name: str = "wan_v2",
    quantization_config: Optional[QuantizationConfig] = None,
) -> Dict[str, Any]:
    """
    Export model in ComfyUI-compatible format.

    This creates:
    1. model.safetensors - The model weights
    2. config.json - Model configuration
    3. quantization.json - Quantization metadata (if applicable)

    Args:
        model: WAN transformer model
        output_dir: Directory to save files
        model_name: Name for the model files
        quantization_config: Optional quantization configuration

    Returns:
        Dictionary of export statistics
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create serializer
    serializer = WANModelSerializer(quantization_config)

    # Save model weights
    model_path = output_dir / f"{model_name}.safetensors"
    save_stats = serializer.save_to_safetensors(
        model,
        model_path,
        quantize=quantization_config is not None,
    )

    # Save configuration
    config = create_comfyui_config(model)
    config_path = output_dir / f"{model_name}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Save quantization config if applicable
    if quantization_config is not None:
        quant_path = output_dir / f"{model_name}_quantization.json"
        with open(quant_path, "w") as f:
            json.dump(asdict(quantization_config), f, indent=2)

    print(f"Exported model to {output_dir}")
    print(f"  - Model: {model_path} ({save_stats['total_mb']:.1f} MB)")
    print(f"  - Config: {config_path}")
    if quantization_config:
        print(f"  - Quantization: {quant_path}")

    return save_stats
