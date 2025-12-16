# """ WAN 2.2 Architecture Documentation

## Implementation Strategy and Design Decisions

This document codifies the architectural decisions and compatibility strategy for the WAN 2.2
Image-to-Video transformer implementation.

## Architecture Overview

The implementation is split into three strategic layers:

### 1. Pure PyTorch Layer (`wan_attention_pure.py`)

- **Purpose**: Inference-optimized, branch-free computational graph
- **Characteristics**:
  - Zero framework dependencies beyond PyTorch
  - Fully static graph (torch.compile, TorchScript, CUDAGraph compatible)
  - Explicit shape annotations throughout
  - No dynamic control flow
  - Optimized for NVFP4 quantization readiness

### 2. Compatibility Layer (`wan_attention_diffusers.py`)

- **Purpose**: Framework compatibility and configuration management
- **Role**: "Mess absorption zone" - handles all real-world complexity
- **Responsibilities**:
  - Diffusers pipeline integration
  - Weight conversion and remapping
  - Configuration-time graph specialization
  - Hardware detection and routing
  - Serialization format compatibility

### 3. Serialization Layer (`wan_serialization.py`)

- **Purpose**: Universal model distribution
- **Formats**:
  - Safetensors (HuggingFace standard)
  - ComfyUI compatibility
  - GGUF export (future)
  - TensorRT plans (future)
- **Quantization Interfaces**:
  - FP8 (E4M3/E5M2)
  - NVFP4 (Blackwell optimization)
  - Nunchaku
  - SVDQuant

## Key Design Principles

### 1. Configuration-Time Specialization

Instead of runtime conditionals, the compatibility layer allocates different block types based on
configuration:

```python
# Bad: Runtime branching
def forward(self, x, use_image=True):
    if use_image:
        x = self.image_projection(x)
    return x

# Good: Configuration-time specialization
def __init__(self, config):
    if config.use_image:
        self.blocks = ImageAwareBlocks()
    else:
        self.blocks = TextOnlyBlocks()
```

This ensures the inference graph remains static and compileable.

### 2. Fused Operations

Key optimizations that eliminate memory bottlenecks:

1. **Fused QKV Projection**: Single matmul instead of three
2. **Pre-projected Image Features**: Computed once, used by all blocks
3. **Batched KV Computation** (future): All 48 layers in one operation

### 3. NVFP4 Readiness

The architecture anticipates NVFP4 quantization on Blackwell:

1. **Memory Layout**: Optimized for 128-bit aligned access patterns
2. **Tensor Shapes**: Multiples of 8 for tensor core efficiency
3. **Pre-computed Caches**: Ready for FP4 storage
4. **Minimal Bandwidth**: Designed for 4x memory reduction

## Compatibility Strategy

### HuggingFace Diffusers

We maintain compatibility through a thin wrapper that:

1. Accepts diffusers configuration format
2. Provides expected method signatures
3. Returns diffusers output types
4. Handles weight conversion from stock models

The compatibility layer is intentionally "messy" - it's designed to absorb all the complexity of
real-world deployment so the pure implementation remains clean.

### ComfyUI

Serialization to safetensors with proper naming conventions ensures ComfyUI can load our models
directly. Key mappings:

- `.self_attn.` → `.attn1.`
- `.cross_attn.` → `.attn2.`
- Fused weights are properly split when needed

### Weight Conversion

The `from_pretrained_stock` method handles all the complexity of converting from various model
versions:

1. Detects fused vs unfused projections
2. Handles different normalization configurations
3. Maps layer names appropriately
4. Validates dimensions and shapes

## Quantization Strategy

### Current State

Quantization interfaces are defined but not implemented, awaiting:

1. Blackwell hardware availability
2. CUDA 13 with NVFP4 support
3. Calibration datasets

### Implementation Path

```python
# Phase 1: Define interfaces (DONE)
class NVFP4Quantization:
    def quantize_tensor(...) -> NotImplemented

# Phase 2: Implement with CUDA kernels (PENDING)
class NVFP4Quantization:
    def quantize_tensor(...):
        return cuda_nvfp4_quantize(tensor)

# Phase 3: Integrate with inference
model = model.quantize(NVFP4Config())
```

## Testing Philosophy

### Layer-by-Layer Testing

Where possible, we test individual components:

- Attention modules
- Normalization layers
- Feed-forward networks
- Position embeddings

### Fusion Boundaries

Some optimizations prevent component testing:

- Fused QKV projections must be tested as a unit
- Pre-computed KV cache changes the testing boundary
- Quantized operations may require different tolerances

### Property-Based Testing

Using Hypothesis, we verify invariants:

- Shape preservation
- Attention weights sum to 1
- Normalization statistics
- Gradient flow

### Reference Implementation Comparison

Where available, we compare against:

- Stock diffusers implementation
- Original paper specifications
- Known good outputs

## Performance Targets

### Current (FP16 on Ampere/Hopper)

- Single block forward: \<10ms
- Full model (48 blocks): \<500ms
- Memory usage: \<8GB for batch size 1

### Target (NVFP4 on Blackwell)

- Single block forward: \<2ms
- Full model (48 blocks): \<100ms
- Memory usage: \<2GB for batch size 1
- Throughput: 1.2+ PFLOPS sustained

## Deployment Strategy

### Phase 1: Compatibility (COMPLETE)

- Diffusers pipeline support
- ComfyUI integration
- Safetensors serialization

### Phase 2: Optimization (IN PROGRESS)

- torch.compile integration
- CUDAGraph capture
- Memory layout optimization

### Phase 3: Quantization (PENDING)

- NVFP4 implementation
- KV cache compression
- Batched operations

### Phase 4: Platform Expansion (FUTURE)

- GGUF export
- TensorRT optimization
- ONNX compatibility
- Edge deployment

## Hardware Detection and Routing

The compatibility layer will detect hardware and route accordingly:

```python
def create_optimized_model(config):
    if detect_blackwell():
        return WanModelNVFP4(config)
    elif detect_ampere_or_hopper():
        return WanModelFP16(config)
    elif detect_intel_arc():
        return WanModelXPU(config)
    else:
        return WanModelCPU(config)
```

## Memory Optimization Strategies

### Current Optimizations

1. **Fused Operations**: Reduce intermediate tensor allocation
2. **In-place Operations**: Where safe, modify tensors in-place
3. **Gradient Checkpointing**: For training, trade compute for memory

### Future Optimizations

1. **FP4 Storage**: 4x memory reduction
2. **Chunked Attention**: Process attention in chunks for long sequences
3. **CPU Offloading**: For very large models
4. **Pipeline Parallelism**: Split model across GPUs

## Benchmarking and Validation

### Correctness Validation

- Unit tests for each component
- Integration tests for full model
- Numerical accuracy tests with different dtypes
- Comparison with reference implementation

### Performance Validation

- Throughput benchmarks (tokens/second)
- Latency measurements (ms per forward pass)
- Memory usage profiling
- Power efficiency metrics

### Quality Validation

- Perceptual metrics for generated videos
- FID/IS scores for image quality
- Temporal consistency metrics
- User studies (when applicable)

## Configuration Examples

### Minimal Inference Configuration

```python
config = {
    "dim": 5120,
    "num_heads": 40,
    "num_layers": 48,
    "patch_size": (1, 2, 2),
    "mode": "inference",
    "dtype": "fp16"
}
```

### Optimized Blackwell Configuration

```python
config = {
    "dim": 5120,
    "num_heads": 40, 
    "num_layers": 48,
    "patch_size": (1, 2, 2),
    "mode": "inference",
    "dtype": "nvfp4",
    "use_flash_attn": True,
    "use_cudagraph": True,
    "kv_cache_dtype": "fp4"
}
```

### Training Configuration

```python
config = {
    "dim": 5120,
    "num_heads": 40,
    "num_layers": 48,
    "patch_size": (1, 2, 2),
    "mode": "training",
    "dtype": "bfloat16",
    "gradient_checkpointing": True,
    "dropout": 0.1
}
```

## Error Handling Philosophy

### Fail Fast in Development

- Assertions for shape mismatches
- Type checking for tensor dtypes
- Validation of configuration parameters

### Graceful Degradation in Production

- Fallback from NVFP4 to FP16 if unavailable
- CPU fallback for unsupported operations
- Warning logs for suboptimal configurations

## Conclusion

This architecture prioritizes:

1. **Inference Performance**: Every decision optimizes for speed
2. **Deployment Flexibility**: Runs everywhere, optimized for Blackwell
3. **Maintainability**: Clear separation of concerns
4. **Future-Proofing**: Ready for NVFP4 and beyond

The three-layer architecture ensures we can maintain compatibility with the ecosystem while
aggressively optimizing for next-generation hardware. """
