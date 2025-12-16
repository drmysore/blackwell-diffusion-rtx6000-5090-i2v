# `// fxy // mldev // style guide`

## Strategy and Motivation

We use C++ for machine learning inference when we need extreme performance along one or more
dimensions: minimal latency, maximum throughput, or optimal hardware utilization. We operate in
regimes where Python overhead is unacceptable and where every microsecond matters. Our C++ inference
stack provides low-friction access to hardware accelerators, efficient memory management, and
best-in-class optimization techniques.

Modern ML inference in C++ is challenging because the best practices are often proprietary - hidden
inside companies like NVIDIA, Google, and Meta. The gap between academic papers and production
systems is vast. This document bridges that gap, with particular emphasis on diffusion models and
generative AI workloads.

This document is aimed at three audiences:

- Experienced C++ programmers entering ML inference
- ML engineers needing production-grade C++ inference
- AI agents generating high-performance inference code

## The Economics of Code in Agent-Heavy Development

**In a codebase with heavy agent contribution, traditional economics invert:**

- Code is written once by agents in seconds
- Code is read hundreds of times by humans and agents
- Code is debugged under production pressure by tired humans
- Code is modified by agents who lack the original context

**Every ambiguity compounds exponentially.**

### The Fundamental Principle

```cpp
// This costs an agent 0.1 seconds to write, a human 10 seconds to debug:
auto m = model{};
if (m.s > 0) infer(m);

// Better with AAA (Almost Always Auto), but still needs descriptive names:
auto transformer_model = fxy::models::vision_transformer{};
if (transformer_model.num_layers > 0) {
    run_transformer_inference(transformer_model);
}

// Best: AAA with descriptive names and initializer lists
auto diffusion_model = fxy::models::load_dit_model(model_path);
auto generation_config = fxy::models::dit_config{
    .num_steps = 50,
    .guidance_scale = 7.5f,
    .use_flash_attention = true
};
auto result = diffusion_model.generate(prompt, generation_config);
```

**With AAA, descriptive variable names become even more important.**

### Why Config Parsing Is Sacred

Configuration parsing is the most critical code in any inference system because:

1. **Multiplication Effect**: One config bug affects every inference request
2. **Trust Boundary**: External model configs that everything trusts implicitly
3. **Silent Corruption**: Config errors cause wrong predictions, not crashes
4. **Production Impact**: Misconfigured models serve incorrect results at scale

Config parsing should be **human-written**, **brutally simple**, and **fail-fast**.

## High Level Choices

1. **Almost Always Auto (AAA)** - Use `auto` and initializer lists wherever reasonable
2. **Global namespace prefix** - Use `::` for global functions like `::memcpy`, `::printf`
3. **Fully qualified names** - No `using namespace`, absolute clarity
4. **C++23 features** - Use modern constructs maximally
5. **Hardware-aware design** - Optimize for GPU/TPU characteristics, especially for FLOPS-bound
   workloads
6. **Name for `grep`** - Every identifier must be globally searchable

## Naming Conventions

### The Disambiguation Imperative

In an agent-heavy codebase, names must be:

- **Globally unique** within their semantic domain
- **Self-documenting** without context
- **Searchable** with basic tools

```cpp
// BAD: Will create confusion at scale
class encoder;
auto weights = load();
int process(tensor& t);

// GOOD: Unambiguous even with 100 agents contributing
class transformer_encoder;
auto model_weights = fxy::model_storage::load_model_weights();
auto process_image_batch(fxy::tensor::image_tensor& batch) -> int;
```

### Core Naming Rules

- **snake_case** for everything: `tensor_buffer`, `model_weights`, `run_inference()`
- **Full words** over abbreviations: `configuration` not `config`, `buffer` not `buf`
- **Domain prefixes** for common concepts: `gpu_tensor`, `cpu_tensor`
- **member_suffix\_** for members: `batch_size_`, `sequence_length_`, `device_id_`
- **Preserve acronyms**: `BERT_tokenizer` not `BertTokenizer`

### The Three-Letter Rule

If an abbreviation is less than 4 characters, it's too short:

```cpp
// BAD
auto cfg = load_cfg();
auto mdl = get_mdl();
auto res = inf(req);

// GOOD  
auto configuration = fxy::model_configuation::load_configuration();
auto model = fxy::model_storage::get_model();
auto result = fxh::inference::run_inference(request);
```

### Standard Abbreviations (Use Sparingly)

Only when the full name would be absurd:

- `idx/jdx` - index (prefer descriptive names like `batch_index`)
- `ptr` - pointer (only in compounds like `data_ptr`)
- `ctx` - context (only when type makes it unambiguous)

## Code Organization

### Directory Structure Guidelines

```
fxy/
├── core/           # Foundation utilities (tensor, memory, profiling)
├── ops/            # Neural network operators (conv, attention, etc.)
├── runtime/        # Inference runtime and graph execution
├── backends/       # Hardware backends (cuda, cpu, metal)
├── models/         # Model-specific implementations
├── serving/        # Model serving infrastructure
├── benchmarks/     # Performance benchmarks (*_benchmark.cpp)
└── examples/       # Integration examples
```

- Keep modules flat until complexity demands subdirectories
- Test files live alongside source: `attention.cpp`, `attention_test.cpp`
- Kernel implementations: `attention_cuda.cu`, `attention_cpu.cpp`
- Header-only implementations use: `tensor-inl.h`

### Headers

```cpp
#pragma once

#include <memory>
#include <span>
#include <vector>

#include "fxy/core/tensor.h"
#include "fxy/core/result.h"

namespace fxy::inference {

class model_executor {  // Full descriptive names
public:
    model_executor() = default;

    // Full words in function names
    auto load_model_from_path(std::filesystem::path model_path) noexcept 
        -> fxy::core::result<void>;
    
    auto set_batch_size(std::size_t batch_size) noexcept -> void;

    // Clear parameter names with types
    auto execute_forward_pass(
        std::span<const fxy::core::tensor> input_tensors,
        fxy::core::device_type device = fxy::core::device_type::gpu) noexcept
        -> fxy::core::result<std::vector<fxy::core::tensor>>;

private:
    // Clear member names with units where applicable
    std::unique_ptr<compute_graph> computation_graph_{};
    std::size_t max_batch_size_{1};
    std::size_t max_sequence_length_{2048};
    fxy::core::device_type preferred_device_{fxy::core::device_type::gpu};
};

}  // namespace fxy::inference
```

### Implementation

```cpp
#include "fxy/inference/model_executor.h"

#include <format>

#include "fxy/core/logging.h"
#include "fxy/backends/cuda/device.h"

namespace fxy::inference {

auto model_executor::load_model_from_path(
    std::filesystem::path model_path) noexcept -> fxy::core::result<void> {
    
    if (!std::filesystem::exists(model_path)) {
        return fxy::core::fail(
            std::format("[fxy] [inference] [executor] model not found: {}",
                       model_path.string()));
    }

    // AAA style - let the compiler deduce types
    auto model_format = detect_model_format(model_path);
    if (!model_format) {
        return fxy::core::fail(
            std::format("[fxy] [inference] [executor] unknown model format: {}",
                       model_path.string()));
    }

    auto loaded_graph = load_computation_graph(model_path, *model_format);
    if (!loaded_graph) {
        return fxy::core::fail(
            std::format("[fxy] [inference] [executor] failed to load graph: {}",
                       loaded_graph.error().what()));
    }
    
    // Global namespace functions prefixed with ::
    ::printf("[fxy] Model loaded successfully\n");
    
    computation_graph_ = std::move(loaded_graph.value());
    return fxy::core::ok();
}

}  // namespace fxy::inference
```

## Modern C++23 Patterns for ML

### Core Hardware Realities

Modern ML inference has unique hardware constraints that vary by workload:

**For Diffusion Models (DiT architectures):**

- **FLOPS-bound, not bandwidth-bound** - Modern diffusion transformers saturate compute before
  memory
- **Kernel fusion is critical** - Each kernel launch costs microseconds; fusing operations is
  essential
- **Quantization requires co-design** - 4-bit inference needs fused low-rank + low-bit branches (see
  SVDQuant/Nunchaku)
- **Batch size 1 is still compute-bound** - Unlike LLMs, single-image generation saturates ALUs

**General GPU characteristics:**

- **Cache lines are 64 bytes** - This is the unit of memory transfer
- **Tensor cores dominate throughput** - FP16/INT8/INT4 operations via tensor cores
- **Kernel launch overhead** - Microseconds per launch add up quickly
- **Warp divergence kills performance** - All threads in a warp must take the same path

### AAA and Modern C++ Style

**AAA (Almost Always Auto) is the default:**

```cpp
// DO: Use auto everywhere possible
auto model = load_diffusion_model(path);
auto result = model.generate({prompt, negative_prompt});
auto latents = result.get_latents();

// Use explicit types only when necessary for clarity
std::span<const std::uint8_t> image_bytes = result.get_image_data();
```

**When to break AAA:**

```cpp
// DO: Use explicit types for numeric precision
std::int32_t batch_size = 1;  // Not auto - precision matters
std::float16_t learning_rate = 0.001f;  // Explicit half precision
std::uint8_t* image_data = buffer.data();  // Pointer arithmetic needs clarity

// DO: Use explicit types for API boundaries
auto compute(std::span<const float> input) -> fxy::core::result<tensor> {
    // Parameter and return types should be explicit
    auto internal_calc = do_something(input);  // But AAA internally
    return fxy::core::ok(internal_calc);
}
```

**Initializer lists everywhere:**

```cpp
// DO: Uniform initialization prevents narrowing
auto config = inference_config{
    .batch_size = 1,
    .num_steps = 50,
    .guidance_scale = 7.5f,
    .device = device_type::cuda
};

auto tensor = fxy::core::tensor{
    {2, 3, 224, 224},  // shape
    fxy::core::dtype::float16
};
```

**Global namespace prefix:**

```cpp
// DO: Prefix global functions with ::
::memcpy(dest.data(), src.data(), size);
::printf("[fxy] Kernel fusion saved %d launches\n", saved_launches);
auto file = ::fopen(path.c_str(), "rb");
```

### Memory Management Patterns

```cpp
// DO: Use memory pools for tensor allocations
class tensor_allocator {
    struct memory_pool {
        std::vector<std::unique_ptr<aligned_buffer>> free_buffers_;
        std::mutex pool_mutex_;
    };
    std::array<memory_pool, 32> pools_by_size_class_;  // Power-of-2 buckets

public:
    auto allocate(std::size_t size_bytes) -> fxy::core::result<memory_handle>;
    auto deallocate(memory_handle handle) -> void;
};

// DON'T: Allocate per inference
auto bad_inference(tensor input) -> tensor {
    auto temp = allocate_tensor(input.shape());  // Allocation in hot path!
    // ...
}
```

### Error Handling Philosophy

We don't throw exceptions in inference paths. We use `fxy::core::result<T>`:

```cpp
// When failure is recoverable - return result
auto load_weights(std::filesystem::path weights_path) noexcept
    -> fxy::core::result<weight_tensor> {
    if (!std::filesystem::exists(weights_path)) {
        return fxy::core::fail<weight_tensor>("weights not found: {}", 
                                             weights_path.string());
    }
    // Load weights...
    return fxy::core::ok(weight_tensor{...});
}

// When failure is unrecoverable - fail fast
if (!device_context) {
    fxy::fatal("Device context lost - cannot continue inference");
}
```

### Operator Fusion for Diffusion Models

```cpp
// DO: Fuse operations to minimize kernel launches (SVDQuant/Nunchaku style)
class fused_dit_block {
    // Fuses attention + FFN + normalization in a single kernel
    auto forward(const tensor& input) -> tensor {
        // Single fused kernel for DiT block
        return fxy::ops::fused_dit_block(
            input, 
            q_proj_weight_, k_proj_weight_, v_proj_weight_,
            ffn_weights_, 
            norm_params_
        );
    }
};

// DON'T: Separate operations with multiple kernel launches
auto forward(const tensor& input) -> tensor {
    auto norm1 = layer_norm(input);           // Kernel launch 1
    auto attn = attention(norm1);             // Kernel launch 2
    auto residual1 = add(input, attn);        // Kernel launch 3
    auto norm2 = layer_norm(residual1);       // Kernel launch 4
    auto ffn_out = feed_forward(norm2);       // Kernel launch 5
    return add(residual1, ffn_out);           // Kernel launch 6
}
```

### Quantization with Low-Rank Decomposition

```cpp
// DO: Co-design quantization with kernel fusion (SVDQuant pattern)
class svd_quantized_linear {
    // 4-bit main branch + 16-bit low-rank branch, fused execution
    auto forward(const tensor& input) -> tensor {
        // Fused kernel: quantize input + 4-bit matmul + low-rank correction
        return fxy::ops::fused_svd_linear(
            input,
            quantized_weight_int4_,    // 4-bit weights
            low_rank_u_,                // 16-bit SVD components
            low_rank_v_,
            scale_,
            zero_point_
        );
    }
    
private:
    tensor_int4 quantized_weight_int4_{};
    tensor_fp16 low_rank_u_{};
    tensor_fp16 low_rank_v_{};
    tensor_fp32 scale_{};
    tensor_int32 zero_point_{};
};
```

## Agent-Human Collaboration Patterns

### The Comment Convention

**This convention helps identify code provenance at a glance:**

- Agents: Properly capitalized comments
- Humans: lowercase comments

```cpp
// This is agent-generated code following standard patterns
auto attention_module = create_multi_head_attention(configuration);

// human note: flash attention requires specific memory alignment
if (configuration.use_flash_attention) {
    ensure_alignment(attention_module, 128);
}
```

### Guidelines for Modern C++ in Agent-Heavy Codebases

Both humans and agents should follow AAA and modern C++ patterns:

```cpp
// Good style for both humans and agents - AAA with descriptive names
auto input_batch = prepare_input_batch();
auto model_output = model.forward(input_batch);
auto predictions = apply_softmax(model_output);

// Only use explicit types when clarity demands it
std::span<const std::float16_t> weights = model.get_weights();
```

### Critical Path Marking

Identify code requiring human review:

```cpp
// CRITICAL PATH: Kernel implementation - human review required
namespace fxy::ops::attention {
    // Numerical precision issues here affect all transformer models
    auto scaled_dot_product_attention(
        const tensor& queries,
        const tensor& keys,
        const tensor& values) -> tensor {
        // Human-optimized implementation
    }
}

// AUXILIARY: Logging utilities - agent generation acceptable  
namespace fxy::utils::logging {
    // Agent can generate this boilerplate
}
```

## Testing Philosophy: Property-Based Testing for Numerics

### The Foundation of Numerical Testing

**In production ML systems, property-based testing is how tensor code is actually validated.** Unit
tests catch obvious bugs; property tests catch the subtle numerical disasters that corrupt models in
production.

### Core Properties Every Tensor Operation Must Satisfy

```cpp
// Property: Matrix multiplication associativity
TEST_CASE("matmul associativity") {
    property_test([](const tensor& a, const tensor& b, const tensor& c) {
        // Shapes must be compatible
        assume(a.cols() == b.rows() && b.cols() == c.rows());
        
        auto left_first = matmul(matmul(a, b), c);
        auto right_first = matmul(a, matmul(b, c));
        
        // Must be numerically close, not exact (floating point!)
        return tensors_allclose(left_first, right_first, 
                               /*rtol=*/1e-5, /*atol=*/1e-7);
    });
}

// Property: Softmax outputs sum to 1.0
TEST_CASE("softmax normalization") {
    property_test([](const tensor& logits) {
        assume(logits.rank() == 2);  // batch x classes
        
        auto output = softmax(logits, /*axis=*/-1);
        
        for (size_t batch_idx = 0; batch_idx < output.shape()[0]; ++batch_idx) {
            float sum = 0.0f;
            for (size_t class_idx = 0; class_idx < output.shape()[1]; ++class_idx) {
                sum += output.at({batch_idx, class_idx});
            }
            
            // Sum must be 1.0 within numerical tolerance
            if (std::abs(sum - 1.0f) > 1e-6f) {
                return false;
            }
        }
        return true;
    });
}
```

### Gradient Properties

```cpp
// Property: Gradient of sum equals ones
TEST_CASE("sum gradient correctness") {
    property_test([](const tensor& input) {
        auto sum_output = sum_all(input);
        auto gradient = compute_gradient(sum_output, input);
        
        // Gradient should be all ones
        return tensor_all_close_to_scalar(gradient, 1.0f, /*tol=*/1e-7);
    });
}

// Property: ReLU gradient is 0 or 1
TEST_CASE("relu gradient property") {
    property_test([](const tensor& input) {
        auto output = relu(input);
        auto gradient = compute_gradient(output, input);
        
        // Each gradient element must be exactly 0 or 1
        for (auto grad_val : gradient) {
            if (grad_val != 0.0f && grad_val != 1.0f) {
                return false;
            }
        }
        return true;
    });
}
```

### Numerical Stability Properties

```cpp
// Property: Operations must handle edge cases gracefully
TEST_CASE("numerical stability under extreme values") {
    // Test with denormals
    property_test([](float scale) {
        assume(scale > 0 && scale < 1e-30);  // Denormal range
        
        tensor tiny = tensor::constant({100, 100}, scale);
        auto result = layer_norm(tiny);
        
        // Must not produce NaN or Inf
        return !has_nan(result) && !has_inf(result);
    });
    
    // Test with large values
    property_test([](float scale) {
        assume(scale > 1e10 && scale < 1e30);
        
        tensor huge = tensor::constant({100, 100}, scale);
        auto result = softmax(huge);
        
        // Softmax must handle large values without overflow
        return !has_nan(result) && !has_inf(result) && 
               tensor_sum(result) ≈ huge.shape()[0];  // Sum over classes = batch_size
    });
}
```

### Shape Invariants

```cpp
// Property: Convolution output shape formula
TEST_CASE("convolution shape invariants") {
    property_test([](size_t h, size_t w, size_t k, size_t s, size_t p) {
        assume(h > 0 && w > 0 && k > 0 && s > 0);
        assume(k <= h + 2*p && k <= w + 2*p);  // Kernel can't be larger than padded input
        
        tensor input({1, 3, h, w});
        tensor kernel({64, 3, k, k});
        
        auto output = conv2d(input, kernel, /*stride=*/s, /*pad=*/p);
        
        // Verify output shape matches formula
        size_t expected_h = (h + 2*p - k) / s + 1;
        size_t expected_w = (w + 2*p - k) / s + 1;
        
        return output.shape() == shape{1, 64, expected_h, expected_w};
    });
}
```

### Consistency Properties

```cpp
// Property: CPU and GPU implementations must match
TEST_CASE("cpu gpu consistency") {
    property_test([](const tensor& input, const tensor& weights) {
        assume(input.cols() == weights.rows());
        
        auto cpu_result = matmul_cpu(input, weights);
        auto gpu_result = matmul_gpu(input, weights);
        
        // GPU and CPU must produce identical results (within tolerance)
        return tensors_allclose(cpu_result, gpu_result, 
                               /*rtol=*/1e-5, /*atol=*/1e-7);
    });
}

// Property: Different batch sizes produce consistent per-sample results
TEST_CASE("batch independence") {
    property_test([](const tensor& sample) {
        assume(sample.rank() == 3);  // Single sample: CHW
        
        // Run single sample
        auto single_result = model.forward(sample.unsqueeze(0)).squeeze(0);
        
        // Run in batch of 8
        tensor batch = tensor::stack({sample, sample, sample, sample,
                                     sample, sample, sample, sample});
        auto batch_results = model.forward(batch);
        
        // Each batch result must match single sample result
        for (size_t i = 0; i < 8; ++i) {
            if (!tensors_allclose(batch_results[i], single_result, 
                                 /*rtol=*/1e-6, /*atol=*/1e-8)) {
                return false;
            }
        }
        return true;
    });
}
```

### The Five-Minute Rule

If you can't understand what agent-generated code does in 5 minutes, regenerate it with better
structure.

## Performance Guidelines for Generative Models

### Diffusion Models are Different

Unlike language models, diffusion models have fundamentally different performance characteristics:

```cpp
// Diffusion models are FLOPS-bound, not memory-bound
// This changes everything about optimization strategy

// DO: Optimize for compute efficiency
auto optimize_dit_block(const dit_config& config) {
    // Fuse everything possible to maximize FLOPS utilization
    return fused_dit_implementation{
        .use_tensor_cores = true,
        .fuse_all_ops = true,
        .minimize_kernel_launches = true
    };
}

// DON'T: Over-optimize for memory bandwidth (less critical for DiT)
auto optimize_dit_block_wrong(const dit_config& config) {
    // Memory optimizations alone won't help much
    return memory_optimized_dit{
        .use_flash_attention = true,  // Still good, but not sufficient
        .optimize_cache_reuse = true  // Secondary concern
    };
}
```

### Kernel Fusion is King

```cpp
// Example: SVDQuant-style fused quantization
// This single kernel does what would normally take 4 kernel launches
template<int BLOCK_SIZE = 256>
__global__ void fused_svd_quant_kernel(
    const half* __restrict__ input,
    const int4* __restrict__ weight_quantized,
    const half* __restrict__ low_rank_u,
    const half* __restrict__ low_rank_v,
    half* __restrict__ output,
    int M, int N, int rank) {
    
    // Shared memory for input tile
    __shared__ half input_tile[BLOCK_SIZE];
    
    // 1. Load and quantize input (normally kernel 1)
    auto quantized_input = quantize_to_int4(input[threadIdx.x]);
    
    // 2. 4-bit matmul (normally kernel 2)
    auto low_precision_result = int4_matmul_tile(
        quantized_input, weight_quantized);
    
    // 3. Low-rank correction (normally kernel 3)
    auto low_rank_correction = compute_low_rank_correction(
        input_tile, low_rank_u, low_rank_v, rank);
    
    // 4. Add and dequantize (normally kernel 4)
    output[threadIdx.x] = dequantize_and_add(
        low_precision_result, low_rank_correction);
}
```

### Profiling First

```cpp
// Profile with diffusion-specific metrics
{
    fxy::profiler::scope_timer timer("dit_block_forward");
    
    // Track both compute and memory metrics
    auto metrics = fxy::profiler::gpu_metrics{
        .track_tensor_core_usage = true,
        .track_achieved_flops = true,
        .track_memory_bandwidth = true,  // Still track, but not primary
        .track_kernel_launches = true    // Critical for diffusion
    };
    
    output = dit_block(input);
    
    // For diffusion models, we want:
    // - Tensor core usage > 80%
    // - FLOPS utilization > 70%
    // - Minimal kernel launches
}
```

## Configuration Philosophy

### Model Configuration

```cpp
struct model_configuration {
    // Model architecture
    std::size_t num_layers = 12;
    std::size_t hidden_dim = 768;
    std::size_t num_heads = 12;
    std::size_t vocab_size = 50257;
    
    // Inference settings
    std::size_t max_batch_size = 32;
    std::size_t max_sequence_length = 2048;
    
    // Optimization flags
    bool use_flash_attention = true;
    bool use_int8_quantization = false;
    bool enable_cuda_graphs = true;
    
    // Validate all constraints
    auto validate() const -> fxy::status {
    
        if (hidden_dim % num_heads != 0) {
            return fxy::fail("hidden_dim must be divisible by num_heads");
        }
        
        if (max_batch_size == 0) {
            return fxy::fail("max_batch_size must be positive");
        }
        
        return fxy::ok();
    }
};
```

### Runtime Configuration

```cpp
// Parse and validate at startup
auto initialize_inference_runtime(const std::string& config_path) 
    -> fxy::core::result<inference_runtime> {
    
    auto config_data = fxy::core::read_file(config_path);
    if (!config_data) {
        fxy::fatal("Cannot read configuration: {}", config_path);
    }
    
    auto configuration = parse_json_config<model_configuration>(*config_data);
    if (!configuration) {
        fxy::fatal("Invalid configuration: {}", configuration.error());
    }
    
    auto validation = configuration->validate();
    if (!validation) {
        fxy::fatal("Configuration validation failed: {}", validation.error());
    }
    
    return create_runtime(*configuration);
}
```

## Logging

Hierarchical tagging for structured logs:

```cpp
fxy::info("[fxy] [inference] [attention] computing self-attention batch_size={}", 
         batch_size);
fxy::debug("[fxy] [memory] [allocator] allocated {}MB on device {}", 
          size_mb, device_id);
fxy::error("[fxy] [runtime] [error] kernel launch failed: {}", 
          error_message);
```

Format: `[project] [subsystem] [component] message`

## Anti-Patterns to Avoid

### The Kernel Launch Anti-Pattern

```cpp
// BAD: Too many kernel launches for DiT block
for (const auto& layer : dit_layers) {
    auto norm_out = layer_norm(input);       // Launch 1
    auto q = linear(norm_out, q_weight);     // Launch 2
    auto k = linear(norm_out, k_weight);     // Launch 3
    auto v = linear(norm_out, v_weight);     // Launch 4
    auto attn = attention(q, k, v);          // Launch 5
    // ... many more launches
}

// GOOD: Fused DiT block
auto output = fused_dit_forward(input, all_weights);  // Single launch
```

### The Quantization Without Fusion Anti-Pattern

```cpp
// BAD: Naive quantization adds overhead
auto quantized_input = quantize_to_int4(input);        // Extra launch
auto result = int4_matmul(quantized_input, weights);   // Main compute
auto output = dequantize(result);                      // Extra launch

// GOOD: Fused quantization (SVDQuant style)
auto output = fused_quant_matmul(input, weights_int4, low_rank_branch);
```

### The Copy Anti-Pattern

```cpp
// BAD: Unnecessary copies even with AAA
auto get_predictions(tensor logits) -> tensor {  // Copy!
    auto softmaxed = softmax(logits);           // Another copy!
    return softmaxed;
}

// GOOD: Move semantics with AAA
auto get_predictions(tensor&& logits) -> tensor {
    return softmax(std::move(logits));  // Reuse memory
}
```

### The Missing :: Prefix Anti-Pattern

```cpp
// BAD: Ambiguous global calls
memcpy(dest, src, size);  // Which memcpy?
printf("Done\n");          // Which printf?

// GOOD: Explicit global namespace
::memcpy(dest, src, size);
::printf("Done\n");

// BETTER: Modern C++
std::memcpy(dest, src, size);
std::println("Done");
```

## Summary

In an agent-heavy ML inference codebase for generative models:

1. **Every name must be globally unambiguous**
2. **Diffusion models are FLOPS-bound - optimize compute, not just memory**
3. **Every kernel launch costs microseconds - fusion is critical**
4. **Every numerical computation must be tested with properties, not examples**
5. **AAA and initializer lists make code cleaner and safer**

Write code as if 100 agents will be pattern-matching against it tomorrow, and a tired human will be
debugging a production model failure at 3am. Because both will happen.

For diffusion models specifically: remember that you're compute-bound even at batch size 1. Study
implementations like SVDQuant/Nunchaku that achieve 3-10× speedups through aggressive kernel fusion
and quantization co-design.

The most dangerous bugs in ML systems are not crashes - they're silent numerical corruptions that
produce plausible but wrong results. Property-based testing is your defense against these disasters.

## Required Reading/Watching

### ML Performance

- [SVDQuant: 4-Bit Diffusion Models](https://arxiv.org/abs/2411.05007) - SOTA quantization for
  diffusion
- [FlashAttention: Fast and Memory-Efficient Attention](https://arxiv.org/abs/2205.14135)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)

### Property-Based Testing

- [QuickCheck: A Lightweight Tool for Random Testing](https://dl.acm.org/doi/10.1145/351240.351266)
- [Hypothesis: Property-based testing for Python](https://hypothesis.readthedocs.io/)
- [Testing with Expectations - Jane Street Blog](https://blog.janestreet.com/testing-with-expectations/)

### Systems Programming for ML

- [The Architecture of Open Source Applications](https://aosabook.org/en/)
- [CppCon Talks on ML and Performance](https://www.youtube.com/user/CppCon)
- [Stanford CS courses on ML Systems](https://cs.stanford.edu/)

## Living List of Great Code

**Tier 1** (Study Every Line)

- [GGML](https://github.com/ggerganov/ggml) - Tensor library for LLMs, exemplary C code
- [ONNXRuntime](https://github.com/microsoft/onnxruntime) - Production inference, great abstractions
- [XLA](https://github.com/openxla/xla) - Google's ML compiler, state of the art
- [Nunchaku](https://github.com/mit-han-lab/nunchaku) - SOTA 4-bit diffusion inference with kernel
  fusion

**Tier 2** (Domain Excellence)

- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA's LLM inference
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient LLM inference on CPU
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) - Universal LLM deployment engine
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention innovation

**Tier 3** (Specific Excellence)

- [SVDQuant/DeepCompressor](https://github.com/mit-han-lab/deepcompressor) - Quantization library
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - IO-aware attention
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - Optimized transformer inference
- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) - NVIDIA's transformer kernels
- [DeepSpeed-Inference](https://github.com/microsoft/DeepSpeed) - Distributed inference

**Property Testing Examples**

- [ArrayFire Tests](https://github.com/arrayfire/arrayfire/tree/master/test) - Excellent property
  tests
- [JAX Tests](https://github.com/google/jax/tree/main/tests) - Comprehensive numerical properties
- [PyTorch Tests](https://github.com/pytorch/pytorch/tree/main/test) - Large-scale testing
  infrastructure

**Kernel Implementation Studies**

- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA templates for linear algebra
- [Triton](https://github.com/openai/triton) - Python-like GPU kernel language
- [cccl](https://github.com/NVIDIA/cccl) - CUDA building blocks

**Required Papers**

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The transformer paper
- [What Every Computer Scientist Should Know About Floating-Point](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- [The Roofline Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf) -
  Performance analysis

**What Makes Inference Code "Great"**

1. **Kernel fusion mastery** - Minimizes launches, maximizes FLOPS utilization
2. **Numerical correctness** - Satisfies mathematical properties via property tests
3. **Quantization co-design** - Algorithms and kernels designed together (like SVDQuant)
4. **Hardware saturation** - Achieves >70% of theoretical FLOPS on target hardware
5. **Production readiness** - Handles errors, monitoring, deployment
