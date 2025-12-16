import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, NVFP4BlockScaling
import time
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple
import pynvml


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single configuration"""

    config: str
    batch_size: int
    seq_len: int
    hidden_dim: int
    dtype: str
    latency_ms: float
    throughput_tflops: float
    memory_gb: float
    bandwidth_gb_s: float
    power_watts: float = 0.0
    efficiency_tflops_per_watt: float = 0.0
    rmse: float = 0.0
    snr_db: float = 0.0


class NVMLMonitor:
    """Monitor GPU power and memory during benchmarks"""

    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.baseline_power = self._get_power()

    def _get_power(self):
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W

    def get_power_delta(self):
        return self._get_power() - self.baseline_power

    def get_memory_used(self):
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return info.used / (1024**3)  # bytes to GB


@contextmanager
def cuda_timer():
    """Accurate CUDA timing with events"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    globals()["_last_timing"] = elapsed_ms


def benchmark_nvfp4():
    """Comprehensive NVFP4 benchmark for Blackwell architectures"""

    # Initialize monitoring
    monitor = NVMLMonitor()
    results = []

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count

    print(f"=" * 80)
    print(f"NVFP4 BLACKWELL BENCHMARK")
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: sm_{compute_capability[0]}{compute_capability[1]}")
    print(f"SM Count: {sm_count}")
    print(f"=" * 80)

    # Test configurations - sizes relevant to FLUX/WAN models
    configs = [
        # (batch, seq_len, hidden_dim, out_dim) - all dims multiple of 16 for NVFP4
        ("FLUX-dev context", 1, 256, 3072, 3072),  # Typical FLUX.1-dev layer
        ("FLUX-schnell", 1, 512, 2048, 2048),  # FLUX.1-schnell attention
        ("WAN 2.2 block", 1, 1024, 1536, 1536),  # WAN video model
        ("Multi-batch FLUX", 4, 256, 3072, 3072),  # Batched inference
        ("Large context", 1, 2048, 4096, 4096),  # Extended context
        ("Production batch", 8, 512, 2048, 2048),  # Production workload
        ("Extreme VRAM", 16, 1024, 8192, 8192),  # Push VRAM limits
    ]

    # FP4 configuration
    fp4_format = Format.E2M1
    fp4_recipe = NVFP4BlockScaling(fp4_format=fp4_format)

    # Warmup
    print("\nWarming up GPU...")
    warmup_x = torch.randn(1, 128, 768, dtype=torch.bfloat16).cuda()
    warmup_linear = te.Linear(768, 768, bias=True, params_dtype=torch.bfloat16).cuda()
    for _ in range(50):
        with torch.no_grad():
            with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
                _ = warmup_linear(warmup_x)
    torch.cuda.synchronize()

    print(
        f"\n{'Config':<20} {'Shape':<20} {'BF16 ms':<10} {'FP4 ms':<10} "
        f"{'Speedup':<8} {'TFLOPS':<10} {'GB/s':<10} {'W':<8} {'RMSE':<10}"
    )
    print("-" * 120)

    for config_name, batch, seq_len, hidden_in, hidden_out in configs:
        torch.cuda.empty_cache()

        # Create tensors
        x = torch.randn(batch, seq_len, hidden_in, dtype=torch.bfloat16).cuda()
        linear = te.Linear(
            hidden_in, hidden_out, bias=True, params_dtype=torch.bfloat16
        ).cuda()

        # Memory usage
        mem_before = monitor.get_memory_used()

        # BF16 baseline timing
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                out_bf16 = linear(x)
            torch.cuda.synchronize()

            # Benchmark
            with cuda_timer():
                for _ in range(100):
                    out_bf16 = linear(x)
        bf16_ms = _last_timing / 100

        # FP4 timing
        with torch.no_grad():
            with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
                # Warmup
                for _ in range(10):
                    out_fp4 = linear(x)
                torch.cuda.synchronize()

                # Benchmark with power monitoring
                power_samples = []
                with cuda_timer():
                    for _ in range(100):
                        out_fp4 = linear(x)
                        if _ % 10 == 0:
                            power_samples.append(monitor.get_power_delta())
        fp4_ms = _last_timing / 100
        avg_power = np.mean(power_samples) if power_samples else 0

        # Calculate metrics
        speedup = bf16_ms / fp4_ms
        flops = 2 * batch * seq_len * hidden_in * hidden_out  # GEMM FLOPs
        tflops = (flops / fp4_ms) / 1e9  # ms to s, then to TFLOPS

        # Memory bandwidth (approximate)
        bytes_moved = (
            batch * seq_len * hidden_in + hidden_in * hidden_out
        ) * 2  # FP4 ~ 0.5 bytes
        bandwidth_gb_s = (bytes_moved / fp4_ms) / 1e6  # ms to s, bytes to GB

        mem_after = monitor.get_memory_used()
        mem_used = mem_after - mem_before

        # Accuracy metrics
        with torch.no_grad():
            out_bf16_test = linear(x)
            with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
                out_fp4_test = linear(x)

        rmse = torch.sqrt(torch.mean((out_bf16_test - out_fp4_test) ** 2)).item()
        signal_power = torch.mean(out_bf16_test**2)
        noise_power = torch.mean((out_bf16_test - out_fp4_test) ** 2)
        snr = (
            10 * torch.log10(signal_power / noise_power)
            if noise_power > 0
            else float("inf")
        )

        # Store result
        result = BenchmarkResult(
            config=config_name,
            batch_size=batch,
            seq_len=seq_len,
            hidden_dim=hidden_in,
            dtype="NVFP4",
            latency_ms=fp4_ms,
            throughput_tflops=tflops,
            memory_gb=mem_used,
            bandwidth_gb_s=bandwidth_gb_s,
            power_watts=avg_power,
            efficiency_tflops_per_watt=tflops / avg_power if avg_power > 0 else 0,
            rmse=rmse,
            snr_db=snr.item() if not torch.isinf(snr) else 99.99,
        )
        results.append(result)

        # Print row
        shape_str = f"{batch}x{seq_len}x{hidden_in}"
        print(
            f"{config_name:<20} {shape_str:<20} {bf16_ms:<10.3f} {fp4_ms:<10.3f} "
            f"{speedup:<8.2f}x {tflops:<10.2f} {bandwidth_gb_s:<10.1f} "
            f"{avg_power:<8.1f} {rmse:<10.6f}"
        )

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    avg_speedup = np.mean([r.latency_ms for r in results if "BF16" not in r.config])
    max_tflops = max(r.throughput_tflops for r in results)
    max_bandwidth = max(r.bandwidth_gb_s for r in results)
    avg_snr = np.mean([r.snr_db for r in results if r.snr_db < 99])

    print(f"Peak TFLOPS: {max_tflops:.2f}")
    print(f"Peak Bandwidth: {max_bandwidth:.1f} GB/s")
    print(f"Average SNR: {avg_snr:.1f} dB")

    # Check for sm_120 tensor core utilization
    if compute_capability[0] == 12:  # Blackwell
        theoretical_fp4_tflops = sm_count * 2 * 1.5  # Rough estimate for RTX 5090
        utilization = (max_tflops / theoretical_fp4_tflops) * 100
        print(f"Estimated Tensor Core Utilization: {utilization:.1f}%")

        if utilization < 50:
            print("\n⚠️  LOW UTILIZATION DETECTED!")
            print("Possible issues:")
            print("- NVFP4 may not be using tcgen05.mma instructions")
            print("- Cluster shape might be (1,1,1) instead of optimal (2,2,1)")
            print("- Not using persistent kernels or warp specialization")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR FLUX OPTIMIZATION")
    print("=" * 80)

    # Find sweet spot for FLUX
    flux_results = [r for r in results if "FLUX" in r.config]
    if flux_results:
        best_flux = max(flux_results, key=lambda r: r.throughput_tflops)
        print(f"Best FLUX config: {best_flux.config}")
        print(f"  - Latency: {best_flux.latency_ms:.3f} ms")
        print(f"  - Throughput: {best_flux.throughput_tflops:.2f} TFLOPS")
        print(f"  - Memory: {best_flux.memory_gb:.2f} GB")
        print(f"  - RMSE: {best_flux.rmse:.6f}")

        # Projection for full FLUX pipeline
        layers_in_flux = 38  # Approximate transformer blocks in FLUX.1-dev
        projected_latency = best_flux.latency_ms * layers_in_flux
        print(f"\nProjected FLUX.1-dev full pipeline:")
        print(f"  - {projected_latency:.1f} ms total latency")
        print(
            f"  - Target: <200ms ✓"
            if projected_latency < 200
            else f"  - Target: <200ms ✗ ({projected_latency / 200:.1f}x over)"
        )

    return results


if __name__ == "__main__":
    try:
        results = benchmark_nvfp4()

        # Optional: Save results for tracking
        import json

        with open("nvfp4_benchmark_results.json", "w") as f:
            json.dump(
                [{k: v for k, v in r.__dict__.items()} for r in results], f, indent=2
            )
        print("\n✓ Results saved to nvfp4_benchmark_results.json")

    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        print("\nPossible issues:")
        print("- Transformer Engine not installed: pip install transformer-engine")
        print("- Not on Blackwell GPU (requires sm_120/sm_120a)")
        print("- CUDA 13 / PyTorch 2.5+ required")
        print("- pynvml not installed: pip install pynvml")
