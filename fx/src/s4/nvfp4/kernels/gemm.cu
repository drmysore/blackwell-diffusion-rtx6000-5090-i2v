#include <cuda.h>
#include <cuda_fp8.h>
#include <cutlass/arch/mma_sm120.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/numeric_types.h>

namespace s4::nvfp4::kernels {

// no abstractions over physics
using nvfp4_t = cutlass::float_e2m1_t;

template <int M, int N, int K>
__global__ void gemm_native(const nvfp4_t* __restrict__ a, const nvfp4_t* __restrict__ b,
                            __half* __restrict__ c, const __half* __restrict__ scale_a,
                            const __half* __restrict__ scale_b, int lda, int ldb, int ldc) {

  // blackwell tensor cores eat nvfp4 natively
  // no conversion, no overhead, no compromise

  using namespace cute;

  // tile configuration for rtx 5090
  using TileShape = Shape<_256, _128, _64>;
  using ClusterShape = Shape<_2, _2, _1>;

  // native nvfp4 mma on sm_120
  using MMA = SM120_16x8x32_F32E2M1E2M1F32_TN;

  // shared memory for cooperative fetching
  extern __shared__ nvfp4_t smem[];

  // thread block coordinates
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  // global tile coordinates
  const int tile_m = blockIdx.x * TileShape::kM;
  const int tile_n = blockIdx.y * TileShape::kN;

  // tma descriptors for async copy
  CUtensorMap* tma_a = nullptr;
  CUtensorMap* tma_b = nullptr;

  // create tma descriptors (blackwell feature)
  if (threadIdx.x == 0) {
    // 5d tensor for swizzled nvfp4 layout
    cudaTmaCreateDescriptor(&tma_a, a,
                            CUDA_R_8U,  // nvfp4 packed
                            5,          // dimensions
                            ...         // blackwell-optimized layout
    );
  }

  // cooperative matrix fragments
  fragment<matrix_a> a_frag;
  fragment<matrix_b> b_frag;
  fragment<accumulator> c_frag;

// main loop - no dequantization needed
#pragma unroll
  for (int k = 0; k < K; k += TileShape::kK) {
    // async copy with tma
    copy_async(smem_a, tma_a + k, ...);
    copy_async(smem_b, tma_b + k, ...);

    // commit and wait
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // load from smem to registers
    load_matrix_sync(a_frag, smem_a, ...);
    load_matrix_sync(b_frag, smem_b, ...);

    // native nvfp4 tensor core operation
    // hardware handles everything
    mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

// epilogue with scaling
#pragma unroll
  for (int i = 0; i < c_frag.num_elements; i++) {
    c_frag.x[i] *= scale_a[...] * scale_b[...];
  }

  // store to global
  store_matrix_sync(c + ..., c_frag, ldc, mem_row_major);
}

// launch configuration for blackwell
void launch_gemm(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  // validate blackwell
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  if (prop.major < 12) {
    throw std::runtime_error("nvfp4 requires sm_120+");
  }

  // dimensions
  const int M = a.size(0);
  const int N = b.size(1);
  const int K = a.size(1);

  // tile configuration
  dim3 grid(M / 256, N / 128);
  dim3 block(128, 2);  // 2 warps per cta

  // shared memory for double buffering
  size_t smem_size = 2 * (256 * 64 + 128 * 64) * sizeof(nvfp4_t);

  // launch with optimal l2 cache config
  cudaFuncSetAttribute(gemm_native<256, 128, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  gemm_native<256, 128, 64><<<grid, block, smem_size>>>(
      a.data_ptr<nvfp4_t>(), b.data_ptr<nvfp4_t>(), c.data_ptr<__half>(), ..., M, N, K);
}

}  // namespace s4::nvfp4::kernels
