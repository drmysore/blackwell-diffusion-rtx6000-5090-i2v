#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <torch/types.h>

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace s4::nvfp4 {

torch::Tensor blackwell_fp4_gemm_prequantized(
    torch::Tensor A_fp4,     // [M*K/2] packed FP4 data (2 values per byte)
    torch::Tensor B_fp4,     // [N*K/2] packed FP4 data
    torch::Tensor A_scales,  // FP8 E4M3 scale factors
    torch::Tensor B_scales,  // FP8 E4M3 scale factors
    int64_t M, int64_t N, int64_t K, float alpha = 1.0f, float beta = 0.0f) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

  using namespace cute;

  TORCH_CHECK(A_fp4.is_cuda() && B_fp4.is_cuda(), "FP4 data must be CUDA tensors");
  TORCH_CHECK(A_scales.is_cuda() && B_scales.is_cuda(), "Scale factors must be CUDA tensors");
  
  TORCH_CHECK(A_fp4.dtype() == torch::kUInt8 && B_fp4.dtype() == torch::kUInt8,
              "FP4 data must be uint8");
  
  TORCH_CHECK(A_scales.dtype() == torch::kUInt8 && B_scales.dtype() == torch::kUInt8,
              "Scale factors must be uint8 (FP8 E4M3)");

  // Validate sizes
  TORCH_CHECK(A_fp4.numel() == (M * K + 1) / 2, "A_fp4 size mismatch");
  TORCH_CHECK(B_fp4.numel() == (N * K + 1) / 2, "B_fp4 size mismatch");

  // Alignment requirements
  TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 128 == 0, "M, N, K must be divisible by 128");

  auto stream = at::cuda::getCurrentCUDAStream();

  // CUTLASS type definitions
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  constexpr int AlignmentA = 32;
  constexpr int AlignmentB = 32;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD = AlignmentC;

  using MmaTileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  constexpr int InputSFVectorSize = 16;

  // Epilogue configuration
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  // Mainloop configuration
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Set up scale factor layouts
  using Sm1xxConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  auto problem_shape = cute::make_shape(M, N, K, 1);
  auto computed_layout_SFA = Sm1xxConfig::tile_atom_to_shape_SFA(problem_shape);
  auto computed_layout_SFB = Sm1xxConfig::tile_atom_to_shape_SFB(problem_shape);

  // Verify scale factor sizes
  size_t expected_sfa_size = size(filter_zeros(computed_layout_SFA));
  size_t expected_sfb_size = size(filter_zeros(computed_layout_SFB));

  TORCH_CHECK(A_scales.numel() == expected_sfa_size, "A scale factor size mismatch: expected ",
              expected_sfa_size, ", got ", A_scales.numel());
  
  TORCH_CHECK(B_scales.numel() == expected_sfb_size, "B scale factor size mismatch: expected ",
              expected_sfb_size, ", got ", B_scales.numel());

  // Create output tensors
  auto C = torch::zeros({M, N}, torch::dtype(torch::kBFloat16).device(A_fp4.device()));
  auto D = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A_fp4.device()));

  // Set up strides
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  // Configure GEMM arguments with pre-quantized data
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<ElementA::DataType*>(A_fp4.data_ptr()), stride_A,
       reinterpret_cast<ElementB::DataType*>(B_fp4.data_ptr()), stride_B,
       reinterpret_cast<ElementA::ScaleFactorType*>(A_scales.data_ptr()), computed_layout_SFA,
       reinterpret_cast<ElementB::ScaleFactorType*>(B_scales.data_ptr()), computed_layout_SFB},
      {{alpha, beta},
       reinterpret_cast<ElementC*>(C.data_ptr()),
       stride_C,
       reinterpret_cast<ElementD*>(D.data_ptr()),
       stride_D}};

  // Initialize and run GEMM
  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace =
      torch::empty({(int64_t)workspace_size}, torch::dtype(torch::kUInt8).device(A_fp4.device()));

  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS cannot implement this configuration");

  status = gemm.initialize(arguments, workspace.data_ptr<uint8_t>(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize CUTLASS GEMM");

  status = gemm(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run CUTLASS GEMM");

  return D;

#else
  TORCH_CHECK(false, "Blackwell FP4 GEMM requires CUDA 12.8+ and SM120/SM121");
  return torch::Tensor();
#endif
}


}  // namespace s4::nvfp4
