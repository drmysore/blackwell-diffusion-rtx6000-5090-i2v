#include "helper.h"

#include <cstdint>

#include <spdlog/spdlog.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"
#include "src/fx/examples/nvfp4/gemm_nvfp4_nvfp4_nvidia.cuh"

using namespace cute;

// A matrix configuration
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // Element type for A matrix operand
using LayoutATag = cutlass::layout::RowMajor;                  // Layout type for A matrix operand
constexpr int AlignmentA =
    32;  // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // Element type for B matrix operand
using LayoutBTag = cutlass::layout::ColumnMajor;               // Layout type for B matrix operand
constexpr int AlignmentB =
    32;  // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using ElementD = cutlass::float_e2m1_t;     // Element type for D matrix operand
using ElementSFD = cutlass::float_ue8m0_t;  // Element type for SFD matrix operand

using ElementC = cutlass::bfloat16_t;          // Element type for C matrix operand
using LayoutCTag = cutlass::layout::RowMajor;  // Layout type for C matrix operand
using LayoutDTag = cutlass::layout::RowMajor;  // Layout type for D matrix operand
using LayoutSFDTag = LayoutDTag;  // Layout type for SFD should be same as D matrix operand

constexpr int AlignmentD =
    128 / cutlass::sizeof_bits<ElementD>::value;  // Memory access granularity/alignment of C matrix
                                                  // in units of elements (up to 16 bytes)
constexpr int AlignmentC =
    128 / cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of C matrix
                                                  // in units of elements (up to 16 bytes)
// Kernel functional config
using ElementAccumulator = float;      // Element type for internal accumulation
using ElementCompute = float;          // Element type for internal accumulation
using ArchTag = cutlass::arch::Sm120;  // Tag indicating the minimum SM that supports the intended
                                       // feature
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;  // Operator class tag

// Kernel Perf config
using ThreadBlockShape = Shape<_128, _128, _128>;  // Threadblock's tile size
using ClusterShape = Shape<_1, _1, _1>;            // Shape of the threadblocks in a cluster

constexpr int InputSFVectorSize = 16;
constexpr int OutputSFVectorSize = InputSFVectorSize;

// D = alpha * acc + beta * C
//      With BlockScaleFactor generation.
using FusionOperation =
    cutlass::epilogue::fusion::LinCombBlockScaleFactor<OutputSFVectorSize, ElementD, ElementCompute,
                                                       ElementSFD, LayoutSFDTag, ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto, FusionOperation>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<std::int64_t>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<std::int64_t, std::int64_t, std::int64_t, std::int64_t>,  // problem shape...
    CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::
    LayoutSFA;  // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.

using StrideB = typename Gemm::GemmKernel::StrideB;
using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::
    LayoutSFB;  // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.

using StrideC = typename Gemm::GemmKernel::StrideC;
using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));

using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));

using FusionOp = typename Gemm::EpilogueOutputOp;
constexpr bool IsBlockScaleSupported = FusionOp::IsBlockScaleSupported;
using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
using LayoutSFD = typename SfdOutputCfg::LayoutSF;

auto stride_A = StrideA{};
auto layout_A = LayoutA{};
auto layout_SFA = LayoutSFA{};

auto stride_B = StrideB{};
auto layout_B = LayoutB{};
auto layout_SFB = LayoutSFB{};

auto stride_C = StrideC{};
auto layout_C = LayoutC{};

auto stride_D = StrideD{};
auto layout_D = LayoutD{};
auto layout_SFD = LayoutSFD{};

auto seed = std::uint64_t{};

// The HostTensors are only used for allocating memory on host and device, and transferring data
// between host and device Use cute::Tensor and cute::Layout for iterating thru the matrix elements
cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;

cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;

cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;

// Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
cutlass::HostTensor<ElementSFD, cutlass::layout::PackedVectorLayout> block_SFD;

// Reference Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_reference_D;
cutlass::HostTensor<ElementSFD, cutlass::layout::PackedVectorLayout> block_reference_SFD;

// Matrix-wide normalization constant
cutlass::HostTensor<ElementCompute, cutlass::layout::PackedVectorLayout> block_Normconst;

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

template <typename Element, typename Layout>
auto initialize_block(cutlass::TensorView<Element, Layout> view, uint64_t seed) -> bool {

  auto scope_min = double{};
  auto scope_max = double{};
  constexpr std::int64_t bits_input = cutlass::sizeof_bits<Element>::value;

  if constexpr (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  } else if constexpr (bits_input <= 6) {
    scope_max = 2;
    scope_min = -2;
  } else if constexpr (bits_input <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
      scope_max = 4;
      scope_min = 1;
    } else {
      scope_max = 1;
      scope_min = -1;
    }
  } else {
    scope_max = 4;
    scope_min = -4;
  }

  cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, 0);
  return true;
}

auto initialize(fx::nvidia::examples::options options) -> void {
  // using namespace cute;

  // For SFA and SFB tensors layouts
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  // For SFD tensor layout
  using Sm1xxBlockScaledOutputConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  layout_A = cute::make_layout(make_shape(options.m, options.k, 1), stride_A);
  layout_B = cute::make_layout(make_shape(options.n, options.k, 1), stride_B);
  layout_C = cute::make_layout(make_shape(options.m, options.n, 1), stride_C);
  layout_D = cute::make_layout(make_shape(options.m, options.n, 1), stride_D);

  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(options.m, options.n, options.k, 1));

  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(options.m, options.n, options.k, 1));

  layout_SFD =
      SfdOutputCfg::tile_atom_to_shape_SFD(cute::make_shape(options.m, options.n, options.k, 1));

  block_A.reset(cutlass::make_Coord(size(layout_A)));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_C.reset(cutlass::make_Coord(size(layout_C)));
  block_D.reset(cutlass::make_Coord(size(layout_D)));
  block_reference_D.reset(cutlass::make_Coord(size(layout_D)));
  block_reference_SFD.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));
  block_Normconst.reset(cutlass::make_Coord(1));
  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
  block_SFD.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));

  initialize_block(block_A.host_view(), seed + 2021);
  initialize_block(block_B.host_view(), seed + 2022);
  initialize_block(block_C.host_view(), seed + 2023);
  initialize_block(block_SFA.host_view(), seed + 2024);
  initialize_block(block_SFB.host_view(), seed + 2025);
  block_Normconst.at(cutlass::make_Coord(0)) = 2;

  block_A.sync_device();
  block_B.sync_device();
  block_C.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();
  block_SFD.sync_device();
  block_Normconst.sync_device();
}

auto arguments_from_options(fx::nvidia::examples::options options) noexcept ->
    typename Gemm::Arguments {

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, 1},
      {// Mainloop arguments
       block_A.device_data(), stride_A, block_B.device_data(), stride_B, block_SFA.device_data(),
       layout_SFA, block_SFB.device_data(), layout_SFB},
      {// Epilogue arguments
       {options.alpha, options.beta},
       block_C.device_data(),
       stride_C,
       block_D.device_data(),
       stride_D}};

  if constexpr (IsBlockScaleSupported) {
    arguments.epilogue.thread.block_scale_factor_ptr = block_SFD.device_data();
    arguments.epilogue.thread.norm_constant_ptr = block_Normconst.device_data();
  }

  return arguments;
}

auto verify(fx::nvidia::examples::options options) noexcept -> bool {

  auto tensor_A = cute::make_tensor(make_iterator(block_A.host_data()), layout_A);
  auto tensor_SFA = cute::make_tensor(block_SFA.host_data(), layout_SFA);

  auto tensor_B = cute::make_tensor(make_iterator(block_B.host_data()), layout_B);
  auto tensor_SFB = cute::make_tensor(block_SFB.host_data(), layout_SFB);

  cutlass::reference::host::GettBlockScalingMainloopParams<ElementAccumulator, decltype(tensor_A),
                                                           decltype(tensor_SFA), decltype(tensor_B),
                                                           decltype(tensor_SFB)>
      mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

  auto tensor_C = cute::make_tensor(make_iterator(block_C.host_data()), layout_C);

  auto tensor_D = cute::make_tensor(make_iterator(block_reference_D.host_data()), layout_D);
  auto tensor_SFD = cute::make_tensor(block_reference_SFD.host_data(), layout_SFD);

  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementAccumulator, ElementAccumulator, ElementAccumulator, decltype(tensor_C),
      decltype(tensor_D), decltype(tensor_SFD), cute::Int<OutputSFVectorSize>,
      cutlass::reference::host::SfStrategy::SfDGen>
      epilogue_params{options.alpha, options.beta, tensor_C,
                      tensor_D,      tensor_SFD,   block_Normconst.at(cutlass::make_Coord(0))};

  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  block_D.sync_host();

  bool passed =
      cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_D.host_view());

  passed &= (cutlass::reference::host::TensorNorm(block_reference_D.host_view()) > 0);
  passed &= (cutlass::reference::host::TensorNorm(block_D.host_view()) > 0);

  return passed;
}

template <typename Gemm>
auto run(fx::nvidia::examples::options options) -> fx::nvidia::examples::result {

  initialize(options);
  auto arguments = arguments_from_options(options);
  std::size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = cutlass::device_memory::allocation<uint8_t>{workspace_size};

  spdlog::info("[fx] [nvidia] [nvfp4] workspace allocated with size {}", workspace_size);

  auto gemm = Gemm{};
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());

  ::cudaDeviceSynchronize();

  auto result = fx::nvidia::examples::result{};
  // result.passed = verify(options);
  // spdlog::info("[fx] [nvidia] [nvfp4] test passed: {}", result.passed);
  // if (!result.passed) {
  //   ::exit(-1);
  // }

  if (options.iterations > 0) {
    auto timer = GpuTimer{};
    timer.start();

    for (std::size_t idx = 0; idx < options.iterations; ++idx) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }

    timer.stop();

    const auto elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);


    spdlog::info("[fx] [nvidia] problem size: {}x{}x{}", options.m, options.n, options.k);
    spdlog::info("[fx] [nvidia] average: {}ms", result.avg_runtime_ms);
    spdlog::info("[fx] [nvidia] TFLOPs: {}", result.gflops / 1000.0);
  }

  return result;
}

namespace fx::nvidia::examples {
auto launch(fx::nvidia::examples::options options) -> fx::nvidia::examples::result {
  auto rv = run<Gemm>(options);
  return rv;
}
}  // namespace fx::nvidia::examples
