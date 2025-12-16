#include <torch/extension.h>

torch::Tensor tensor_add(torch::Tensor a, torch::Tensor b) {
  return a + b;
}

torch::Tensor custom_gelu(torch::Tensor input) {
  return 0.5 * input *
         (1.0 + torch::tanh(std::sqrt(2.0 / M_PI) * (input + 0.044715 * torch::pow(input, 3))));
}

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
  return torch::matmul(a, b);
}

PYBIND11_MODULE(extension, m) {
  m.def("tensor_add", &tensor_add, "Add two tensors");
  m.def("custom_gelu", &custom_gelu, "Custom GELU activation");
  m.def("matmul", &matmul, "Matrix multiplication");
}
