#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void launch_cosine_sampler_forward_kernel(
    const torch::TensorBase &output, const torch::TensorBase &input, const torch::TensorBase &grid, const torch::TensorBase &offset,
    int64_t padding_mode, bool align_corners);


void launch_cosine_sampler_backward_kernel(
    const torch::TensorBase& grad_input, const torch::TensorBase &grad_grid,
    const torch::TensorBase& grad_output, const torch::TensorBase& input,
    const torch::TensorBase& grid, const torch::TensorBase &offset, int64_t padding_mode, bool align_corners,
    bool input_requires_grad);


void launch_cosine_sampler_backward_backward_kernel(
    const torch::TensorBase& grad_input,
    const torch::TensorBase& grad_grid,
    const torch::TensorBase& grad_grad_out,
    const torch::TensorBase& input,
    const torch::TensorBase& grid,
    const torch::TensorBase& grad_out_input,
    const torch::TensorBase& grad_out_grid,
    const torch::TensorBase& grad_output,
    const torch::TensorBase &offset,
    int64_t padding_mode,
    const bool align_corners,
    const bool input_requires_grad);

void launch_cosine_sampler_backward_backward_backward_kernel(
    const torch::TensorBase& gInput,
    const torch::TensorBase& ggOut,
    const torch::TensorBase& input,
    const torch::TensorBase& grid,
    const torch::TensorBase& gOut,
    const torch::TensorBase& gOutggOut,
    const torch::TensorBase& gOutGrid,
    const torch::TensorBase& gOutgGrid,
    const torch::TensorBase &offset,
    int64_t padding_mode,
    const bool align_corners,
    const bool input_requires_grad);

torch::Tensor cosine_sampler_forward(torch::Tensor input, torch::Tensor grid, torch::Tensor offset,
                    int64_t padding_mode, bool align_corners){

    CHECK_INPUT(input)
    CHECK_INPUT(grid)
    CHECK_INPUT(offset)
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input)) ;

    auto in_size = input.sizes();
    auto grid_size = grid.sizes();
    auto output = torch::empty({in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());

    launch_cosine_sampler_forward_kernel(output, input, grid, offset, padding_mode, align_corners);
    return output;

}

std::tuple<torch::Tensor, torch::Tensor> cosine_sampler_backward(torch::Tensor grad_output, torch::Tensor input,
                                                                 torch::Tensor grid, torch::Tensor offset, int64_t padding_mode, bool align_corners,
                                                                 bool input_requires_grad) {
  CHECK_INPUT(grad_output)
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  CHECK_INPUT(offset)
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  torch::Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return torch::Tensor();
    }
  })();
  auto grad_grid = torch::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_cosine_sampler_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, offset, padding_mode, align_corners, input_requires_grad);
  return std::make_tuple(grad_input, grad_grid);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cosine_sampler_backward_backward(torch::Tensor grad_out_input, torch::Tensor grad_out_grid,
                                                                          torch::Tensor input, torch::Tensor grid, torch::Tensor grad_output,
                                                                          torch::Tensor offset,
                                                                          int64_t padding_mode, bool align_corners, bool input_requires_grad) {
  CHECK_INPUT(grad_out_input)
  CHECK_INPUT(grad_out_grid)
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  CHECK_INPUT(grad_output)
  CHECK_INPUT(offset)
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  
  auto grad_input = torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grid = torch::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grad_out = torch::zeros_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_cosine_sampler_backward_backward_kernel(grad_input, grad_grid, grad_grad_out, input, grid,
                                                 grad_out_input, grad_out_grid, grad_output, offset,
                                                 padding_mode, align_corners, input_requires_grad);
  return std::make_tuple(grad_input, grad_grid, grad_grad_out);
}

std::tuple<torch::Tensor, torch::Tensor> cosine_sampler_backward_backward_backward(torch::Tensor input, torch::Tensor grid, torch::Tensor gOut, torch::Tensor gOutggOut,
                                             torch::Tensor gOutGrid, torch::Tensor gOutgGrid,
                                              torch::Tensor offset, int64_t padding_mode, bool align_corners, bool input_requires_grad) {
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  CHECK_INPUT(gOut)
  CHECK_INPUT(gOutggOut)
  CHECK_INPUT(gOutgGrid)
  CHECK_INPUT(gOutGrid)
  CHECK_INPUT(offset)
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  
  auto gInput = torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto ggOut = torch::zeros_like(gOut, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  
  launch_cosine_sampler_backward_backward_backward_kernel(gInput, ggOut,  input, grid, 
                                                 gOut, gOutggOut, gOutGrid,gOutgGrid, offset,
                                                 padding_mode, align_corners, input_requires_grad);
  return std::make_tuple(gInput, ggOut);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cosine_sampler_forward, "Cosine sampler forward (CUDA)");
    m.def("backward", &cosine_sampler_backward, "Cosine sampler backward (CUDA)");
    m.def("backward_backward", &cosine_sampler_backward_backward, "Cosine sampler backward backward (CUDA)");
    m.def("backward_backward_backward", &cosine_sampler_backward_backward_backward, "Cosine sampler backward backward backward (CUDA)");
}