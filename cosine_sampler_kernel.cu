/*
  Based on https://github.com/pytorch/pytorch/blob/v1.12.0/aten/src/ATen/native/cuda/GridSampler.cu
*/

#include <torch/extension.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <math_constants.h>

// like at::native::safe_add_3d but without bound check
template<typename scalar_t, typename index_t>
static __forceinline__ __device__
void add_3d(scalar_t *data, int d, int h, int w,
            int sD, int sH, int sW,
            scalar_t delta,
            const index_t NC_offset_inp,
            const index_t memory_span) {
  at::native::fastAtomicAdd(data,
                NC_offset_inp + d * sD + h * sH + w * sW,
                memory_span,
                delta,
                true);
}

template<typename scalar_t, typename index_t>
static __forceinline__ __device__
void add_2d(scalar_t *data, int h, int w,
            int sH, int sW,
            scalar_t delta,
            const index_t NC_offset_inp,
            const index_t memory_span) {
  at::native::fastAtomicAdd(data,
                NC_offset_inp + h * sH + w * sW,
                memory_span,
                delta,
                true);
}


// Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
// where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
// if align_corners: -1 and +1 get sent to the centers of the corner pixels
//     -1 --> 0
//     +1 --> (size - 1)
//     scale_factor = (size - 1) / 2
// if not align_corners: -1 and +1 get sent to the image edges
//     -1 --> -0.5
//     +1 --> (size - 1) + 0.5 == size - 0.5
//     scale_factor = size / 2
template <typename scalar_t>
static inline __device__ scalar_t grid_sampler_unnormalize(scalar_t coord, int64_t size,
                                                bool align_corners, scalar_t offset) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return (((coord + 1) / 2) * (size - 2)) + offset;
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return (((coord + 1) * size - 1) / 2) + offset;
  }
}

// grid_sampler_unnormalize_set_grad works the same as grid_sampler_unnormalize
// except that it also returns the `d output / d input` via pointer argument
// `grad_in`.
// This is useful in the backward pass of grid_sampler.
template <typename scalar_t>
static inline __device__ scalar_t grid_sampler_unnormalize_set_grad(scalar_t coord, int64_t size,
                                                         bool align_corners, scalar_t *grad_in, scalar_t offset) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    *grad_in = static_cast<scalar_t>(size - 2) / 2;
    return ((coord + 1) / 2) * (size - 2) + offset;
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    *grad_in = static_cast<scalar_t>(size) / 2;
    return (((coord + 1) * size - 1) / 2) + offset;
  }
}

// Clips coordinates to between 0 and clip_limit - 1
template<typename scalar_t>
static inline __device__ scalar_t clip_coordinates(scalar_t in, int64_t clip_limit) {
  return std::min(static_cast<scalar_t>(clip_limit - 1), std::max(in, static_cast<scalar_t>(0)));
}

// clip_coordinates_set_grad works similarly to clip_coordinates except that
// it also returns the `d output / d input` via pointer argument `grad_in`.
// This is useful in the backward pass of grid_sampler.
template<typename scalar_t>
static inline __device__ scalar_t clip_coordinates_set_grad(scalar_t in, int64_t clip_limit,
                                                 scalar_t *grad_in) {
  // Note that it is important for the gradient calculation that borders
  // are considered out of bounds.
  if (in <= static_cast<scalar_t>(0)) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  } else {
    scalar_t max = static_cast<scalar_t>(clip_limit - 1);
    if (in >= max) {
      *grad_in = static_cast<scalar_t>(0);
      return max;
    } else {
      *grad_in = static_cast<scalar_t>(1);
      return in;
    }
  }
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template<typename scalar_t>
static inline __device__ scalar_t reflect_coordinates(scalar_t in, int64_t twice_low,
                                           int64_t twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = std::fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = std::fmod(in, span);
  int flips = static_cast<int>(std::floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

// reflect_coordinates_set_grad works similarly to reflect_coordinates except
// that it also returns the `d output / d input` via pointer argument
// `grad_in`.
// This is useful in the backward pass of grid_sampler.
template<typename scalar_t>
static inline __device__ scalar_t reflect_coordinates_set_grad(scalar_t in, int64_t twice_low,
                                                    int64_t twice_high, scalar_t *grad_in) {
  if (twice_low == twice_high) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  }
  int grad_in_mult_;
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = in - min;
  if (in < static_cast<scalar_t>(0)) {
    grad_in_mult_ = -1;
    in = -in;
  } else {
    grad_in_mult_ = 1;
  }
  // `fmod` returns same sign as `in`, which is positive after the `if` above.
  scalar_t extra = std::fmod(in, span);
  int flips = static_cast<int>(std::floor(in / span));
  if (flips % 2 == 0) {
    *grad_in = static_cast<scalar_t>(grad_in_mult_);
    return extra + min;
  } else {
    *grad_in = static_cast<scalar_t>(-grad_in_mult_);
    return span - extra + min;
  }
}

// Mapping the out-of-boundary points back into boundary
// This would only affect padding_mode=border or reflection
template<typename scalar_t>
static inline __device__ scalar_t compute_coordinates(scalar_t coord, int64_t size,
                                           at::native::detail::GridSamplerPadding padding_mode,
                                           bool align_corners) {
  if (padding_mode == at::native::detail::GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == at::native::detail::GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2*(size - 2));
    } else {
      coord = reflect_coordinates(coord, -1, 2*size - 1);
    }
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  }
  return coord;
}

// Computes the pixel source index value for a grid coordinate
template <typename scalar_t>
static inline __device__ scalar_t grid_sampler_compute_source_index(
    scalar_t coord,
    int64_t size,
    at::native::detail::GridSamplerPadding padding_mode,
    bool align_corners, scalar_t offset) {
  coord = grid_sampler_unnormalize(coord, size, align_corners, offset);
  coord = compute_coordinates(coord, size, padding_mode, align_corners);
  return coord;
}

// grid_sampler_compute_source_index_set_grad works similarly to
// grid_sampler_compute_source_index except that it also returns the
// `d output / d input` via pointer argument `grad_in`.
// This is useful in the backward pass of grid_sampler.
template <typename scalar_t>
static inline __device__ scalar_t grid_sampler_compute_source_index_set_grad(
    scalar_t coord,
    int64_t size,
    at::native::detail::GridSamplerPadding padding_mode,
    bool align_corners,
    scalar_t *grad_in, scalar_t offset) {
  scalar_t grad_clip, grad_refl;
  coord = grid_sampler_unnormalize_set_grad(coord, size, align_corners, grad_in, offset);
  if (padding_mode == at::native::detail::GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_clip;
  } else if (padding_mode == at::native::detail::GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates_set_grad(coord, 0, 2*(size - 2), &grad_refl);
    } else {
      coord = reflect_coordinates_set_grad(coord, -1, 2*size - 1, &grad_refl);
    }
    // clip coordinates to image borders
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_refl * grad_clip;
  }
  return coord;
}


__device__ inline float smoothstep(float val) {
	return val * val * (3.0f - 2.0f * val);
  // return 0.5*(1-cos(CUDART_PI_F*val));
}

__device__ inline float smoothstep_derivative(float val) {
	return 6 * val * (1.0f - val);
}

__device__ inline float smoothstep_2nd_derivative(float val) {
	return 6.0f - 12.0f * val;
}

__device__ inline float cosine(float val) {
  return 0.5f*(1-cos(CUDART_PI_F*val));
}

__device__ inline float cosine_derivative(float val) {
	return  0.5f * CUDART_PI_F *sin(CUDART_PI_F*val);
}

__device__ inline float cosine_2nd_derivative(float val) {
	return  0.5f * CUDART_PI_F * CUDART_PI_F *cos(CUDART_PI_F*val);
}
__device__ inline float cosine_3rd_derivative(float val) {
	return  -0.5f * CUDART_PI_F * CUDART_PI_F * CUDART_PI_F *sin(CUDART_PI_F*val);
}



template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(512)
  __global__ void cosine_sampler_kernel(
    const index_t nthreads,
    at::cuda::detail::TensorInfo<scalar_t, index_t> input,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> output,
    at::cuda::detail::TensorInfo<scalar_t, index_t> offset,
    const at::native::detail::GridSamplerPadding padding_mode,
    bool align_corners,
    bool apply_cosine_step){

        index_t C = input.sizes[1];
        index_t inp_H = input.sizes[2];
        index_t inp_W = input.sizes[3];
        index_t out_H = grid.sizes[1];
        index_t out_W = grid.sizes[2];
        index_t inp_sN = input.strides[0];
        index_t inp_sC = input.strides[1];
        index_t inp_sH = input.strides[2];
        index_t inp_sW = input.strides[3];
        index_t grid_sN = grid.strides[0];
        index_t grid_sH = grid.strides[1];
        index_t grid_sW = grid.strides[2];
        index_t grid_sCoor = grid.strides[3];
        index_t out_sN = output.strides[0];
        index_t out_sC = output.strides[1];
        index_t out_sH = output.strides[2];
        index_t out_sW = output.strides[3];
        
        index_t off_sN = offset.strides[0];

        CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
            const index_t w = index % out_W;
            const index_t h = (index / out_W) % out_H;
            const index_t n = index / (out_H * out_W);
            const index_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

            // get the corresponding input x, y co-ordinates from grid
            scalar_t x = grid.data[grid_offset];
            scalar_t y = grid.data[grid_offset + grid_sCoor];


            // int a = threadIdx.x;
            // int b = blockIdx.x;

            // if (a ==0 && b == 0) {
            //     printf("%d", sizeof(offset.data[n * off_sN]));
            // }
            scalar_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, 1, offset.data[n * off_sN]); //at::native::
            scalar_t iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, 1, offset.data[n * off_sN]);

            index_t ix_left = static_cast<index_t>(::floor(ix));
            index_t iy_top = static_cast<index_t>(::floor(iy));
            index_t ix_right = ix_left +1;
            index_t iy_bottom = iy_top +1;

            scalar_t dx_right = ix_right - ix;
            scalar_t dy_bottom = iy_bottom - iy;

            if(apply_cosine_step){
                dx_right = cosine(ix_right - ix);
                dy_bottom = cosine(iy_bottom - iy);
            }
            scalar_t dx_left = 1.0f - dx_right;
            scalar_t dy_top = 1.0f - dy_bottom;

            scalar_t nw = dx_right * dy_bottom;
            scalar_t ne = dx_left * dy_bottom;
            scalar_t sw = dx_right * dy_top;
            scalar_t se = dx_left * dy_top;

            auto inp_ptr_NC = input.data + n*inp_sN;
            auto out_ptr_NCHW = output.data + n*out_sN + h*out_sH + w*out_sW;

            for (index_t c= 0; c< C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC){
                *out_ptr_NCHW = static_cast<scalar_t>(0);
                if (at::native::within_bounds_2d(iy_top, ix_left, inp_H, inp_W)){
                    *out_ptr_NCHW += inp_ptr_NC[iy_top*inp_sH + ix_left * inp_sW] * nw;
                }
                if (at::native::within_bounds_2d(iy_top, ix_right, inp_H, inp_W)){
                    *out_ptr_NCHW += inp_ptr_NC[iy_top*inp_sH + ix_right *inp_sW] * ne;
                }
                if (at::native::within_bounds_2d(iy_bottom, ix_left, inp_H, inp_W)){
                    *out_ptr_NCHW += inp_ptr_NC[iy_bottom*inp_sH + ix_left * inp_sW] * sw;
                }
                if (at::native::within_bounds_2d(iy_bottom, ix_right, inp_H, inp_W)){
                    *out_ptr_NCHW += inp_ptr_NC[iy_bottom*inp_sH + ix_right*inp_sW] * se;
                }
            }

    }
}


template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void cosine_sampler_backward_kernel(
    const index_t nthreads,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_output,
    at::cuda::detail::TensorInfo<scalar_t, index_t> input,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_input,  // initialized to zeros (or unused if input_requires_grad is false)
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_grid,   // initialized to empty
    at::cuda::detail::TensorInfo<scalar_t, index_t> offset,
    const at::native::detail::GridSamplerPadding padding_mode,
    bool align_corners,
    bool apply_cosine_step,
    const index_t grad_input_memory_span,
    const bool input_requires_grad) {

    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];
    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];
    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];
    index_t gOut_sN = grad_output.strides[0];
    index_t gOut_sC = grad_output.strides[1];
    index_t gOut_sH = grad_output.strides[2];
    index_t gOut_sW = grad_output.strides[3];
    // gInp_* (and NC_offset below) are not really needed if input_requires_grad is false.
    int64_t gInp_sN = 0;
    int64_t gInp_sC = 0;
    int64_t gInp_sH = 0;
    int64_t gInp_sW = 0;
    index_t off_sN = offset.strides[0];
    if (input_requires_grad) {
      gInp_sN = grad_input.strides[0];
      gInp_sC = grad_input.strides[1];
      gInp_sH = grad_input.strides[2];
      gInp_sW = grad_input.strides[3];
    }
    index_t gGrid_sW = grad_grid.strides[2];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t n = index / (out_H * out_W);
      const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y co-ordinates from grid
      scalar_t x = grid.data[grid_offset];
      scalar_t y = grid.data[grid_offset + grid_sCoor];

      // multipliers for gradients on ix and iy
      scalar_t gix_mult, giy_mult;
      scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gix_mult, offset.data[n * off_sN]);
      scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &giy_mult, offset.data[n * off_sN]);

      // if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        index_t ix_left = static_cast<index_t>(::floor(ix));
        index_t iy_top = static_cast<index_t>(::floor(iy));
        index_t ix_right = ix_left + 1;
        index_t iy_bottom = iy_top + 1;

        scalar_t dx_right = ix_right - ix; // ix - ix_left; // ix_right - ix;
        scalar_t dy_bottom = iy_bottom - iy;// iy - iy_top; // iy_bottom - iy;
    


        float dx_right_derivative = 1.0f;
        float dy_bottom_derivative = 1.0f;

        if (apply_cosine_step) {
          dx_right_derivative = cosine_derivative(dx_right);
          dy_bottom_derivative = cosine_derivative(dy_bottom);
          dx_right = cosine(dx_right);
          dy_bottom = cosine(dy_bottom);
        }
        // } elif(apply_cosine_step){
        //   dx_right = smoothstep(ix_right - ix);
        //   dy_bottom = smoothstep(iy_bottom - iy);
        // }
        scalar_t dx_left = 1.0f - dx_right;
        scalar_t dy_top = 1.0f - dy_bottom;

        // get surfaces to each neighbor:
        scalar_t nw = dx_right * dy_bottom; //(ix_right - ix)    * (iy_bottom - iy);  
        scalar_t ne = dx_left * dy_bottom; //(ix    - ix_left) * (iy_bottom - iy);  
        scalar_t sw =  dx_right * dy_top; //(ix_right - ix)    * (iy    - iy_top);
        scalar_t se =  dx_left * dy_top; //(ix    - ix_left) * (iy    - iy_top);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
        scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
        index_t NC_offset;
    
        if (input_requires_grad) {
              NC_offset = n * gInp_sN;
         }
        // index_t NC_offset = n * gInp_sN;
        scalar_t *inp_ptr_NC = input.data + n * inp_sN;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
          scalar_t gOut = *gOut_ptr_NCHW;

          if (input_requires_grad) {
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            at::native::safe_add_2d(grad_input.data, iy_top, ix_left, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut, NC_offset, grad_input_memory_span);
            at::native::safe_add_2d(grad_input.data, iy_top, ix_right, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut, NC_offset, grad_input_memory_span);
            at::native::safe_add_2d(grad_input.data, iy_bottom, ix_left, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut, NC_offset, grad_input_memory_span);
            at::native::safe_add_2d(grad_input.data, iy_bottom, ix_right, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut, NC_offset, grad_input_memory_span);
          }

          // calculate grad_grid
          if (at::native::within_bounds_2d(iy_top, ix_left, inp_H, inp_W)) {
            scalar_t nw_val = inp_ptr_NC[iy_top * inp_sH + ix_left * inp_sW];
            gix -= nw_val *  dy_bottom * gOut; 
            giy -= nw_val *  dx_right* gOut; 
          }
          if (at::native::within_bounds_2d(iy_top, ix_right, inp_H, inp_W)) {
            scalar_t ne_val = inp_ptr_NC[iy_top * inp_sH + ix_right * inp_sW];
            gix += ne_val *  dy_bottom * gOut; 
            giy -= ne_val *  (1-dx_right)* gOut;
          }
          if (at::native::within_bounds_2d(iy_bottom, ix_left, inp_H, inp_W)) {
            scalar_t sw_val = inp_ptr_NC[iy_bottom * inp_sH + ix_left * inp_sW];
            gix -= sw_val *  (1-dy_bottom)* gOut;
            giy += sw_val *  (dx_right)* gOut; 
          }
          if (at::native::within_bounds_2d(iy_bottom, ix_right, inp_H, inp_W)) {
            scalar_t se_val = inp_ptr_NC[iy_bottom * inp_sH + ix_right * inp_sW];
            gix += se_val *  (1-dy_bottom)* gOut; 
            giy += se_val *  (1-dx_right)* gOut; 
          }
        // }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = gix_mult * gix * dx_right_derivative;
        gGrid_ptr_NHW[1] = giy_mult * giy * dy_bottom_derivative;
      
      }
    }
  }

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void cosine_sampler_backward_backward_kernel(
    const index_t nthreads,
    at::cuda::detail::TensorInfo<scalar_t, index_t> gInput, // initialized to empty
    at::cuda::detail::TensorInfo<scalar_t, index_t> gGrid, // initialized to zeros
    at::cuda::detail::TensorInfo<scalar_t, index_t> ggOut, // initialized to zeros
    at::cuda::detail::TensorInfo<scalar_t, index_t> input,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> gOutInput,
    at::cuda::detail::TensorInfo<scalar_t, index_t> gOutGrid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> gOut,
    at::cuda::detail::TensorInfo<scalar_t, index_t> offset,
    const at::native::detail::GridSamplerPadding padding_mode,
    bool align_corners,
    bool apply_cosine_step,
    bool input_requires_grad,
    const index_t gInput_memory_span,
    const index_t ggOut_memory_span) {

    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];
    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];

    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];

    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];

    index_t gOut_sN = gOut.strides[0];
    index_t gOut_sC = gOut.strides[1];
    index_t gOut_sH = gOut.strides[2];
    index_t gOut_sW = gOut.strides[3];

    index_t off_sN = offset.strides[0];

    index_t gOutInput_sN = 0;
    index_t gOutInput_sC = 0;

    // index_t gGrid_sN = gGrid.strides[0]; // 원래 3
    // index_t gGrid_sH = gGrid.strides[1]; // 원래 3
    index_t gGrid_sW = gGrid.strides[2]; // 원래 3
    // index_t gGrid_sCoor = gGrid.strides[3]; // 원래 3

    // index_t gOutGrid_sN = gOutGrid.strides[0]; // 원래 3
    // index_t gOutGrid_sH = gOutGrid.strides[1]; // 원래 3
    index_t gOutGrid_sW = gOutGrid.strides[2]; // 원래 3
    // index_t gOutGrid_sCoor = gOutGrid.strides[3]; // 원래 3

    if (input_requires_grad) {
        gOutInput_sN = gOutInput.strides[0];
        gOutInput_sC = gOutInput.strides[1];
    }

    index_t gInp_sN = gInput.strides[0];
    index_t gInp_sC = gInput.strides[1];
    index_t gInp_sH = gInput.strides[2];
    index_t gInp_sW = gInput.strides[3];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
        const index_t w = index % out_W;
        const index_t h = (index / out_W) % out_H;
        const index_t n = index / (out_H * out_W);
        const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;
        // const auto gOutGrid_offset = n * gOutGrid_sN + h * gOutGrid_sH + w * gOutGrid_sW;
        // const auto gGrid_offset = n * gGrid_sN + h * gGrid_sH + w * gGrid_sW;


        scalar_t x = grid.data[grid_offset];
        scalar_t y = grid.data[grid_offset + grid_sCoor];

        // multipliers for gradients on ix and iy
        scalar_t dL_dix_mult, dL_diy_mult;
        scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &dL_dix_mult, offset.data[n * off_sN]);
        scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &dL_diy_mult, offset.data[n * off_sN]);

        // if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        index_t ix_left = static_cast<index_t>(::floor(ix));
        index_t iy_top = static_cast<index_t>(::floor(iy));
        index_t ix_right = ix_left + 1;
        index_t iy_bottom = iy_top + 1;

        scalar_t dx_right = ix_right - ix; 
        scalar_t dy_bottom = iy_bottom - iy;
    
        float dx_right_2nd_derivative = 0.0f;
        float dy_bottom_2nd_derivative = 0.0f;

        float dx_right_derivative = -dL_dix_mult;
        float dy_bottom_derivative = -dL_diy_mult;

        if (apply_cosine_step) {
            dx_right_2nd_derivative = dL_dix_mult * dL_dix_mult * cosine_2nd_derivative(dx_right);
            dy_bottom_2nd_derivative = dL_diy_mult * dL_diy_mult * cosine_2nd_derivative(dy_bottom);
            dx_right_derivative *= cosine_derivative(dx_right);
            dy_bottom_derivative *= cosine_derivative(dy_bottom);

            dx_right = cosine(dx_right);
            dy_bottom = cosine(dy_bottom);
        }

        scalar_t dx_left = 1.0f - dx_right;
        scalar_t dy_top = 1.0f - dy_bottom;
        scalar_t dx_left_derivative = -dx_right_derivative;
        scalar_t dy_top_derivative = -dy_bottom_derivative;
        scalar_t dx_left_2nd_derivative = -dx_right_2nd_derivative;
        scalar_t dy_top_2nd_derivative = -dy_bottom_2nd_derivative;

        index_t index_corners[2][2] = {{ix_left, iy_top},
                                        {ix_right, iy_bottom}};
        scalar_t pos_corners[2][6] = {{dx_right, dy_bottom,
                                    dx_right_derivative, dy_bottom_derivative,
                                    dx_right_2nd_derivative, dy_bottom_2nd_derivative},
                                    {dx_left, dy_top,
                                    dx_left_derivative, dy_top_derivative,
                                    dx_left_2nd_derivative, dy_top_2nd_derivative}};

        scalar_t surface_coefficients[4] = {};
        scalar_t out_derivatives[4][4] = {};

        #pragma unroll
        for (int shift = 0; shift < 4; shift++) {
        int px = (shift >> 0) & 1;  // 0 1 0 1
        int py = (shift >> 1) & 1;  // 0 0 1 1

        surface_coefficients[shift] = pos_corners[px][0] * pos_corners[py][1]; // 
        out_derivatives[0][shift] =  pos_corners[py][1] * pos_corners[px][2]; // dOut_dx / surf_weight
        out_derivatives[1][shift] = pos_corners[py][1] * pos_corners[px][4]; // d2Out_dx2 / surf_weight
        // out_derivatives[2][shift] = pos_corners[py][3] * pos_corners[px][2]; // d2Out_dxdy / surf_weight

        out_derivatives[2][shift] = pos_corners[px][0] * pos_corners[py][3]; // dOut_dy / surf_weight
        out_derivatives[3][shift] = pos_corners[px][0] * pos_corners[py][5]; // d2Out_dy2 / surf_weight
        // out_derivatives[5][shift] = pos_corners[px][2] * pos_corners[py][3]; // d2Out_dydx / surf_weight
        }

        scalar_t d2L_dix2 = static_cast<scalar_t>(0), d2L_diy2 = static_cast<scalar_t>(0);
        index_t offset_out_DHW =  h * gOut_sH + w * gOut_sW;
        scalar_t *gOut_ptr_NCDHW = gOut.data + n * gOut_sN + offset_out_DHW;
        index_t NC_offset_inp = n * gInp_sN;
        index_t NC_offset_out = n * gOut_sN;
        scalar_t *inp_ptr_NC = input.data + n * inp_sN;

        scalar_t *gOutInput_ptr_NC = NULL;

        if (input_requires_grad) {
            gOutInput_ptr_NC = gOutInput.data + n * gOutInput_sN;
        }

        scalar_t *gOutGrid_ptr_NDHW = gOutGrid.data +  index * gOutGrid_sW;
        scalar_t *gGrid_ptr_NDHW = gGrid.data + index * gGrid_sW;

    for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, inp_ptr_NC += inp_sC, gOutInput_ptr_NC += gOutInput_sC, NC_offset_inp += gInp_sC, NC_offset_out += gOut_sC) {
        scalar_t gOut = *gOut_ptr_NCDHW;

        #pragma unroll
        for (int shift = 0; shift < 4; shift++) { // 0, 1, 2, 3
            int px = (shift >> 0) & 1;              // [0, 1, 0, 1]
            int py = (shift >> 1) & 1;              // [0, 0, 1, 1]

            index_t ix = index_corners[px][0]; // l r l r 
            index_t iy = index_corners[py][1]; // t t b b
                                            // -> nw ne sw se
            // Slightly unprecise naming: in fact these are divided by surf_weight.
            scalar_t dOut_dx = out_derivatives[0][shift]; // E.g. variable "dOut_dx" is mathematically "dOut/dx * 1/surf_weight"
            scalar_t d2Out_dx2 = out_derivatives[1][shift];
            // scalar_t d2Out_dxdy = out_derivatives[2][shift];
            scalar_t dOut_dy = out_derivatives[2][shift];
            scalar_t d2Out_dy2 = out_derivatives[3][shift];
            // scalar_t d2Out_dydx = out_derivatives[5][shift];
            scalar_t surface_coeff = surface_coefficients[shift];

        if (at::native::within_bounds_2d(iy, ix, inp_H, inp_W)) {
            index_t inp_el = iy * inp_sH + ix * inp_sW;
            scalar_t surf_weight = inp_ptr_NC[inp_el];

            scalar_t dL_dx = gOut * dOut_dx;
            scalar_t dL_dy = gOut * dOut_dy;

            scalar_t gOutGrid_x = gOutGrid_ptr_NDHW[0];
            scalar_t gOutGrid_y = gOutGrid_ptr_NDHW[1];

            scalar_t ggOut_delta = surf_weight * (dOut_dx * gOutGrid_x
                                                        + dOut_dy * gOutGrid_y); // u_x_C

            if (gOutInput_ptr_NC != NULL) {
                scalar_t gOutInput = gOutInput_ptr_NC[inp_el];
                ggOut_delta += gOutInput * surface_coeff; // u_c
                // d2L_dix2 += dL_dx * gOutInput;
                // d2L_diy2 += dL_dy * gOutInput;
            }

            at::native::fastAtomicAdd(ggOut.data,
                                        NC_offset_out + offset_out_DHW,
                                        ggOut_memory_span,
                                        ggOut_delta,
                                        true);
            
            // u_xx, u_yy
            d2L_dix2 += surf_weight * gOut * (d2Out_dx2 * gOutGrid_x);
            d2L_diy2 += surf_weight * gOut * (d2Out_dy2 * gOutGrid_y);

            // at::native::safe_add_2d(grad_input.data, iy, ix, gInp_sH, gInp_sW, inp_H, inp_W, dL_dx * gOutGrid_x + dL_dy * gOutGrid_y, NC_offset_inp, grad_input_memory_span);
            
            // cell로 미분
            add_2d(gInput.data, iy, ix, gInp_sH, gInp_sW, dL_dx * gOutGrid_x + dL_dy * gOutGrid_y, NC_offset_inp, gInput_memory_span);
        }
      }
    }

    gGrid_ptr_NDHW[0] = d2L_dix2;
    gGrid_ptr_NDHW[1] = d2L_diy2;
  }
}




template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void cosine_sampler_backward_backward_backward_kernel(
    const index_t nthreads,
    at::cuda::detail::TensorInfo<scalar_t, index_t> gInput, // initialized to empty
    at::cuda::detail::TensorInfo<scalar_t, index_t> ggOut, 
    // at::cuda::detail::TensorInfo<scalar_t, index_t> gGrid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> input,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> gOut,
    at::cuda::detail::TensorInfo<scalar_t, index_t> gOutggOut, 
    at::cuda::detail::TensorInfo<scalar_t, index_t> gOutGrid, 
    at::cuda::detail::TensorInfo<scalar_t, index_t> gOutgGrid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> offset,
    const at::native::detail::GridSamplerPadding padding_mode,
    bool align_corners,
    bool apply_cosine_step,
    bool input_requires_grad,
    const index_t gInput_memory_span,
    const index_t ggOut_memory_span) {

    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];
    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];

    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];

    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];

    index_t gOut_sN = gOut.strides[0];
    index_t gOut_sC = gOut.strides[1];
    index_t gOut_sH = gOut.strides[2];
    index_t gOut_sW = gOut.strides[3];

    index_t gOutggOut_sN = gOutggOut.strides[0];
    index_t gOutggOut_sC = gOutggOut.strides[1];
    index_t gOutggOut_sH = gOutggOut.strides[2];
    index_t gOutggOut_sW = gOutggOut.strides[3];

    index_t off_sN = offset.strides[0];

    // index_t gOutgInput_sN = 0;
    // index_t gOutgInput_sC = 0;

    // index_t gGrid_sW = gGrid.strides[2]; // 원래 3
    index_t gOutGrid_sW = gOutGrid.strides[2]; // 원래 3
    index_t gOutgGrid_sW = gOutgGrid.strides[2];

    // if (input_requires_grad) {
    //     gOutgInput_sN = gOutgInput.strides[0];
    //     gOutgInput_sC = gOutgInput.strides[1];
    // }

    index_t gInp_sN = gInput.strides[0];
    index_t gInp_sC = gInput.strides[1];
    index_t gInp_sH = gInput.strides[2];
    index_t gInp_sW = gInput.strides[3];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
        const index_t w = index % out_W;
        const index_t h = (index / out_W) % out_H;
        const index_t n = index / (out_H * out_W);
        const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;


        scalar_t x = grid.data[grid_offset];
        scalar_t y = grid.data[grid_offset + grid_sCoor];

        // multipliers for gradients on ix and iy
        scalar_t dL_dix_mult, dL_diy_mult;
        scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &dL_dix_mult, offset.data[n * off_sN]);
        scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &dL_diy_mult, offset.data[n * off_sN]);

        // if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        index_t ix_left = static_cast<index_t>(::floor(ix));
        index_t iy_top = static_cast<index_t>(::floor(iy));
        index_t ix_right = ix_left + 1;
        index_t iy_bottom = iy_top + 1;

        scalar_t dx_right = ix_right - ix; 
        scalar_t dy_bottom = iy_bottom - iy;
    
        float dx_right_2nd_derivative = 0.0f;
        float dy_bottom_2nd_derivative = 0.0f;

        float dx_right_derivative = -dL_dix_mult;
        float dy_bottom_derivative = -dL_diy_mult;

        
        if (apply_cosine_step) {

            dx_right_2nd_derivative = dL_dix_mult * dL_dix_mult * cosine_2nd_derivative(dx_right);
            dy_bottom_2nd_derivative = dL_diy_mult * dL_diy_mult * cosine_2nd_derivative(dy_bottom);
            dx_right_derivative *= cosine_derivative(dx_right);
            dy_bottom_derivative *= cosine_derivative(dy_bottom);

            dx_right = cosine(dx_right);
            dy_bottom = cosine(dy_bottom);
        }

        scalar_t dx_left = 1.0f - dx_right;
        scalar_t dy_top = 1.0f - dy_bottom;
        scalar_t dx_left_derivative = -dx_right_derivative;
        scalar_t dy_top_derivative = -dy_bottom_derivative;
        scalar_t dx_left_2nd_derivative = -dx_right_2nd_derivative;
        scalar_t dy_top_2nd_derivative = -dy_bottom_2nd_derivative;

        index_t index_corners[2][2] = {{ix_left, iy_top},
                                        {ix_right, iy_bottom}};
        scalar_t pos_corners[2][6] = {{dx_right, dy_bottom,
                                    dx_right_derivative, dy_bottom_derivative,
                                    dx_right_2nd_derivative, dy_bottom_2nd_derivative},
                                    {dx_left, dy_top,
                                    dx_left_derivative, dy_top_derivative,
                                    dx_left_2nd_derivative, dy_top_2nd_derivative}};

        // scalar_t surface_coefficients[4] = {};
        scalar_t out_derivatives[4][4] = {};

        #pragma unroll
        for (int shift = 0; shift < 4; shift++) {
        int px = (shift >> 0) & 1;  // 0 1 0 1
        int py = (shift >> 1) & 1;  // 0 0 1 1

        // surface_coefficients[shift] = pos_corners[px][0] * pos_corners[py][1]; // 
        out_derivatives[0][shift] = pos_corners[py][1] * pos_corners[px][2]; // dOut_dx / surf_weight
        out_derivatives[1][shift] =  pos_corners[py][1] * pos_corners[px][4]; // d2Out_dx2 / surf_weight
        // out_derivatives[2][shift] = pos_corners[py][1] * pos_corners[px][6]; // d3Out_dx3 / surf_weight

        out_derivatives[2][shift] = pos_corners[px][0] * pos_corners[py][3]; // dOut_dy / surf_weight
        out_derivatives[3][shift] = pos_corners[px][0] * pos_corners[py][5]; // d2Out_dy2 / surf_weight
        // out_derivatives[5][shift] = pos_corners[px][0] * pos_corners[py][7]; // d3Out_dy3 / surf_weight
        }

        scalar_t ggOut_delta = static_cast<scalar_t>(0);//, d3L_dix3 = static_cast<scalar_t>(0), d3L_diy3 = static_cast<scalar_t>(0);
        index_t offset_out_DHW =  h * gOut_sH + w * gOut_sW;
        scalar_t *gOut_ptr_NCDHW = gOut.data + n * gOut_sN + offset_out_DHW;

        index_t offset_ggout_DHW =  h * gOutggOut_sH + w * gOutggOut_sW;
        scalar_t *gOutggOut_ptr_NCDHW = gOutggOut.data + n * gOutggOut_sN + offset_ggout_DHW;

        index_t NC_offset_inp = n * gInp_sN;
        index_t NC_offset_out = n * gOut_sN;
        scalar_t *inp_ptr_NC = input.data + n * inp_sN;

        // scalar_t *gOutgInput_ptr_NC = NULL;

        // if (input_requires_grad) {
        //     gOutgInput_ptr_NC = gOutgInput.data + n * gOutgInput_sN;
        // }

        scalar_t *gOutGrid_ptr_NDHW = gOutGrid.data +  index * gOutGrid_sW;
        
        //////////////////////////////////////////////
        scalar_t *gOutgGrid_ptr_NDHW = gOutgGrid.data +  index * gOutgGrid_sW;
        //////////////////////////////////////////////
        // scalar_t *gGrid_ptr_NDHW = gGrid.data + index * gGrid_sW;
    for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gOutggOut_ptr_NCDHW += gOutggOut_sC, inp_ptr_NC += inp_sC, NC_offset_inp += gInp_sC, NC_offset_out += gOut_sC) {
        scalar_t gOut = *gOut_ptr_NCDHW;
        scalar_t gOutggOut_scalar = *gOutggOut_ptr_NCDHW;

        #pragma unroll
        for (int shift = 0; shift < 4; shift++) { // 0, 1, 2, 3
            int px = (shift >> 0) & 1;              // [0, 1, 0, 1]
            int py = (shift >> 1) & 1;              // [0, 0, 1, 1]

            index_t ix = index_corners[px][0]; // l r l r 
            index_t iy = index_corners[py][1]; // t t b b
                                            // -> nw ne sw se
            // Slightly unprecise naming: in fact these are divided by surf_weight.
            scalar_t dOut_dx = out_derivatives[0][shift]; // E.g. variable "dOut_dx" is mathematically "dOut/dx * 1/surf_weight"
            scalar_t d2Out_dx2 = out_derivatives[1][shift];
            // scalar_t d3Out_dx3 = out_derivatives[2][shift];
            scalar_t dOut_dy = out_derivatives[2][shift];
            scalar_t d2Out_dy2 = out_derivatives[3][shift];
            // scalar_t d3Out_dy3 = out_derivatives[5][shift];
            // scalar_t surface_coeff = surface_coefficients[shift];

        if (at::native::within_bounds_2d(iy, ix, inp_H, inp_W)) {
            index_t inp_el = iy * inp_sH + ix * inp_sW;
            scalar_t surf_weight = inp_ptr_NC[inp_el];

            // scalar_t dL_dx = gOut * dOut_dx;
            // scalar_t dL_dy = gOut * dOut_dy;

            scalar_t gOutGrid_x = gOutGrid_ptr_NDHW[0];
            scalar_t gOutGrid_y = gOutGrid_ptr_NDHW[1];
            //////////////////////////////////////////////
            scalar_t gOutgGrid_x = gOutgGrid_ptr_NDHW[0];
            scalar_t gOutgGrid_y = gOutgGrid_ptr_NDHW[1];
            //////////////////////////////////////////////
            
            if (gOutgGrid_x ==1 && gOutGrid_x !=1){
              printf("????!!!");
              //  ggOut_delta =  surf_weight * (d2Out_dx2 * gOutgGrid_x*gOutGrid_x + d2Out_dy2 * gOutgGrid_y*gOutGrid_y); // (dOut_dx * gOutgGrid_x + dOut_dy * gOutgGrid_y) +  
            }else{
               ggOut_delta =  surf_weight * (d2Out_dx2 * gOutgGrid_x*gOutGrid_x + d2Out_dy2 * gOutgGrid_y*gOutGrid_y); // (dOut_dx * gOutgGrid_x + dOut_dy * gOutgGrid_y) +  
            }

            // if (gOutgInput_ptr_NC != NULL) {
            //     scalar_t gOutgInput = gOutgInput_ptr_NC[inp_el];
            //     ggOut_delta += gOutgInput*(dOut_dx * gOutGrid_x + dOut_dy * gOutGrid_y) ; // u_c
            //     // d2L_dix2 += dL_dx * gOutInput;
            //     // d2L_diy2 += dL_dy * gOutInput;
            // }

            at::native::fastAtomicAdd(ggOut.data,
                                        NC_offset_out + offset_out_DHW,
                                        ggOut_memory_span,
                                        ggOut_delta,
                                        true);
            
            // u_xx, u_yy
            // scalar_t d2L_dix2 = (d2Out_dx2);
            // scalar_t d2L_diy2 = (d2Out_dy2);
            
            // d3L_dix3 += gOut *  (d3Out_dx3)* gOutgGrid_x ;
            // d3L_diy3 += gOut *  (d3Out_dy3)* gOutgGrid_y ;

            // at::native::safe_add_2d(grad_input.data, iy, ix, gInp_sH, gInp_sW, inp_H, inp_W, dL_dx * gOutGrid_x + dL_dy * gOutGrid_y, NC_offset_inp, grad_input_memory_span);
            
            // cell로 미분
            add_2d(gInput.data, iy, ix, gInp_sH, gInp_sW,  gOut * (d2Out_dx2 * gOutgGrid_x *gOutGrid_x + d2Out_dy2 * gOutgGrid_y*gOutGrid_y), NC_offset_inp, gInput_memory_span);
        }
      }
    }

    // gGrid_ptr_NDHW[0] = d3L_dix3;
    // gGrid_ptr_NDHW[1] = d3L_diy3;
  }
}





void launch_cosine_sampler_forward_kernel(
    const torch::TensorBase &output, const torch::TensorBase &input, const torch::TensorBase &grid, const torch::TensorBase &offset,
    int64_t padding_mode, bool align_corners, bool apply_cosine_step) {
  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  int64_t count = N * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cosine_sampler_cuda", [&] {
      if (at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(grid) &&
          at::native::canUse32BitIndexMath(output)) {
        cosine_sampler_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            at::cuda::detail::getTensorInfo<scalar_t, int>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grid),
            at::cuda::detail::getTensorInfo<scalar_t, int>(output),
            at::cuda::detail::getTensorInfo<scalar_t, int>(offset),
            static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
            align_corners,
            apply_cosine_step);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        cosine_sampler_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grid),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(output),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(offset),
            static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
            align_corners,
            apply_cosine_step);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}

void launch_cosine_sampler_backward_kernel(
    const torch::TensorBase &grad_input, const torch::TensorBase &grad_grid,
    const torch::TensorBase& grad_output, const torch::TensorBase& input,
    const torch::TensorBase& grid, const torch::TensorBase &offset, int64_t padding_mode,
    bool align_corners, bool apply_cosine_step, bool input_requires_grad) {
  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  int64_t count = N * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cosine_sampler_backward_cuda", [&] {
      if (at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(grid) &&
          at::native::canUse32BitIndexMath(grad_output)) {
        cosine_sampler_backward_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grad_output),
            at::cuda::detail::getTensorInfo<scalar_t, int>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grid),
            input_requires_grad ? at::cuda::detail::getTensorInfo<scalar_t, int>(grad_input) : at::cuda::detail::TensorInfo<scalar_t, int>(),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grad_grid),
            at::cuda::detail::getTensorInfo<scalar_t, int>(offset),
            static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
            align_corners,
            apply_cosine_step,
            /*grad_input_memory_span =*/input_requires_grad ? static_cast<int>(grad_input.numel()) : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        cosine_sampler_backward_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_output),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grid),
            input_requires_grad ? at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_input) : at::cuda::detail::TensorInfo<scalar_t, int64_t>(),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_grid),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(offset),
            static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
            align_corners,
            apply_cosine_step,
            /*grad_input_memory_span =*/input_requires_grad ? grad_input.numel() : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}


void launch_cosine_sampler_backward_backward_kernel(
    const torch::TensorBase& gInput,
    const torch::TensorBase& gGrid,
    const torch::TensorBase& ggOut,
    const torch::TensorBase& input,
    const torch::TensorBase& grid,
    const torch::TensorBase& gOutInput,
    const torch::TensorBase& gOutGrid,
    const torch::TensorBase& gOut, 
    const torch::TensorBase &offset,
    int64_t padding_mode,
    const bool align_corners,
    const bool apply_cosine_step,
    const bool input_requires_grad) {
  auto N = input.size(0);
  // auto D = grid.size(1);
  auto H = grid.size(1);
  auto W = grid.size(2);
  int64_t count = N  * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cosine_sampler_backward_backward_kernel", [&] {
      if (at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(grid) && at::native::canUse32BitIndexMath(gOut)
          && at::native::canUse32BitIndexMath(gOutInput) && at::native::canUse32BitIndexMath(gOutGrid)) {
        cosine_sampler_backward_backward_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            at::cuda::detail::getTensorInfo<scalar_t, int>(gInput),
            at::cuda::detail::getTensorInfo<scalar_t, int>(gGrid),
            at::cuda::detail::getTensorInfo<scalar_t, int>(ggOut),
            at::cuda::detail::getTensorInfo<scalar_t, int>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grid),
            input_requires_grad ? at::cuda::detail::getTensorInfo<scalar_t, int>(gOutInput) : at::cuda::detail::TensorInfo<scalar_t, int>(),
            at::cuda::detail::getTensorInfo<scalar_t, int>(gOutGrid),
            at::cuda::detail::getTensorInfo<scalar_t, int>(gOut),
            at::cuda::detail::getTensorInfo<scalar_t, int>(offset),
            static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
            align_corners,
            apply_cosine_step,
            input_requires_grad,
            static_cast<int>(gInput.numel()),
            static_cast<int>(ggOut.numel()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        cosine_sampler_backward_backward_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(gInput),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(gGrid),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(ggOut),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grid),
            input_requires_grad ? at::cuda::detail::getTensorInfo<scalar_t, int64_t>(gOutInput) : at::cuda::detail::TensorInfo<scalar_t, int64_t>(),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(gOutGrid),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(gOut),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(offset),
            static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
            align_corners,
            apply_cosine_step,
            input_requires_grad,
            gInput.numel(),
            ggOut.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}

void launch_cosine_sampler_backward_backward_backward_kernel(
    const torch::TensorBase& gInput,
    const torch::TensorBase& ggOut,
    // const torch::TensorBase& gGrid,
    const torch::TensorBase& input,
    const torch::TensorBase& grid,
    // const torch::TensorBase& gOutGrid,
    const torch::TensorBase& gOut,
    const torch::TensorBase& gOutggOut,
    const torch::TensorBase& gOutGrid, 
    const torch::TensorBase& gOutgGrid,
    const torch::TensorBase &offset,
    int64_t padding_mode,
    const bool align_corners,
    const bool apply_cosine_step,
    const bool input_requires_grad) {
  auto N = input.size(0);
  // auto D = grid.size(1);
  auto H = grid.size(1);
  auto W = grid.size(2);
  int64_t count = N  * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cosine_sampler_backward_backward_backward_kernel", [&] {
      if (at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(grid) && at::native::canUse32BitIndexMath(gOut)
          && at::native::canUse32BitIndexMath(gOutgGrid)) {
        cosine_sampler_backward_backward_backward_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            at::cuda::detail::getTensorInfo<scalar_t, int>(gInput),
            at::cuda::detail::getTensorInfo<scalar_t, int>(ggOut),
            // at::cuda::detail::getTensorInfo<scalar_t, int>(gGrid),
            at::cuda::detail::getTensorInfo<scalar_t, int>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grid),
            // at::cuda::detail::getTensorInfo<scalar_t, int>(gOutGrid),
            at::cuda::detail::getTensorInfo<scalar_t, int>(gOut),
            at::cuda::detail::getTensorInfo<scalar_t, int>(gOutggOut),
            at::cuda::detail::getTensorInfo<scalar_t, int>(gOutGrid),
            at::cuda::detail::getTensorInfo<scalar_t, int>(gOutgGrid),
            // input_requires_grad ? at::cuda::detail::getTensorInfo<scalar_t, int>(gOutgInput) : at::cuda::detail::TensorInfo<scalar_t, int>(),
            at::cuda::detail::getTensorInfo<scalar_t, int>(offset),
            static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
            align_corners,
            apply_cosine_step,
            input_requires_grad,
            static_cast<int>(gInput.numel()),
            static_cast<int>(ggOut.numel()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        cosine_sampler_backward_backward_backward_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(gInput),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(ggOut),
            // at::cuda::detail::getTensorInfo<scalar_t, int64_t>(gGrid),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grid),
            // at::cuda::detail::getTensorInfo<scalar_t, int64_t>(gOutGrid),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(gOut),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(gOutggOut),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(gOutGrid),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(gOutgGrid),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(offset),
            // input_requires_grad ? at::cuda::detail::getTensorInfo<scalar_t, int64_t>(gOutgInput) : at::cuda::detail::TensorInfo<scalar_t, int64_t>(),
            static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
            align_corners,
            apply_cosine_step,
            input_requires_grad,
            gInput.numel(),
            ggOut.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}
