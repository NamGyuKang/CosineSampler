#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <iostream>
#include <math_constants.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <memory.h>

#define CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)
#define CUDA_MAX_THREADS at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock
#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                         \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;           \
  for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

using namespace std;


template <typename scalar_t>
__global__ void normalize_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> idx, 
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ret, const int len) {
  int n = blockIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  // 5 1 100000  c = 100000  n = 5
  if (c < idx.size(2) * idx.size(0)) { // issue. 순수 c로만 잡는게 맞지 않나? n은 어차피 블록단인데... 근데 size(2)로만 하면 범위가 안맞음. 밑에 블록이랑 봐야할듯
    ret[n][0][c] = ((idx[n][0][c]+1)/2)*len;
  }
}

template <typename scalar_t>
__global__ void normalize_offset_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> idx, 
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ret, const int len, 
torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> offset) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  
  const int n_points = idx.size(2);
  const int n_cells = idx.size(0);
  int n = tmp / n_points;
  int c = tmp % n_points;  
  if (c < n_points && n < n_cells) { // issue. 순수 c로만 잡는게 맞지 않나? n은 어차피 블록단인데... 근데 size(2)로만 하면 범위가 안맞음. 밑에 블록이랑 봐야할듯
    ret[n][0][c] = ((idx[n][0][c]+1)/2)*len + offset[n];
  }
}

template <typename scalar_t>
__global__ void step_bilinear_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix, 
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ret) { // backward 계산 떄문에 일부러 5 넣어줌
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  
  const int n_points = ix.size(2);
  const int n_cells = ix.size(0);
  int n = tmp / n_points;
  int c = tmp % n_points;  

  if (c < n_points && n < n_cells) {
    ret[0][n][0][c] = ix_right[n][0][c] - ix[n][0][c];
    ret[1][n][0][c] = 1 - ret[0][n][0][c];
    ret[2][n][0][c] = iy_bottom[n][0][c] - iy[n][0][c];
    ret[3][n][0][c] = 1 - ret[2][n][0][c];
  }
}

// ix : it, iy : ix
template <typename scalar_t>
__global__ void step_cosine_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix, 
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ret) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  
  const int n_points = ix.size(2);
  const int n_cells = ix.size(0);
  int n = tmp / n_points;
  int c = tmp % n_points;  

  if (c < n_points && n < n_cells) {
    scalar_t tmp_ix = ix_right[n][0][c] - ix[n][0][c];
    scalar_t tmp_iy = iy_bottom[n][0][c] - iy[n][0][c];
    ret[0][n][0][c] = 0.5*(1-cos(CUDART_PI_F*tmp_ix)); 
    ret[1][n][0][c] = 1 - ret[0][n][0][c];
    ret[2][n][0][c] = 0.5*(1-cos(CUDART_PI_F*tmp_iy)); 
    ret[3][n][0][c] = 1 - ret[2][n][0][c];
  }
}

template <typename scalar_t>
__global__ void step_smooth_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix, 
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ret) {
  int n = blockIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (c < ix.size(2) * ix.size(0)) {
    scalar_t tmp_ix = ix_right[n][0][c] - ix[n][0][c];
    scalar_t tmp_iy = iy_bottom[n][0][c] - iy[n][0][c];
    ret[0][n][0][c] = pow(tmp_ix,2) * (3-2*tmp_ix); 
    ret[1][n][0][c] = 1 - ret[0][n][0][c];
    ret[2][n][0][c] = pow(tmp_iy,2) * (3-2*tmp_iy); 
    ret[3][n][0][c] = 1 - ret[2][n][0][c];
  }
}

// ix : it, iy : ix
template <typename scalar_t>
__global__ void get_corner_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix, // issue 함수명, 변수명
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ret) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  
  const int n_points = ix.size(2);
  const int n_cells = ix.size(0);
  int n = tmp / n_points;
  int c = tmp % n_points;  

  if (c < n_points && n < n_cells) {
    ret[0][n][0][c] = floor(ix[n][0][c]);
    ret[1][n][0][c] = ret[0][n][0][c] + 1; // floor를 tmp로 받고 걔를 +1할까? 메모리 접근 줄이게? gpu도 메모리 접근 오래 걸리나?
    ret[2][n][0][c] = floor(iy[n][0][c]);
    ret[3][n][0][c] = ret[2][n][0][c] + 1;
  }
}


template <typename scalar_t>
__global__ void compute_point_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dx_right, // issue 함수명, 변수명
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dx_left,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dy_bottom,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dy_top,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ret) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  
  const int n_points = dx_right.size(2);
  const int n_cells = dx_right.size(0);
  int n = tmp / n_points;
  int c = tmp % n_points;  

  if (c < n_points && n < n_cells) {
    ret[0][n][0][c] = dx_right[n][0][c]*dy_bottom[n][0][c];
    ret[1][n][0][c] = dx_left[n][0][c]*dy_bottom[n][0][c];
    ret[2][n][0][c] = dx_right[n][0][c]*dy_top[n][0][c];
    ret[3][n][0][c] = dx_left[n][0][c]*dy_top[n][0][c];
  }
}

template <typename scalar_t> 
__global__ void gather_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input, // issue 함수명, 변수명
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_left,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_top,
const int IW, const int W, const int H, const int C, const int n_points, const int n_cells,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ret) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;
  int n = tmp / n_points;
  int c = tmp % n_points;

  if (c < ix_right.size(2) && n < n_cells ) {  // 이게 size(2)가 HW로 바뀌면서 c도 잘 바뀌었는지 화깅ㄴ해야 하는데.. issue 
  // 이젠 ix_right 얘네가 다 N, 1, HW임!!!

    long idx_nw = iy_top[n][0][c] * IW + ix_left[n][0][c];
    long idx_ne = iy_top[n][0][c] * IW + ix_right[n][0][c];
    long idx_sw = iy_bottom[n][0][c] * IW + ix_left[n][0][c];
    long idx_se = iy_bottom[n][0][c] * IW + ix_right[n][0][c];

#pragma unroll
    for (int i = 0; i<C; i++) {
      ret[0][n][i][c] =  input[n][i][idx_nw];   // 이것도 한줄로 가능함. input자체를 stack해서 보내면!!!
      ret[1][n][i][c] =  input[n][i][idx_ne];
      ret[2][n][i][c] =  input[n][i][idx_sw];
      ret[3][n][i][c] =  input[n][i][idx_se];
      // printf("%d %d %d--\n %d %d %d %d\n%f %f %f %f\n", n, i, c, idx_nw, idx_ne, idx_sw, idx_se, input[n][i][idx_nw],input[n][i][idx_ne],input[n][i][idx_sw],input[n][i][idx_se]);
    }
  }
}

template <typename scalar_t>
__global__ void interpolate_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> nw_val,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> nw,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ne_val,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ne,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> sw_val,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> sw,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> se_val,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> se,
const int C, const int n_points, const int n_cells,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> intpl_val) {
  // int n = gridDim.y;  // 무조건 010임;;
  // int n = blockIdx.y * blockDim.y + threadIdx.y;
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  // 얘가 800만 까지 쭉 감. 400까지는 갔음

  int n = tmp / n_points;
  int c = tmp % n_points;  // 이게 빠를지 for문으로해서 loop unrolling하는게 빠를지?

// for(int n=0; n < n_cells; n++) {
  if (c < nw_val.size(3) && n < n_cells) { 
#pragma unroll
    for (int i = 0; i<C; i++) {
      intpl_val[n][i][0][c] = nw_val[n][i][0][c] * nw[n][0][0][c] + ne_val[n][i][0][c] * ne[n][0][0][c] +
      sw_val[n][i][0][c] * sw[n][0][0][c] + se_val[n][i][0][c] * se[n][0][0][c];
    }
  }

// }  
}

// % 비싸니까 최대한 안쓰게 디자인 ㄱㄱ

// APP plz
// template <typename scalar_t> 
// __global__ void cell_backward_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_left,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_top,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> nw,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ne,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> sw,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> se,
// const int IW, const int C, const int n_points, const int n_cells,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ret) { // volatile 못쓸것같은뎀ㄴㅇ러나이런미ㅏ
//   int tmp = blockIdx.x * blockDim.x + threadIdx.x;
//   int n = tmp / n_points;
//   int c = tmp % n_points;

//   unsigned int tid = threadIdx.x;
//   // 이번 블록에 할당되는 데이터의 첫 지점을 찾아야 함
//   int first = blockIdx.x * blockDim.x;
//   int n_first = first / n_points;
//   int c_first = first % n_points;

//   // scalar_t* iData = 

//   int cell_dim = 1;
//   // volatile scalar_t* plz = new scalar_t [10 * 2 * 1 * 9]; // 이걸 shared memory로
//   __shared__ scalar_t plz[10 * 2 * 1 * 9];  // 얘는 항상 1024로 선언돼야 할듯. 애매하게 작게 들어오면 그걸로 할당했다가 뒤에서 + 64맞고 자멸할수도
//   volatile __shared__ scalar_t total[2*1*9];
//   // scalar_t plz[1024][2][3][9];
//   // torch::Tensor plz = torch.zeros({1024, 2, 3, 9});

//   /* grad랑 nw다 flat 됐다고 가정
//     접근은 c + i*E + n*C*E    c + n*E
//   */



//   if (c < grad.size(3) && n < n_cells) { // issue!!!! 이거 사이즈 호출하지말고 첨부터 걍 값 받아오면 되는데?;;
//     // printf("1tid:%d n:%d c:%d\n", tid, n, c);
//     int idx_nw = iy_top[n][0][c] * IW + ix_left[n][0][c]; /// t0 ~ t1023   idx
//     // printf("2tid:%d n:%d c:%d idx: %d\n", tid, n, c, idx_nw);
//     __syncthreads();
//     // printf("3tid:%d n:%d c:%d idx: %d\n", tid, n, c, idx_nw);
//     // printf("tid:%d n:%d c:%d idx:%d c+4: %d c-4: %d  &n:%p  &c: %p\n", tid, idx_nw, n, c, *(&c + 4), *(&c-4), &n, &c);
//     // printf("tid:%d n:%d c:%d idx:%d \n", tid,  n, c, idx_nw);
//     // long idx_ne = iy_top[n][0][c] * IW + ix_right[n][0][c];
//     // long idx_sw = iy_bottom[n][0][c] * IW + ix_left[n][0][c];
//     // long idx_se = iy_bottom[n][0][c] * IW + ix_right[n][0][c];
// // #pragma unroll 
//     for (int i = 0; i<C; i++) { 
//       // plz[tid][n][i][idx_nw] += grad[n][i][0][c] * nw[n][0][0][c]; // tid가 다르니 중복접근 x
//       plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw] = grad[n][i][0][c] * nw[n][0][0][c]; // val

//       printf("tid:%d, val: %f, idx_nw: %d, n:%d, i:%d, c:%d, loc: %d = %d + %d + %d + %d   data load\n", tid, plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw], idx_nw, n,i,c, tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw, tid * 2 * cell_dim * 9 , n*cell_dim*9 ,i * c, idx_nw);
//     }
//     __syncthreads();
//     if (tid == 1) {
//       for (int t = 0; t < 10; t++) {
//         for (int i = 0; i< 2; i++) {
//           for (int j = 0; j< cell_dim; j++) {
//             for (int k = 0; k<9; k++) {
//               printf("%f ", plz[t*2*cell_dim*9 + i*cell_dim*9 + j * cell_dim + k]);
//               // printf("%d %d %d store\n", i, j, k);
//             }
//           }
//         }
//       printf("\n");
//       }
//       printf("\n");
//       // ret[0][0][0] = plz[0];
//     }
//     /*
//     load 끝. 이제 각 스레드당 셀이 하나씩 있고 그 중 한 값만 채워진 상태. 즉 tid=0의 경우엔 1024*n*c*IW*IH개의 ele 중 하나만 채워져있음; 너무 sparse한데..
//     최대로 보면 1024개의 스레드가 존재하고 96*4*256 = 98304개의 엘리먼트 중 1024*4 = 4096개만 값을 지니고 나머지 9.4만개는 0임
//       --> 생각해보니까 1024*4가 아닌게 nw ne 다 따로 돌려야함. idx_nw랑 idx_ne간의 중복이 있을 수도 있어서..
//     근데 이게 모든 엘리먼트를 돌진 않지 않나?
//     */ 

//     // 일단 여기까지는 데이터 불러온거라고 생각하면 됨. 아래부터 summation

//     if (blockDim.x >= 1024 && tid < 512) {
//       for (int i = 0; i<C; i++) { 
//         // plz[tid][n][i][idx_nw] += plz[tid + 512][n][i][idx_nw];
//         plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw] += plz[(tid+512) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//         printf("tid:%d, val: %f at tid < 512\n", tid, plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw]);
//       }
//     }
//     __syncthreads();
    
    
//     if (blockDim.x >= 512 && tid < 256) {
//       for (int i = 0; i<C; i++) { 
//         // plz[tid][n][i][idx_nw] += plz[tid + 256][n][i][idx_nw];
//         plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw] += plz[(tid+256) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//         printf("tid:%d, val: %f at tid < 256\n", tid, plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw]);
//       }
//     }
//     __syncthreads();
    
//     if (blockDim.x >= 256 && tid < 128) {
//       for (int i = 0; i<C; i++) { 
//         // plz[tid][n][i][idx_nw] += plz[tid + 128][n][i][idx_nw];
//         plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw] += plz[(tid+128) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//         printf("tid:%d, val: %f at tid < 128\n", tid, plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw]);
//       }
//     }
//     __syncthreads();
    
//     if (blockDim.x >= 128 && tid < 64) {
//       for (int i = 0; i<C; i++) { 
//         // plz[tid][n][i][idx_nw] += plz[tid + 64][n][i][idx_nw];
//         plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw] += plz[(tid+64) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//         printf("tid:%d, val: %f at tid < 64\n", tid, plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw]);
//       }
//     }
//     __syncthreads();

//     if (tid < 32) {
//         volatile scalar_t* vsmem = plz; // tensor는 volatile이 안먹음; 그럼 여기서 무조건 scalar_t로 선언을 해야하는디 
//         for (int i = 0; i<C; i++) {
//           // plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw] += plz[(tid+32) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//           // printf("tid:%d, add: %f, to:%d, from:%d, from_tid: %d  sum: %f idx: %d at tid == 32\n", tid,plz[(tid+32) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw],tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw, (tid+32) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw,  tid+32, plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw], idx_nw);
//           //  __syncthreads();
//           // plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw] += plz[(tid+16) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//           // printf("tid:%d, add: %f, to:%d, from:%d, from_tid: %d  sum: %f idx: %d at tid == 16\n", tid,plz[(tid+16) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw],tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw, (tid+16) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw,  tid+16, plz[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw], idx_nw);
//           //  __syncthreads();
//           vsmem[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw] += vsmem[(tid+8) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//           // total[n*cell_dim*9 + i * c + idx_nw] = vsmem[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//           printf("tid:%d, add: %f, to:%d, from:%d, from_tid: %d  sum: %f idx: %d at tid == 8\n", tid, vsmem[(tid+8) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw],tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw, (tid+8) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw, tid+8, vsmem[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw], idx_nw);
//            __syncthreads();
//           vsmem[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw] += vsmem[(tid+4) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//           // total[n*cell_dim*9 + i * c + idx_nw] = vsmem[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//           printf("tid:%d, add: %f, to:%d, from:%d, from_tid: %d  sum: %f idx: %d at tid == 4\n", tid, vsmem[(tid+4) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw],tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw, (tid+4) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw, tid+4, vsmem[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw], idx_nw);
//            __syncthreads();
//           vsmem[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw] += vsmem[(tid+2) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//           // total[n*cell_dim*9 + i * c + idx_nw] = vsmem[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//           printf("tid:%d, add: %f, to:%d, from:%d, from_tid: %d  sum: %f idx: %d at tid == 2\n", tid, vsmem[(tid+2) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw],tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw, (tid+2) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw, tid+2, vsmem[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw], idx_nw);
//            __syncthreads();
//           vsmem[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw] += vsmem[(tid+1) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//           total[n*cell_dim*9 + i * c + idx_nw] = vsmem[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw];
//           printf("tid:%d, add: %f, to:%d, from:%d, from_tid: %d  sum: %f idx: %d at tid == 1\n", tid, vsmem[(tid+1) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw],tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw, (tid+1) * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw, tid+1, vsmem[tid * 2 * cell_dim * 9 + n*cell_dim*9 + i * c + idx_nw], idx_nw);
//            __syncthreads();
//           // plz[tid][n][i][idx_nw] += plz[tid + 32][n][i][idx_nw];
//           // plz[tid][n][i][idx_nw] += plz[tid + 16][n][i][idx_nw];
//           // plz[tid][n][i][idx_nw] += plz[tid +  8][n][i][idx_nw];
//           // plz[tid][n][i][idx_nw] += plz[tid +  4][n][i][idx_nw];
//           // plz[tid][n][i][idx_nw] += plz[tid +  2][n][i][idx_nw];
//           // plz[tid][n][i][idx_nw] += plz[tid +  1][n][i][idx_nw];
//         }
//     }

//     // if (tid < 32) {
//     //   /// tid마다 idx가 다름
//     //   vsmem[0] = vsmem[tid][idx_nw]
//     // }


//     if (tid == 0) {
//       for (int i = 0; i< 2; i++) {
//         for (int j = 0; j< cell_dim; j++) {
//           for (int k = 0; k<9; k++) {
//             ret[i][j][k] = total[ tid*2*cell_dim*9+ i*cell_dim*9 + j * cell_dim + k];
//             printf("%f stored at (%d, %d, %d) flat: %d\n", total[ tid*2*cell_dim*9+ i*cell_dim*9 + j * cell_dim + k], i, j, k, i*cell_dim*9 + j * cell_dim + k);
//             // printf("%d %d %d store\n", i, j, k);
//           }
//         }
//       }
//       // ret[0][0][0] = plz[0];
//     }

//   }
// }

// Base

__device__ float atomicAddDouble(float* address, float val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


template <typename scalar_t> 
__global__ void cell_backward_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_left,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_top,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> nw,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ne,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> sw,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> se,
const int IW, const int C, const int n_points, const int n_cells, const int t_num,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ret) { // volatile 못쓸것같은뎀ㄴㅇ러나이런미ㅏ
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;
  int n = tmp / n_points;
  int c = tmp % n_points;

  if (c < grad.size(3) && n < n_cells) { // issue!!!! 이거 사이즈 호출하지말고 첨부터 걍 값 받아오면 되는데?;;
    long idx_nw = iy_top[n][0][c] * IW + ix_left[n][0][c]; // 원래는 롱
    long idx_ne = iy_top[n][0][c] * IW + ix_right[n][0][c];
    long idx_sw = iy_bottom[n][0][c] * IW + ix_left[n][0][c];
    long idx_se = iy_bottom[n][0][c] * IW + ix_right[n][0][c];
// #pragma unroll 
    for (int i = 0; i<C; i++) { 
      atomicAdd(&ret[n][i][idx_nw],grad[n][i][0][c] * nw[n][0][0][c]);
      atomicAdd(&ret[n][i][idx_ne],grad[n][i][0][c] * ne[n][0][0][c]);
      atomicAdd(&ret[n][i][idx_sw],grad[n][i][0][c] * sw[n][0][0][c]);
      atomicAdd(&ret[n][i][idx_se],grad[n][i][0][c] * se[n][0][0][c]);
      // ret[n][i][idx_nw] += grad[n][i][0][c] * nw[n][0][0][c];
      // ret[n][i][idx_ne] += grad[n][i][0][c] * ne[n][0][0][c];
      // ret[n][i][idx_sw] += grad[n][i][0][c] * sw[n][0][0][c];
      // ret[n][i][idx_se] += grad[n][i][0][c] * se[n][0][0][c];
      // __threadfence();
      // __syncthreads();
    }

  }
}


// torch scatter
// template <typename scalar_t> 
// __global__ void cell_backward_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_left,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_top,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> nw,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ne,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> sw,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> se,
// const int IW, const int grid_size, const int C, const int n_points, const int n_cells, 
// torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> idx,
// torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> val) { // volatile 못쓸것같은뎀ㄴㅇ러나이런미ㅏ
// // int *idx, float* val, int *idx_ne, float* val_ne, int *idx_sw, float* val_sw, 
// // int *idx_se, float* val_se) { // volatile 못쓸것같은뎀ㄴㅇ러나이런미ㅏ
//   int tmp = blockIdx.x * blockDim.x + threadIdx.x;
//   int n = tmp / n_points;
//   int c = tmp % n_points;

//   // 이건 grad 등에 대해서 구하니까 input 크기가 아니라 grad 크기( N C 1 E)
//   // 근데 idx, val의 좌표(주소)는 grad의 크기에 맞게 되는게 맞음. 하지만 idx에 들어가는 값 자체는 input에 맞아야함!!

//   if (c < grad.size(3) && n < n_cells) { // issue!!!! 이거 사이즈 호출하지말고 첨부터 걍 값 받아오면 되는데?;; npoints
//     int offset_idx = c + n * n_points; 
//     int offset_dir = n_cells * C * n_points;
//     // int offset_idx = c + n * grid_size; 
//     // int offset_dir = n_cells * C * grid_size;
// #pragma unroll 
//     for (int i = 0; i<C; i++) { 
//       long offset_cell = n * C * grid_size + i * grid_size;
//       int offset_val = c + i*n_points + n*C*n_points;
//       // int offset_cell = n * C * n_points + i * n_points;
//       printf("tid: %d %d %d %f\n", threadIdx.x, offset_val, iy_top[n][0][c] * IW + ix_left[n][0][c] + offset_cell, grad[n][i][0][c] * nw[n][0][0][c]);
//       idx[offset_val] = iy_top[n][0][c] * IW + ix_left[n][0][c] + offset_cell;
//       idx[offset_val + offset_dir] = iy_top[n][0][c] * IW + ix_right[n][0][c] + offset_cell;
//       idx[offset_val + offset_dir * 2] = iy_bottom[n][0][c] * IW + ix_left[n][0][c] + offset_cell;
//       idx[offset_val + offset_dir * 3] = iy_bottom[n][0][c] * IW + ix_right[n][0][c] + offset_cell;

//       // int offset_val = c + i*grid_size + n*C*grid_size;
//       // int offset_val = c + i*grid_size + n*C*grid_size;
//       val[offset_val] = grad[n][i][0][c] * nw[n][0][0][c];
//       val[offset_val + offset_dir] = grad[n][i][0][c] * ne[n][0][0][c];
//       val[offset_val + offset_dir * 2] = grad[n][i][0][c] * sw[n][0][0][c];
//       val[offset_val + offset_dir * 3] = grad[n][i][0][c] * se[n][0][0][c];
//       printf("tid: %d  %d %f\t %d %d\n", threadIdx.x,  idx[offset_val], val[offset_val], offset_val, offset_cell);
//       printf("tid: %d  %d %f\t %d %d\n", threadIdx.x,  idx[offset_val+ offset_dir], val[offset_val + offset_dir], offset_val, offset_cell);
//       printf("tid: %d  %d %f\t %d %d\n", threadIdx.x,  idx[offset_val+ offset_dir*2], val[offset_val + offset_dir*2], offset_val, offset_cell);
//       printf("tid: %d  %d %f\t %d %d\n", threadIdx.x,  idx[offset_val+ offset_dir*3], val[offset_val + offset_dir*3], offset_val, offset_cell);
//     }

//   }
// }
// // thrust sort - reduce
// template <typename scalar_t> 
// __global__ void cell_backward_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_left,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_top,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> nw,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ne,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> sw,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> se,
// const int IW, const int grid_size, const int C, const int n_points, const int n_cells, 
// int *idx, float* val) { // volatile 못쓸것같은뎀ㄴㅇ러나이런미ㅏ
// // int *idx, float* val, int *idx_ne, float* val_ne, int *idx_sw, float* val_sw, 
// // int *idx_se, float* val_se) { // volatile 못쓸것같은뎀ㄴㅇ러나이런미ㅏ
//   int tmp = blockIdx.x * blockDim.x + threadIdx.x;
//   int n = tmp / n_points;
//   int c = tmp % n_points;

//   // 이건 grad 등에 대해서 구하니까 input 크기가 아니라 grad 크기( N C 1 E)
//   // 근데 idx, val의 좌표(주소)는 grad의 크기에 맞게 되는게 맞음. 하지만 idx에 들어가는 값 자체는 input에 맞아야함!!

//   if (c < grad.size(3) && n < n_cells) { // issue!!!! 이거 사이즈 호출하지말고 첨부터 걍 값 받아오면 되는데?;; npoints
//     int offset_idx = c + n * n_points; 
//     int offset_dir = n_cells * C * n_points;
//     // int offset_idx = c + n * grid_size; 
//     // int offset_dir = n_cells * C * grid_size;
// #pragma unroll 
//     for (int i = 0; i<C; i++) { 
//       int offset_cell = n * C * grid_size + i * grid_size;
//       int offset_val = c + i*n_points + n*C*n_points;
//       // int offset_cell = n * C * n_points + i * n_points;
//       idx[offset_val] = iy_top[n][0][c] * IW + ix_left[n][0][c] + offset_cell;
//       idx[offset_val + offset_dir] = iy_top[n][0][c] * IW + ix_right[n][0][c] + offset_cell;
//       idx[offset_val + offset_dir * 2] = iy_bottom[n][0][c] * IW + ix_left[n][0][c] + offset_cell;
//       idx[offset_val + offset_dir * 3] = iy_bottom[n][0][c] * IW + ix_right[n][0][c] + offset_cell;

//       // int offset_val = c + i*grid_size + n*C*grid_size;
//       // int offset_val = c + i*grid_size + n*C*grid_size;
//       val[offset_val] = grad[n][i][0][c] * nw[n][0][0][c];
//       val[offset_val + offset_dir] = grad[n][i][0][c] * ne[n][0][0][c];
//       val[offset_val + offset_dir * 2] = grad[n][i][0][c] * sw[n][0][0][c];
//       val[offset_val + offset_dir * 3] = grad[n][i][0][c] * se[n][0][0][c];
//       // printf("tid: %d  %d %f\t %d %d\n", threadIdx.x,  idx[offset_val], grad[n][i][0][c] * nw[n][0][0][c], offset_val, offset_cell);
//       // printf("tid: %d  %d %f\t %d %d\n", threadIdx.x,  idx[offset_val+ offset_dir], grad[n][i][0][c] * ne[n][0][0][c], offset_val, offset_cell);
//       // printf("tid: %d  %d %f\t %d %d\n", threadIdx.x,  idx[offset_val+ offset_dir*2], grad[n][i][0][c] * sw[n][0][0][c], offset_val, offset_cell);
//       // printf("tid: %d  %d %f\t %d %d\n", threadIdx.x,  idx[offset_val+ offset_dir*3], grad[n][i][0][c] * se[n][0][0][c], offset_val, offset_cell);
//     }

//   }
// }

template <typename scalar_t> // val이 float이 아니라 스칼라가 돼야하네
__global__ void migrate_kernel(int* idx, float* val, const int num_ele, 
torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ret,
const int n_cells) {
    int tmp = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("pid %d: %d\n", threadIdx.x, idx[0]);
    // printf("pid %d: %f\n", threadIdx.x, val[1]);
    // 접근을 어떻게..? 제한자가 있어야 하는디?
    if (tmp < num_ele) {
      ret[idx[tmp]] = val[tmp];
      // printf("pid: %d idx: %d ret: %f\n", threadIdx.x, idx[tmp], ret[tmp]);
    }

}

// APP1
// template <typename scalar_t> 
// __global__ void cell_backward_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_left,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_top,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> nw,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ne,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> sw,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> se,
// const int IW, const int C, const int n_points, const int n_cells, const int t_num,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> val,
// torch::PackedTensorAccessor32<int, 4, torch::RestrictPtrTraits> idx) { // 일단 3으로 받고 나중에 view해서 4차원으로
//   int tmp = blockIdx.x * blockDim.x + threadIdx.x;
//   int n = tmp / n_points;
//   int c = tmp % n_points;

//   if (c < grad.size(3) && n < n_cells) { // issue!!!! 이거 사이즈 호출하지말고 첨부터 걍 값 받아오면 되는데?;;
//     int idx_nw = iy_top[n][0][c] * IW + ix_left[n][0][c]; // 원래는 롱
//     int idx_ne = iy_top[n][0][c] * IW + ix_right[n][0][c];
//     int idx_sw = iy_bottom[n][0][c] * IW + ix_left[n][0][c];
//     int idx_se = iy_bottom[n][0][c] * IW + ix_right[n][0][c];
// #pragma unroll 
//     for (int i = 0; i<C; i++) { 
//       val[0][n][i][c] = grad[n][i][0][c] * nw[n][0][0][c];
//       val[1][n][i][c] = grad[n][i][0][c] * ne[n][0][0][c];
//       val[2][n][i][c] = grad[n][i][0][c] * sw[n][0][0][c];
//       val[3][n][i][c] = grad[n][i][0][c] * se[n][0][0][c];

//       idx[0][n][i][c] = idx_nw;
//       idx[1][n][i][c] = idx_ne;
//       idx[2][n][i][c] = idx_sw;
//       idx[3][n][i][c] = idx_se;
//     }

//   }
// }


// APP2
// template <typename scalar_t> 
// __global__ void cell_backward_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_left,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_top,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> nw,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ne,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> sw,
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> se,
// const int IW, const int C, const int n_points, const int n_cells, const int t_num,
// torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ret) { // 일단 3으로 받고 나중에 view해서 4차원으로
//   int tmp = blockIdx.x * blockDim.x + threadIdx.x;
//   int n = tmp / n_points;
//   int c = tmp % n_points;

//   // __shared__ scalar_t sm[1024][96][4][256];
//   // const int iwih = 9;
//   // __shared__ scalar_t sm[t_num][n_cells][C][iwih]; // 1024, 9가 아님;; 외부에서 스레드개수 받아야댐
//   int tid = threadIdx.x;
//   if (c < grad.size(3) && n < n_cells) { 
//     int idx_nw = iy_top[n][0][c] * IW + ix_left[n][0][c];
//     int idx_ne = iy_top[n][0][c] * IW + ix_right[n][0][c];
//     int idx_sw = iy_bottom[n][0][c] * IW + ix_left[n][0][c];
//     int idx_se = iy_bottom[n][0][c] * IW + ix_right[n][0][c];
      
// #pragma unroll 
//     for (int i = 0; i<C; i++) { 
//       float tmp = ret[n][i][idx_nw]; 
//       __syncthreads();
//       __threadfence();
//       ret[n][i][idx_nw] = tmp + grad[n][i][0][c] * nw[n][0][0][c];
//       __syncthreads();
//       __threadfence();
//       // ret[n][i][idx_ne] += grad[n][i][0][c] * ne[n][0][0][c];
//       // ret[n][i][idx_sw] += grad[n][i][0][c] * sw[n][0][0][c];
//       // ret[n][i][idx_se] += grad[n][i][0][c] * se[n][0][0][c];
//       printf("%d %d %d %f %f %s\n", n, i, idx_nw, grad[n][i][0][c] * nw[n][0][0][c], ret[n][i][idx_nw], "nw");
//       // printf("%d %d %d %f %f %s\n", n, i, idx_ne, grad[n][i][0][c] * ne[n][0][0][c], ret[n][i][idx_ne], "ne");
//       // printf("%d %d %d %f %f %s\n", n, i, idx_sw, grad[n][i][0][c] * sw[n][0][0][c], ret[n][i][idx_sw], "sw");
//       // printf("%d %d %d %f %f %s\n", n, i, idx_se, grad[n][i][0][c] * se[n][0][0][c], ret[n][i][idx_se], "se");
//     }

//   }
// }


template <typename scalar_t>
__global__ void get_point_backward_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> nw_val,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ne_val,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> sw_val,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> se_val,
const int C,
torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> ret) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  

  const int n_points = nw_val.size(3);
  const int n_cells = nw_val.size(0);
  int n = tmp / n_points;
  int c = tmp % n_points;  // issue. 포문 쓰지 말고 나누기 잘 하면 i의 좌표도 구할 수 있지 않나?

// for(int n=0; n < n_cells; n++) {
  if (c < n_points && n < n_cells) {   // 여기서 size를 매번 호출하느게 성능상 문제가 있을지?
#pragma unroll
    for (int i = 0; i<C; i++) {
      ret[0][n][i][0][c] = grad[n][i][0][c] * nw_val[n][i][0][c];
      ret[1][n][i][0][c] = grad[n][i][0][c] * ne_val[n][i][0][c];
      ret[2][n][i][0][c] = grad[n][i][0][c] * sw_val[n][i][0][c];
      ret[3][n][i][0][c] = grad[n][i][0][c] * se_val[n][i][0][c];
    }
  }

// }  
}

template <typename scalar_t>
__global__ void interpolate_backward_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> d_points,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dx_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dy_bottom,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy,
const int IW, const int IH,
torch::PackedTensorAccessor32<scalar_t, 6, torch::RestrictPtrTraits> d_grad) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  

  const int n_points = d_points.size(4);
  const int n_cells = d_points.size(1); // 5
  const int C = d_points.size(2); // 여기서 size를 매번 호출하느게 성능상 문제가 있을지? && 계산이랑
  int n = tmp / n_points;
  int c = tmp % n_points;  
  

// for(int n=0; n < n_cells; n++) { // 이게 빠를지 for문으로해서 loop unrolling하는게 빠를지?
  if (c < n_points && n < n_cells) {   
#pragma unroll
    for (int i = 0; i<C; i++) {
      // bilinear_new
      // d_grad[0][i][n][0][c] = -IW * 0.5 * (dy_bottom[n][0][c] *(d_points[0][i][n][0][c] - d_points[1][i][n][0][c]) + \
      //       (1 - dy_bottom[n][0][c]) * (d_points[2][i][n][0][c] - d_points[3][i][n][0][c]));
      // d_grad[1][i][n][0][c] = -IH * 0.5 * (dx_right[n][0][c] *(d_points[0][i][n][0][c] - d_points[2][i][n][0][c]) + \
      //       (1 - dx_right[n][0][c]) * (d_points[1][i][n][0][c] - d_points[3][i][n][0][c]));

      // cosine new      
      // d_grad[0][i][n][0][c] = -CUDART_PI_F * 0.5 * sin((ix_right[n][0][c] - ix[n][0][c])*CUDART_PI_F) * IW * 0.5 * (dy_bottom[n][0][c] *(d_points[0][i][n][0][c] - d_points[1][i][n][0][c]) + \
      //       (1 - dy_bottom[n][0][c]) * (d_points[2][i][n][0][c] - d_points[3][i][n][0][c]));
      // d_grad[1][i][n][0][c] = -CUDART_PI_F * 0.5 * sin((iy_bottom[n][0][c] - iy[n][0][c])*CUDART_PI_F) * IH * 0.5 * (dx_right[n][0][c] *(d_points[0][i][n][0][c] - d_points[2][i][n][0][c]) + \
      //       (1 - dx_right[n][0][c]) * (d_points[1][i][n][0][c] - d_points[3][i][n][0][c]));
      // n i 버전
      d_grad[0][n][i][0][c][0] = -CUDART_PI_F * 0.5 * sin((ix_right[n][0][c] - ix[n][0][c])*CUDART_PI_F) * IW * 0.5 * (dy_bottom[n][0][c] *(d_points[0][n][i][0][c] - d_points[1][n][i][0][c]) + \
            (1 - dy_bottom[n][0][c]) * (d_points[2][n][i][0][c] - d_points[3][n][i][0][c]));
      d_grad[1][n][i][0][c][0] = -CUDART_PI_F * 0.5 * sin((iy_bottom[n][0][c] - iy[n][0][c])*CUDART_PI_F) * IH * 0.5 * (dx_right[n][0][c] *(d_points[0][n][i][0][c] - d_points[2][n][i][0][c]) + \
            (1 - dx_right[n][0][c]) * (d_points[1][n][i][0][c] - d_points[3][n][i][0][c])); // grad를 안곱한 값(2차용)
      d_grad[2][n][i][0][c][0] = d_grad[0][n][i][0][c][0] * grad[n][i][0][c];   // 실제 미분값
      d_grad[3][n][i][0][c][0] = d_grad[1][n][i][0][c][0] * grad[n][i][0][c];


      // 상하좌우
      // d_grad[0][i][n][0][c] = dx_right[n][0][c] * d_points[2][i][n][0][c] + \
      //       dx_left[n][0][c] * d_points[3][i][n][0][c];
      // d_grad[1][i][n][0][c] = dx_right[n][0][c] * d_points[0][i][n][0][c] + \
      //       dx_left[n][0][c] * d_points[1][i][n][0][c];
      // d_grad[2][i][n][0][c] = dy_bottom[n][0][c] * d_points[1][i][n][0][c] + \
      //       dy_top[n][0][c] * d_points[3][i][n][0][c];
      // d_grad[3][i][n][0][c] = dy_bottom[n][0][c] * d_points[0][i][n][0][c] + \
      //       dy_top[n][0][c] * d_points[2][i][n][0][c];
      // bilinear
      // d_grad[0][i][n][0][c] = IW / 2 * (dy_bottom[n][0][c] *(d_points[1][i][n][0][c] - d_points[0][i][n][0][c]) + \
      //       dy_top[n][0][c] * (d_points[3][i][n][0][c] - d_points[2][i][n][0][c]));
      // d_grad[1][i][n][0][c] = IH / 2 * (dx_right[n][0][c] *(d_points[2][i][n][0][c] - d_points[0][i][n][0][c]) + \
      //       dx_left[n][0][c] * (d_points[3][i][n][0][c] - d_points[1][i][n][0][c]));
      // cosine
      // d_grad[0][i][n][0][c] = IW / 2 * CUDART_PI_F / 2 * sin((ix_right[n][0][c] - ix[n][0][c])*CUDART_PI_F) * (dy_bottom[n][0][c] * (d_points[1][i][n][0][c] - d_points[0][i][n][0][c]) + \
      //       dy_top[n][0][c] * (d_points[3][i][n][0][c] - d_points[2][i][n][0][c]));
      // d_grad[1][i][n][0][c] = IH / 2 * CUDART_PI_F / 2 * sin((iy_bottom[n][0][c] - iy[n][0][c])*CUDART_PI_F) * (dx_right[n][0][c] * (d_points[2][i][n][0][c] - d_points[0][i][n][0][c]) + \
      //       dx_left[n][0][c] * (d_points[3][i][n][0][c] - d_points[1][i][n][0][c]));
    }
  }


// }  
}

// template <typename scalar_t>
// __global__ void get_point_backward_backward_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> intpl, // 쌩 intpl
// torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_points,
// const int C,
// torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> ret) {
//   int tmp = blockIdx.x * blockDim.x + threadIdx.x;  

//   const int n_points = nw_val.size(3);
//   const int n_cells = nw_val.size(0);
//   int n = tmp / n_points;
//   int c = tmp % n_points;  // issue. 포문 쓰지 말고 나누기 잘 하면 i의 좌표도 구할 수 있지 않나?

// // for(int n=0; n < n_cells; n++) {
//   if (c < n_points && n < n_cells) {   // 여기서 size를 매번 호출하느게 성능상 문제가 있을지?
// #pragma unroll
//     for (int i = 0; i<C; i++) {
//       ret[0][n][i][0][c] = -2 * d_points[0][n][i][0][c] * tanh(intpl[n][i][0][c]);  // 이것도 미리 구해서 하는게;;
//       ret[1][n][i][0][c] = -2 * d_points[1][n][i][0][c] * tanh(intpl[n][i][0][c]);
//       ret[2][n][i][0][c] = -2 * d_points[2][n][i][0][c] * tanh(intpl[n][i][0][c]);
//       ret[3][n][i][0][c] = -2 * d_points[3][n][i][0][c] * tanh(intpl[n][i][0][c]);
//     }
//   }

// // }  
// }

template <typename scalar_t>
__global__ void interpolate_backward_backward_kernel(
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> saved_grad_out,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> x_grad,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> y_grad,
torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> d_points,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dx_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dy_bottom,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy,
const int IW, const int IH,
torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> dd_grad) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  

  const int n_points = d_points.size(4);
  const int n_cells = d_points.size(1); // 5
  const int C = d_points.size(2); // 여기서 size를 매번 호출하느게 성능상 문제가 있을지? && 계산이랑
  int n = tmp / n_points;
  int c = tmp % n_points;  // 이거 그냥 tmp - n 으로 구하는게 시간적으로 낫지 않나?
  

  if (c < n_points && n < n_cells) {   
#pragma unroll
    for (int i = 0; i<C; i++) {
      // bilinear_new
      // dd_grad[0][i][n][0][c] = -IW * 0.5 * (dy_bottom[n][0][c] *(d_points[0][i][n][0][c] - d_points[1][i][n][0][c]) + \
      //       (1 - dy_bottom[n][0][c]) * (d_points[2][i][n][0][c] - d_points[3][i][n][0][c]));
      // dd_grad[1][i][n][0][c] = -IH * 0.5 * (dx_right[n][0][c] *(d_points[0][i][n][0][c] - d_points[2][i][n][0][c]) + \
      //       (1 - dx_right[n][0][c]) * (d_points[1][i][n][0][c] - d_points[3][i][n][0][c]));

      // // cosine new       C N 1 E
      // dd_grad[0][c][0] += CUDART_PI_F * CUDART_PI_F * 0.5 * cos((ix_right[n][0][c] - ix[n][0][c])*CUDART_PI_F) * IW * 0.5 * (dy_bottom[n][0][c] *(d_points[0][n][i][0][c] - d_points[1][n][i][0][c]) + \
      //       (1 - dy_bottom[n][0][c]) * (d_points[2][n][i][0][c] - d_points[3][n][i][0][c])) * IW * 0.5 * saved_grad_out[n][i][0][c] ;

      // dd_grad[1][c][0] += CUDART_PI_F * CUDART_PI_F * 0.5 * cos((iy_bottom[n][0][c] - iy[n][0][c])*CUDART_PI_F) * IH * 0.5 * (dx_right[n][0][c] *(d_points[0][n][i][0][c] - d_points[2][n][i][0][c]) + \
      //       (1 - dx_right[n][0][c]) * (d_points[1][n][i][0][c] - d_points[3][n][i][0][c])) * IH * 0.5  * saved_grad_out[n][i][0][c] ;
      
      // grad 합치기 전 백업
      dd_grad[0][n][i][0][c] = CUDART_PI_F * CUDART_PI_F * 0.5 * cos((ix_right[n][0][c] - ix[n][0][c])*CUDART_PI_F) * IW * 0.5 * (dy_bottom[n][0][c] *(d_points[0][n][i][0][c] - d_points[1][n][i][0][c]) + \
            (1 - dy_bottom[n][0][c]) * (d_points[2][n][i][0][c] - d_points[3][n][i][0][c])) * IW * 0.5 * x_grad[i][0][c] * saved_grad_out[n][i][0][c] ;

      dd_grad[1][n][i][0][c] = CUDART_PI_F * CUDART_PI_F * 0.5 * cos((iy_bottom[n][0][c] - iy[n][0][c])*CUDART_PI_F) * IH * 0.5 * (dx_right[n][0][c] *(d_points[0][n][i][0][c] - d_points[2][n][i][0][c]) + \
            (1 - dx_right[n][0][c]) * (d_points[1][n][i][0][c] - d_points[3][n][i][0][c])) * IH * 0.5 * y_grad[i][0][c]* saved_grad_out[n][i][0][c] ;
      
      

      // cosine with tanh
      // scalar_t alpha_prime_x = CUDART_PI_F * CUDART_PI_F * 0.5 * cos((ix_right[n][0][c] - ix[n][0][c])*CUDART_PI_F) * IW * 0.5 * (dy_bottom[n][0][c] *(d_points[0][i][n][0][c] - d_points[1][i][n][0][c]) + \
      //       (1 - dy_bottom[n][0][c]) * (d_points[2][i][n][0][c] - d_points[3][i][n][0][c])) * IW * 0.5 * x_grad[i][n][0][c] ;
      // scalar_t alpha_prime_y = CUDART_PI_F * CUDART_PI_F * 0.5 * cos((iy_bottom[n][0][c] - iy[n][0][c])*CUDART_PI_F) * IH * 0.5 * (dx_right[n][0][c] *(d_points[0][i][n][0][c] - d_points[2][i][n][0][c]) + \
      //       (1 - dx_right[n][0][c]) * (d_points[1][i][n][0][c] - d_points[3][i][n][0][c])) * IH * 0.5 * y_grad[i][n][0][c];
      // scalar_t tanh_intpl = tanh(intpl[i][n][0][c]) // intpl이 n i 0 c가 아니라 i n 0 c 로 들어가있음!!!

      // scalar_t alpha = -CUDART_PI_F * 0.5 * sin((ix_right[n][0][c] - ix[n][0][c])*CUDART_PI_F) * IW * 0.5 * (dy_bottom[n][0][c] *(d_points[4][i][n][0][c] - d_points[5][i][n][0][c]) + \
      //       (1 - dy_bottom[n][0][c]) * (d_points[6][i][n][0][c] - d_points[7][i][n][0][c]));
      // dd_grad[0][i][n][0][c] = alpha_prime_x - (alpha_prime_x * pow(tanh_intpl, 2) + 2 * d_grad[i][n][0][c] * tanh_intpl * alpha); // alpha는 intpl을 x로 미분한 값
    }
  }

// }  


}
template <typename scalar_t>
__global__ void grad_backward_backward_kernel(
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_grad,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> x_grad,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> y_grad,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dd_grad) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  

  const int n_points = x_grad.size(2);
  // const int n_cells = x_grad.size(0); // 5
  const int C = x_grad.size(0); // 여기서 size를 매번 호출하느게 성능상 문제가 있을지? && 계산이랑
  int n = tmp / n_points;
  int c = tmp % n_points;  // 이거 그냥 tmp - n 으로 구하는게 시간적으로 낫지 않나?
  

  if (c < n_points) {   // && n < n_cells
#pragma unroll
    for (int i = 0; i<C; i++) {
      dd_grad[i][0][c] = d_grad[0][i][0][c] * x_grad[i][0][c] + d_grad[1][i][0][c] * y_grad[i][0][c];
    }
  }

// }  
}



torch::Tensor normalize_cuda(torch::Tensor input, int len) {
  int ele_num = input.numel();
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (input.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads));  // 원래는 grid.size(0)이었음. 상고나없긴 함. 근데 에러가 왜뜨지?
  torch::Tensor ret = torch::empty_like(input, input.options());
    AT_DISPATCH_FLOATING_TYPES(input.type(), "normalize_kernel", ([&] {
      normalize_kernel<scalar_t><<<blocks, threads>>>(input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
      ret.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), len);
    }));
  cudaDeviceSynchronize();
  return ret;
}

torch::Tensor normalize_offset_cuda(torch::Tensor input, int len, torch::Tensor offset) {
  int ele_num = input.numel();
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (input.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads));  // 원래는 grid.size(0)이었음. 상고나없긴 함. 근데 에러가 왜뜨지?
  torch::Tensor ret = torch::empty_like(input, input.options());
    AT_DISPATCH_FLOATING_TYPES(input.type(), "normalize_offset_kernel", ([&] {
      normalize_offset_kernel<scalar_t><<<blocks, threads>>>(input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
      ret.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), len,
      offset.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
    }));
  cudaDeviceSynchronize();
  return ret;
}

torch::Tensor get_corner_cuda(torch::Tensor ix, torch::Tensor iy) {
  const int ele_num = ix.numel(); // 정육면체라는 가정 하에.
  const int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  const dim3 blocks = (ix.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads));
  torch::Tensor ret = torch::zeros({4, ix.size(0), ix.size(1), ix.size(2)}, ix.options());
  AT_DISPATCH_FLOATING_TYPES(ix.type(), "get_corner_kernel", ([&] {
    get_corner_kernel<scalar_t><<<blocks, threads>>>(ix.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    iy.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    ret.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
 
  return ret;
}


// void step_bilinear(torch::Tensor ix, torch::Tensor ix_right, torch::Tensor iy, torch::Tensor iy_bottom,torch::Tensor ret, 
// dim3 blocks, int threads) {
//   AT_DISPATCH_FLOATING_TYPES(ix.type(), "step_bilinear", ([&] {
//     step_bilinear_kernel<scalar_t><<<blocks, threads>>>(ix.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     ix_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     iy.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     iy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     ret.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
//   }));
// }

// void step_smooth(torch::Tensor ix, torch::Tensor ix_right, torch::Tensor iy, torch::Tensor iy_bottom,torch::Tensor ret, 
// dim3 blocks, int threads) {
//   AT_DISPATCH_FLOATING_TYPES(ix.type(), "step_smooth", ([&] {
//     step_smooth_kernel<scalar_t><<<blocks, threads>>>(ix.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     ix_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     iy.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     iy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     ret.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
//   }));
// }

// void step_cosine(torch::Tensor ix, torch::Tensor ix_right, torch::Tensor iy, torch::Tensor iy_bottom,torch::Tensor ret, 
// dim3 blocks, int threads) {
//   AT_DISPATCH_FLOATING_TYPES(ix.type(), "step_cosine", ([&] {
//     step_cosine_kernel<scalar_t><<<blocks, threads>>>(ix.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     ix_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     iy.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     iy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     ret.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
//   }));
// }

torch::Tensor get_weight_cuda(torch::Tensor ix, torch::Tensor ix_right, torch::Tensor iy, torch::Tensor iy_bottom) {
  torch::Tensor weight = torch::empty({4, ix.size(0), ix.size(1), ix.size(2)}, ix.options());

  int ele_num = ix.numel();
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (ix.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads));
  AT_DISPATCH_FLOATING_TYPES(ix.type(), "step_cosine", ([&] {
    step_cosine_kernel<scalar_t><<<blocks, threads>>>(ix.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    ix_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    iy.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    iy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
  return weight;
}


torch::Tensor get_point_cuda(torch::Tensor dx_right, torch::Tensor dx_left, torch::Tensor dy_bottom, torch::Tensor dy_top) {
  int ele_num = dx_right.numel();
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (dx_right.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads));

  torch::Tensor point = torch::empty({4, dx_right.size(0), dx_right.size(1), dx_right.size(2)}, dx_right.options());

  AT_DISPATCH_FLOATING_TYPES(dx_right.type(), "compute_point_kernel", ([&] {
    compute_point_kernel<scalar_t><<<blocks, threads>>>(dx_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    dx_left.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    dy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    dy_top.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    point.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
  return point;
}

torch::Tensor gather_cuda(torch::Tensor input, torch::Tensor ix_right, torch::Tensor ix_left, torch::Tensor iy_bottom, torch::Tensor iy_top,
const int IW, const int W, const int H, const int C) {
  int ele_num = ix_right.numel();
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (ix_right.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads)); // 4 맞는디;;
  torch::Tensor ret = torch::empty({4, ix_right.size(0), C, ix_right.size(2)}, ix_right.options());
  const int n_points = ix_right.size(2);
  const int n_cells = ix_right.size(0);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "gather_kernel", ([&] {
    gather_kernel<scalar_t><<<blocks, threads>>>(input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    ix_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    ix_left.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    iy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    iy_top.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    IW, W, H, C, n_points, n_cells,
    ret.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
  return ret;
} 

torch::Tensor interpolate_cuda(torch::Tensor nw, torch::Tensor nw_val,torch::Tensor ne, torch::Tensor ne_val,
torch::Tensor sw, torch::Tensor sw_val,torch::Tensor se, torch::Tensor se_val, const int C) {
  int ele_num = nw_val.numel();
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  const int n_points = nw_val.size(3);
  const int n_cells = nw_val.size(0);
  dim3 blocks = (nw_val.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads) ); // issue. 아니 근데 나머지는 왜 잘되지? n_cell로 안나눠줘도?
  torch::Tensor ret = torch::zeros_like(nw_val, nw_val.options());
  AT_DISPATCH_FLOATING_TYPES(nw_val.type(), "interpolate_kernel", ([&] {
    interpolate_kernel<scalar_t><<<blocks, threads>>>(nw_val.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    nw.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    ne_val.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    ne.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    sw_val.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    sw.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    se_val.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    se.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(), C, n_points, n_cells,
    ret.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
  return ret;
}

// APP1
torch::Tensor cell_backward_cuda(torch::Tensor input, torch::Tensor grad, torch::Tensor ix_right, torch::Tensor ix_left, torch::Tensor iy_bottom, torch::Tensor iy_top,
torch::Tensor nw, torch::Tensor ne, torch::Tensor sw, torch::Tensor se,
const int IW, const int C) {
  int ele_num = ix_right.numel();
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (ix_right.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads)); // 4 맞는디;;
  torch::Tensor ret = torch::zeros_like(input, input.options());
  torch::Tensor val = torch::zeros({4, grad.size(0), grad.size(1), grad.size(3)}, grad.options());
  torch::Tensor idx = torch::zeros({4, grad.size(0), grad.size(1), grad.size(3)}, torch::kInt32);
  // torch::Tensor ret = torch::zeros({ix_right.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads), input.size(0), input.size(1), input.size(2)}, input.options());
  const int n_points = ix_right.size(2);
  const int n_cells = ix_right.size(0);


  AT_DISPATCH_FLOATING_TYPES(input.type(), "cell_backward_kernel", ([&] {
    cell_backward_kernel<scalar_t><<<blocks, threads>>>(grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    ix_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    ix_left.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    iy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    iy_top.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    nw.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    ne.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    sw.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    se.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    IW, C, n_points, n_cells, threads,
    ret.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
  // // ret = ret.view({n_cells, C, IW, IH });
  return ret;
} 


// APP3 cuda scatter
// std::vector<torch::Tensor> cell_backward_cuda(torch::Tensor input, torch::Tensor grad, torch::Tensor ix_right, torch::Tensor ix_left, torch::Tensor iy_bottom, torch::Tensor iy_top,
// torch::Tensor nw, torch::Tensor ne, torch::Tensor sw, torch::Tensor se,
// const int IW, const int IH, const int C) {
//   int ele_num = grad.numel(); 
//   int input_num = input.numel();
//   int threads = std::min<int>(CUDA_MAX_THREADS, std::max<int>(ele_num, input_num)); 
//   dim3 blocks = (ix_right.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads)); // 4 맞는디;;
//   cout<<"thread: "<<threads<< " bloacks: " << CUDA_N_BLOCKS_NEEDED(ele_num, threads) << endl;
//   std::vector<torch::Tensor> ret;
//   torch::Tensor idx = torch::empty(ele_num, grad.options());
//   idx = idx.to(torch::kInt64);
//   torch::Tensor val = torch::empty(ele_num, grad.options());
//   const int n_points = ix_right.size(2);
//   const int n_cells = ix_right.size(0);
//   // float* ret = 0;
//   // cudaMalloc((void**)&ret, sizeof(float)*2*3*9);


//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);

//   cudaEventRecord(start);
//   cout <<"Total ele: " << ele_num<<endl;
//   cout <<"ix_r ele: " << ix_right.numel()<<endl;

//   float time;
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);
//   cudaEventElapsedTime(&time, start, stop);
//   cudaEventDestroy(start);
//   cudaEventDestroy(stop);

//   cout <<"init "<<time << endl;

//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);
//   cudaEventRecord(start);
//   int grid_size = IH * IW;
//   AT_DISPATCH_FLOATING_TYPES(input.type(), "cell_backward_kernel", ([&] {
//     cell_backward_kernel<scalar_t><<<blocks, threads>>>(grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//     ix_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     ix_left.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     iy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     iy_top.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     nw.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//     ne.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//     sw.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//     se.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//     IW, grid_size, C, n_points, n_cells,
//     idx.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
//     val.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
//     // idx_nw, val_nw, idx_ne, val_ne,idx_sw, val_sw,idx_ne, val_se);
//   }));
//   cudaDeviceSynchronize();
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);
//   cudaEventElapsedTime(&time, start, stop);
//   cout <<"cell_back "<<time << endl;
//   cudaEventDestroy(start);
//   cudaEventDestroy(stop);
//   ret.push_back(idx);
//   ret.push_back(val);
//   for (int i = 0; i< 5; i++) {
//     cout<<i<<" ";
//     cout << idx[i] << " " << val[i]<<endl;
//   }

  
//   return ret;
// } 


// torch::Tensor cell_backward_cuda(torch::Tensor input, torch::Tensor grad, torch::Tensor ix_right, torch::Tensor ix_left, torch::Tensor iy_bottom, torch::Tensor iy_top,
// torch::Tensor nw, torch::Tensor ne, torch::Tensor sw, torch::Tensor se,
// const int IW, const int IH, const int C) {
//   int ele_num = grad.numel(); 
//   int input_num = input.numel();
//   int threads = std::min<int>(CUDA_MAX_THREADS, std::max<int>(ele_num, input_num)); 
//   dim3 blocks = (ix_right.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads)); // 4 맞는디;;
//   cout<<"thread: "<<threads<< " bloacks: " << CUDA_N_BLOCKS_NEEDED(ele_num, threads) << endl;
//   torch::Tensor ret = torch::zeros(input.numel(), input.options());
//   // torch::Tensor ret = torch::zeros_like(input, input.options());
//   // torch::Tensor ret = torch::zeros({ix_right.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads), input.size(0), input.size(1), input.size(2)}, input.options());
//   const int n_points = ix_right.size(2);
//   const int n_cells = ix_right.size(0);
//   // float* ret = 0;
//   // cudaMalloc((void**)&ret, sizeof(float)*2*3*9);


//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);

//   cudaEventRecord(start);
//   cout <<"Total ele: " << ele_num<<endl;
//   cout <<"ix_r ele: " << ix_right.numel()<<endl;
//   // int *idx = new int[ele_num * 4];
//   // float* val = new float[ele_num * 4];
//   int *out_idx = new int[input_num];
//   float* out_val = new float[input_num];
//   memset(out_idx,0,sizeof(int)*input_num );

//   int* idx;
//   float* val;
//   // int* out_idx;
//   // float* out_val;
//   // float* ret;
//   cudaMalloc((void**)&idx, 4 * ele_num * sizeof(int));
//   cudaMalloc((void**)&val, 4 * ele_num * sizeof(float));
//   // cudaMalloc((void**)&out_idx, 4 * ele_num * sizeof(int));
//   // cudaMalloc((void**)&out_val, 4 * ele_num * sizeof(float));
//   // cudaMalloc((void**)&ret, 4 * input_num * sizeof(float));
//   cudaMemset(idx, 1, 4 * ele_num * sizeof(int));
//   cudaMemset(val, 1, 4 * ele_num * sizeof(float));
//   // cudaMemset(ret, 0, 4 * ele_num * sizeof(float));
//   int * host_idx = (int*)malloc(sizeof(int) * 4 * ele_num);
//   float * host_val = (float*)malloc(sizeof(float) * 4 * ele_num);
//   // int *idx_nw = new int[ele_num];
//   // float* val_nw = new float[ele_num];
//   // int *idx_ne = new int[ele_num];
//   // float* val_ne = new float[ele_num];
//   // int *idx_sw = new int[ele_num];
//   // float* val_sw = new float[ele_num];
//   // int *idx_se = new int[ele_num];
//   // float* val_se = new float[ele_num];
//   // memset(idx, 1, sizeof(int)*ele_num * 4); // 해당 안되는 곳은 -1로 초기화
//   // memset(val, 1, sizeof(float)*ele_num * 4);
//   // memset(idx_nw, 1, sizeof(int)*ele_num); // 해당 안되는 곳은 -1로 초기화
//   // memset(idx_nw, 1, sizeof(int)*ele_num);
//   // memset(idx_ne, 1, sizeof(int)*ele_num);
//   // memset(idx_ne, 1, sizeof(int)*ele_num);
//   // memset(val_sw, 1, sizeof(float)*ele_num);
//   // memset(val_sw, 1, sizeof(float)*ele_num);
//   // memset(val_se, 1, sizeof(float)*ele_num);
//   // memset(val_se, 1, sizeof(float)*ele_num);

//   // cout<<"memset"<<endl;
//   // for (int i = 0; i< ele_num*4; i++) {
//   //   cout << idx[i] << " ";
//   // }
//   // cout<<endl;
//   float time;
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);
//   cudaEventElapsedTime(&time, start, stop);
//   cudaEventDestroy(start);
//   cudaEventDestroy(stop);

//   cout <<"init "<<time << endl;

//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);
//   cudaEventRecord(start);
//   int grid_size = IH * IW;
//   AT_DISPATCH_FLOATING_TYPES(input.type(), "cell_backward_kernel", ([&] {
//     cell_backward_kernel<scalar_t><<<blocks, threads>>>(grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//     ix_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     ix_left.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     iy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     iy_top.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//     nw.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//     ne.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//     sw.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//     se.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//     IW, grid_size, C, n_points, n_cells,
//     idx, val);
//     // idx_nw, val_nw, idx_ne, val_ne,idx_sw, val_sw,idx_ne, val_se);
//   }));
//   cudaDeviceSynchronize();
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);
//   cudaEventElapsedTime(&time, start, stop);
//   cout <<"cell_back "<<time << endl;
//   cudaEventDestroy(start);
//   cudaEventDestroy(stop);

//   cudaMemcpy(host_idx, idx, sizeof(int) * 4 * ele_num, cudaMemcpyDeviceToHost);
//   cudaMemcpy(host_val, val, sizeof(float) * 4 * ele_num, cudaMemcpyDeviceToHost);
  
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);
//   cudaEventRecord(start);
//   thrust::sort_by_key(thrust::host, host_idx, host_idx + ele_num * 4, host_val);
//   cout<<"sort"<<endl;
//   // for (int i = 0; i< ele_num*4; i++) {
//   //   cout << host_idx[i] << " "<< host_val[i] <<endl;
//   // }
//   cudaDeviceSynchronize();
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);
//   cudaEventElapsedTime(&time, start, stop);
//   cout <<"sort "<<time << endl;
//   cudaEventDestroy(start);
//   cudaEventDestroy(stop);
//   cout<<endl;


//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);
//   cudaEventRecord(start);
//   thrust::pair<int*,float*> new_end;
//   new_end = thrust::reduce_by_key(thrust::host, host_idx, host_idx + ele_num*4, host_val, out_idx, out_val);
//   cudaDeviceSynchronize();
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);
//   cudaEventElapsedTime(&time, start, stop);
//   cout <<"reduce "<<time << endl;
//   cudaEventDestroy(start);
//   cudaEventDestroy(stop);  
//   int owowow = new_end.first - out_idx;
//   cout << owowow << "!!!!\n";
//   cout << *new_end.first <<" "<<*new_end.second<<endl;
//   cout << *(new_end).first <<" "<<*(new_end).second<<endl;


//   // cout << "redice\n";
//   // for (int i = 0; i< 18; i++) {
//   //   cout << out_idx[i] << " " << out_val[i] << endl;
//   // }
//   int* device_out_idx;
//   float* device_out_val;
//   cudaMalloc((void**)&device_out_idx, sizeof(int)* input_num);
//   cudaMalloc((void**)&device_out_val, sizeof(float)* input_num);
//   cudaMemcpy(device_out_idx, out_idx, sizeof(int) * input_num, cudaMemcpyHostToDevice);
//   cudaMemcpy(device_out_val, out_val, sizeof(float) * input_num, cudaMemcpyHostToDevice);
//   printf("cpy done\n"); 
//   // idx가 18보다 작으니까..
//   AT_DISPATCH_FLOATING_TYPES(input.type(), "migrate_kernel", ([&] {
//   migrate_kernel<scalar_t><<<blocks, threads>>>(device_out_idx, device_out_val, input_num,
//   ret.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
//   n_cells);
//   }));
//   if (out_idx[0] == 0) {
//     // printf("overlap\n");
//     ret[0] = out_val[0];
//   }

//   cudaDeviceSynchronize();

//   // cout <<"plz...."<<endl;
//   // for (int i = 0; i< input.numel(); i++) {
//   //   cout << ret[i] << " ";
//   // }
//   cout << endl;
//   // cout << endl;
//   // for (int i = 0; i< input.numel(); i++) {
//   //   cout << out_idx[i] << " ";
//   // }
//   // cout << endl;
//   // cout << endl;
//   // for (int i = 0; i< input.numel(); i++) {
//   //   cout << out_val[i] << " ";
//   // }
//   // cout << endl;
  
  
//   // out_val 사이에 0을 넣어야 함
//   // 무지성이면 걍 다 0으로 초기화해놓고 out_idx보고 넣으면 되긴 함 => gather 
//   // 만약 0이 들어와있다면 걍 out_val 리턴하면 돼서 좋긴한데.. 일단은 무지성 게더 ㄱ => arr이 아니라 vec라서 안댐. 
//   // 커널로 하면? 커널 내에서 옮기면 빠르지 않나?
//   // 파트 별로 시간 측정하기!!!! policy에서 device와 host의 차이?
//   cout << endl;
  


//   // // ret = ret.view({n_cells, C, IW, IH });
//   return ret;
// } 


torch::Tensor get_point_backward_cuda(torch::Tensor grad, torch::Tensor nw_val, torch::Tensor ne_val, torch::Tensor sw_val,  
torch::Tensor se_val, const int C) {
  int ele_num = grad.numel() / C;   // 어차피 repeat 으로 생성된애는 커널 내부에서 for문으로 똑같이 넣음.
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (grad.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads));

  torch::Tensor d_points = torch::empty({4, grad.size(0), grad.size(1),grad.size(2),grad.size(3)}, torch::requires_grad().device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "get_point_backward_kernel", ([&] {
    get_point_backward_kernel<scalar_t><<<blocks, threads>>>(grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    nw_val.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    ne_val.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    sw_val.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    se_val.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(), 
    C,
    d_points.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
  }));
  // d_points = d_points.view({d_points.size(0), d_points.size(2), d_points.size(1), d_points.size(3), d_points.size(4)});
  // d_points = d_points.transpose(1,2);
  cudaDeviceSynchronize();
  return d_points;
}

// issue 일단은 grad 남겼음 나중에 지우기 
torch::Tensor interpolate_backward_cuda(torch::Tensor grad, torch::Tensor d_points, torch::Tensor dx_right, torch::Tensor dy_bottom,
torch::Tensor ix_right, torch::Tensor iy_bottom, torch::Tensor ix, torch::Tensor iy,
const int N, const int C, const int IW, const int IH) {

  int ele_num = grad.numel() / C;   // 어차피 repeat 으로 생성된애는 커널 내부에서 for문으로 똑같이 넣음.
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (grad.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads));
  torch::Tensor d_grad = torch::zeros({4, grad.size(0), grad.size(1),grad.size(2),grad.size(3), 1}, torch::requires_grad().device(torch::kCUDA));
  // torch::Tensor d_grad = torch::zeros({2, grad.size(1), grad.size(0),grad.size(2),grad.size(3)}, torch::requires_grad().device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "interpolate_backward_kernel", ([&] {
    interpolate_backward_kernel<scalar_t><<<blocks, threads>>>(grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    d_points.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
    dx_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    dy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    ix_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    iy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    ix.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    iy.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    IW, IH,
    d_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
  return d_grad;
}



torch::Tensor interpolate_backward_backward_cuda(torch::Tensor saved_grad_out, torch::Tensor d_points, torch::Tensor x_grad, torch::Tensor y_grad, torch::Tensor dx_right, 
torch::Tensor dy_bottom, torch::Tensor ix_right, torch::Tensor iy_bottom, torch::Tensor ix, torch::Tensor iy,
const int IW, const int IH) {
  int ele_num = saved_grad_out.numel();
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (saved_grad_out.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads));
  torch::Tensor dd_grad = torch::zeros({2, saved_grad_out.size(0), saved_grad_out.size(1),saved_grad_out.size(2),saved_grad_out.size(3)}, saved_grad_out.options());
  // torch::Tensor dd_grad = torch::zeros({2, saved_grad_out.size(3), 1}, saved_grad_out.options());
  AT_DISPATCH_FLOATING_TYPES(saved_grad_out.type(), "interpolate_backward_backward_kernel", ([&] {
    interpolate_backward_backward_kernel<scalar_t><<<blocks, threads>>>(saved_grad_out.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    x_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    y_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    d_points.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
    dx_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    dy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    ix_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    iy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    ix.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    iy.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    IW, IH,
    dd_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
  // dd_grad = dd_grad.view({dd_grad.size(0), dd_grad.size(2), dd_grad.size(1), dd_grad.size(3), dd_grad.size(4)}); // transpose 쓰기..
  return dd_grad;
}

torch::Tensor grad_backward_backward_cuda(torch::Tensor d_grad, torch::Tensor x_grad, torch::Tensor y_grad) {
  int ele_num = x_grad.numel();
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (x_grad.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads));
  torch::Tensor dd_grad = torch::zeros({x_grad.size(0),x_grad.size(1),x_grad.size(2)}, x_grad.options());
  AT_DISPATCH_FLOATING_TYPES(x_grad.type(), "grad_backward_backward_kernel", ([&] {
    grad_backward_backward_kernel<scalar_t><<<blocks, threads>>>(d_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    x_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    y_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    dd_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
  return dd_grad;
}

