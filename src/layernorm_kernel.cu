#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/

/*
  (JHY): helper functions
*/
__device__ float4 operator+(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ float4 operator-(float4 a, float4 b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__device__ float4 operator*(float4 a, float4 b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__device__ float4 operator*(float4 a, float scalar) {
  return make_float4(a.x * scalar, a.y * scalar, a.z * scalar, a.w * scalar);
}
__device__ float4 operator+(float4 a, float addition) {
  return make_float4(a.x + addition, a.y + addition, a.z + addition, a.w + addition);
}
__device__ float4 operator-(float4 a, float subtr) {
  return make_float4(a.x - subtr, a.y - subtr, a.z - subtr, a.w - subtr);
}
__device__ float  sum_reduce(float4 a){
  return a.x + a.y + a.z + a.w;
}


template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1
  float l_sum[2] = {0.0, 0.0};
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;  
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum[0] += val.x + val.y + val.z + val.w;
    l_sum[1] += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }

  // Step 2
  blockReduce<ReduceType::kSum, 2>(l_sum);
  float mean_ = l_sum[0] / (hidden_size * 4.0);
  float var_ = l_sum[1] / (hidden_size * 4.0) - mean_ * mean_;
  float mul_fac = __fdividef(1.0f, __fsqrt_rn(var_ + LN_EPSILON));

  if (threadIdx.x == 0){
    // write to means and vars array
    *(vars + blockIdx.x) = var_;
    *(means + blockIdx.x) = mean_;
  }

  // Step 3
  float4 *ln_res_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    ln_res_f4[idx] = scale_f4[idx] * ((inp_f4[idx] - mean_) * mul_fac) + bias_f4[idx];
  }
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2; // thus we can group 4 adjacent elements together in hidden_dim
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  // 4. Assign the final result to the correct position in the global output

  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float vars_local[TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];
  __shared__ float means_local[TILE_DIM];
  __shared__ float betta_grad_[TILE_DIM];
  __shared__ float gamma_grad_[TILE_DIM];
  const int        row_roof = (rows + TILE_DIM - 1) / TILE_DIM * TILE_DIM;

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  const int col_off = threadIdx.y;
  const int row_off = threadIdx.x;
  const int col = blockIdx.x * blockDim.x + col_off;
  
  // initialize the result array
  if (row_off == 0){
    betta_grad_[col_off] = gamma_grad_[col_off] = 0.0;
  }

  // add across the row_axis
  for(int row = row_off; row < row_roof; row += blockDim.y){
    float betta_grad__ = 0.0, gamma_grad__ = 0.0;
    // first load global array into shared memory
    if (row < rows){
      if (col_off == 0){
        vars_local[row_off] = vars[row];
        means_local[row_off] = means[row];
      }
      betta_buffer[row_off][col_off] = out_grad[row * width + col];
      gamma_buffer[row_off][col_off] = inp[row * width + col];
    }
    else{
      if (col_off == 0){
        vars_local[row_off] = 1.0f;
        means_local[row_off] = 0.0f;
      }
      betta_buffer[row_off][col_off] = 0.0;
      gamma_buffer[row_off][col_off] = 0.0;
    }
    __syncthreads();

    // then compute the local grad
    betta_grad__ = betta_buffer[row_off][col_off];
    gamma_grad__ = __fdividef(
                    gamma_buffer[row_off][col_off] - means_local[row_off], 
                    __fsqrt_rn(
                      vars_local[row_off] + LN_EPSILON
                    )
                  ) * betta_grad__;
    __syncthreads();
              
    // then reduce inside a warp
    for (int i = 1; i < TILE_DIM; i <<= 1) {
      betta_grad__ += g.shfl_xor(betta_grad__, i);
      gamma_grad__ += g.shfl_xor(gamma_grad__, i);
    }

    // add to the shared memory to record
    if (row_off == 0){
      betta_grad_[col_off] += betta_grad__;
      gamma_grad_[col_off] += gamma_grad__;
    }
    
  }

  // write back to the global memory
  if (row_off == 0){
    betta_grad[col] = betta_grad_[col_off];
    gamma_grad[col] = gamma_grad_[col_off];
  }
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *vars,
                               const T *means, int hidden_size) {
  
  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient
  
  
  // Step 1: load vars and mean
  const int row = blockIdx.x;
  const int col_base = threadIdx.x;
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + row * hidden_size;  
  const float4 *out_grad_f4 = reinterpret_cast<const float4 *>(out_grad) + row * hidden_size; 
  float4 *inp_grad_f4 = reinterpret_cast<float4 *>(inp_grad) + row * hidden_size; // need to save the output
  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);
  __shared__ float local_mean, local_sigma_inv;
  if (col_base == 0){
    local_mean = means[row];
    local_sigma_inv = __fdividef(1.0f, __fsqrt_rn(vars[row] + LN_EPSILON));
  }
  __syncthreads();

  // Step 2.1: Walk through the region this thread for, do a local reduce
  float l_sum[2] = {0.0, 0.0};
  for (uint idx = col_base; idx < hidden_size; idx += blockDim.x) {
    float4 g_o_mul_gamma = out_grad_f4[idx] * gamma_f4[idx];
    l_sum[0] += sum_reduce(g_o_mul_gamma);
    l_sum[1] += sum_reduce(g_o_mul_gamma * ((inp_f4[idx] - local_mean) * local_sigma_inv));
  }

  // Step 2.2: Block Reduce
  blockReduce<ReduceType::kSum, 2>(l_sum);


  // Step 3: Write back result
  for (uint idx = col_base; idx < hidden_size; idx += blockDim.x) {
    inp_grad_f4[idx] = (
      (
        out_grad_f4[idx] 
          * 
        gamma_f4[idx] 
          * 
        local_sigma_inv
      ) 
        - 
      (
        ((inp_f4[idx] - local_mean) * local_sigma_inv) 
          * 
        l_sum[1] 
          + 
        l_sum[0]
      ) 
        * 
      __fdividef(local_sigma_inv, hidden_size * 4.0f)
    );
  }

}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  // The result is then multiplied by TILE_DIM to ensure that the grid size is a multiple of TILE_DIM.
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM));
  /*
    Prevously, the code in the above line is: dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
    IDK which idiot comes up with that...
    Actually if it's that way, each block will be responsible for a single col's reduction, thus we do not need a 2D shared Matrix...
    Such a stupid a** h*** (* TILE_DIM)
  */
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
