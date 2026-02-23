#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdint.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Block size for matrix multiplication
#define BLOCK_SIZE 16
#define TILE_SIZE 16

// ============================================================================
// INT8 Quantized Matrix Multiplication Kernel
// ============================================================================
__global__ void int8_matmul_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    int M, int N, int K,
    float scale_a, float scale_b, float scale_c
) {
    // Shared memory for tiles
    __shared__ int8_t tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t tile_b[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    int32_t sum = 0;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile A
        int a_row = row;
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (a_row < M && a_col < K) {
            tile_a[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        
        // Load tile B
        int b_row = tile * TILE_SIZE + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N) {
            tile_b[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += (int32_t)tile_a[threadIdx.y][k] * (int32_t)tile_b[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        // Apply scaling and convert to output format
        float result = (float)sum * scale_a * scale_b * scale_c;
        C[row * N + col] = (int32_t)result;
    }
}

// ============================================================================
// INT4 Packed Quantized Matrix Multiplication Kernel
// ============================================================================
__global__ void int4_matmul_kernel(
    const uint8_t* __restrict__ A_packed,  // Packed INT4: 2 values per byte
    const uint8_t* __restrict__ B_packed,
    int32_t* __restrict__ C,
    int M, int N, int K,
    float scale_a, float scale_b, float scale_c
) {
    __shared__ uint8_t tile_a[TILE_SIZE][TILE_SIZE / 2];  // Packed: 2 values per byte
    __shared__ uint8_t tile_b[TILE_SIZE / 2][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    int32_t sum = 0;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load and unpack tile A
        int a_row = row;
        int a_col_packed = (tile * TILE_SIZE + threadIdx.x * 2) / 2;
        int a_col_offset = (tile * TILE_SIZE + threadIdx.x * 2) % 2;
        
        if (a_row < M && a_col_packed < (K + 1) / 2) {
            uint8_t packed_val = A_packed[a_row * ((K + 1) / 2) + a_col_packed];
            int8_t val1 = (packed_val & 0x0F) - 8;  // Unpack and dequantize (assuming zero-point 8)
            int8_t val2 = ((packed_val >> 4) & 0x0F) - 8;
            tile_a[threadIdx.y][threadIdx.x / 2] = packed_val;
        } else {
            tile_a[threadIdx.y][threadIdx.x / 2] = 0;
        }
        
        // Load and unpack tile B
        int b_row_packed = (tile * TILE_SIZE + threadIdx.y * 2) / 2;
        int b_col = col;
        
        if (b_row_packed < (K + 1) / 2 && b_col < N) {
            uint8_t packed_val = B_packed[b_row_packed * N + b_col];
            tile_b[threadIdx.y / 2][threadIdx.x] = packed_val;
        } else {
            tile_b[threadIdx.y / 2][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product with unpacking
        for (int k = 0; k < TILE_SIZE / 2; ++k) {
            uint8_t a_packed = tile_a[threadIdx.y][k];
            int8_t a1 = (a_packed & 0x0F) - 8;
            int8_t a2 = ((a_packed >> 4) & 0x0F) - 8;
            
            uint8_t b_packed = tile_b[k][threadIdx.x];
            int8_t b1 = (b_packed & 0x0F) - 8;
            int8_t b2 = ((b_packed >> 4) & 0x0F) - 8;
            
            sum += (int32_t)a1 * (int32_t)b1 + (int32_t)a2 * (int32_t)b2;
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        float result = (float)sum * scale_a * scale_b * scale_c;
        C[row * N + col] = (int32_t)result;
    }
}

// ============================================================================
// Dequantization Kernel (INT8/INT4 -> FP16/FP32)
// ============================================================================
__global__ void dequantize_kernel(
    const int8_t* __restrict__ quantized,
    half* __restrict__ output,
    int size,
    float scale,
    float zero_point
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float dequantized = ((float)quantized[idx] - zero_point) * scale;
        output[idx] = __float2half(dequantized);
    }
}

__global__ void dequantize_fp32_kernel(
    const int8_t* __restrict__ quantized,
    float* __restrict__ output,
    int size,
    float scale,
    float zero_point
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = ((float)quantized[idx] - zero_point) * scale;
    }
}

// ============================================================================
// Quantization Kernel (FP32 -> INT8)
// ============================================================================
__global__ void quantize_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    int size,
    float scale,
    float zero_point
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float quantized = input[idx] / scale + zero_point;
        quantized = fmaxf(-128.0f, fminf(127.0f, quantized));  // Clamp to INT8 range
        output[idx] = (int8_t)roundf(quantized);
    }
}

// ============================================================================
// Fused Quantized MatMul + Bias + Activation (ReLU)
// ============================================================================
__global__ void fused_int8_matmul_bias_relu_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K,
    float scale_a, float scale_b, float scale_c
) {
    __shared__ int8_t tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t tile_b[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    int32_t sum = 0;
    
    // Matrix multiplication
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int a_row = row;
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (a_row < M && a_col < K) {
            tile_a[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        
        int b_row = tile * TILE_SIZE + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N) {
            tile_b[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += (int32_t)tile_a[threadIdx.y][k] * (int32_t)tile_b[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result with bias and ReLU
    if (row < M && col < N) {
        float result = (float)sum * scale_a * scale_b * scale_c;
        if (bias != nullptr) {
            result += bias[col];
        }
        result = fmaxf(0.0f, result);  // ReLU activation
        C[row * N + col] = result;
    }
}

// ============================================================================
// C Wrapper Functions
// ============================================================================
extern "C" {

void int8_matmul_cuda(
    const int8_t* A,
    const int8_t* B,
    int32_t* C,
    int M, int N, int K,
    float scale_a, float scale_b, float scale_c,
    cudaStream_t stream
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    int8_matmul_kernel<<<grid, block, 0, stream>>>(
        A, B, C, M, N, K, scale_a, scale_b, scale_c
    );
}

void int4_matmul_cuda(
    const uint8_t* A_packed,
    const uint8_t* B_packed,
    int32_t* C,
    int M, int N, int K,
    float scale_a, float scale_b, float scale_c,
    cudaStream_t stream
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    int4_matmul_kernel<<<grid, block, 0, stream>>>(
        A_packed, B_packed, C, M, N, K, scale_a, scale_b, scale_c
    );
}

void dequantize_cuda(
    const int8_t* quantized,
    half* output,
    int size,
    float scale,
    float zero_point,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dequantize_kernel<<<blocks, threads, 0, stream>>>(
        quantized, output, size, scale, zero_point
    );
}

void dequantize_fp32_cuda(
    const int8_t* quantized,
    float* output,
    int size,
    float scale,
    float zero_point,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dequantize_fp32_kernel<<<blocks, threads, 0, stream>>>(
        quantized, output, size, scale, zero_point
    );
}

void quantize_cuda(
    const float* input,
    int8_t* output,
    int size,
    float scale,
    float zero_point,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    quantize_kernel<<<blocks, threads, 0, stream>>>(
        input, output, size, scale, zero_point
    );
}

void fused_int8_matmul_bias_relu_cuda(
    const int8_t* A,
    const int8_t* B,
    const float* bias,
    float* C,
    int M, int N, int K,
    float scale_a, float scale_b, float scale_c,
    cudaStream_t stream
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_int8_matmul_bias_relu_kernel<<<grid, block, 0, stream>>>(
        A, B, bias, C, M, N, K, scale_a, scale_b, scale_c
    );
}

} // extern "C"




