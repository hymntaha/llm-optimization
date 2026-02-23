#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// INT8 quantized matrix multiplication: C = A @ B (with scaling)
// A: [M x K] int8_t
// B: [K x N] int8_t
// C: [M x N] int32_t
void int8_matmul_cuda(
    const int8_t* A,
    const int8_t* B,
    int32_t* C,
    int M, int N, int K,
    float scale_a, float scale_b, float scale_c,
    cudaStream_t stream
);

// INT4 packed quantized matrix multiplication: C = A @ B (with scaling)
// A_packed: [M x (K+1)/2] uint8_t (packed INT4)
// B_packed: [(K+1)/2 x N] uint8_t (packed INT4)
// C: [M x N] int32_t
void int4_matmul_cuda(
    const uint8_t* A_packed,
    const uint8_t* B_packed,
    int32_t* C,
    int M, int N, int K,
    float scale_a, float scale_b, float scale_c,
    cudaStream_t stream
);

// Dequantize INT8 to FP16
void dequantize_cuda(
    const int8_t* quantized,
    half* output,
    int size,
    float scale,
    float zero_point,
    cudaStream_t stream
);

// Dequantize INT8 to FP32
void dequantize_fp32_cuda(
    const int8_t* quantized,
    float* output,
    int size,
    float scale,
    float zero_point,
    cudaStream_t stream
);

// Quantize FP32 to INT8
void quantize_cuda(
    const float* input,
    int8_t* output,
    int size,
    float scale,
    float zero_point,
    cudaStream_t stream
);

// Fused INT8 matrix multiplication + bias + ReLU activation
void fused_int8_matmul_bias_relu_cuda(
    const int8_t* A,
    const int8_t* B,
    const float* bias,
    float* C,
    int M, int N, int K,
    float scale_a, float scale_b, float scale_c,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_H




