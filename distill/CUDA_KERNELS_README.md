# CUDA Quantization Kernels

High-performance CUDA kernels for quantized matrix operations, optimized for LLM inference and training.

## Features

- **INT8 Quantized Matrix Multiplication**: Efficient INT8 matrix multiplication with configurable scaling
- **INT4 Packed Quantized Matrix Multiplication**: Memory-efficient INT4 operations (2 values per byte)
- **Quantization/Dequantization**: Fast conversion between FP32/FP16 and INT8/INT4
- **Fused Operations**: Combined matrix multiplication + bias + ReLU activation
- **Optimized for Modern GPUs**: Supports compute capabilities 7.5+ (Turing, Ampere, Ada, Hopper)

## Compilation

### Prerequisites

- NVIDIA GPU with CUDA support (compute capability 7.5+)
- CUDA Toolkit (11.0+)
- GCC/G++ compiler

### Build

```bash
# Simple compilation
make

# Or manually with nvcc
nvcc -O3 -arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_89 -arch=sm_90 \
     --compiler-options -fPIC -shared \
     cuda_kernels.cu -o cuda_kernels.so -lcudart -lcublas
```

### Verify Installation

```bash
make test
```

## Usage

### Python Interface

```python
import torch
from cuda_wrapper import int8_matmul, quantize_tensor, dequantize_tensor

# Ensure CUDA is available
assert torch.cuda.is_available()
device = torch.device("cuda")

# Create test matrices
M, K, N = 1024, 2048, 512
A = torch.randn(M, K, device=device)
B = torch.randn(K, N, device=device)

# Quantize matrices
A_int8, scale_a = quantize_tensor(A)
B_int8, scale_b = quantize_tensor(B)

# Perform quantized matrix multiplication
C_int32 = int8_matmul(
    A_int8, B_int8,
    scale_a=scale_a,
    scale_b=scale_b,
    scale_c=1.0
)

# Dequantize result
C_fp32 = dequantize_tensor(C_int32, scale_a * scale_b)
```

### Direct C/C++ Usage

```cpp
#include "cuda_kernels.h"
#include <cuda_runtime.h>

// Allocate device memory
int8_t *d_A, *d_B;
int32_t *d_C;
cudaMalloc(&d_A, M * K * sizeof(int8_t));
cudaMalloc(&d_B, K * N * sizeof(int8_t));
cudaMalloc(&d_C, M * N * sizeof(int32_t));

// Copy data to device
cudaMemcpy(d_A, h_A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);

// Perform matrix multiplication
float scale_a = 1.0f / 127.0f;
float scale_b = 1.0f / 127.0f;
float scale_c = 1.0f;

int8_matmul_cuda(
    d_A, d_B, d_C,
    M, N, K,
    scale_a, scale_b, scale_c,
    nullptr  // default stream
);

// Copy result back
cudaMemcpy(h_C, d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
```

## Kernel Details

### INT8 Matrix Multiplication

- **Input**: `A [M×K] int8`, `B [K×N] int8`
- **Output**: `C [M×N] int32`
- **Scaling**: `C = (A * scale_a) @ (B * scale_b) * scale_c`
- **Tile Size**: 16×16 threads per block
- **Shared Memory**: Optimized tile-based access pattern

### INT4 Packed Matrix Multiplication

- **Input**: `A_packed [M×(K+1)/2] uint8`, `B_packed [(K+1)/2×N] uint8`
- **Output**: `C [M×N] int32`
- **Packing**: 2 INT4 values per uint8 byte (lower 4 bits, upper 4 bits)
- **Memory Savings**: 50% reduction compared to INT8

### Performance Considerations

1. **Memory Alignment**: Ensure inputs are properly aligned for optimal performance
2. **Stream Usage**: Use CUDA streams for asynchronous execution
3. **Batch Processing**: Process multiple matrices in parallel using streams
4. **Tile Size**: Current implementation uses 16×16 tiles (optimal for most GPUs)

## Benchmarking

Example benchmark script:

```python
import torch
import time
from cuda_wrapper import int8_matmul, quantize_tensor

def benchmark_int8_matmul(M, K, N, iterations=100):
    device = torch.device("cuda")
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    
    A_int8, scale_a = quantize_tensor(A)
    B_int8, scale_b = quantize_tensor(B)
    
    # Warmup
    for _ in range(10):
        _ = int8_matmul(A_int8, B_int8, scale_a, scale_b)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = int8_matmul(A_int8, B_int8, scale_a, scale_b)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iterations * 1000  # ms
    gflops = (2 * M * K * N) / (avg_time * 1e6)  # GFLOPS
    
    print(f"M={M}, K={K}, N={N}: {avg_time:.3f} ms, {gflops:.2f} GFLOPS")

# Run benchmarks
benchmark_int8_matmul(1024, 2048, 512)
benchmark_int8_matmul(2048, 4096, 1024)
```

## Integration with PyTorch

For seamless integration with PyTorch models:

```python
import torch
import torch.nn as nn
from cuda_wrapper import int8_matmul, quantize_tensor, dequantize_tensor

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('weight_int8', None)
        self.register_buffer('weight_scale', None)
        
    def quantize_weights(self):
        self.weight_int8, self.weight_scale = quantize_tensor(self.weight.data)
        
    def forward(self, x):
        if self.weight_int8 is None:
            return nn.functional.linear(x, self.weight, self.bias)
        
        # Quantize input
        x_int8, x_scale = quantize_tensor(x)
        
        # Quantized matmul
        output_int32 = int8_matmul(
            x_int8, self.weight_int8,
            x_scale, self.weight_scale, 1.0
        )
        
        # Dequantize and add bias
        output = dequantize_tensor(output_int32, x_scale * self.weight_scale)
        return output + self.bias
```

## Troubleshooting

### Compilation Errors

- **"nvcc not found"**: Ensure CUDA Toolkit is installed and in PATH
- **"sm_XX not supported"**: Remove unsupported architecture flags or update CUDA Toolkit
- **"undefined reference"**: Check that CUDA libraries are properly linked

### Runtime Errors

- **"CUDA kernels library not found"**: Ensure `cuda_kernels.so` is in the same directory or in LD_LIBRARY_PATH
- **"CUDA out of memory"**: Reduce matrix sizes or use batch processing
- **"Invalid device function"**: Recompile with correct architecture flags for your GPU

### Performance Issues

- Use `nvidia-smi` to check GPU utilization
- Profile with `nsys` or `nvprof` to identify bottlenecks
- Ensure matrices are properly aligned (multiples of 16 for optimal performance)

## License

This code is provided as-is for research and educational purposes.

## References

- NVIDIA CUDA Programming Guide
- Quantization techniques: GPTQ, AWQ, SmoothQuant
- Matrix multiplication optimization strategies




