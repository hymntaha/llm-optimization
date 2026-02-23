#!/usr/bin/env python3
"""
Python wrapper for CUDA quantization kernels.
This module provides a Python interface to the CUDA kernels for quantized matrix operations.
"""

import ctypes
import numpy as np
import torch
from typing import Optional, Tuple

# Try to load the CUDA library
try:
    _lib = ctypes.CDLL('./cuda_kernels.so')
except OSError:
    try:
        _lib = ctypes.CDLL('./libcuda_kernels.so')
    except OSError:
        _lib = None
        print("Warning: CUDA kernels library not found. Please compile cuda_kernels.cu first.")


def _setup_cuda_functions():
    """Setup CUDA function signatures"""
    if _lib is None:
        return None
    
    # INT8 matrix multiplication
    _lib.int8_matmul_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # M, N, K
        ctypes.c_float, ctypes.c_float, ctypes.c_float,  # scales
        ctypes.c_void_p  # stream
    ]
    _lib.int8_matmul_cuda.restype = None
    
    # INT4 matrix multiplication
    _lib.int4_matmul_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_void_p
    ]
    _lib.int4_matmul_cuda.restype = None
    
    # Dequantize
    _lib.dequantize_fp32_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_void_p
    ]
    _lib.dequantize_fp32_cuda.restype = None
    
    # Quantize
    _lib.quantize_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_void_p
    ]
    _lib.quantize_cuda.restype = None
    
    return _lib


_lib = _setup_cuda_functions()


def int8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: float = 1.0,
    scale_b: float = 1.0,
    scale_c: float = 1.0,
    stream: Optional[torch.cuda.Stream] = None
) -> torch.Tensor:
    """
    Perform INT8 quantized matrix multiplication: C = A @ B
    
    Args:
        A: [M, K] int8 tensor
        B: [K, N] int8 tensor
        scale_a: Scale factor for A
        scale_b: Scale factor for B
        scale_c: Scale factor for output
        stream: CUDA stream (optional)
    
    Returns:
        C: [M, N] int32 tensor
    """
    if _lib is None:
        raise RuntimeError("CUDA kernels library not loaded")
    
    assert A.dtype == torch.int8 and B.dtype == torch.int8
    assert A.is_cuda and B.is_cuda
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Matrix dimensions mismatch: A[{M}x{K}] @ B[{K2}x{N}]"
    
    C = torch.zeros((M, N), dtype=torch.int32, device=A.device)
    
    cuda_stream = stream.cuda_stream if stream else None
    
    _lib.int8_matmul_cuda(
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        M, N, K,
        scale_a, scale_b, scale_c,
        cuda_stream
    )
    
    return C


def quantize_tensor(
    tensor: torch.Tensor,
    scale: Optional[float] = None,
    zero_point: float = 0.0
) -> Tuple[torch.Tensor, float]:
    """
    Quantize a FP32 tensor to INT8
    
    Args:
        tensor: Input FP32 tensor
        scale: Quantization scale (if None, computed automatically)
        zero_point: Quantization zero point
    
    Returns:
        quantized: INT8 tensor
        scale: Quantization scale used
    """
    if _lib is None:
        # Fallback to PyTorch quantization
        if scale is None:
            scale = tensor.abs().max().item() / 127.0
        quantized = (tensor / scale + zero_point).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale
    
    assert tensor.dtype == torch.float32
    assert tensor.is_cuda
    
    if scale is None:
        scale = tensor.abs().max().item() / 127.0
    
    quantized = torch.zeros_like(tensor, dtype=torch.int8)
    
    _lib.quantize_cuda(
        tensor.data_ptr(),
        quantized.data_ptr(),
        tensor.numel(),
        scale, zero_point,
        None
    )
    
    return quantized, scale


def dequantize_tensor(
    quantized: torch.Tensor,
    scale: float,
    zero_point: float = 0.0
) -> torch.Tensor:
    """
    Dequantize an INT8 tensor to FP32
    
    Args:
        quantized: Input INT8 tensor
        scale: Quantization scale
        zero_point: Quantization zero point
    
    Returns:
        dequantized: FP32 tensor
    """
    if _lib is None:
        # Fallback to PyTorch dequantization
        return (quantized.float() - zero_point) * scale
    
    assert quantized.dtype == torch.int8
    assert quantized.is_cuda
    
    dequantized = torch.zeros_like(quantized, dtype=torch.float32)
    
    _lib.dequantize_fp32_cuda(
        quantized.data_ptr(),
        dequantized.data_ptr(),
        quantized.numel(),
        scale, zero_point,
        None
    )
    
    return dequantized


def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack INT4 values into uint8 (2 values per byte)
    
    Args:
        tensor: INT4 tensor (values should be in range [-8, 7])
    
    Returns:
        packed: uint8 tensor with shape [..., (size+1)//2]
    """
    # Shift values to [0, 15] range
    tensor_shifted = (tensor + 8).clamp(0, 15)
    
    # Reshape to group pairs
    shape = tensor.shape
    size = tensor.numel()
    pairs = (size + 1) // 2
    
    tensor_flat = tensor_shifted.flatten()
    packed = torch.zeros(pairs, dtype=torch.uint8, device=tensor.device)
    
    for i in range(pairs):
        idx1 = i * 2
        idx2 = idx1 + 1
        
        val1 = tensor_flat[idx1].item() if idx1 < size else 0
        val2 = tensor_flat[idx2].item() if idx2 < size else 0
        
        packed[i] = val1 | (val2 << 4)
    
    return packed.reshape(*shape[:-1], pairs)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack uint8 values to INT4 (2 values per byte)
    
    Args:
        packed: uint8 tensor with packed INT4 values
    
    Returns:
        unpacked: INT4 tensor
    """
    shape = packed.shape
    size = packed.numel()
    
    packed_flat = packed.flatten()
    unpacked = torch.zeros(size * 2, dtype=torch.int8, device=packed.device)
    
    for i in range(size):
        val = packed_flat[i].item()
        unpacked[i * 2] = (val & 0x0F) - 8
        unpacked[i * 2 + 1] = ((val >> 4) & 0x0F) - 8
    
    return unpacked.reshape(*shape[:-1], shape[-1] * 2)


# Example usage
if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available. This example requires CUDA.")
        exit(1)
    
    print("CUDA Quantization Kernels Example")
    print("=" * 50)
    
    # Create test tensors
    M, K, N = 128, 256, 64
    device = torch.device("cuda")
    
    # Test quantization/dequantization
    print("\n1. Testing Quantization/Dequantization:")
    fp32_tensor = torch.randn(M, K, device=device) * 2.0
    quantized, scale = quantize_tensor(fp32_tensor)
    dequantized = dequantize_tensor(quantized, scale)
    
    mse = ((fp32_tensor - dequantized) ** 2).mean().item()
    print(f"   Original shape: {fp32_tensor.shape}")
    print(f"   Quantized shape: {quantized.shape}, dtype: {quantized.dtype}")
    print(f"   Scale: {scale:.6f}")
    print(f"   MSE: {mse:.6f}")
    
    # Test INT8 matrix multiplication
    if _lib is not None:
        print("\n2. Testing INT8 Matrix Multiplication:")
        A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
        B_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
        
        # Compute scales
        scale_a = 1.0 / 127.0
        scale_b = 1.0 / 127.0
        scale_c = 1.0
        
        C_int32 = int8_matmul(A_int8, B_int8, scale_a, scale_b, scale_c)
        print(f"   A: {A_int8.shape}, B: {B_int8.shape}")
        print(f"   C: {C_int32.shape}, dtype: {C_int32.dtype}")
        print(f"   C sample values: {C_int32[:3, :3]}")
    
    print("\nDone!")




