# Makefile for compiling CUDA quantization kernels

NVCC = nvcc
CUDA_PATH ?= /usr/local/cuda
CUDA_INC = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_89 -arch=sm_90 \
             --compiler-options -fPIC -shared -Xcompiler -fPIC \
             -std=c++17 -lineinfo

# Include paths
INCLUDES = -I$(CUDA_INC)

# Library paths
LIBRARIES = -L$(CUDA_LIB) -lcudart -lcublas

# Source and output files
CUDA_SRC = cuda_kernels.cu
OUTPUT_LIB = cuda_kernels.so

# Default target
all: $(OUTPUT_LIB)

# Compile CUDA kernels into shared library
$(OUTPUT_LIB): $(CUDA_SRC) cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBRARIES) $(CUDA_SRC) -o $(OUTPUT_LIB)
	@echo "Successfully compiled $(OUTPUT_LIB)"

# Clean build artifacts
clean:
	rm -f $(OUTPUT_LIB) *.o

# Install (copy to system location)
install: $(OUTPUT_LIB)
	cp $(OUTPUT_LIB) /usr/local/lib/
	ldconfig

# Test compilation
test:
	@echo "Testing CUDA compilation..."
	$(NVCC) --version
	@echo "CUDA_PATH: $(CUDA_PATH)"

.PHONY: all clean install test




