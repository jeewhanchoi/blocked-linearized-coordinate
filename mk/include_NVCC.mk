# For H100
# NVCC           = nvcc -gencode=arch=compute_90,code=sm_90
# For A100
NVCC           = nvcc -gencode=arch=compute_80,code=sm_80
CUDA_HOST_CXX ?= g++

CUDA_OPENMP_FLAG ?= -fopenmp
CUOPENMP    = -Xcompiler $(CUDA_OPENMP_FLAG)
# CUDA = /usr/local/cuda
CUDA = /packages/cuda/12.4.1
CUBLAS_LIBS = -L$(CUDA)/lib64 -lcudart -lcublas -lcusolver -lcurand
CUDA_INCLUDE = $(CUDA)/include

NVCCFLAGS   = -O3 -std=c++17 --allow-unsupported-compiler
NVCCLFLAGS   = $(CUOPENMP)
NVCCINCLUDES = -I$(CUDA_INCLUDE)
NVCCLIBS     = $(CUBLAS_LIBS)
