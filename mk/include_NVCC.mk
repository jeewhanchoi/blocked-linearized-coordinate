# For H100
# NVCC           = nvcc -gencode=arch=compute_90,code=sm_90
# For A100
NVCC           = nvcc -gencode=arch=compute_80,code=sm_80
CUDA_HOST_CXX  = g++

CUOPENMP   = -Xcompiler -fopenmp
CUDA = /usr/local/cuda
#CUDA = /packages/cuda/12.4.1
CUBLAS_LIBS = -L$(CUDA)/lib64 -lcudart -lcublas -lcusolver -lcurand
CUDA_INCLUDE = $(CUDA)/include

NVCCFLAGS   = -O3 -std=c++17
NVCCLFLAGS   = $(CUOPENMP)
NVCCINCLUDES = -I$(CUDA_INCLUDE)
NVCCLIBS     = $(CUBLAS_LIBS)
