NVCC       = nvcc -gencode=arch=compute_70,code=sm_70

CUOPENMP   = -Xcompiler -fopenmp
CUDA = /usr/local/cuda
#CUDA =/swtools/cuda/latest
CUBLAS_LIBS = -L$(CUDA)/lib64 -lcudart -lcublas -lcusolver -lcurand
CUDA_INCLUDE = $(CUDA)/include

NVCCFLAGS   = -O3 -std=c++17
NVCCLFLAGS   = $(CUOPENMP) 
NVCCINCLUDES = -I$(CUDA_INCLUDE)
NVCCLIBS     = $(CUBLAS_LIBS)
