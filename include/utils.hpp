// This file has miscellaneous methods

#pragma once
#include "cusolverDn.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include <iostream>

#define WARP_SIZE 32
#define TILE_SIZE WARP_SIZE
#define STASH_SIZE 4
#define NUM_STREAMS 8 // set CUDA_DEVICE_MAX_CONNECTIONS env = 1 to 32 (default is 8)
#define LVL3_MAX_MODE_LENGTH 100

// Double precision atomic add for Kepler architecture and below
__device__ double atomicAdd(double* a, double b);


// Copies a vector to the GPU, returning a pointer to it
//
// Parameters:
//  - vector: a vector on the CPU to copy over
//  - n: the length of the vector to copy over
//  - name: the name of the vector to use for debug errors
template <typename T> T* make_device_copy(T* vector, IType n, std::string name);


// Verifies that two vectors, one on CPU host and one on GPU device, are equal
//
// Parameters:
//  - x_cpu: a pointer to host memory
//  - x_gpu: a pointer to device memory
//  - n: the length of the device and host vectors to check for equality
// Returns: 
//  - 1 if they are equal, 0 if they are not equal
template <typename T> bool vectors_equal(T* x_cpu, T* x_gpu, IType n);


// Verifies that two vectors, one on CPU host and one on GPU device, are equal
//
// The tolerance is adjusted based on the current value, with preference placed
// for the CPU value. The actual tolerance used per entry is the following:
//
//      max(abs(x_cpu[i]) * tolerance, tolerance);
//
// Parameters:
//  - x_cpu: a pointer to host memory
//  - x_gpu: a pointer to device memory
//  - n: the length of the device and host vectors to check for equality
//  - tolerance: the tolerance by which to accept the float
// Returns: 
//  - 1 if they are equal, 0 if they are not equal
template <typename T> bool vectors_equal(T* x_cpu, T* x_gpu, IType n, FType tolerance);


// Prints the contents of a vector (i.e. array) 
//
// Parameters:
//  - x: the vector to display
//  - n: the length of the vector
//  - name: the name of the vector to display
// Returns:
//  - none
template <typename T> void print_vector(T* x, IType n, std::string name);


// Prints the contents of a matrix stored in column-major order
//
// Parameters:
//  - mtx: the matrix in column-major order to display
//  - m: the number of rows of the matrix
//  - n: the number of columns of the matrix
//  - name: the name of the matrix to display
// Returns:
//  - none
template <typename T> void print_matrix_col_maj(T* mtx, IType m, IType n, std::string name);


// Prints the contents of a matrix stored in row-major order
//
// Parameters:
//  - mtx: the matrix in row-major order to display
//  - m: the number of rows of the matrix
//  - n: the number of columns of the matrix
//  - name: the name of the matrix to display
// Returns:
//  - none
template <typename T> void print_matrix_row_maj(T* mtx, IType m, IType n, std::string name);


// Crashes if the given CUDA error status is not successful
//
// Parameters:
//  - status: the status returned from a CUDA call
//  - message: a provided error string to display alongside the error
// Returns:
//  - none
void check_cuda(cudaError_t status, std::string message);


// Crashes if the given cuBLAS error status is not successful
//
// Parameters:
//  - status: the status returned from a cuBLAS call
//  - message: a provided error string to display alongside the error
// Returns:
//  - none
void check_cublas(cublasStatus_t status, std::string message);


// Crashes if the given cuSOLVER error status is not successful
//
// Parameters:
//  - status: the status returned from a cuSOLVER call
//  - message: a provided error string to display alongside the error
// Returns:
//  - none
void check_cusolver(cusolverStatus_t status, std::string message);
