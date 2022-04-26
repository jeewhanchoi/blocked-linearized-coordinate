
#pragma once
#include "cusolverDn.h"
#include "cublas_v2.h"
#include "cblas.h"


// Computes the CANDECOMP/PARAFAC decomposition of a tensor with the alternating least squares algorithm
//
// Parameters:
//  - X: the BLCO tensor stored entirely in host memory
//  - A: a kruskal tensor initated with values. This will be overwritten
//  - maxiter: the maximum number of iterations to perform
//  - tolerance: prematurely quits if the change in fit goes below this
//  - kernel: the ID of the MTTKRP kernel to use
//  - stream_data: whether data should be streamed during MTTKRP 
//  - do_batching: whether MTTKRP kernel launch should be batched
//  - thread_cf: the thread coarsening factor to use for relevent MTTKRP kernels
//  - nprtn: the number of partitions to use for relevent MTTKRP kernels
// Returns:
//  - The final fit
FType cpals_blco_dev(blcotensor* X, KruskalModel* A, int maxiter, FType tolerance, int kernel, 
        bool stream_data, bool do_batching, int thread_cf, int nprtn);


// Computes the fit metric between the given Kruskal tensor and sparse tensor
//
// This is the same fit metric as seen in MATLAB Tensor Toolbox. The formula is:
//     1 - (norm(X - A) / norm(X))
//
// This must be used following an entire CP-ALS iteration because it relies on the 
// the residual result of the most recent MTTKRP
//
// Parameters:
//  - cublasHandle: a handle to the cuBLAS context
//  - v_stream: a handle to the stream to run this routine on. This stream will be synchronized
//  - grams: the Gram matrices for each factor matrix
//  - A: the Kruskal tensor
//  - X: the BLCO tensor stored entirely in host memory
//  - work: a length `A->N` x `A->N` array to be used as a scratchpad. Must be on device memory
//  - last_mttkrp: the result of the most recent MTTKRP
//  - X_norm: the norm of `X` (the original tensor)
//  - mode: the most of the most recent iteration
//  - blocks: the number of CUDA blocks to use
// Returns:
//  - The computed fit metric between the original tensor and the Kruskal tensor
//
FType kruskal_fit_gpu(cublasHandle_t cublasHandle, cudaStream_t v_stream, FType** grams, KruskalModel* A, blcotensor* X, FType* work, FType* last_mttkrp, FType X_norm, IType mode, IType blocks);


// Computes the Hadamard product of two vectors, i.e. x <-- x .* y
//
// The stream will not be synchronized.
//
// Parameters:
//  - stream: the cuda stream to use
//  - x: the result vector to accumulate to
//  - y: the vector to scale x by
//  - n: the length of each vector
void vector_hadamard(cudaStream_t stream, FType* x, FType* y, IType n);


// Fills a vector with the specified value
//
// The stream will not be synchronized.
//
// Parameters:
//  - stream: the cuda stream to use
//  - x: the vector to fill
//  - n: the length of the vector
//  - val: the value to fill
void value_fill(cudaStream_t stream, FType* x, IType n, FType val);


// Sums up a vector. The result will be placed in the first entry of the vector
//
// The rest of the vector will be treated as scratch and overwritten as necessary
//
// The stream will not be synchronized.
//
// Parameters
//  - stream: the cuda stream to use
//  - x: the vector to sum
//  - n: the length of the vector
void vecsum(cudaStream_t stream, FType* x, IType n);


// Computes the Moore-Penrose pseudoinverse of a matrix, GPU side
//
// The size of the work array depends on cusolverDnDgesvdj_bufferSize
// Add `2n^2 + n` to the value returned to get the full required work length
//
// cuBLAS pointer mode will be set to be device mode upon exit
//
// Parameters:
//  - cusolverHandle: a handle to the cuSOLVER context
//  - cublasHandle: a handle to the cuBLAS context
//  - stream: a handle to the stream to run this routine on. This stream will be synchronized
//  - A: an `n` x `n` row-major matrix that will be overwritten with its inverse. 
//       Must be on device memory
//  - n: the length of the square matrix
//  - work: a length `lwork` array to be used as a scratchpad. Must be on device memory
//  - lwork: the length of the `work` array. Must be at least `7n^2 + 2n` in size
//  - info: represents the status of the SVD (see gesvd from LAPACK for more information) 
//          Must be on device memory
//  - gesvd_info: Function parameters for the underlying gesvdj function
// Returns:
//  - none
void pseudoinverse_gpu(cusolverDnHandle_t cusolverHandle, cublasHandle_t cublasHandle, 
    cudaStream_t stream, FType* A, IType n, FType* work, IType lwork, int* info, gesvdjInfo_t gesvd_info);


// Performs columnwise L2 normalization on a matrix, storing norms for each column in a given vector
//
// The stream will not be synchronized.
//
// Parameters:
//  - stream: a handle to the stream to run this routine on. This stream will be synchronized
//  - blocks: the number of CUDA blocks to create
//  - mat: the `mode_length` x `rank` matrix to normalize
//  - rank: The width of the matrix
//  - mode_length: the height of the matrix
//  - lambda: the vector to store norms in (will be overwritten)
void columnwise_normscal(cudaStream_t stream, IType blocks, FType* mat, IType rank, IType mode_length, FType* lambda);


// Performs the routine C <-- A^T * A, GPU side
//
// The stream will not be synchronized.
//
// Parameters:
//  - handle: the handle to the cuBLAS context
//  - stream: a handle to the stream to run this routine on. This stream will be synchronized
//  - A: the input matrix in row-major order, size m x n
//  - C: the output matrix in row-major order to accumulate to, size n x n
//  - m: the number of columns in A
//  - n: the number of rows in A (and the column and row count of C)
// Returns:
//  - none
void ata_gpu(cublasHandle_t handle, cudaStream_t stream, const FType* A, FType* C, IType m, IType n);
