#include "common.hpp"
#include "utils.hpp"
#include "kruskal_model.hpp"
#include "blco.hpp"
#include "cpd_gpu.hpp"
#include "alto_dev.hpp"
#include "cooperative_groups.h"
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 128 // TODO tune?

static const FType one = 1.0;
static const FType zero = 0.0;

using namespace cooperative_groups;

__global__ void columnwise_normscal_kernel(FType* mat, IType rank, IType mode_length, FType* lambda);
__global__ void columnwise_inner_product_kernel(FType* U, FType* V, FType* output, IType rank, IType mode_length);

FType cpals_blco_dev(blcotensor* at_host, KruskalModel* A, int maxiter, FType tolerance, int kernel, bool stream_data, bool do_batching, int thread_cf, int nprtn) {

    IType R = A->rank;
    IType N = at_host->N;

    printf("\nCopying to GPU device\n");

    double wtime_s, wtime;
    wtime_s = omp_get_wtime();

    printf("--> Init BLCO on GPU\n");

    IType num_streams = NUM_STREAMS; 
    IType max_block_size = at_host->max_block_size;
    if (!stream_data) {
        num_streams = at_host->block_count;
        max_block_size = 0;
    }
    blcotensor* at_dev = gen_blcotensor_device(at_host, num_streams, do_batching, max_block_size);
    if (!stream_data) {
        printf("--> Send over BLCO tensor\n");
        send_blcotensor_over(at_dev->streams[0], at_dev, at_host);
    }
    send_masks_over(at_host);

    // Generate partial matrices if kernel uses it
    FType** partial_matrices = NULL;
    FType** partial_matrices_dev = NULL;
    if (kernel == 3 || kernel == 10) {
        printf("--> Allocate partial matrices\n");
        for (int i = 0; i < num_streams; i++) {
            at_dev->blocks[i]->pmatrices_staging_ptr = create_da_mem_dev(kernel, -1, A->rank, nprtn, at_host);
            at_dev->blocks[i]->pmatrices = make_device_copy(at_dev->blocks[i]->pmatrices_staging_ptr, nprtn, "partial matrices pointer");
        }
        if (num_streams > 1) {
            partial_matrices = create_da_mem_dev(kernel, -1, A->rank, num_streams, at_host);
            partial_matrices_dev = make_device_copy(partial_matrices, num_streams, "partial matrices pointer");
        }
    }

    // Factor matrices
    printf("--> Copy over Kruskal model (factor matrices)\n");
    KruskalModel* M_dev = make_device_copy(A);

    // Setup CUDA contexts
    printf("--> Prep auxiliary data\n");
    cudaStream_t v_stream;
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;
    check_cuda(cudaStreamCreate(&v_stream), "cudaStreamCreate");
    check_cublas(cublasCreate(&cublasHandle), "cublasCreate");
    check_cusolver(cusolverDnCreate(&cusolverHandle), "cusolverDnCreate");
    cublasSetStream(cublasHandle, v_stream);
    cusolverDnSetStream(cusolverHandle, v_stream);

    // Allocate MTTKRP res space
    FType* mttkrp_res;
    IType max_length = at_host->modes[0];
    for (IType i = 1; i < at_host->N; i++) max_length = std::max(max_length, at_host->modes[i]);
    check_cuda(cudaMalloc(&mttkrp_res, sizeof(FType) * max_length * R), "cudaMalloc mttkrp result matrix");

    // Allocate pseudoinverse array + work
    FType *V, *work;
    check_cuda(cudaMalloc(&V, sizeof(FType) * R * R), "cudaMalloc V");
    int work_int = 0;
    check_cublas(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    gesvdjInfo_t gesvd_info;
    check_cusolver(cusolverDnCreateGesvdjInfo(&gesvd_info), "cusolverDnCreateGesvdjInfo");
    cusolverDnXgesvdjSetMaxSweeps(gesvd_info, 15); // As recommended by cuSOLVER docs
    #ifdef USE_32BIT_TYPE
        check_cusolver(cusolverDnSgesvdj_bufferSize(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, 1, R, R, V, R, NULL, NULL, R, NULL, R, &work_int, gesvd_info), "cusolverDnSgesvdj_bufferSize");
    #else
        check_cusolver(cusolverDnDgesvdj_bufferSize(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, 1, R, R, V, R, NULL, NULL, R, NULL, R, &work_int, gesvd_info), "cusolverDnDgesvdj_bufferSize");
    #endif
    IType work_length = 2 * R * R + R + work_int;
    check_cuda(cudaMalloc(&work, sizeof(FType) * work_length), "cudaMalloc work");

    // Allocate info ptr
    int* info;
    check_cuda(cudaMalloc(&info, sizeof(int)), "cudaMalloc info");

    // Precompute norms
    // This used to be dnrm2/snrm2 BLAS call but it is breaking with
    // vector lengths in the billions so we implement it ourselves
    // Accumulate to double for precision, same as SPLATT
    double _norm = 0.0;
    #pragma omp parallel for reduction (+:_norm)
    for (IType i = 0; i < at_host->total_nnz; i++) _norm += at_host->values[i] * at_host->values[i];
    FType X_norm = (FType) sqrt(_norm);

    // Init and precompute grams
    FType** grams = new FType*[N];
    assert(grams);
    for (IType i = 0; i < N; i++) {
        check_cuda(cudaMalloc(grams + i, sizeof(FType) * R * R), "cudaMalloc grams");
    }
    for (IType n = 0; n < N; n++) ata_gpu(cublasHandle, v_stream, M_dev->U[n], grams[n], at_host->modes[n], R);
    check_cuda(cudaStreamSynchronize(v_stream), "cudaStreamSynchronize grams");

    // Get device info
    int device_id = 0, normscal_kernel_blocks = 0, iprod_kernel_blocks = 0;
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice");
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&normscal_kernel_blocks, columnwise_normscal_kernel, BLOCK_SIZE, R * sizeof(FType));
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&iprod_kernel_blocks, columnwise_inner_product_kernel, BLOCK_SIZE, R * sizeof(FType));
    normscal_kernel_blocks *= deviceProp.multiProcessorCount;
    iprod_kernel_blocks *= deviceProp.multiProcessorCount;

    // Init GPU timers
    cudaEvent_t start, stop;
    float millis;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Iterate modes largest to smallest to minimize kruskal fit time
    IType* mode_order = new IType[N];
    for (IType i = 0; i < N; i++) mode_order[i] = i;
    std::sort(mode_order, mode_order + N, [at_host](IType a, IType b) { return at_host->modes[a] > at_host->modes[b]; });
    //for (IType i = 0; i < N; i++) printf("Order: mode %d is %d (length %d)\n", i, mode_order[i], at_host->modes[mode_order[i]]);

    wtime = omp_get_wtime() - wtime_s;
    printf("--> Total initialization = %0.3f (s)\n\n", wtime);

    FType fit = 0, fit_old = 0;

    printf("Begin CP-ALS decomposition, rank=%d maxiter=%d tolerance=%f\n", R, maxiter, tolerance);
    
    for (IType count = 0; count < maxiter; count++) {
        cudaEventRecord(start);

        for (IType _n = 0; _n < N; _n++) {
            IType n = mode_order[_n];
            // Compute MTTKRP. Set output matrix to auxiliary array for pseudoinverse
            check_cuda(cudaMemset(mttkrp_res, 0, sizeof(FType) * R * at_host->modes[n]), "cudaMemset 0 mttkrp result");
            mttkrp_alto_dev_onemode<IType>(at_host, at_dev, A, mttkrp_res, kernel, n, thread_cf, stream_data, nprtn, M_dev->U, M_dev->U_dev, partial_matrices, partial_matrices_dev);
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize mttkrp_blco");

            // Compute hadamards
            value_fill(v_stream, V, R * R, 1.0);
            for (IType i = 0; i < N; i++) if (i != n) {
                vector_hadamard(v_stream, V, grams[i], R * R);
            }

            // Take pseudoinverse
            pseudoinverse_gpu(cusolverHandle, cublasHandle, v_stream, 
                V, R, work, work_length, info, gesvd_info);
            //check_cuda(cudaStreamSynchronize(v_stream), "cudaStreamSynchronize pseudoinverse_gpu");
            
            // Multiply V^-1 by mttkrp_res
            #ifdef USE_32BIT_TYPE
                check_cublas(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, R, 
                    at_host->modes[n], R, &one, V, R, mttkrp_res, R, &zero, M_dev->U[n], R), "cublasSgemm");
            #else
                check_cublas(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, R, 
                    at_host->modes[n], R, &one, V, R, mttkrp_res, R, &zero, M_dev->U[n], R), "cublasDgemm");
            #endif

            // Normalize, only supports L2 for now
            columnwise_normscal(v_stream, normscal_kernel_blocks, M_dev->U[n], R, at_host->modes[n], M_dev->lambda);

            // Update grams again
            ata_gpu(cublasHandle, v_stream, M_dev->U[n], grams[n], at_host->modes[n], R);
            check_cuda(cudaStreamSynchronize(v_stream), "cudaStreamSynchronize ata_gpu");
        }

        fit = kruskal_fit_gpu(cublasHandle, v_stream, grams, M_dev, at_host, work, mttkrp_res, X_norm, mode_order[N - 1], iprod_kernel_blocks);

        cudaEventRecord(stop);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize cpd");
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&millis, start, stop);
        printf("--> Iter %3d: f = %0.6f f-delta = %0.6f time = %0.6f, (s)\n", count, fit, (fit - fit_old), millis / 1000.0f);

        if (abs(fit - fit_old) < tolerance || isnan(fit)) break;
        fit_old = fit;
    }

    printf("Final fit: %0.9f\n", fit);

    // Write back
    printf("--> Copy back\n");
    for (IType i = 0; i < N; i++) {
        check_cuda(cudaMemcpy(A->U[i], M_dev->U[i], sizeof(FType) * R * at_host->modes[i], cudaMemcpyDeviceToHost), "cudaMemcpy mats back");
    }
    check_cuda(cudaMemcpy(A->lambda, M_dev->lambda, sizeof(FType) * R, cudaMemcpyDeviceToHost), "cudaMemcpy lambda back");
    
    // Delete partial matrices
    if (partial_matrices) {
        for (IType i = 0; i < num_streams; i++) cudaFree(partial_matrices[i]);
        cudaFree(partial_matrices_dev);
        delete [] partial_matrices;
    }
    for (int i = 0; i < num_streams; i++) {
        blco_block* blk = at_dev->blocks[i];
        if (blk->pmatrices_staging_ptr) {
            for (IType j = 0; j < nprtn; j++) cudaFree(blk->pmatrices_staging_ptr[j]);
            cudaFree(blk->pmatrices);
            delete [] blk->pmatrices_staging_ptr;
        }
    }

    // Cleanup
    printf("--> Cleanup \n");
    delete_blcotensor_device(at_dev);
    destroy_kruskal_model_dev(M_dev);
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);
    cusolverDnDestroyGesvdjInfo(gesvd_info);
    cudaStreamDestroy(v_stream);
    cudaFree(V);
    cudaFree(work);
    cudaFree(mttkrp_res);
    cudaFree(info);

    for (IType i = 0; i < N; i++) cudaFree(grams[i]);
    delete[] grams;

    delete[] mode_order;

    return fit;
}

__global__ void columnwise_normscal_kernel(FType* mat, IType rank, IType mode_length, FType* lambda) {

    extern __shared__ FType s_lambda[]; // Length rank
    thread_block tb = this_thread_block();
    grid_group grid = this_grid();
    IType lane = tb.thread_index().x;
    IType global_id = grid.thread_rank();

    for (IType i = lane; i < rank; i += tb.size()) s_lambda[i] = 0;
    tb.sync();

    // Thread block local L2 normalization
    for (IType i = global_id; i < rank * mode_length; i += grid.size()) {
        IType col = i % rank;
        FType val = mat[i];

        // Atomic update to shared lambda
        atomicAdd(s_lambda + col, val * val);
    }
    tb.sync();

    // Write to global
    for (IType i = lane; i < rank; i += tb.size()) atomicAdd(lambda + i, s_lambda[i]);

    grid.sync();

    // Sqrt
    if (global_id < rank) lambda[global_id] = sqrt(lambda[global_id]);

    grid.sync();

    // Fetch lambda and invert
    for (IType i = lane; i < rank; i += tb.size()) {
        FType l = lambda[i];

        // If the norm was zero the the entire column is just zeros
        s_lambda[i] = (l == 0) ? 0 : 1 / lambda[i];
    }
    tb.sync();
    
    // Scale columns
    for (IType i = global_id; i < rank * mode_length; i += grid.size()) {
        IType col = i % rank;
        FType val = mat[i];

        mat[i] = val * s_lambda[col];
    }
}
void columnwise_normscal(cudaStream_t stream, IType blocks, FType* mat, IType rank, IType mode_length, FType* lambda) {
    // Move params onto the stack TODO find more elegant solution to this
    FType* stack_mat = mat;
    IType stack_rank = rank;
    IType stack_mode_length = mode_length;
    FType* stack_lambda = lambda;
    
    IType shared_mem = rank * sizeof(FType);
    void *kernel_args[] = {&stack_mat, &stack_rank, &stack_mode_length, &stack_lambda};
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    check_cuda(cudaLaunchCooperativeKernel((void*) columnwise_normscal_kernel, dimGrid, dimBlock, kernel_args, shared_mem, stream), "columnwise_normscal_kernel launch");
}


__global__ void columnwise_inner_product_kernel(FType* U, FType* V, FType* output, IType rank, IType mode_length) {
    extern __shared__ FType s_lambda[]; // Length rank
    thread_block tb = this_thread_block();
    grid_group grid = this_grid();
    IType lane = tb.thread_index().x;
    IType global_id = grid.thread_rank();

    for (IType i = lane; i < rank; i += tb.size()) s_lambda[i] = 0;
    tb.sync();

    // TB dot product to local lambda
    for (IType i = global_id; i < rank * mode_length; i += grid.size()) {
        IType col = i % rank;
        FType val1 = U[i];
        FType val2 = V[i];

        atomicAdd(s_lambda + col, val1 * val2);
    }

    tb.sync();

    // Write to global
    for (IType i = lane; i < rank; i += tb.size()) atomicAdd(output + i, s_lambda[i]);
}
FType kruskal_fit_gpu(cublasHandle_t cublasHandle, cudaStream_t v_stream, FType** grams, KruskalModel* A, blcotensor* at_host, FType* work, FType* last_mttkrp, FType X_norm, IType mode, IType blocks) {
    // Same routine as in MATLAB Tensor Toolbox

    check_cuda(cudaMemsetAsync(work, 0, sizeof(FType) * A->rank, v_stream), "cudaMemset work");

    FType iprod, norm_a, fit;

    FType* stack_U = A->U[mode];
    FType* stack_V = last_mttkrp;
    FType* stack_work = work;
    IType stack_rank = A->rank;
    IType stack_mode_length = at_host->modes[mode];
    
    IType shared_mem = A->rank * sizeof(FType);
    void *kernel_args[] = {&stack_U, &stack_V, &stack_work, &stack_rank, &stack_mode_length};
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    check_cuda(cudaLaunchCooperativeKernel((void*) columnwise_inner_product_kernel, dimGrid, dimBlock, kernel_args, shared_mem, v_stream), "columnwise_normscal_kernel launch");

    check_cublas(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    #ifdef USE_32BIT_TYPE
        check_cublas(cublasSdot(cublasHandle, A->rank, work, 1, A->lambda, 1, &iprod), "cublasSdot");
    #else
        check_cublas(cublasDdot(cublasHandle, A->rank, work, 1, A->lambda, 1, &iprod), "cublasDdot");
    #endif
    //check_cuda(cudaStreamSynchronize(v_stream), "cudaStreamSynchronize inner product");

    // Calculate correlation coefficients, use work as workspace
    check_cuda(cudaMemsetAsync(work, 0, sizeof(FType) * A->rank * A->rank, v_stream), "cudaMemset");
    #ifdef USE_32BIT_TYPE
        check_cublas(cublasSger(cublasHandle, A->rank, A->rank, &one, A->lambda, 1, A->lambda, 1, work, A->rank), "cublasSger");
    #else 
        check_cublas(cublasDger(cublasHandle, A->rank, A->rank, &one, A->lambda, 1, A->lambda, 1, work, A->rank), "cublasDger");
    #endif

    // Hadamard vector accumulate of all factor matrices
    for (IType i = 0; i < A->mode; i++) {
        vector_hadamard(v_stream, work, grams[i], A->rank * A->rank);
    }

    // Sum matrix
    vecsum(v_stream, work, A->rank * A->rank);
    check_cuda(cudaStreamSynchronize(v_stream), "cudaStreamSynchronize kruskal fit");
    check_cuda(cudaMemcpy(&norm_a, work, sizeof(FType), cudaMemcpyDeviceToHost), "cudaMemcpy result");
    norm_a = sqrt(abs(norm_a));

    // Calculate fit
    fit = norm_a * norm_a - 2 * iprod;
    if (X_norm != 0) {
        fit = sqrt(abs(fit + X_norm * X_norm));
        fit = 1 - (fit / X_norm); // See MATLAB formula
    }
    return fit;
}


// Takes the reciprocal of each vector entry, sets to zero if smaller than tol
__global__ void reciprocal_vector_kernel(FType* v, IType n, FType tol) {
    IType index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        v[index] = (fabs(v[index]) > tol) ? 1.0 / v[index] : 0;
    }
}


// Hadamard update, i.e. x <-- x .* y
__global__ void hadamard_kernel(FType* x, FType* y, IType n) {
    IType index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) x[index] *= y[index];
}
void vector_hadamard(cudaStream_t stream, FType* x, FType* y, IType n) {
    hadamard_kernel <<<n / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(x, y, n);
    check_cuda(cudaGetLastError(), "hadamard_kernel launch");
}


// Fills every element of vector x with the same value
__global__ void value_fill_kernel(FType* x, IType n, FType val) {
    IType index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) x[index] = val;
}
void value_fill(cudaStream_t stream, FType* x, IType n, FType val) {
    value_fill_kernel <<<n / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(x, n, val);
    check_cuda(cudaGetLastError(), "value_fill_kernel launch");
}


// Based off the NVIDIA parallel reduction slides
__global__ void reduce2(FType* x, IType n) {
    // Calculate id
    IType thread_id = threadIdx.x;
    IType index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    extern __shared__ FType temp[];
    if (index < n) temp[thread_id] = x[index];
    else temp[thread_id] = 0;
    __syncthreads();

    // Perform reduction in shared mem
    for (IType i = blockDim.x >> 1; i > 0; i >>= 1) {
        if (thread_id < i) temp[thread_id] += temp[thread_id + i];
        __syncthreads();
    }

    // Write back into vector
    if (thread_id == 0) x[blockIdx.x] = temp[0];
}
void vecsum(cudaStream_t stream, FType* x, IType n) {
    IType blocks;
    while (n > BLOCK_SIZE) {
        blocks = 1 + ((n - 1) / BLOCK_SIZE);
        reduce2 <<<blocks, BLOCK_SIZE, sizeof(FType) * BLOCK_SIZE, stream>>>(x, n);
        n = blocks;
    }
    if (n > 0) reduce2 <<<1, BLOCK_SIZE, sizeof(FType) * BLOCK_SIZE, stream>>>(x, n);
    check_cuda(cudaGetLastError(), "reduce2 launch");
}


// Use cusolverDnDgesvdj_bufferSize to calculate needed buffer size
void pseudoinverse_gpu(cusolverDnHandle_t cusolverHandle, cublasHandle_t cublasHandle,
    cudaStream_t stream, FType* A, IType n, FType* work, IType lwork, int* info, gesvdjInfo_t gesvd_info) {

    // I tried Cholesky / QR / LU factorization
    // They scale poorly to larger matrices compared to svd + gemm

    FType* U = work;
    FType* V = U + n * n;
    FType* S = V + n * n;
    work = S + n;
    lwork -= (2 * n * n + n);

    // Gen SVD
    check_cublas(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode");
    #ifdef USE_32BIT_TYPE
        check_cusolver(cusolverDnSgesvdj(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, 
            0, n, n, A, n, S, U, n, V, n, work, lwork, info, gesvd_info), "cusolverDnSgesvdj");
    #else 
        check_cusolver(cusolverDnDgesvdj(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, 
            0, n, n, A, n, S, U, n, V, n, work, lwork, info, gesvd_info), "cusolverDnDgesvdj");
    #endif
    check_cuda(cudaStreamSynchronize(stream), "cusolverDngesvdj execute");

    /*
    int info_host;
    check_cuda(cudaMemcpy(&info_host, info, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy info");
    if (info_host < 0) {
        printf("Warning: gesvdj failed at %d\n", -info_host);
    } else if (info_host > 0) {
        printf("Warning: gesvdj failed to converge with info %d. Increase sweeps\n", info_host);
    }
    */

    // Multiply U by S^-1 (scale rows of U by reciprocal of S);
    IType blocks = n / BLOCK_SIZE + 1;
    FType s = 0; // Get largest singular value
    check_cuda(cudaMemcpy(&s, S, sizeof(FType), cudaMemcpyDeviceToHost), "memcpy");
    s = n * (nextafter(s, s + 1) - s);
    reciprocal_vector_kernel <<<blocks, BLOCK_SIZE, 0, stream>>>(S, n, s);
    //check_cuda(cudaGetLastError(), "reciprocal_vector launch");
    check_cuda(cudaStreamSynchronize(stream), "reciprocal_vector execute");
    for (IType i = 0; i < n; i++) {
        cublasSetStream(cublasHandle, stream);
        #ifdef USE_32BIT_TYPE
            check_cublas(cublasSscal(cublasHandle, n, S + i, U + i * n, 1), "cublasSscal");
        #else 
            check_cublas(cublasDscal(cublasHandle, n, S + i, U + i * n, 1), "cublasDscal");
        #endif
    }
    check_cuda(cudaStreamSynchronize(stream), "cublasDscal execute");

    // Multiply U by V (we multiply V by U^T to convert col to row major)
    check_cublas(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    cublasSetStream(cublasHandle, stream);
    #ifdef USE_32BIT_TYPE
        check_cublas(cublasSgemm(
            cublasHandle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            n, n, n,
            &one, 
            V, n,
            U, n,
            &zero, 
            A, n), "cublasSgemm"
        );
    #else 
        check_cublas(cublasDgemm(
            cublasHandle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            n, n, n,
            &one, 
            V, n,
            U, n,
            &zero, 
            A, n), "cublasDgemm"
        );
    #endif
    check_cuda(cudaStreamSynchronize(stream), "cublasDgemm execute");
}


__global__ void copy_lower_to_upper_kernel(IType dimx, IType dimy, FType* a) {
	IType ix = blockIdx.x * blockDim.x + threadIdx.x;
	IType iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (iy > ix && iy < dimy && ix < dimx) {
		IType id_dest = iy * dimy + ix;
		IType id_src = ix * dimx + iy;
		a[id_dest] = a[id_src];
	}
}


void ata_gpu(cublasHandle_t handle, cudaStream_t stream, const FType* A, FType* C, IType m, IType n) {
    check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    cublasSetStream(handle, stream);

    // TODO figure out why syrk is super slow
    #ifdef USE_32BIT_TYPE
        check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, m, &one, A, n, A, n, &zero, C, n), "cublasSgemm");
    #else
        check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, m, &one, A, n, A, n, &zero, C, n), "cublasSgemm");
    #endif
    
    /*
    #ifdef USE_32BIT_TYPE
        check_cublas(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, m, &one, A, n, &zero, C, n), "cublasSsyrk");
    #else
        check_cublas(cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, m, &one, A, n, &zero, C, n), "cublasDsyrk");
    #endif
    dim3 grid(n/32 + 1, n/32 + 1), threadblock(32,32);
    copy_lower_to_upper_kernel <<<grid, threadblock, 0, stream>>> (n, n, C);
    check_cuda(cudaGetLastError(), "copy_lower_to_upper_kernel launch");
    */
}
