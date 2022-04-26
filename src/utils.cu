#include "common.hpp"
#include "utils.hpp"
#include <iomanip>
#include <cmath>
#include <cassert>


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif


template <typename T> T* make_device_copy(T* vector, IType n, std::string name) {
    T* d_vector = nullptr;
    check_cuda(cudaMalloc(&d_vector, sizeof(T) * n), "cudaMalloc " + name);
    check_cuda(cudaMemcpy(d_vector, vector, sizeof(T) * n, cudaMemcpyHostToDevice), "cudaMemcpy " + name);
    return d_vector;
}
template IType* make_device_copy(IType* vector, IType n, std::string name);
template FType* make_device_copy(FType* vector, IType n, std::string name);
template FType** make_device_copy(FType** vector, IType n, std::string name);
template FType*** make_device_copy(FType*** vector, IType n, std::string name);
template unsigned long* make_device_copy(unsigned long* vector, IType n, std::string name);
template int* make_device_copy(int* vector, IType n, std::string name);


template <typename T> bool vectors_equal(T* x_cpu, T* x_gpu, IType n) {
    T* x_gpu_host = new T[n];
    assert(x_gpu_host);
    check_cuda(
        cudaMemcpy(x_gpu_host, x_gpu, sizeof(T) * n, cudaMemcpyDeviceToHost),
        "cudaMemcpy"
    );

    bool equal = 1;
    for (IType i = 0; i < n; i++) {
        if (x_cpu[i] != x_gpu_host[i]) {
            std::cerr << "Warning: index " << i << " failed to match" << std::endl;
            std::cerr << "CPU " << x_cpu[i] << " vs GPU " << x_gpu_host[i] << std::endl;
            equal = 0;
            break;
        }
    }
    
    delete [] x_gpu_host;
    return equal;
}
template bool vectors_equal(IType* x_cpu, IType* x_gpu, IType n);


template <typename T> bool vectors_equal(T* x_cpu, T* x_gpu, IType n, FType tolerance) {
    T* x_gpu_host = new T[n];
    assert(x_gpu_host);
    check_cuda(
        cudaMemcpy(x_gpu_host, x_gpu, sizeof(T) * n, cudaMemcpyDeviceToHost),
        "cudaMemcpy"
    );

    //print_vector(x_cpu, n, "CPU");
    //print_vector(x_gpu_host, n, "GPU");

    bool equal = 1;
    FType tol;
    for (IType i = 0; i < n; i++) {
        tol = max(abs(x_cpu[i]) * tolerance, tolerance);
        if (abs(x_cpu[i] - x_gpu_host[i]) > tol) {
            std::cerr << "Warning: index " << i << " failed to match" << std::endl;
            std::cerr << "CPU " << std::setprecision(20) << x_cpu[i] << " vs GPU " << std::setprecision(20) << x_gpu_host[i];
            std::cerr << ", tolerance " << tol << ", difference " << abs(x_cpu[i] - x_gpu_host[i]) << std::endl;
            equal = 0;
            break;
        }
    }
    
    delete [] x_gpu_host;
    return equal;
}
template bool vectors_equal(FType* x_cpu, FType* x_gpu, IType n, FType tolerance);


template <typename T> void print_vector(T* x, IType n, std::string name) {
    std::cout << "\"" << name << "\": Vector with " << n << " elements" << std::endl;
    std::cout << " ";
    for (IType i = 0; i < n; i++) std::cout << " " << x[i];
    std::cout << std::endl;
}
template void print_vector(FType* x, IType n, std::string name);
template void print_vector(IType* x, IType n, std::string name);


template <typename T> void print_matrix_col_maj(T* mtx, IType m, IType n, std::string name) {
    std::cout << "\"" << name << "\": " << m << " x " << n << " matrix" << std::endl;
    for (IType i = 0; i < m; i++) {
        std::cout << " ";
        for (IType j = 0; j < n; j++) {
            std::cout << " " << mtx[j * m + i];
        }
        std::cout << std::endl;
    }
}
template void print_matrix_col_maj(FType* mtx, IType m, IType n, std::string name);
template void print_matrix_col_maj(IType* mtx, IType m, IType n, std::string name);


template <typename T> void print_matrix_row_maj(T* mtx, IType m, IType n, std::string name) {
    std::cout << "\"" << name << "\": " << m << " x " << n << " matrix" << std::endl;
    for (IType i = 0; i < m; i++) {
        std::cout << " ";
        for (IType j = 0; j < n; j++) {
            std::cout << " " << mtx[i * m + j];
        }
        std::cout << std::endl;
    }
}
template void print_matrix_row_maj(FType* mtx, IType m, IType n, std::string name);
template void print_matrix_row_maj(IType* mtx, IType m, IType n, std::string name);


void check_cuda(cudaError_t status, std::string message) {
    if (status != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(status);
        std::cerr << ". " << message << std::endl;
        exit(EXIT_FAILURE);
    }
}


std::string cublasGetErrorString(cublasStatus_t status) {
    // Source: https://stackoverflow.com/questions/13041399/equivalent-of-cudageterrorstring-for-cublas
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}
void check_cublas(cublasStatus_t status, std::string message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Error: " << cublasGetErrorString(status);
        std::cerr << ". " << message << std::endl;
        exit(EXIT_FAILURE);
    }
}


std::string cusolverGetErrorString(cusolverStatus_t status) {
    switch(status) {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }
    return "unknown error";
}
void check_cusolver(cusolverStatus_t status, std::string message) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error: " << cusolverGetErrorString(status);
        std::cerr << ". " << message << std::endl;
        exit(EXIT_FAILURE);
    }
}
