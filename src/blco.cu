#include "omp.h"
#include "blco.hpp"
#include "utils.hpp"
//#include "cblas.h"
#include <algorithm>
#include <iostream>

#ifndef ALT_PEXT
#error ALTERNATIVE_PEXT must be set to run GPU code
#endif

blcotensor* gen_blcotensor_device(blcotensor* X_host, IType block_count, IType MB, bool do_batching) {
    blcotensor* X = new blcotensor;
    X->N = X_host->N;
    X->modes = make_device_copy(X_host->modes, X->N, "cudaMemcpy modes");
    X->mode_masks = make_device_copy(X_host->mode_masks, X->N, "cudaMemcpy mode_masks");
    X->mode_pos = make_device_copy(X_host->mode_pos, X->N, "cudaMemcpy mode_pos");
    X->total_nnz = X_host->total_nnz;
    X->block_count = block_count;
    X->blocks = new blco_block*[block_count];
    X->blocks_dev_staging = new blco_block*[block_count];
    check_cuda(cudaMalloc(&X->blocks_dev_ptr, sizeof(blco_block*) * block_count), "cudaMalloc blocks_dev_ptr");
    X->streams = new cudaStream_t[block_count];
    //X->streams = new cudaStream_t[block_count + 1]; // last one is a control/signal stream
    //X->events = (cudaEvent_t *)malloc(block_count * sizeof(cudaEvent_t));

    if (do_batching) {
        IType warp_info_length = 0;
        for (IType i = 0; i < block_count; i++) {
            warp_info_length += (X_host->blocks[i]->nnz + TILE_SIZE - 1) / TILE_SIZE;
        }
        X->warp_info_length = warp_info_length;
        X->warp_info = new IType[X->warp_info_length];
    } else {
        X->warp_info_length = 0;
        X->warp_info = nullptr;
        X->warp_info_gpu = nullptr;
    }

    IType warp = 0;
    for (IType i = 0; i < block_count; i++) {
        IType mb = (MB == 0) ? X_host->blocks[i]->nnz : MB;
        X->max_nnz = max(X->max_nnz, mb);
        X->blocks[i] = gen_block_device(X->N, mb, X);
        check_cuda(cudaMalloc(X->blocks_dev_staging + i, sizeof(blco_block)), "cudaMalloc blocks_dev_staging");
        check_cuda(cudaStreamCreate(X->streams + i), "cudaStreamCreate");

        if (do_batching) for (IType j = 0; j < mb; j += TILE_SIZE) {
            // Encode block ID and offset
            X->warp_info[warp++] = j * block_count + i;
        }
        //check_cuda(cudaEventCreateWithFlags(X->events + i, cudaEventDisableTiming), "cudaEventCreateWithFlags");
    }
    // create the control/signal stream
    //check_cuda(cudaStreamCreate(X->streams + block_count), "cudaStreamCreate");
    if (do_batching) X->warp_info_gpu = make_device_copy(X->warp_info, X->warp_info_length, "cudaMalloc warp_info_gpu");
    return X;
}


void delete_blcotensor_host(blcotensor* tensor) {
    for (IType i = 0; i < tensor->block_count; i++) delete_block_host(tensor->blocks[i]);
    delete [] tensor->blocks;
    delete [] tensor->mode_masks;
    delete [] tensor->mode_pos;
    delete [] tensor->modes;
    cudaFreeHost(tensor->coords);
    cudaFreeHost(tensor->values);
    delete tensor;
}


void delete_blcotensor_device(blcotensor* tensor) {
    for (IType i = 0; i < tensor->block_count; i++) {
        delete_block_device(tensor->blocks[i]);
        cudaStreamDestroy(tensor->streams[i]);
        cudaFree(tensor->blocks_dev_staging[i]);
        //cudaEventDestroy(tensor->events[i]);
    }
    //cudaStreamDestroy(tensor->streams[tensor->block_count]);
    delete [] tensor->blocks;
    delete [] tensor->blocks_dev_staging;
    cudaFree(tensor->blocks_dev_ptr);
    delete [] tensor->streams;
    cudaFree(tensor->mode_masks);
    cudaFree(tensor->mode_pos);
    cudaFree(tensor->modes);
    cudaFree(tensor->warp_info_gpu);
    delete [] tensor->warp_info;
    delete tensor;
}


blco_block* gen_block_device(IType N, IType nnz, blcotensor* parent) {
    blco_block* b = new blco_block;
    b->nmode = N;
    check_cuda(cudaMalloc(&b->block_coords, sizeof(IType) * N), "cudaMalloc block_coords");
    b->nnz = nnz;
    check_cuda(cudaMalloc(&b->idx, sizeof(IType) * nnz), "cudaMalloc block idx");
    check_cuda(cudaMalloc(&b->vals, sizeof(FType) * nnz), "cudaMalloc block vals");
    b->parent = parent;
    // The blocks themselves no longer need the masks, parent has them
    return b;
}


void delete_block_host(blco_block* block) {
    cudaFreeHost(block->block_coords);
    delete block;
}


void delete_block_device(blco_block* block) {
    cudaFree(block->block_coords);
    cudaFree(block->idx);
    cudaFree(block->vals);
    delete block;
}

void send_blcotensor_over(cudaStream_t stream, blcotensor* X_gpu, blcotensor* X) {
    // Assume GPU has enough space reserved

    for (IType i = 0; i < X->block_count; i++) {
        send_block_over(stream, X_gpu->blocks[i], X->blocks[i]); 
        check_cuda(cudaMemcpyAsync(X_gpu->blocks_dev_staging[i], X_gpu->blocks[i], sizeof(blco_block), cudaMemcpyHostToDevice, stream), "cudaMemcpy blocks_dev_staging");
    }

    check_cuda(cudaMemcpyAsync(X_gpu->blocks_dev_ptr, X_gpu->blocks_dev_staging, sizeof(blco_block*) * X->block_count, cudaMemcpyHostToDevice, stream), "cudaMemcpy blocks_dev_ptr");

    check_cuda(cudaStreamSynchronize(stream), "cudaMemcpy block");
}

void send_block_over(cudaStream_t stream, blco_block* gpu_block, blco_block* cpu_block) {
    check_cuda(cudaMemcpyAsync(gpu_block->idx, cpu_block->idx, cpu_block->nnz * sizeof(IType), cudaMemcpyHostToDevice, stream), "cudaMemcpy block idx");
    check_cuda(cudaMemcpyAsync(gpu_block->vals, cpu_block->vals, cpu_block->nnz * sizeof(FType), cudaMemcpyHostToDevice, stream), "cudaMemcpy block vals");
    check_cuda(cudaMemcpyAsync(gpu_block->block_coords, cpu_block->block_coords, cpu_block->nmode * sizeof(IType), cudaMemcpyHostToDevice, stream), "cudaMemcpy block_coords");
    gpu_block->nnz = cpu_block->nnz;
}
