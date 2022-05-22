#ifndef BLCO_HPP_
#define BLCO_HPP_
#include "alto.hpp"
#include "utils.hpp"
#include "cublas_v2.h"

// Forward declare to resolve circular dependency
struct blcotensor;

// A single block in the BLCO format. Note we force IType as the LIT type
struct blco_block : public AltoTensor<IType> {
    // Length `nmodes` array, the coordinates of this block in the BLCO tensor
    IType* block_coords = nullptr;

    FType** pmatrices_staging_ptr = nullptr;
    FType** pmatrices = nullptr;

    // The BLCO tensor holding this block
    blcotensor* parent = nullptr;
};

// A sparse tensor in BLCO format
struct blcotensor {
    // The number of modes the tensor has
    IType N = 0;

    // Length `N` array where each element is length of the corresponding mode
    IType* modes = nullptr;

    // Length `N` array of masks per mode
    IType* mode_masks;

    // Length `N` array of shift offsets for modes
    int* mode_pos;

    // Same as `modes` but modes are number of bits required for each mode
    //IType* modes_bitcount = nullptr;

    // Total number of non-zeros across all blocks
    IType total_nnz = 0;

    // Number of nonzero elements in largest block
    IType max_nnz = 0;

    // The number of blocks in this tensor
    IType block_count = 0;

    // The maximum number of nonzero elements in each block
    IType max_block_size = 0;

    // Length `block_count` array, pointers to the blocks themselves
    blco_block** blocks = nullptr;

    // Length `block_count` array, pointers to the blocks themselves, on the GPU
    blco_block** blocks_dev_staging = nullptr;

    // Same as blocks_dev, but on the GPU
    blco_block** blocks_dev_ptr = nullptr;

    // Length ceil(`nnz` / TILE_SIZE) array, block information for each tile
    IType* tile_info;

    // Length `total_nnz` array, the linearized coordinates of each value
    IType* coords = nullptr;

    // Length `total_nnz` array, the values corresponding to each coordinate
    FType* values = nullptr;

    // Length `block_count` array, the GPU streams associated with each block
    cudaStream_t* streams = nullptr;

    // Length `block_count` array, the GPU events/signals associated with each block
    //cudaEvent_t* events = nullptr;

    IType* warp_info = nullptr;
    IType* warp_info_gpu = nullptr;
    IType warp_info_length = 0;
};

// Generates the blcotensor format with given max block size.
//
// Parameters:
//  - tensor: the sptensor to generate the format from
//  - max_block_size: the maximum block size per block
// Returns:
//  - The newly generated blcotensor
template <typename LIT>
blcotensor* gen_blcotensor_host(SparseTensor* tensor, IType max_block_size);

// Allocates memory on the GPU for the blcotensor format
//
// If MB is zero, then it is assumed that the data will not be streamed, and block
// sizes allocated will match the CPU side
//
// Parameters:
//  - X_host: the generated blcotensor on the host
//  - block_count: the number of blocks to allocate on the GPU. This should
//                 match the number of cuda streams in use
//  - MB: the size (number of nonzero elements stored) of each block
//  - do_batching: whether to batch kernel launches
blcotensor* gen_blcotensor_device(blcotensor* X_host, IType block_count, IType MB, bool do_batching);


// Deletes a sparse blcotensor stored on host side
//
// Parameters:
//  - tensor: the tensor to delete
// Returns:
//  - none
void delete_blcotensor_host(blcotensor* tensor);


// Deletes a sparse blcotensor stored on the device side
//
// Parameters:
//  - tensor: the tensor to delete
// Returns:
//  - none

void delete_blcotensor_device(blcotensor* tensor);


// Synchronously transfers the given blcotensor on the CPU to the GPU. This means
// all blocks will be copied over
//
// This assumes the GPU block has sufficient space allocated for the data
//
// Parameters:
//  - stream: the CUDA stream to use
//  - X_gpu: The blcotensor on the GPU to be overwritten by the CPU blcotensor
//  - X: The blcotensor on the CPU to transfer over
// Returns:
//  - none
void send_blcotensor_over(cudaStream_t stream, blcotensor* X_gpu, blcotensor* X);


// Asynchronously transfers the given block on CPU to the GPU, overwriting the GPU block
//
// This assumes the GPU block has sufficient space allocated for the data
//
// Parameters:
//  - stream: the CUDA stream to use
//  - gpu_block: The block on the GPU to be overwritten by the CPU block
//  - cpu_block: The block on the CPU to transfer over
// Returns:
//  - none
void send_block_over(cudaStream_t stream, blco_block* gpu_block, blco_block* cpu_block);


// Generates a block stored on the host
//
// Parameters:
//  - N: The number of dimensions of the block
//  - nnz: The number of nonzero elements in the block
//  - parent: The parent of the block, if any
// Returns:
//  - The newly created block
blco_block* gen_block_host(IType N, IType nnz, blcotensor* parent);


// Generates a block stored on the device
//
// Parameters:
//  - N: The number of dimensions of the block
//  - nnz: The number of nonzero elements in the block
//  - parent: The parent of the block, if any
// Returns:
//  - The newly created block
blco_block* gen_block_device(IType N, IType nnz, blcotensor* parent);


// Deletes a block stored on the host side
//
// Parameters:
//  - block: the block to delete
// Returns:
//  - none
void delete_block_host(blco_block* block);


// Deletes a block stored on the device side
//
// Parameters:
//  - block: the block to delete
// Returns:
//  - none
void delete_block_device(blco_block* block);


// Deletes a blcotensor stored on the host
//
// This method will also delete the blocks, which is assumed to be on the CPU
void delete_blcotensor_host(blcotensor* block);


template <typename LIT>
static inline blcotensor* gen_blcotensor_host(SparseTensor* spt, IType max_block_size) {
    double wtime_s, wtime;

    // Init ALTO
    AltoTensor<LIT>* _at;
    gen_alto(spt, &_at);
    assert(_at);

    // Init blocked lco tensor
    assert(_at->nmode <= MAX_NUM_MODES);
    int nmode = _at->nmode;
    IType nnz = _at->nnz;
    blcotensor* X = new blcotensor;
    X->N = nmode;
    X->modes = new IType[X->N];
    X->total_nnz = nnz;
    X->max_block_size = max_block_size;

    check_cuda(cudaMallocHost(&X->coords, sizeof(IType) * nnz), "cudaMallocHost coords"); // Pinned mem
    check_cuda(cudaMallocHost(&X->values, sizeof(FType) * nnz), "cudaMallocHost values");
    for (IType i = 0; i < nmode; i++) X->modes[i] = _at->dims[i];
    X->mode_pos = new int[X->N];
    X->mode_masks = new IType[X->N];

    // Count bits in lower half
    wtime_s = omp_get_wtime();
    int truncated_bitcounts[MAX_NUM_MODES];
    for (int i = 0; i < nmode; i++) {
        IType mask = lhalf(_at->mode_masks[i]); // Truncate LIT --> IType
        truncated_bitcounts[i] = popcount(mask);
    }

    // Construct mask and pos
    for (int i = 0; i < nmode; i++) {
        X->mode_masks[i] = ((IType) 1 << truncated_bitcounts[i]) - 1;
        X->mode_pos[i] = (i == 0) ? 0 : X->mode_pos[i - 1] + truncated_bitcounts[i - 1];
    }

    // Determine possible number of blocks, prep histogram
    int block_count = 0;
    for (int i = 0; i < nmode; i++) block_count += (sizeof(IType) * 8) - clz(_at->dims[i] - 1);
    block_count = block_count - sizeof(IType) * 8; // possible negative at this point
    block_count = std::max(0, block_count);
    block_count = 1 << block_count;
    IType* block_histogram = new IType[block_count];
    IType* block_prefix_sum = new IType[block_count + 1];
    #pragma omp parallel for
    for (IType i = 0; i < block_count; i++) block_histogram[i] = 0;

    wtime = omp_get_wtime() - wtime_s;
    printf("BLCO: Setup time = %f (s)\n", wtime);

    // Construct block histogram. OpenMP 4.5+ required
    wtime_s = omp_get_wtime();
    if (block_count > 1) {
       #pragma omp parallel for reduction(+:block_histogram[:block_count])
       for (IType i = 0; i < nnz; i++) {
           IType block = uhalf(_at->idx[i]); // Get upper half, truncate LIT --> IType
           block_histogram[block] += 1;
       }
    } else {
		block_histogram[0] = X->total_nnz;
    }
    wtime = omp_get_wtime() - wtime_s;
    printf("BLCO: Histogram time = %f (s)\n", wtime);

    // Construct prefix sum and count total blocks
    wtime_s = omp_get_wtime();
    IType total_blocks_split = 0;
    block_prefix_sum[0] = 0;
    if (max_block_size <= 0) max_block_size = nnz;
    for (IType i = 0; i < block_count; i++) {
        // ceil(block_histogram / max_block_size)
        total_blocks_split += (block_histogram[i] + max_block_size - 1) / max_block_size;
        block_prefix_sum[i + 1] = block_prefix_sum[i] + block_histogram[i];
    }

    // Relinearize and copy into BLCO
    #pragma omp parallel for schedule(static)
    for (IType i = 0; i < nnz; i++) {
        LIT index = _at->idx[i];
        IType new_index = 0;
        for (int n = 0; n < nmode; ++n) {
            IType mode_idx = (IType) pext(index, _at->mode_masks[n]) & X->mode_masks[n];
            new_index |= (mode_idx << X->mode_pos[n]);
        }

        X->coords[i] = new_index;
        X->values[i] = _at->vals[i];
    }
    wtime = omp_get_wtime() - wtime_s;
    printf("BLCO: Relinearize time = %f (s)\n", wtime);

    // Construct split blocks
    wtime_s = omp_get_wtime();
    blco_block** blocks = new blco_block*[total_blocks_split];
    X->blocks = blocks;
    X->block_count = total_blocks_split;
    IType curr_blocks = 0;
    for (IType block = 0; block < block_count; block++) {
        IType start = block_prefix_sum[block];
        IType end = block_prefix_sum[block + 1];
        IType nnz = end - start;
        if (nnz > 0) {
            // Generate block coordinates, tricky because we can only use "block"
            IType block_coords[MAX_NUM_MODES];
            for (int i = 0; i < X->N; i++) {
                IType mode_mask = uhalf(_at->mode_masks[i]); // Truncate LIT --> IType
                IType mode_idx = pext(block, mode_mask);
                mode_idx <<= truncated_bitcounts[i];
                block_coords[i] = mode_idx;
            }

            // Generate split block, indices are just offset pointers into main array
            for (IType stride = 0; stride < nnz; stride += max_block_size) {
                IType split_block_nnz = std::min(max_block_size, nnz - stride);
                blco_block* blk = gen_block_host(X->N, split_block_nnz, X);
                blk->idx = X->coords + start + stride;
                blk->vals = X->values + start + stride;
                blocks[curr_blocks] = blk;
                curr_blocks++;
                for (IType i = 0; i < X->N; i++) blk->block_coords[i] = block_coords[i];
            }
        }
    }

    wtime = omp_get_wtime() - wtime_s;
    printf("BLCO: Blocking time = %f (s)\n", wtime);
    printf("Total blocks: %d\n\n", total_blocks_split);

    destroy_alto(_at);
    delete [] block_histogram;
    delete [] block_prefix_sum;
    return X;
}

// Won't compile if static is present
inline blco_block* gen_block_host(IType N, IType nnz, blcotensor* parent) {
    blco_block* b = new blco_block;
    check_cuda(cudaMallocHost(&b->block_coords, sizeof(IType) * N), "cudaMallocHost block_coords");
    b->nmode = N;
    b->nnz = nnz;
    b->parent = parent;
    // The blocks themselves no longer need the masks, parent has them
    return b;
}

#endif
