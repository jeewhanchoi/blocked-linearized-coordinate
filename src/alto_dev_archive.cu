#if 0
#include "alto_dev.hpp"
#include "utils.hpp"
#include <cassert>
#include <cooperative_groups.h>
#include "blco.hpp"

namespace cg = cooperative_groups;

#define INVALID_ID	((IType) -1)

__constant__ IType ALTO_MASKS[MAX_NUM_MODES];
__constant__  int ALTO_POS[MAX_NUM_MODES];    

template <typename LIT>
__device__ inline IType alt_pext(LIT x, int pos, IType mask) {
    return (x >> pos) & mask;
}

__device__ void mutex_lock(int *mutex) {
    unsigned int ns = 8;
    while (atomicCAS(mutex, 0, 1) != 0) {
         __nanosleep(ns);
         if (ns < 256) ns *= 2;
    }
}

__device__ void mutex_unlock(int *mutex) {
    atomicExch(mutex, 0);
}

// Register-based conflict resolution using tile-based execution
template <typename LIT>
__global__ void mttkrp_lvl1_3d_kernel(FType* f0, FType* f1, FType* f2, FType* output, const int tmode, 
        const IType rank, const LIT* __restrict__ lidx, const FType* __restrict__ values, const IType nnz) 
{
    auto block = cg::this_thread_block();
    // Used for warp level primitives (must be static or coalesced group)
    auto tile = cg::tiled_partition<TILE_SIZE>(block);
    const int tid = block.thread_rank();

    // Setup caches (stash)
    extern __shared__ int count[]; // [block.size()] 
    IType *nnz_idx = (IType*) (count + block.size()); // [block.size() * 4] 
    IType *nnz_out = (IType*) (nnz_idx + 3 * block.size()); // [block.size()] 
    FType* nnz_val = (FType*) (nnz_out + block.size()); // [block.size()] 
    //block.sync();

    // Identify block-level workload
    IType curr_elem = block.group_index().x * block.size(); // Index of start element
    IType end_elem = min(nnz, curr_elem + block.size()); // Index of last element
       
    const int mp0 = ALTO_POS[0];
    const int mp1 = ALTO_POS[1];
    const int mp2 = ALTO_POS[2];
    const LIT mm0 = ALTO_MASKS[0];
    const LIT mm1 = ALTO_MASKS[1];
    const LIT mm2 = ALTO_MASKS[2];

    // Iterate workload
    while (curr_elem < end_elem) {
        // Threads collaborate to perform On-the-fly delinerization, sorting, and segmented scan. 
        count[tid] = 0;
        
        //const LIT idx = lidx[curr_elem+tid];
        LIT idx;
        IType x, y, z, output_row;
        if (curr_elem+tid < end_elem) { 
            idx = lidx[curr_elem+tid];
            x = alt_pext(idx, mp0, mm0, bc0);
            y = alt_pext(idx, mp1, mm1, bc1);
            z = alt_pext(idx, mp2, mm2, bc2);
            if (tmode == 0) output_row = x;
            else if (tmode == 1) output_row = y;
            else output_row = z;    
        } else {
            x = y = z = output_row = (IType) -1;
        }
        
        // Find subgroups of threads with same key (output_row).
        block.sync();
        int sg_mask = tile.match_any(output_row);
        auto sg = cg::labeled_partition(tile, sg_mask);
        int sg_rank = sg.thread_rank();
        int sg_id = sg.meta_group_rank();
        if (sg_rank == 0) count[sg_id+1] = sg.size(); // OOB writes will be overwritten later.
        block.sync();
        
        // Scan for counting sort
        sg_mask = count[tid];
        //block.sync();
        #pragma unroll
        for (int j = 1; j < tile.size(); j <<= 1) {
            int temp = tile.shfl_up(sg_mask, j);
            if (tid >= j) sg_mask += temp;
        }
        count[tid] = sg_mask;
        block.sync();
 
        // Sorted rank
        sg_rank += count[sg_id];
        
        // Strided access to facilitate broadcast later
        nnz_idx[sg_rank * 3]  = x;
        nnz_idx[sg_rank * 3 + 1]  = y;
        nnz_idx[sg_rank * 3 + 2 ]  = z;
        nnz_out[sg_rank] = output_row;
        if (curr_elem+tid < end_elem) nnz_val[sg_rank] = values[curr_elem+tid];
        
        // Segmented scan structure (reuse sg_mask).
        if (sg.thread_rank() == 0) sg_mask = 1<<sg_rank;
        else sg_mask = 0;
        #pragma unroll
        for (int j = tile.size()/2; j > 0; j >>= 1) {
            sg_mask |= tile.shfl_down(sg_mask, j);
        }
        sg_mask = tile.shfl(sg_mask, 0);
        
        // Now threads perform rank-wise operations.
        int n = 0;
        while (n < block.size() &&  (curr_elem + n) < end_elem) {
            //block.sync();
            
            // Perform update
            const IType output_row = nnz_out[n];
            const int next_n = n;
            for (IType i = tid; i < rank; i += block.size()) {               
                // Register-based accumlation
                FType value = 0.0;
                n = next_n;
                do {
                    // Broadcast 
                    FType val = nnz_val[n];
                    x = nnz_idx[n * 3];
                    y = nnz_idx[n * 3 + 1];
                    z = nnz_idx[n * 3 + 2];
                    
                    if (tmode == 0) val *= f1[rank * y + i] * f2[rank * z + i];
                    else if (tmode == 1) val *= f0[rank * x + i] * f2[rank * z + i];
                    else val *= f0[rank * x + i] * f1[rank * y + i];                    
                    
                    value += val;
                    ++n;
                } while (n < block.size() && !(sg_mask & (1<<n)));
                atomicAdd(output + output_row * rank + i, value);
            } // rank
            // broadcast n
            n = tile.shfl(n, 0);
        } // block.size()
        curr_elem += block.size();
    } // curr_elem < end_elem
}

template <typename LIT>
__global__ void mttkrp_lvl1_4d_kernel(FType* f0, FType* f1, FType* f2, FType* f3, FType* output, const int tmode, 
        const IType rank, const LIT* __restrict__ lidx, const FType* __restrict__ values, const IType nnz, IType* block_coords) 
{
    auto block = cg::this_thread_block();
    // Used for warp level primitives (must be static or coalesced group)
    auto tile = cg::tiled_partition<TILE_SIZE>(block);
    const int tid = block.thread_rank();

    // Setup caches (stash)
    extern __shared__ int count[]; // [block.size()] 
    IType *nnz_idx = (IType*) (count + block.size()); // [block.size() * 4] 
    IType *nnz_out = (IType*) (nnz_idx + 4 * block.size()); // [block.size()] 
    FType* nnz_val = (FType*) (nnz_out + block.size()); // [block.size()] 
    //block.sync();

    // Identify block-level workload
    IType curr_elem = block.group_index().x * block.size(); // Index of start element
    IType end_elem = min(nnz, curr_elem + block.size()); // Index of last element
       
    const int mp0 = ALTO_POS[0];
    const int mp1 = ALTO_POS[1];
    const int mp2 = ALTO_POS[2];
    const int mp3 = ALTO_POS[3];
    const LIT mm0 = ALTO_MASKS[0];
    const LIT mm1 = ALTO_MASKS[1];
    const LIT mm2 = ALTO_MASKS[2];
    const LIT mm3 = ALTO_MASKS[3];
    const IType bc0 = 

    // Iterate workload
    while (curr_elem < end_elem) {
        // Threads collaborate to perform On-the-fly delinerization, sorting, and segmented scan. 
        count[tid] = 0;
        
        //const LIT idx = lidx[curr_elem+tid];
        LIT idx;
        IType x, y, z, w, output_row;
        if (curr_elem+tid < end_elem) { 
            idx = lidx[curr_elem+tid];
            x = alt_pext(idx, mp0, mm0, bc0);
            y = alt_pext(idx, mp1, mm1, bc1);
            z = alt_pext(idx, mp2, mm2, bc2);
            w = alt_pext(idx, mp3, mm3, bc3);
            if (tmode == 0) output_row = x;
            else if (tmode == 1) output_row = y;
            else if (tmode == 2) output_row = z;
            else output_row = w;
        } else {
            x = y = z = w = output_row = (IType) -1;
        }
        
        // Find subgroups of threads with same key (output_row).
        block.sync();
        int sg_mask = tile.match_any(output_row);
        auto sg = cg::labeled_partition(tile, sg_mask);
        int sg_rank = sg.thread_rank();
        int sg_id = sg.meta_group_rank();
        if (sg_rank == 0) count[sg_id+1] = sg.size(); // OOB writes will be overwritten later.
        block.sync();
        
        // Scan for counting sort
        sg_mask = count[tid];
        //block.sync();
        #pragma unroll
        for (int j = 1; j < tile.size(); j <<= 1) {
            int temp = tile.shfl_up(sg_mask, j);
            if (tid >= j) sg_mask += temp;
        }
        count[tid] = sg_mask;
        block.sync();
 
        // Sorted rank
        sg_rank += count[sg_id];
        
        // Strided access to facilitate broadcast later
        nnz_idx[sg_rank * 4]  = x;
        nnz_idx[sg_rank * 4 + 1]  = y;
        nnz_idx[sg_rank * 4 + 2 ]  = z;
        nnz_idx[sg_rank * 4 + 3 ]  = w;
        nnz_out[sg_rank] = output_row;
        if (curr_elem+tid < end_elem) nnz_val[sg_rank] = values[curr_elem+tid];
        
        // Segmented scan structure (reuse sg_mask).
        if (sg.thread_rank() == 0) sg_mask = 1<<sg_rank;
        else sg_mask = 0;
        #pragma unroll
        for (int j = tile.size()/2; j > 0; j >>= 1) {
            sg_mask |= tile.shfl_down(sg_mask, j);
        }
        sg_mask = tile.shfl(sg_mask, 0);
        
        // Now threads perform rank-wise operations.
        int n = 0;
        while (n < block.size() &&  (curr_elem + n) < end_elem) {
            //block.sync();
            
            // Perform update
            const IType output_row = nnz_out[n];
            const int next_n = n;
            for (IType i = tid; i < rank; i += block.size()) {               
                // Register-based accumlation
                FType value = 0.0;
                n = next_n;
                do {
                    // Broadcast 
                    FType val = nnz_val[n];
                    x = nnz_idx[n * 4];
                    y = nnz_idx[n * 4 + 1];
                    z = nnz_idx[n * 4 + 2];
                    w = nnz_idx[n * 4 + 3];
                    
                    if (tmode == 0) val *= f1[rank * y + i] * f2[rank * z + i] * f3[rank * w + i];
                    else if (tmode == 1) val *= f0[rank * x + i] * f2[rank * z + i] * f3[rank * w + i];
                    else if (tmode == 2) val *= f0[rank * x + i] * f1[rank * y + i] * f3[rank * w + i];                 
                    else val *= f0[rank * x + i] * f1[rank * y + i] * f2[rank * z + i];
                    
                    value += val;
                    ++n;
                } while (n < block.size() && !(sg_mask & (1<<n)));
                atomicAdd(output + output_row * rank + i, value);
            } // rank
            // broadcast n
            n = tile.shfl(n, 0);
        } // block.size()
        curr_elem += block.size();
    } // curr_elem < end_elem
    //block.sync();
}

// Register-based conflict resolution using tile-based execution
template <typename LIT>
void mttkrp_lvl1(blco_block* at, FType** mats_staging_ptr, FType** mats_dev, int target_mode, IType rank) 
{
    int nnz_block = TILE_SIZE;
    IType blocks = (at->nnz + nnz_block -1) / nnz_block;
    FType* output = mats_staging_ptr[target_mode];
    int smem_sz = nnz_block * (sizeof(FType) + (at->nmode+1) * sizeof(IType) + sizeof(int)) ;

    if (at->nmode == 3) {
        mttkrp_lvl1_3d_kernel<<<blocks, nnz_block, smem_sz, 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], 
            output, target_mode, rank, at->idx, at->vals, at->nnz, at->block_coords);
    } else if (at->nmode == 4) {
        mttkrp_lvl1_4d_kernel<<<blocks, nnz_block, smem_sz, 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], 
            output, target_mode, rank, at->idx, at->vals, at->nnz, at->block_coords);
    } else {
        printf("!ERROR! Only order 3 and 4 tensors are supported\n");
    }
    check_cuda(cudaGetLastError(), "mttkrp_lvl1_kernel launch. Exceeded shared mem space?");
}

// Register- and smem-based conflict resolution using tile-based execution with thread coarsening
template <typename LIT>
__global__ void mttkrp_lvl2_3d_kernel(FType* f0, FType* f1, FType* f2, FType* output, const int tmode, 
        const IType rank, const LIT* __restrict__ lidx, const FType* __restrict__ values, const IType nnz, IType THREAD_CF) 
{
    auto block = cg::this_thread_block();
    // Used for warp level primitives (must be static or coalesced group)
    auto tile = cg::tiled_partition<TILE_SIZE>(block);
    const int tid = block.thread_rank();

    // Setup caches (stash)
    extern __shared__ int count[]; // [block.size()] 
    IType *nnz_idx = (IType*) (count + block.size()); // [block.size() * 4] 
    IType *nnz_out = (IType*) (nnz_idx + 3 * block.size()); // [block.size()] 
    FType* nnz_val = (FType*) (nnz_out + block.size()); // [block.size()] 
    FType* data = (FType*) (nnz_val + block.size()); // [rank * STASH_SIZE]
    IType* tags = (IType*) (data + rank * STASH_SIZE); // Output row ID [STASH_SIZE]
    for (int i = tid; i < STASH_SIZE; i += block.size()) {
        tags[i] = INVALID_ID;
    }
    //block.sync();

    // Identify block-level workload
    IType curr_elem = block.group_index().x * block.size() * THREAD_CF; // Index of start element
    IType end_elem = min(nnz, curr_elem + block.size() * THREAD_CF); // Index of last element
       
    const int mp0 = ALTO_POS[0];
    const int mp1 = ALTO_POS[1];
    const int mp2 = ALTO_POS[2];
    const LIT mm0 = ALTO_MASKS[0];
    const LIT mm1 = ALTO_MASKS[1];
    const LIT mm2 = ALTO_MASKS[2];

    // Iterate workload
    while (curr_elem < end_elem) {
        // Threads collaborate to perform On-the-fly delinerization, sorting, and segmented scan. 
        count[tid] = 0;
        
        //const LIT idx = lidx[curr_elem+tid];
        LIT idx;
        IType x, y, z, output_row;
        if (curr_elem+tid < end_elem) { 
            idx = lidx[curr_elem+tid];
            x = alt_pext(idx, mp0, mm0, bc0);
            y = alt_pext(idx, mp1, mm1, bc1);
            z = alt_pext(idx, mp2, mm2, bc2);
            if (tmode == 0) output_row = x;
            else if (tmode == 1) output_row = y;
            else output_row = z;    
        } else {
            x = y = z = output_row = (IType) -1;
        }
        
        // Find subgroups of threads with same key (output_row).
        block.sync();
        int sg_mask = tile.match_any(output_row);
        auto sg = cg::labeled_partition(tile, sg_mask);
        int sg_rank = sg.thread_rank();
        int sg_id = sg.meta_group_rank();
        if (sg_rank == 0) count[sg_id+1] = sg.size(); // OOB writes will be overwritten later.
        block.sync();
        
        // Scan for counting sort
        sg_mask = count[tid];
        //block.sync();
        #pragma unroll
        for (int j = 1; j < tile.size(); j <<= 1) {
            int temp = tile.shfl_up(sg_mask, j);
            if (tid >= j) sg_mask += temp;
        }
        count[tid] = sg_mask;
        block.sync();
 
        // Sorted rank
        sg_rank += count[sg_id];
        
        // Strided access to facilitate broadcast later
        nnz_idx[sg_rank * 3]  = x;
        nnz_idx[sg_rank * 3 + 1]  = y;
        nnz_idx[sg_rank * 3 + 2 ]  = z;
        nnz_out[sg_rank] = output_row;
        if (curr_elem+tid < end_elem) nnz_val[sg_rank] = values[curr_elem+tid];
        
        // Segmented scan structure (reuse sg_mask).
        if (sg.thread_rank() == 0) sg_mask = 1<<sg_rank;
        else sg_mask = 0;
        #pragma unroll
        for (int j = tile.size()/2; j > 0; j >>= 1) {
            sg_mask |= tile.shfl_down(sg_mask, j);
        }
        sg_mask = tile.shfl(sg_mask, 0);
        
        // Now threads perform rank-wise operations.
        int n = 0;
        while (n < block.size() &&  (curr_elem + n) < end_elem) {
            //block.sync();
            
            // Prep stash line
            const IType output_row = nnz_out[n];
            int stash_line = (int) output_row & (STASH_SIZE - 1); // Modulo hash function
            if (tags[stash_line] == INVALID_ID) {
               // Initialize cache line
               for (IType i = tid; i < rank; i += block.size()) {
                   data[stash_line * rank + i] = 0.0;
               }
               if (tid == 0) tags[stash_line] = output_row;
            }
            else if (tags[stash_line] != output_row) {
                // Evict cache line to global mem (evict-first policy)
                for (IType i = tid; i < rank; i += block.size()) {
                    atomicAdd(output + tags[stash_line] * rank + i, data[stash_line * rank + i]);
                    data[stash_line * rank + i] = 0.0;
                } 
               if (tid == 0) tags[stash_line] = output_row;
             }
            block.sync();
        
            // Perform update
            const int next_n = n;
            for (IType i = tid; i < rank; i += block.size()) {               
                // Register-based accumlation
                FType value = 0.0;
                n = next_n;
                do {
                    // Broadcast 
                    FType val = nnz_val[n];
                    x = nnz_idx[n * 3];
                    y = nnz_idx[n * 3 + 1];
                    z = nnz_idx[n * 3 + 2];
                    
                    if (tmode == 0) val *= f1[rank * y + i] * f2[rank * z + i];
                    else if (tmode == 1) val *= f0[rank * x + i] * f2[rank * z + i];
                    else val *= f0[rank * x + i] * f1[rank * y + i];                    
                    
                    value += val;
                    ++n;
                } while (n < block.size() && !(sg_mask & (1<<n)));
                data[stash_line * rank + i] += value;                   
            } // rank
            // broadcast n
            n = tile.shfl(n, 0);
        } // block.size()
        curr_elem += block.size();
    } // curr_elem < end_elem
    //block.sync();

    // Write STASH to global
    #pragma unroll
    for (int stash_line = 0; stash_line < STASH_SIZE; stash_line++) {
        IType output_row = tags[stash_line];
        if (output_row != INVALID_ID) {
            for (IType i = tid; i < rank; i += block.size()) {
                atomicAdd(output + output_row * rank + i, data[stash_line * rank + i]);
            }
        }
    }
}

template <typename LIT>
__global__ void mttkrp_lvl2_4d_kernel(FType* f0, FType* f1, FType* f2, FType* f3, FType* output, const int tmode, 
        const IType rank, const LIT* __restrict__ lidx, const FType* __restrict__ values, const IType nnz, IType THREAD_CF) 
{
    auto block = cg::this_thread_block();
    // Used for warp level primitives (must be static or coalesced group)
    auto tile = cg::tiled_partition<TILE_SIZE>(block);
    const int tid = block.thread_rank();

    // Setup caches (stash)
    extern __shared__ int count[]; // [block.size()] 
    IType *nnz_idx = (IType*) (count + block.size()); // [block.size() * 4] 
    IType *nnz_out = (IType*) (nnz_idx + 4 * block.size()); // [block.size()] 
    FType* nnz_val = (FType*) (nnz_out + block.size()); // [block.size()] 
    FType* data = (FType*) (nnz_val + block.size()); // [rank * STASH_SIZE]
    IType* tags = (IType*) (data + rank * STASH_SIZE); // Output row ID [STASH_SIZE]
    for (int i = tid; i < STASH_SIZE; i += block.size()) {
        tags[i] = INVALID_ID;
    }
    //block.sync();

    // Identify block-level workload
    IType curr_elem = block.group_index().x * block.size() * THREAD_CF; // Index of start element
    IType end_elem = min(nnz, curr_elem + block.size() * THREAD_CF); // Index of last element
       
    const int mp0 = ALTO_POS[0];
    const int mp1 = ALTO_POS[1];
    const int mp2 = ALTO_POS[2];
    const int mp3 = ALTO_POS[3];
    const LIT mm0 = ALTO_MASKS[0];
    const LIT mm1 = ALTO_MASKS[1];
    const LIT mm2 = ALTO_MASKS[2];
    const LIT mm3 = ALTO_MASKS[3];

    // Iterate workload
    while (curr_elem < end_elem) {
        // Threads collaborate to perform On-the-fly delinerization, sorting, and segmented scan. 
        count[tid] = 0;
        
        //const LIT idx = lidx[curr_elem+tid];
        LIT idx;
        IType x, y, z, w, output_row;
        if (curr_elem+tid < end_elem) { 
            idx = lidx[curr_elem+tid];
            x = alt_pext(idx, mp0, mm0, bc0);
            y = alt_pext(idx, mp1, mm1, bc1);
            z = alt_pext(idx, mp2, mm2, bc2);
            w = alt_pext(idx, mp3, mm3, bc3);
            if (tmode == 0) output_row = x;
            else if (tmode == 1) output_row = y;
            else if (tmode == 2) output_row = z;
            else output_row = w;
        } else {
            x = y = z = w = output_row = (IType) -1;
        }
        
        // Find subgroups of threads with same key (output_row).
        block.sync();
        int sg_mask = tile.match_any(output_row);
        auto sg = cg::labeled_partition(tile, sg_mask);
        int sg_rank = sg.thread_rank();
        int sg_id = sg.meta_group_rank();
        if (sg_rank == 0) count[sg_id+1] = sg.size(); // OOB writes will be overwritten later.
        block.sync();
        
        // Scan for counting sort
        sg_mask = count[tid];
        //block.sync();
        #pragma unroll
        for (int j = 1; j < tile.size(); j <<= 1) {
            int temp = tile.shfl_up(sg_mask, j);
            if (tid >= j) sg_mask += temp;
        }
        count[tid] = sg_mask;
        block.sync();
 
        // Sorted rank
        sg_rank += count[sg_id];
        
        // Strided access to facilitate broadcast later
        nnz_idx[sg_rank * 4]  = x;
        nnz_idx[sg_rank * 4 + 1]  = y;
        nnz_idx[sg_rank * 4 + 2 ]  = z;
        nnz_idx[sg_rank * 4 + 3 ]  = w;
        nnz_out[sg_rank] = output_row;
        if (curr_elem+tid < end_elem) nnz_val[sg_rank] = values[curr_elem+tid];
        
        // Segmented scan structure (reuse sg_mask).
        if (sg.thread_rank() == 0) sg_mask = 1<<sg_rank;
        else sg_mask = 0;
        #pragma unroll
        for (int j = tile.size()/2; j > 0; j >>= 1) {
            sg_mask |= tile.shfl_down(sg_mask, j);
        }
        sg_mask = tile.shfl(sg_mask, 0);
        
        // Now threads perform rank-wise operations.
        int n = 0;
        while (n < block.size() &&  (curr_elem + n) < end_elem) {
            //block.sync();
            
            // Prep stash line
            const IType output_row = nnz_out[n];
            int stash_line = (int) output_row & (STASH_SIZE - 1); // Modulo hash function
            if (tags[stash_line] == INVALID_ID) {
               // Initialize cache line
               for (IType i = tid; i < rank; i += block.size()) {
                   data[stash_line * rank + i] = 0.0;
               }
               if (tid == 0) tags[stash_line] = output_row;
            }
            else if (tags[stash_line] != output_row) {
                // Evict cache line to global mem (evict-first policy)
                for (IType i = tid; i < rank; i += block.size()) {
                    atomicAdd(output + tags[stash_line] * rank + i, data[stash_line * rank + i]);
                    data[stash_line * rank + i] = 0.0;
                } 
               if (tid == 0) tags[stash_line] = output_row;
             }
            block.sync();
        
            // Perform update
            const int next_n = n;
            for (IType i = tid; i < rank; i += block.size()) {               
                // Register-based accumlation
                FType value = 0.0;
                n = next_n;
                do {
                    // Broadcast 
                    FType val = nnz_val[n];
                    x = nnz_idx[n * 4];
                    y = nnz_idx[n * 4 + 1];
                    z = nnz_idx[n * 4 + 2];
                    w = nnz_idx[n * 4 + 3];
                    
                    if (tmode == 0) val *= f1[rank * y + i] * f2[rank * z + i] * f3[rank * w + i];
                    else if (tmode == 1) val *= f0[rank * x + i] * f2[rank * z + i] * f3[rank * w + i];
                    else if (tmode == 2) val *= f0[rank * x + i] * f1[rank * y + i] * f3[rank * w + i];                 
                    else val *= f0[rank * x + i] * f1[rank * y + i] * f2[rank * z + i];
                    
                    value += val;
                    ++n;
                } while (n < block.size() && !(sg_mask & (1<<n)));
                data[stash_line * rank + i] += value;                   
            } // rank
            // broadcast n
            n = tile.shfl(n, 0);
        } // block.size()
        curr_elem += block.size();
    } // curr_elem < end_elem
    //block.sync();

    // Write STASH to global
    #pragma unroll
    for (int stash_line = 0; stash_line < STASH_SIZE; stash_line++) {
        IType output_row = tags[stash_line];
        if (output_row != INVALID_ID) {
            for (IType i = tid; i < rank; i += block.size()) {
                atomicAdd(output + output_row * rank + i, data[stash_line * rank + i]);
            }
        }
    }
}

// Register- and smem-based conflict resolution using tile-based execution with thread coarsening
template <typename LIT>
void mttkrp_lvl2(AltoTensor<LIT>* at, FType** mats_staging_ptr, FType** mats_dev, int target_mode, IType rank, IType THREAD_CF) 
{
    int nnz_block = TILE_SIZE * THREAD_CF;
    IType blocks = (at->nnz + nnz_block -1) / nnz_block;
    FType* output = mats_staging_ptr[target_mode];
    int smem_sz = TILE_SIZE * (sizeof(FType) + (at->nmode+1) * sizeof(IType) + sizeof(int)) ;
    smem_sz += (rank + 1) * STASH_SIZE * sizeof(FType);

    if (at->nmode == 3) {
        mttkrp_lvl2_3d_kernel<<<blocks, TILE_SIZE, smem_sz, 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], 
            output, target_mode, rank, at->idx, at->vals, at->nnz, THREAD_CF);
    } else if (at->nmode == 4) {
        mttkrp_lvl2_4d_kernel<<<blocks, TILE_SIZE, smem_sz, 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], 
            output, target_mode, rank, at->idx, at->vals, at->nnz, THREAD_CF);
    } else {
        printf("!ERROR! Only order 3 and 4 tensors are supported\n");
    }
    check_cuda(cudaGetLastError(), "mttkrp_lvl2_kernel launch. Exceeded shared mem space?");
}

// Register-, smem- and gmem-based conflict resolution using tile-based execution with thread coarsening
template <typename LIT>
__global__ void mttkrp_lvl3_3d_kernel(FType* f0, FType* f1, FType* f2, FType* output, const int tmode, 
        const IType rank, const LIT* __restrict__ lidx, const FType* __restrict__ values, const IType nnz,
		FType** ofibs, Interval* intervals, IType* prtn_ptr, IType THREAD_CF) 
{
    auto block = cg::this_thread_block();
    // Used for warp level primitives (must be static or coalesced group)
    auto tile = cg::tiled_partition<TILE_SIZE>(block);
    const int tid = block.thread_rank();

    // Setup caches (stash)
    extern __shared__ int count[]; // [block.size()] 
    IType *nnz_idx = (IType*) (count + block.size()); // [block.size() * 4] 
    IType *nnz_out = (IType*) (nnz_idx + 3 * block.size()); // [block.size()] 
    FType* nnz_val = (FType*) (nnz_out + block.size()); // [block.size()] 
    FType* data = (FType*) (nnz_val + block.size()); // [rank * STASH_SIZE]
    IType* tags = (IType*) (data + rank * STASH_SIZE); // Output row ID [STASH_SIZE]
    for (int i = tid; i < STASH_SIZE; i += block.size()) {
        tags[i] = INVALID_ID;
    }
    //block.sync();

    // Determine output partial matrix
    output = ofibs[block.group_index().x];
    IType subspace_start = intervals[block.group_index().x * 3 + tmode].start; 
    
    // Identify block-level workload
    IType curr_elem = prtn_ptr[block.group_index().x] + block.group_index().y * block.size() * THREAD_CF; // Index of start element
    IType end_elem = min(prtn_ptr[block.group_index().x + 1], curr_elem + block.size() * THREAD_CF); // Index of last element    
       
    const int mp0 = ALTO_POS[0];
    const int mp1 = ALTO_POS[1];
    const int mp2 = ALTO_POS[2];
    const LIT mm0 = ALTO_MASKS[0];
    const LIT mm1 = ALTO_MASKS[1];
    const LIT mm2 = ALTO_MASKS[2];

    // Iterate workload
    while (curr_elem < end_elem) {
        // Threads collaborate to perform On-the-fly delinerization, sorting, and segmented scan. 
        count[tid] = 0;
        
        //const LIT idx = lidx[curr_elem+tid];
        LIT idx;
        IType x, y, z, output_row;
        if (curr_elem+tid < end_elem) { 
            idx = lidx[curr_elem+tid];
            x = alt_pext(idx, mp0, mm0, bc0);
            y = alt_pext(idx, mp1, mm1, bc1);
            z = alt_pext(idx, mp2, mm2, bc2);
            if (tmode == 0) output_row = x;
            else if (tmode == 1) output_row = y;
            else output_row = z;    
        } else {
            x = y = z = output_row = (IType) -1;
        }
        
        // Find subgroups of threads with same key (output_row).
        block.sync();
        int sg_mask = tile.match_any(output_row);
        auto sg = cg::labeled_partition(tile, sg_mask);
        int sg_rank = sg.thread_rank();
        int sg_id = sg.meta_group_rank();
        if (sg_rank == 0) count[sg_id+1] = sg.size(); // OOB writes will be overwritten later.
        block.sync();
        
        // Scan for counting sort
        sg_mask = count[tid];
        //block.sync();
        #pragma unroll
        for (int j = 1; j < tile.size(); j <<= 1) {
            int temp = tile.shfl_up(sg_mask, j);
            if (tid >= j) sg_mask += temp;
        }
        count[tid] = sg_mask;
        block.sync();
 
        // Sorted rank
        sg_rank += count[sg_id];
        
        // Strided access to facilitate broadcast later
        nnz_idx[sg_rank * 3]  = x;
        nnz_idx[sg_rank * 3 + 1]  = y;
        nnz_idx[sg_rank * 3 + 2 ]  = z;
        nnz_out[sg_rank] = output_row;
        if (curr_elem+tid < end_elem) nnz_val[sg_rank] = values[curr_elem+tid];
        
        // Segmented scan structure (reuse sg_mask).
        if (sg.thread_rank() == 0) sg_mask = 1<<sg_rank;
        else sg_mask = 0;
        #pragma unroll
        for (int j = tile.size()/2; j > 0; j >>= 1) {
            sg_mask |= tile.shfl_down(sg_mask, j);
        }
        sg_mask = tile.shfl(sg_mask, 0);
        
        // Now threads perform rank-wise operations.
        int n = 0;
        while (n < block.size() &&  (curr_elem + n) < end_elem) {
            //block.sync();
            
            // Prep stash line
            const IType output_row = nnz_out[n];
            int stash_line = (int) output_row & (STASH_SIZE - 1); // Modulo hash function
            if (tags[stash_line] == INVALID_ID) {
               // Initialize cache line
               for (IType i = tid; i < rank; i += block.size()) {
                   data[stash_line * rank + i] = 0.0;
               }
               if (tid == 0) tags[stash_line] = output_row;
            }
            else if (tags[stash_line] != output_row) {
                // Evict cache line to global mem (evict-first policy)
                for (IType i = tid; i < rank; i += block.size()) {
                    atomicAdd(output + (tags[stash_line] - subspace_start) * rank + i, data[stash_line * rank + i]);
                    data[stash_line * rank + i] = 0.0;
                } 
               if (tid == 0) tags[stash_line] = output_row;
             }
            block.sync();
        
            // Perform update
            const int next_n = n;
            for (IType i = tid; i < rank; i += block.size()) {               
                // Register-based accumlation
                FType value = 0.0;
                n = next_n;
                do {
                    // Broadcast 
                    FType val = nnz_val[n];
                    x = nnz_idx[n * 3];
                    y = nnz_idx[n * 3 + 1];
                    z = nnz_idx[n * 3 + 2];
                    
                    if (tmode == 0) val *= f1[rank * y + i] * f2[rank * z + i];
                    else if (tmode == 1) val *= f0[rank * x + i] * f2[rank * z + i];
                    else val *= f0[rank * x + i] * f1[rank * y + i];                    
                    
                    value += val;
                    ++n;
                } while (n < block.size() && !(sg_mask & (1<<n)));
                data[stash_line * rank + i] += value;                   
            } // rank
            // broadcast n
            n = tile.shfl(n, 0);
        } // block.size()
        curr_elem += block.size();
    } // curr_elem < end_elem
    //block.sync();

    // Write STASH to global
    #pragma unroll
    for (int stash_line = 0; stash_line < STASH_SIZE; stash_line++) {
        IType output_row = tags[stash_line];
        if (output_row != INVALID_ID) {
            for (IType i = tid; i < rank; i += block.size()) {
                atomicAdd(output + (output_row - subspace_start) * rank + i, data[stash_line * rank + i]);
            }
        }
    }
}

template <typename LIT>
__global__ void mttkrp_lvl3_4d_kernel(FType* f0, FType* f1, FType* f2, FType* f3, FType* output, const int tmode, 
        const IType rank, const LIT* __restrict__ lidx, const FType* __restrict__ values, const IType nnz,
		FType** ofibs, Interval* intervals, IType* prtn_ptr, IType THREAD_CF) 
{
    auto block = cg::this_thread_block();
    // Used for warp level primitives (must be static or coalesced group)
    auto tile = cg::tiled_partition<TILE_SIZE>(block);
    const int tid = block.thread_rank();

    // Setup caches (stash)
    extern __shared__ int count[]; // [block.size()] 
    IType *nnz_idx = (IType*) (count + block.size()); // [block.size() * 4] 
    IType *nnz_out = (IType*) (nnz_idx + 4 * block.size()); // [block.size()] 
    FType* nnz_val = (FType*) (nnz_out + block.size()); // [block.size()] 
    FType* data = (FType*) (nnz_val + block.size()); // [rank * STASH_SIZE]
    IType* tags = (IType*) (data + rank * STASH_SIZE); // Output row ID [STASH_SIZE]
    for (int i = tid; i < STASH_SIZE; i += block.size()) {
        tags[i] = INVALID_ID;
    }
    //block.sync();

    output = ofibs[block.group_index().x];
    IType subspace_start = intervals[block.group_index().x * 4 + tmode].start; // order 4 tensor
    
    // Identify block-level workload
    IType curr_elem = prtn_ptr[block.group_index().x] + block.group_index().y * block.size() * THREAD_CF; // Index of start element
    IType end_elem = min(prtn_ptr[block.group_index().x + 1], curr_elem + block.size() * THREAD_CF); // Index of last element    
       
    const int mp0 = ALTO_POS[0];
    const int mp1 = ALTO_POS[1];
    const int mp2 = ALTO_POS[2];
    const int mp3 = ALTO_POS[3];
    const LIT mm0 = ALTO_MASKS[0];
    const LIT mm1 = ALTO_MASKS[1];
    const LIT mm2 = ALTO_MASKS[2];
    const LIT mm3 = ALTO_MASKS[3];

    // Iterate workload
    while (curr_elem < end_elem) {
        // Threads collaborate to perform On-the-fly delinerization, sorting, and segmented scan. 
        count[tid] = 0;
        
        //const LIT idx = lidx[curr_elem+tid];
        LIT idx;
        IType x, y, z, w, output_row;
        if (curr_elem+tid < end_elem) { 
            idx = lidx[curr_elem+tid];
            x = alt_pext(idx, mp0, mm0, bc0);
            y = alt_pext(idx, mp1, mm1, bc1);
            z = alt_pext(idx, mp2, mm2, bc2);
            w = alt_pext(idx, mp3, mm3, bc3);
            if (tmode == 0) output_row = x;
            else if (tmode == 1) output_row = y;
            else if (tmode == 2) output_row = z;
            else output_row = w;
        } else {
            x = y = z = w = output_row = (IType) -1;
        }
        
        // Find subgroups of threads with same key (output_row).
        block.sync();
        int sg_mask = tile.match_any(output_row);
        auto sg = cg::labeled_partition(tile, sg_mask);
        int sg_rank = sg.thread_rank();
        int sg_id = sg.meta_group_rank();
        if (sg_rank == 0) count[sg_id+1] = sg.size(); // OOB writes will be overwritten later.
        block.sync();
        
        // Scan for counting sort
        sg_mask = count[tid];
        //block.sync();
        #pragma unroll
        for (int j = 1; j < tile.size(); j <<= 1) {
            int temp = tile.shfl_up(sg_mask, j);
            if (tid >= j) sg_mask += temp;
        }
        count[tid] = sg_mask;
        block.sync();
 
        // Sorted rank
        sg_rank += count[sg_id];
        
        // Strided access to facilitate broadcast later
        nnz_idx[sg_rank * 4]  = x;
        nnz_idx[sg_rank * 4 + 1]  = y;
        nnz_idx[sg_rank * 4 + 2 ]  = z;
        nnz_idx[sg_rank * 4 + 3 ]  = w;
        nnz_out[sg_rank] = output_row;
        if (curr_elem+tid < end_elem) nnz_val[sg_rank] = values[curr_elem+tid];
        
        // Segmented scan structure (reuse sg_mask).
        if (sg.thread_rank() == 0) sg_mask = 1<<sg_rank;
        else sg_mask = 0;
        #pragma unroll
        for (int j = tile.size()/2; j > 0; j >>= 1) {
            sg_mask |= tile.shfl_down(sg_mask, j);
        }
        sg_mask = tile.shfl(sg_mask, 0);
        
        // Now threads perform rank-wise operations.
        int n = 0;
        while (n < block.size() &&  (curr_elem + n) < end_elem) {
            //block.sync();
            
            // Prep stash line
            const IType output_row = nnz_out[n];
            int stash_line = (int) output_row & (STASH_SIZE - 1); // Modulo hash function
            if (tags[stash_line] == INVALID_ID) {
               // Initialize cache line
               for (IType i = tid; i < rank; i += block.size()) {
                   data[stash_line * rank + i] = 0.0;
               }
               if (tid == 0) tags[stash_line] = output_row;
            }
            else if (tags[stash_line] != output_row) {
                // Evict cache line to global mem (evict-first policy)
                for (IType i = tid; i < rank; i += block.size()) {
                    atomicAdd(output + (tags[stash_line] - subspace_start) * rank + i, data[stash_line * rank + i]);
                    data[stash_line * rank + i] = 0.0;
                } 
               if (tid == 0) tags[stash_line] = output_row;
             }
            block.sync();
        
            // Perform update
            const int next_n = n;
            for (IType i = tid; i < rank; i += block.size()) {               
                // Register-based accumlation
                FType value = 0.0;
                n = next_n;
                do {
                    // Broadcast 
                    FType val = nnz_val[n];
                    x = nnz_idx[n * 4];
                    y = nnz_idx[n * 4 + 1];
                    z = nnz_idx[n * 4 + 2];
                    w = nnz_idx[n * 4 + 3];
                    
                    if (tmode == 0) val *= f1[rank * y + i] * f2[rank * z + i] * f3[rank * w + i];
                    else if (tmode == 1) val *= f0[rank * x + i] * f2[rank * z + i] * f3[rank * w + i];
                    else if (tmode == 2) val *= f0[rank * x + i] * f1[rank * y + i] * f3[rank * w + i];                 
                    else val *= f0[rank * x + i] * f1[rank * y + i] * f2[rank * z + i];
                    
                    value += val;
                    ++n;
                } while (n < block.size() && !(sg_mask & (1<<n)));
                data[stash_line * rank + i] += value;                   
            } // rank
            // broadcast n
            n = tile.shfl(n, 0);
        } // block.size()
        curr_elem += block.size();
    } // curr_elem < end_elem
    //block.sync();

    // Write STASH to global
    #pragma unroll
    for (int stash_line = 0; stash_line < STASH_SIZE; stash_line++) {
        IType output_row = tags[stash_line];
        if (output_row != INVALID_ID) {
            for (IType i = tid; i < rank; i += block.size()) {
                atomicAdd(output + (output_row - subspace_start) * rank + i, data[stash_line * rank + i]);
            }
        }
    }
}

// Register-, smem- and gmem-based conflict resolution using tile-based execution with thread coarsening
template <typename LIT>
void mttkrp_lvl3(AltoTensor<LIT>* at, FType** mats_staging_ptr, FType** mats_dev, int target_mode, IType rank, 
		IType mode_length, FType** ofibs_host, FType** ofibs_dev, IType THREAD_CF) 
{
    int nnz_block = TILE_SIZE * THREAD_CF;
    IType nnz_ptrn = (at->nnz + at->nprtn - 1) / at->nprtn;
    IType blocks = (nnz_ptrn + nnz_block - 1) / nnz_block;
    dim3 grid_dim(at->nprtn, blocks);
    FType* output = mats_staging_ptr[target_mode];
    int smem_sz = TILE_SIZE * (sizeof(FType) + (at->nmode+1) * sizeof(IType) + sizeof(int)) ;
    smem_sz += (rank + 1) * STASH_SIZE * sizeof(FType);

    if (at->nmode == 3) {
        mttkrp_lvl3_3d_kernel<<<grid_dim, TILE_SIZE, smem_sz, 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], 
            output, target_mode, rank, at->idx, at->vals, at->nnz,
			ofibs_dev, at->prtn_intervals, at->prtn_ptr, THREAD_CF);
    } else if (at->nmode == 4) {
        mttkrp_lvl3_4d_kernel<<<grid_dim, TILE_SIZE, smem_sz, 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], 
            output, target_mode, rank, at->idx, at->vals, at->nnz,
			ofibs_dev, at->prtn_intervals, at->prtn_ptr, THREAD_CF);
    } else {
        printf("!ERROR! Only order 3 and 4 tensors are supported\n");
    }
    check_cuda(cudaGetLastError(), "mttkrp_lvl3_kernel launch. Exceeded shared mem space?");
    
    partial_matrix_reduction<<<mode_length, TILE_SIZE, 0, 0>>> (mats_staging_ptr[target_mode], at->nmode, target_mode, at->nprtn, rank, ofibs_dev, at->prtn_intervals);
    check_cuda(cudaGetLastError(), "partial_matrix_reduction launch");
}

// Parallelization granularity: each thread block deals with one nonzero at a time
// Atomiclessly write to shared memory (think of shared mem as manually-managed "cache")
// Atomically write to global memory as needed to evict shared memory "cache"
// Finally, after done, write all valid cache lines to global mem
template <typename LIT>
__global__ void mttkrp_stash_3d_kernel(FType* f0, FType* f1, FType* f2, FType* output, int tmode, IType rank, int* mode_pos, LIT* mode_masks,
    LIT* lidx, FType* values, int nmode, IType nnz, int cache_size) 
{

    auto block = cg::this_thread_block();
    int block_rank = block.thread_rank();

    // Setup caches (stash)
    extern __shared__ FType data[]; // Length rank * cache_size + 2 * cache_size
    IType* tags = (IType*) (data + rank * cache_size); // Output row ID
    bool* valid = (bool*) (tags + cache_size); // Whether cache line is active
    for (int i = block_rank; i < cache_size; i += block.size()) {
        valid[i] = false;
    }
    block.sync();

    // Identify block-level workload (currently ceil(nnz/blocks) )
    IType num_compute = max(1 + (nnz - 1) / gridDim.x, (IType) block.size()); // Number of nonzero elements each block should process
    IType curr_elem = block.group_index().x * num_compute; // Index of start element
    IType end_elem = min(nnz, curr_elem + num_compute); // Index of last element
    
    int mp0 = mode_pos[0];
    int mp1 = mode_pos[1];
    int mp2 = mode_pos[2];
    LIT mm0 = mode_masks[0];
    LIT mm1 = mode_masks[1];
    LIT mm2 = mode_masks[2];

    // Iterate workload
    while (curr_elem < end_elem) {
        block.sync();
        LIT idx = lidx[curr_elem];
        FType value = values[curr_elem];

        IType x = alt_pext(idx, mp0, mm0);
        IType y = alt_pext(idx, mp1, mm1);
        IType z = alt_pext(idx, mp2, mm2);

        IType row_output;
        if (tmode == 0) row_output = x;
        else if (tmode == 1) row_output = y;
        else row_output = z;

        // Prep cache line
        IType cache_line = row_output & (cache_size - 1); // Modulo hash function
        if (valid[cache_line] && tags[cache_line] != row_output) {
            // Evict cache line to global mem (evict-first policy)
            for (IType i = block_rank; i < rank; i += block.size()) {
                atomicAdd(output + tags[cache_line] * rank + i, data[cache_line * rank + i]);
                data[cache_line * rank + i] = 0.0;
            }
        }
        if (!valid[cache_line]) {
            // Initiate cache line
            for (IType i = block_rank; i < rank; i += block.size()) {
                data[cache_line * rank + i] = 0.0;
            }
        }
        block.sync();
        if (block_rank == 0) tags[cache_line] = row_output;
        if (block_rank == 0) valid[cache_line] = true;

        // Perform update
        if (tmode == 0) {
            for (IType i = block_rank; i < rank; i += block.size()) {
                FType val = value;
                // Accumulate
                val *= f1[rank * y + i];
                val *= f2[rank * z + i];
                // Write
                data[cache_line * rank + i] += val;
            }
        } else if (tmode == 1) {
            for (IType i = block_rank; i < rank; i += block.size()) {
                FType val = value;
                // Accumulate
                val *= f0[rank * x + i];
                val *= f2[rank * z + i];
                // Write
                data[cache_line * rank + i] += val;
            }
        } else {
            for (IType i = block_rank; i < rank; i += block.size()) {
                FType val = value;
                // Accumulate
                val *= f0[rank * x + i];
                val *= f1[rank * y + i];
                // Write
                data[cache_line * rank + i] += val;
            }
        }

        curr_elem += 1;
    }
    block.sync();

    // Write cache to global
    for (IType cache_line = 0; cache_line < cache_size; cache_line++) {
        if (valid[cache_line]) {
            for (IType i = block_rank; i < rank; i += block.size()) {
                atomicAdd(output + tags[cache_line] * rank + i, data[cache_line * rank + i]);
            }
        }
    }
}


template <typename LIT>
__global__ void mttkrp_stash_4d_kernel(FType* f0, FType* f1, FType* f2, FType* f3, FType* output, int tmode, IType rank, int* mode_pos, LIT* mode_masks,
    LIT* lidx, FType* values, int nmode, IType nnz, IType cache_size) 
{
    auto block = cg::this_thread_block();
    int block_rank = block.thread_rank();

    // Setup caches (stash)
    extern __shared__ FType data[]; // Length rank * cache_size + 2 * cache_size
    IType* tags = (IType*) (data + rank * cache_size); // Output row ID
    bool* valid = (bool*) (tags + cache_size); // Whether cache line is active
    for (int i = block_rank; i < cache_size; i += block.size()) {
        valid[i] = false;
    }
    block.sync();

    // Identify block-level workload (currently ceil(nnz/blocks) )
    IType num_compute = max(1 + (nnz - 1) / gridDim.x, (IType) block.size()); // Number of nonzero elements each block should process
    IType curr_elem = block.group_index().x * num_compute; // Index of start element
    IType end_elem = min(nnz, curr_elem + num_compute); // Index of last element
    
    int mp0 = mode_pos[0];
    int mp1 = mode_pos[1];
    int mp2 = mode_pos[2];
    int mp3 = mode_pos[3];
    LIT mm0 = mode_masks[0];
    LIT mm1 = mode_masks[1];
    LIT mm2 = mode_masks[2];
    LIT mm3 = mode_masks[3];

    // Iterate workload
    while (curr_elem < end_elem) {
        block.sync();
        LIT idx = lidx[curr_elem];
        FType value = values[curr_elem];

        IType x = alt_pext(idx, mp0, mm0);
        IType y = alt_pext(idx, mp1, mm1);
        IType z = alt_pext(idx, mp2, mm2);
        IType w = alt_pext(idx, mp3, mm3);

        IType row_output;
        if (tmode == 0) row_output = x;
        else if (tmode == 1) row_output = y;
        else if (tmode == 2) row_output = z;
        else row_output = w;

        // Prep cache line
        IType cache_line = row_output & (cache_size - 1); // Modulo hash function
        if (valid[cache_line] && tags[cache_line] != row_output) {
            // Evict cache line to global mem (evict-first policy)
            for (IType i = block_rank; i < rank; i += block.size()) {
                atomicAdd(output + tags[cache_line] * rank + i, data[cache_line * rank + i]);
                data[cache_line * rank + i] = 0.0;
            }
        }
        if (!valid[cache_line]) {
            // Initiate cache line
            for (IType i = block_rank; i < rank; i += block.size()) {
                data[cache_line * rank + i] = 0.0;
            }
        }
        block.sync();
        if (block_rank == 0) tags[cache_line] = row_output;
        if (block_rank == 0) valid[cache_line] = true;

        // Perform update
        if (tmode == 0) {
            for (IType i = block_rank; i < rank; i += block.size()) {
                FType val = value;
                // Accumulate
                val *= f1[rank * y + i];
                val *= f2[rank * z + i];
                val *= f3[rank * w + i];
                // Write
                data[cache_line * rank + i] += val;
            }
        } else if (tmode == 1) {
            for (IType i = block_rank; i < rank; i += block.size()) {
                FType val = value;
                // Accumulate
                val *= f0[rank * x + i];
                val *= f2[rank * z + i];
                val *= f3[rank * w + i];
                // Write
                data[cache_line * rank + i] += val;
            }
        } else if (tmode == 2) {
            for (IType i = block_rank; i < rank; i += block.size()) {
                FType val = value;
                // Accumulate
                val *= f0[rank * x + i];
                val *= f1[rank * y + i];
                val *= f3[rank * w + i];
                // Write
                data[cache_line * rank + i] += val;
            }
        } else {
            for (IType i = block_rank; i < rank; i += block.size()) {
                FType val = value;
                // Accumulate
                val *= f0[rank * x + i];
                val *= f1[rank * y + i];
                val *= f2[rank * z + i];
                // Write
                data[cache_line * rank + i] += val;
            }
        }

        curr_elem += 1;
    }
    block.sync();

    // Write cache to global
    for (IType cache_line = 0; cache_line < cache_size; cache_line++) {
        if (valid[cache_line]) {
            for (IType i = block_rank; i < rank; i += block.size()) {
                atomicAdd(output + tags[cache_line] * rank + i, data[cache_line * rank + i]);
            }
        }
    }
}


// Parallelization granularity: each thread block deals with one nonzero at a time
// Atomiclessly write to shared memory (think of shared mem as manually-managed "cache")
// Atomically write to global memory as needed to evict shared memory "cache"
// Finally, after done, write all valid cache lines to global mem
template <typename LIT>
__global__ void mttkrp_stash_kernel(FType** factors, int tmode, IType rank, int* mode_pos, LIT* mode_masks,
    LIT* lidx, FType* values, int nmode, IType nnz, int cache_size) 
{
    auto block = cg::this_thread_block();
    int block_rank = block.thread_rank();

    // Setup caches (stash)
    extern __shared__ FType data[]; // Length rank * cache_size + 2 * cache_size
    IType* tags = (IType*) (data + rank * cache_size); // Output row ID
    bool* valid = (bool*) (tags + cache_size); // Whether cache line is active
    for (IType i = block_rank; i < cache_size; i += block.size()) {
        valid[i] = false;
    }
    block.sync();

    // Identify block-level workload (currently ceil(nnz/blocks) )
    IType num_compute = max(1 + (nnz - 1) / gridDim.x, (IType) block.size()); // Number of nonzero elements each block should process
    IType curr_elem = block.group_index().x * num_compute; // Index of start element
    IType end_elem = min(nnz, curr_elem + num_compute); // Index of last element

    // Iterate workload
    while (curr_elem < end_elem) {
        block.sync();
        LIT idx = lidx[curr_elem];
        FType value = values[curr_elem];

        // Prep cache line
        IType row_output = alt_pext(idx, mode_pos[tmode], mode_masks[tmode]);
        IType cache_line = row_output & (cache_size - 1); // Modulo hash function
        if (valid[cache_line] && tags[cache_line] != row_output) {
            // Evict cache line to global mem (evict-first policy)
            for (IType i = block_rank; i < rank; i += block.size()) {
                atomicAdd(factors[tmode] + tags[cache_line] * rank + i, data[cache_line * rank + i]);
                data[cache_line * rank + i] = 0.0;
            }
        }
        if (!valid[cache_line]) {
            // Initiate cache line
            for (IType i = block_rank; i < rank; i += block.size()) {
                data[cache_line * rank + i] = 0.0;
            }
        }
        block.sync();
        if (block_rank == 0) tags[cache_line] = row_output;
        if (block_rank == 0) valid[cache_line] = true;

        // Perform update
        for (IType i = block_rank; i < rank; i += block.size()) {
            FType val = value;
            // Accumulate
            for (IType n = 0; n < nmode; n++) if (n != tmode) {
                IType row = alt_pext(idx, mode_pos[n], mode_masks[n]);
                val *= factors[n][rank * row + i];
            }

            // Write
            data[cache_line * rank + i] += val;
        }
        curr_elem += 1;
    }
    block.sync();

    // Write cache to global
    for (IType cache_line = 0; cache_line < cache_size; cache_line++) {
        if (valid[cache_line]) {
            for (IType i = block_rank; i < rank; i += block.size()) {
                atomicAdd(factors[tmode] + tags[cache_line] * rank + i, data[cache_line * rank + i]);
            }
        }
    }
}

template <typename LIT>
void mttkrp_stash(AltoTensor<LIT>* at, FType** mats_staging_ptr, FType** mats_dev, int target_mode, IType rank, int block_size, int cache_size) 
{
    int blocks = at->nprtn;
    FType* output = mats_staging_ptr[target_mode];

    if (at->nmode == 3) {
        mttkrp_stash_3d_kernel<<<blocks, block_size, (rank + 2) * cache_size * sizeof(FType), 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], 
            output, target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nmode, at->nnz, cache_size);
    } else if (at->nmode == 4) {
        mttkrp_stash_4d_kernel<<<blocks, block_size, (rank + 2) * cache_size * sizeof(FType), 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], 
            output, target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nmode, at->nnz, cache_size);
    } else {
        mttkrp_stash_kernel<<<blocks, block_size, (rank + 2) * cache_size * sizeof(FType), 0>>>(mats_dev, target_mode, 
            rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nmode, at->nnz, cache_size);
    }
    check_cuda(cudaGetLastError(), "mttkrp_stash_kernel launch. Exceeded shared mem space?");
}

template <typename LIT>
__global__ void mttkrp_hierarchical_3d_kernel(FType* f0, FType* f1, FType* f2, FType** ofibs, int tmode, IType rank, int* mode_pos, LIT* mode_masks,
    LIT* lidx, FType* values, IType nnz, int cache_size, Interval* intervals, LIT* prtn_ptr) {
        
    // Determine output partial matrix
    FType* output = ofibs[blockIdx.x];
    IType subspace_start = intervals[blockIdx.x * 3 + tmode].start; // order 3 tensor
    
    // Block ID is partition
    IType curr_elem = prtn_ptr[blockIdx.x];
    IType end_elem = prtn_ptr[blockIdx.x + 1];

    // Setup caches (stash) and mutexes
    extern __shared__ FType data[]; // Length rank * cache_size + 3 * cache_size
    IType* tags = (IType*) (data + rank * cache_size); // Output row ID
    bool* valid = (bool*) (tags + cache_size); // Whether cache line is active
    int* mutexes = (int*) (valid + cache_size); // Mutexes for each cache row
    for (int i = threadIdx.x; i < cache_size; i += blockDim.x) {
        valid[i] = false;
        mutexes[i] = 0;
    }
    __syncthreads();

    // Fetch masks and offsets
    int mp0 = mode_pos[0];
    int mp1 = mode_pos[1];
    int mp2 = mode_pos[2];
    LIT mm0 = mode_masks[0];
    LIT mm1 = mode_masks[1];
    LIT mm2 = mode_masks[2];

    curr_elem += threadIdx.x;
    while (curr_elem < end_elem) {
        LIT idx = lidx[curr_elem];
        FType tensor_value = values[curr_elem];

        // Delinearize
        IType x = alt_pext(idx, mp0, mm0);
        IType y = alt_pext(idx, mp1, mm1);
        IType z = alt_pext(idx, mp2, mm2);

        // Determine output row / key
        IType out;
        if (tmode == 0) out = x;
        else if (tmode == 1) out = y;
        else out = z;

        // Determine participants / subgroup in reduction tree
        auto coalesced = cg::coalesced_threads();
        int sg_mask = coalesced.match_any(out);
        int lane = coalesced.thread_rank();
        int pos = 1 << lane;
        int sg_leader = __ffs(sg_mask) - 1;

        // Lock cache line and prep it
        IType cache_line = out & (cache_size - 1); // Modulo hash function
        if (lane == sg_leader) { // TODO figure out how to make this parallelized
            mutex_lock(mutexes + cache_line);
            if (valid[cache_line] && tags[cache_line] != out) {
                // Evict cache line to global mem (evict-first policy)
                for (IType i = 0; i < rank; i += 1) {
                    output[(tags[cache_line] - subspace_start) * rank + i] += data[cache_line * rank + i];
                    data[cache_line * rank + i] = 0.0;
                }
            }
            if (!valid[cache_line]) {
                // Initiate cache line
                for (IType i = 0; i < rank; i += 1) {
                    data[cache_line * rank + i] = 0.0;
                }
            }
            tags[cache_line] = out;
            valid[cache_line] = true;
        }
        __syncwarp(sg_mask); // Race condition w/ leader thread without this

        // Iterate rank in agnostic fashion
        for (IType i = 0; i < rank; i++) {

            // Construct output value
            FType value = tensor_value;
            if (tmode == 0) {
                value *= f1[rank * y + i];
                value *= f2[rank * z + i];
            } else if (tmode == 1) {
                value *= f0[rank * x + i];
                value *= f2[rank * z + i];
            } else { // tmode == 2
                value *= f0[rank * x + i];
                value *= f1[rank * y + i];
            }

            int sg_rank = __popc(sg_mask & (pos - 1));
            int sg_higher_lanes = sg_mask & (0xfffffffe << lane);

            // Reduction tree
            while (__any_sync(sg_mask, sg_higher_lanes)) {
                int next = __ffs(sg_higher_lanes);
                FType temp = __shfl_sync(sg_mask, value, next - 1);
                sg_higher_lanes &= __ballot_sync(sg_mask, !(sg_rank & 1)); // Clear odd ranks
                sg_rank >>= 1;
                if (next) value += temp;
            }

            // Leader write
            if (lane == sg_leader) data[cache_line * rank + i] += value;
            __syncwarp(sg_mask);
        }
        if (lane == sg_leader) mutex_unlock(mutexes + cache_line);

        curr_elem += blockDim.x;
        cg::sync(coalesced); // Fixes race condition??
    }

    __syncthreads();

    // Write cache to global
    for (IType cache_line = 0; cache_line < cache_size; cache_line++) {
        if (valid[cache_line]) {
            for (IType i = threadIdx.x; i < rank; i += blockDim.x) {
                output[(tags[cache_line] - subspace_start) * rank + i] += data[cache_line * rank + i];
            }
        }
    }
}

template <typename LIT>
__global__ void mttkrp_hierarchical_4d_kernel(FType* f0, FType* f1, FType* f2, FType* f3, FType** ofibs, int tmode, IType rank, int* mode_pos, LIT* mode_masks,
    LIT* lidx, FType* values, IType nnz, int cache_size, Interval* intervals, LIT* prtn_ptr) {

    // Determine output partial matrix
    FType* output = ofibs[blockIdx.x];
    IType subspace_start = intervals[blockIdx.x * 4 + tmode].start; // order 4 tensor

    // Block ID is partition
    IType curr_elem = prtn_ptr[blockIdx.x];
    IType end_elem = prtn_ptr[blockIdx.x + 1];

    // Setup caches (stash) and mutexes
    extern __shared__ FType data[]; // Length rank * cache_size + 3 * cache_size
    IType* tags = (IType*) (data + rank * cache_size); // Output row ID
    bool* valid = (bool*) (tags + cache_size); // Whether cache line is active
    int* mutexes = (int*) (valid + cache_size); // Mutexes for each cache row
    for (IType i = threadIdx.x; i < cache_size; i += blockDim.x) {
        valid[i] = false;
        mutexes[i] = 0;
    }
    __syncthreads();

    // Fetch masks and offsets
    int mp0 = mode_pos[0];
    int mp1 = mode_pos[1];
    int mp2 = mode_pos[2];
    int mp3 = mode_pos[3];
    LIT mm0 = mode_masks[0];
    LIT mm1 = mode_masks[1];
    LIT mm2 = mode_masks[2];
    LIT mm3 = mode_masks[3];
    
    curr_elem += threadIdx.x;
    while (curr_elem < end_elem) {
        LIT idx = lidx[curr_elem];
        FType tensor_value = values[curr_elem];

        // Delinearize
        IType x = alt_pext(idx, mp0, mm0);
        IType y = alt_pext(idx, mp1, mm1);
        IType z = alt_pext(idx, mp2, mm2);
        IType w = alt_pext(idx, mp3, mm3);

        // Determine output row / key
        IType out;
        if (tmode == 0) out = x;
        else if (tmode == 1) out = y;
        else if (tmode == 2) out = z;
        else out = w;

        // Determine participants / subgroup in reduction tree
        auto coalesced = cg::coalesced_threads();
        int sg_mask = coalesced.match_any(out);
        int lane = coalesced.thread_rank();
        int pos = 1 << lane;
        int sg_leader = __ffs(sg_mask) - 1;

        // Lock cache line and prep it
        IType cache_line = out & (cache_size - 1); // Modulo hash function
        if (lane == sg_leader) {// TODO figure out how to make this parallelized
            mutex_lock(mutexes + cache_line);
            if (valid[cache_line] && tags[cache_line] != out) {
                // Evict cache line to global mem (evict-first policy)
                for (IType i = 0; i < rank; i += 1) { 
                    output[(tags[cache_line] - subspace_start) * rank + i] += data[cache_line * rank + i];
                    data[cache_line * rank + i] = 0.0;
                }
            }
            if (!valid[cache_line]) {
                // Initiate cache line
                for (IType i = 0; i < rank; i += 1) {
                    data[cache_line * rank + i] = 0.0;
                }
            }
            tags[cache_line] = out;
            valid[cache_line] = true;
        }
        __syncwarp(sg_mask); // Race condition w/ leader thread without this

        // Iterate rank in agnostic fashion
        for (IType i = 0; i < rank; i++) {

            // Construct output value
            FType value = tensor_value;
            if (tmode == 0) {
                value *= f1[rank * y + i];
                value *= f2[rank * z + i];
                value *= f3[rank * w + i];
            } else if (tmode == 1) {
                value *= f0[rank * x + i];
                value *= f2[rank * z + i];
                value *= f3[rank * w + i];
            } else if (tmode == 2) {
                value *= f0[rank * x + i];
                value *= f1[rank * y + i];
                value *= f3[rank * w + i];
            } else { // tmode == 3
                value *= f0[rank * x + i];
                value *= f1[rank * y + i];
                value *= f2[rank * z + i];
            }

            int sg_rank = __popc(sg_mask & (pos - 1));
            int sg_higher_lanes = sg_mask & (0xfffffffe << lane);

            // Reduction tree
            while (__any_sync(sg_mask, sg_higher_lanes)) {
                int next = __ffs(sg_higher_lanes);
                FType temp = __shfl_sync(sg_mask, value, next - 1);
                sg_higher_lanes &= __ballot_sync(sg_mask, !(sg_rank & 1)); // Clear odd ranks
                sg_rank >>= 1;
                if (next) value += temp;
            }

            // Leader write
            if (lane == sg_leader) data[cache_line * rank + i] += value;
            __syncwarp(sg_mask);
        }
        if (lane == sg_leader) mutex_unlock(mutexes + cache_line);

        curr_elem += blockDim.x;
        cg::sync(coalesced); // Fixes race condition??
    }

    __syncthreads();

    // Write cache to global
    for (IType cache_line = 0; cache_line < cache_size; cache_line++) {
        if (valid[cache_line]) {
            for (IType i = threadIdx.x; i < rank; i += blockDim.x) {
                output[(tags[cache_line] - subspace_start) * rank + i] += data[cache_line * rank + i];
            }
        }
    }
}


// Thread block per output matrix. Pull based reduction
__global__ void partial_matrix_reduction(FType* output, int nmodes, int target_mode, int nprtn, IType rank, FType** partials, Interval* intervals) {
    auto block = cg::this_thread_block();
    int block_rank = block.thread_rank();
    IType row = block.group_index().x;

    if (threadIdx.x < rank) {
        // Iterate partial matrices
        for (IType i = 0; i < nprtn; i++) {
            if (intervals[i * nmodes + target_mode].start <= row && intervals[i * nmodes + target_mode].stop >= row) {
                // This partial matrix contains our row
                IType j = row - intervals[i * nmodes + target_mode].start;
                for (IType r = block_rank; r < rank; r += block.size()) {
                    output[row * rank + r] += partials[i][j * rank + r];
                }   
            }
        }
    }
}

template <typename LIT>
void mttkrp_hierarchical(AltoTensor<LIT>* at, FType** mats_staging_ptr, FType** mats_dev, int target_mode, IType mode_length, IType rank, FType** ofibs_host, FType** ofibs_dev, int block_size, int cache_size) {
    IType blocks = at->nprtn;

    if (at->nmode == 3) {
        mttkrp_hierarchical_3d_kernel<<<blocks, block_size, (rank + 3) * cache_size * sizeof(FType), 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], 
            ofibs_dev, target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nnz, cache_size, at->prtn_intervals, at->prtn_ptr);
    } else if (at->nmode == 4) {
        mttkrp_hierarchical_4d_kernel<<<blocks, block_size, (rank + 3) * cache_size * sizeof(FType), 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], 
            ofibs_dev, target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nnz, cache_size, at->prtn_intervals, at->prtn_ptr);
    } else {
        printf("!ERROR! Only order 3 and 4 tensors are supported\n");
        exit(1);
    }
    check_cuda(cudaGetLastError(), "mttkrp_hierarchical_kernel launch");

    partial_matrix_reduction<<<mode_length, block_size, 0, 0>>> (mats_staging_ptr[target_mode], at->nmode, target_mode, at->nprtn, rank, ofibs_dev, at->prtn_intervals);
    check_cuda(cudaGetLastError(), "partial_matrix_reduction launch");
}

// mode_masks are bit masks, mode_pos is shift amount
template <typename LIT>
__global__ void mttkrp_partials_3d_kernel(FType* f0, FType* f1, FType* f2, FType** ofibs, int tmode, IType rank, int* mode_pos, LIT* mode_masks, 
    LIT* lidx, FType* values, IType nnz, Interval* intervals, LIT* prtn_ptr) 
{
    auto block = cg::this_thread_block();
    int thread_id = block.thread_rank();
    
    // Determine output partial matrix
    FType* A = ofibs[blockIdx.x];
    IType subspace_start = intervals[blockIdx.x * 3 + tmode].start; // order 3 tensor

    // Block ID is partition
    IType curr_elem = prtn_ptr[blockIdx.x];
    IType end_elem = prtn_ptr[blockIdx.x + 1];

    FType value;
    IType x, y, z;
    LIT idx;
    FType* location;

    int mp0 = mode_pos[0];
    int mp1 = mode_pos[1];
    int mp2 = mode_pos[2];
    LIT mm0 = mode_masks[0];
    LIT mm1 = mode_masks[1];
    LIT mm2 = mode_masks[2];

    if (thread_id < rank) {
        while (curr_elem < end_elem) {
            idx = lidx[curr_elem];
            value = values[curr_elem];
    
            x = alt_pext(idx, mp0, mm0);
            y = alt_pext(idx, mp1, mm1);
            z = alt_pext(idx, mp2, mm2);
    
            if (tmode == 0) {
                value *= f1[rank * y + thread_id];
                value *= f2[rank * z + thread_id];
                location = A + rank * (x - subspace_start) + thread_id;
            } else if (tmode == 1) {
                value *= f0[rank * x + thread_id];
                value *= f2[rank * z + thread_id];
                location = A + rank * (y - subspace_start) + thread_id;
            } else { // tmode == 2
                value *= f0[rank * x + thread_id];
                value *= f1[rank * y + thread_id];
                location = A + rank * (z - subspace_start) + thread_id;
            }

            *location += value;
            curr_elem += 1;
        }
    }
}

template <typename LIT>
__global__ void mttkrp_partials_4d_kernel(FType* f0, FType* f1, FType* f2, FType* f3, FType** ofibs, int tmode, IType rank, int* mode_pos, LIT* mode_masks, 
    LIT* lidx, FType* values, IType nnz, Interval* intervals, LIT* prtn_ptr) 
{
    auto block = cg::this_thread_block();
    int thread_id = block.thread_rank();
    
    // Determine output partial matrix
    FType* A = ofibs[blockIdx.x];
    IType subspace_start = intervals[blockIdx.x * 4 + tmode].start; // order 4 tensor

    // Block ID is partition
    IType curr_elem = prtn_ptr[blockIdx.x];
    IType end_elem = prtn_ptr[blockIdx.x + 1];

    FType value;
    IType x, y, z, w;
    LIT idx;
    FType* location;

    int mp0 = mode_pos[0];
    int mp1 = mode_pos[1];
    int mp2 = mode_pos[2];
    int mp3 = mode_pos[3];
    LIT mm0 = mode_masks[0];
    LIT mm1 = mode_masks[1];
    LIT mm2 = mode_masks[2];
    LIT mm3 = mode_masks[3];

    if (thread_id < rank) {
        while (curr_elem < end_elem) {
            idx = lidx[curr_elem];
            value = values[curr_elem];
    
            x = alt_pext(idx, mp0, mm0);
            y = alt_pext(idx, mp1, mm1);
            z = alt_pext(idx, mp2, mm2);
            w = alt_pext(idx, mp3, mm3);

            if (tmode == 0) {
                value *= f1[rank * y + thread_id];
                value *= f2[rank * z + thread_id];
                value *= f3[rank * w + thread_id];
                location = A + rank * (x - subspace_start) + thread_id;
            } else if (tmode == 1) {
                value *= f0[rank * x + thread_id];
                value *= f2[rank * z + thread_id];
                value *= f3[rank * w + thread_id];
                location = A + rank * (y - subspace_start) + thread_id;
            } else if (tmode == 2) {
                value *= f0[rank * x + thread_id];
                value *= f1[rank * y + thread_id];
                value *= f3[rank * w + thread_id];
                location = A + rank * (z - subspace_start) + thread_id;
            } else { // tmode == 3
                value *= f0[rank * x + thread_id];
                value *= f1[rank * y + thread_id];
                value *= f2[rank * z + thread_id];
                location = A + rank * (w - subspace_start) + thread_id;
            }

            *location += value;
            curr_elem += 1;
        }
    }
}

template <typename LIT>
void mttkrp_partials(AltoTensor<LIT>* at, FType** mats_staging_ptr, FType** mats_dev, int target_mode, IType mode_length, IType rank, FType** ofibs_host, FType** ofibs_dev, int block_size) 
{
    IType blocks = at->nprtn;

    if (at->nmode == 3) {
        mttkrp_partials_3d_kernel<<<blocks, block_size, 0, 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], 
            ofibs_dev, target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nnz, at->prtn_intervals, at->prtn_ptr);
    } else if (at->nmode == 4) {
        mttkrp_partials_4d_kernel<<<blocks, block_size, 0, 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], 
            ofibs_dev, target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nnz, at->prtn_intervals, at->prtn_ptr);
    } else {
        printf("!ERROR! Only order 3 and 4 tensors are supported\n");
        exit(1);
    }

    check_cuda(cudaGetLastError(), "mttkrp_partials_kernel launch");

    partial_matrix_reduction<<<mode_length, block_size, 0, 0>>> (mats_staging_ptr[target_mode], at->nmode, target_mode, at->nprtn, rank, ofibs_dev, at->prtn_intervals);
    check_cuda(cudaGetLastError(), "partial_matrix_reduction launch");
}


// mode_masks are bit masks, mode_pos is shift amount
template <typename LIT>
__global__ void mttkrp_partialswarp_3d_kernel(FType* f0, FType* f1, FType* f2, FType** ofibs, int tmode, IType rank, int* mode_pos, LIT* mode_masks, 
    LIT* lidx, FType* values, IType nnz, Interval* intervals, LIT* prtn_ptr) 
{
    auto block = cg::this_thread_block();
    IType thread_id = block.thread_rank() & 31;
    
    // Determine output partial matrix
    FType* A = ofibs[blockIdx.x];
    IType subspace_start = intervals[blockIdx.x * 3 + tmode].start; // order 3 tensor

    // Block ID is partition
    IType curr_elem = prtn_ptr[blockIdx.x];
    IType end_elem = prtn_ptr[blockIdx.x + 1];

    FType value;
    IType x, y, z;
    LIT idx;
    FType* location;

    IType warp_id = block.thread_rank() >> 5;
    IType num_warps = block.size() >> 5;
    curr_elem += warp_id;

    int mp0 = mode_pos[0];
    int mp1 = mode_pos[1];
    int mp2 = mode_pos[2];
    LIT mm0 = mode_masks[0];
    LIT mm1 = mode_masks[1];
    LIT mm2 = mode_masks[2];

    while (curr_elem < end_elem) {
        __syncwarp();
        idx = lidx[curr_elem];

        x = alt_pext(idx, mp0, mm0);
        y = alt_pext(idx, mp1, mm1);
        z = alt_pext(idx, mp2, mm2);

        for (IType i = thread_id; i < rank; i += 32) {
            value = values[curr_elem];
            if (tmode == 0) {
                value *= f1[rank * y + i];
                value *= f2[rank * z + i];
                location = A + rank * (x - subspace_start) + i;
            } else if (tmode == 1) {
                value *= f0[rank * x + i];
                value *= f2[rank * z + i];
                location = A + rank * (y - subspace_start) + i;
            } else { // tmode == 2
                value *= f0[rank * x + thread_id];
                value *= f1[rank * y + thread_id];
                location = A + rank * (z - subspace_start) + i;
            }

            atomicAdd(location, value);
        }

        curr_elem += num_warps;
    }
}

// mode_masks are bit masks, mode_pos is shift amount
template <typename LIT>
__global__ void mttkrp_partialswarp_4d_kernel(FType* f0, FType* f1, FType* f2, FType* f3, FType** ofibs, int tmode, IType rank, int* mode_pos, LIT* mode_masks, 
    LIT* lidx, FType* values, IType nnz, Interval* intervals, LIT* prtn_ptr)
{
    auto block = cg::this_thread_block();
    int thread_id = block.thread_rank() & 31;
    
    // Determine output partial matrix
    FType* A = ofibs[blockIdx.x];
    IType subspace_start = intervals[blockIdx.x * 4 + tmode].start; // order 4 tensor

    // Block ID is partition
    IType curr_elem = prtn_ptr[blockIdx.x];
    IType end_elem = prtn_ptr[blockIdx.x + 1];

    FType value;
    IType x, y, z, w;
    LIT idx;
    FType* location;

    IType warp_id = block.thread_rank() >> 5;
    IType num_warps = block.size() >> 5;
    curr_elem += warp_id;

    int mp0 = mode_pos[0];
    int mp1 = mode_pos[1];
    int mp2 = mode_pos[2];
    int mp3 = mode_pos[3];
    LIT mm0 = mode_masks[0];
    LIT mm1 = mode_masks[1];
    LIT mm2 = mode_masks[2];
    LIT mm3 = mode_masks[3];

    while (curr_elem < end_elem) {
        __syncwarp();
        idx = lidx[curr_elem];

        x = alt_pext(idx, mp0, mm0);
        y = alt_pext(idx, mp1, mm1);
        z = alt_pext(idx, mp2, mm2);
        w = alt_pext(idx, mp3, mm3);

        for (IType i = thread_id; i < rank; i += 32) {
            value = values[curr_elem];
            if (tmode == 0) {
                value *= f1[rank * y + i];
                value *= f2[rank * z + i];
                value *= f3[rank * w + i];
                location = A + rank * (x - subspace_start) + i;
            } else if (tmode == 1) {
                value *= f0[rank * x + i];
                value *= f2[rank * z + i];
                value *= f3[rank * w + i];
                location = A + rank * (y - subspace_start) + i;
            } else if (tmode == 2) {
                value *= f0[rank * x + i];
                value *= f1[rank * y + i];
                value *= f3[rank * w + i];
                location = A + rank * (z - subspace_start) + i;
            } else {
                value *= f0[rank * x + i];
                value *= f1[rank * y + i];
                value *= f2[rank * z + i];
                location = A + rank * (w - subspace_start) + i;
            }

            atomicAdd(location, value);
        }

        curr_elem += num_warps;
    }
}

template <typename LIT>
void mttkrp_partialswarp(AltoTensor<LIT>* at, FType** mats_staging_ptr, FType** mats_dev, int target_mode, IType mode_length, IType rank, FType** ofibs_host, FType** ofibs_dev, int block_size) 
{
    IType blocks = at->nprtn;

    if (at->nmode == 3) {
        mttkrp_partialswarp_3d_kernel<<<blocks, block_size, 0, 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], 
            ofibs_dev, target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nnz, at->prtn_intervals, at->prtn_ptr);
    } else if (at->nmode == 4) {
        mttkrp_partialswarp_4d_kernel<<<blocks, block_size, 0, 0>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], 
            ofibs_dev, target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nnz, at->prtn_intervals, at->prtn_ptr);
    } else {
        printf("!ERROR! Only order 3 and 4 tensors are supported\n");
        exit(1);
    }

    check_cuda(cudaGetLastError(), "mttkrp_partialswap_kernel launch");

    partial_matrix_reduction<<<mode_length, block_size, 0, 0>>> (mats_staging_ptr[target_mode], at->nmode, target_mode, at->nprtn, rank, ofibs_dev, at->prtn_intervals);
    check_cuda(cudaGetLastError(), "partial_matrix_reduction launch");
}


template<typename LIT> void zero_partials(int target_mode, IType rank, AltoTensor<LIT>* at, FType** ofibs) {
    for (IType p = 0; p < at->nprtn; p++) {
        IType interval_length = at->prtn_intervals[p * at->nmode + target_mode].stop - 
                at->prtn_intervals[p * at->nmode + target_mode].start + 1;
        if (interval_length > 0) check_cuda(cudaMemset(ofibs[p], 0, rank * interval_length), "cudaMemset partial matrix");
    }
}

// Make GPU partial matrices
template <typename LIT> FType** create_da_mem_dev(int target_mode, IType rank, AltoTensor<LIT>* at) {
    int const nprtn = at->nprtn;
	int const nmode = at->nmode;

    FType** ofibs = new FType*[nprtn];
    double total_storage = 0.0;

    //#pragma omp parallel for reduction(+: total_storage)
    for (IType p = 0; p < nprtn; p++) {
        IType alloc = 0;
        if (target_mode == -1) {
			//allocate enough da_mem for all modes
			IType max_num_fibs = 0;
			for (int n = 0; n < nmode; ++n) {
				Interval const intvl = at->prtn_intervals[p * nmode + n];
				IType const num_fibs = intvl.stop - intvl.start + 1;
				max_num_fibs = std::max(max_num_fibs, num_fibs);
			}
            alloc = max_num_fibs;
        } else {
            //TODO: for extremely short modes, it would be better to allocate a full interval (i.e, [0, dims[tmode]])
			Interval const intvl = at->prtn_intervals[p * nmode + target_mode];
			IType const num_fibs = intvl.stop - intvl.start + 1;
            alloc = num_fibs;
        }

        alloc *= rank * sizeof(FType);
        check_cuda(cudaMalloc(&ofibs[p], alloc), "cudaMalloc partial matrix");
        total_storage += ((double) alloc) / (1024.0*1024.0);
    }

    printf("Total storage ofibs: %f MB\n", total_storage);
    return ofibs;
}


// mode_masks are bit masks, mode_pos is shift amount
template <typename LIT>
__global__ void mttkrp_atomic_3d_kernel(FType* A, FType* f0, FType* f1, FType* f2, IType tmode, IType rank, int* mode_pos, LIT* mode_masks, LIT* lidx, FType* values, IType nnz, IType thread_count) {
    // Identify curr element and rank to work with
    IType thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    IType curr_elem = thread_id / rank; // Nonzero index
    thread_id = thread_id % rank; // Column index of result matrix

    FType value;
    IType x, y, z;
    LIT idx;
    FType* location;

    int mp0 = mode_pos[0];
    int mp1 = mode_pos[1];
    int mp2 = mode_pos[2];
    LIT mm0 = mode_masks[0];
    LIT mm1 = mode_masks[1];
    LIT mm2 = mode_masks[2];

    if (curr_elem < thread_count) {
        while (curr_elem < nnz) {
            __syncwarp();
            idx = lidx[curr_elem];
            value = values[curr_elem];

            x = alt_pext(idx, mp0, mm0);
            y = alt_pext(idx, mp1, mm1);
            z = alt_pext(idx, mp2, mm2);

            if (tmode == 0) {
                value *= f1[rank * y + thread_id];
                value *= f2[rank * z + thread_id];
                location = A + rank * x + thread_id;
            } else if (tmode == 1) {
                value *= f0[rank * x + thread_id];
                value *= f2[rank * z + thread_id];
                location = A + rank * y + thread_id;
            } else { // tmode == 2
                value *= f0[rank * x + thread_id];
                value *= f1[rank * y + thread_id];
                location = A + rank * z + thread_id;
            }

            atomicAdd(location, value);
            curr_elem += thread_count;
        }
    }       
}
template <typename LIT>
__global__ void mttkrp_atomic_4d_kernel(FType* A, FType* f0, FType* f1, FType* f2, FType* f3, IType tmode, IType rank, int* mode_pos, LIT* mode_masks, LIT* lidx, FType* values, IType nnz, IType thread_count) {
    // Identify curr element and rank to work with
    IType thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    IType curr_elem = thread_id / rank; // Nonzero index
    thread_id = thread_id % rank; // Column index of result matrix

    int mp0 = mode_pos[0];
    int mp1 = mode_pos[1];
    int mp2 = mode_pos[2];
    int mp3 = mode_pos[3];
    LIT mm0 = mode_masks[0];
    LIT mm1 = mode_masks[1];
    LIT mm2 = mode_masks[2];
    LIT mm3 = mode_masks[3];

    FType value;
    IType x, y, z, w;
    LIT idx;
    FType* location;

    if (curr_elem < thread_count) {
        while (curr_elem < nnz) {
            __syncwarp();
            idx = lidx[curr_elem];
            value = values[curr_elem];

            x = alt_pext(idx, mp0, mm0);
            y = alt_pext(idx, mp1, mm1);
            z = alt_pext(idx, mp2, mm2);
            w = alt_pext(idx, mp3, mm3);

            if (tmode == 0) {
                value *= f1[rank * y + thread_id];
                value *= f2[rank * z + thread_id];
                value *= f3[rank * w + thread_id];
                location = A + rank * x + thread_id;
            } else if (tmode == 1) {
                value *= f0[rank * x + thread_id];
                value *= f2[rank * z + thread_id];
                value *= f3[rank * w + thread_id];
                location = A + rank * y + thread_id;
            } else if (tmode == 2) {
                value *= f0[rank * x + thread_id];
                value *= f1[rank * y + thread_id];
                value *= f3[rank * w + thread_id];
                location = A + rank * z + thread_id;
            } else { // tmode == 3
                value *= f0[rank * x + thread_id];
                value *= f1[rank * y + thread_id];
                value *= f2[rank * z + thread_id];
                location = A + rank * w + thread_id;
            }

            atomicAdd(location, value);
            //*location += value;
            curr_elem += thread_count;
        }
    }       
}

template <typename LIT>
__global__ void mttkrp_atomic_kernel(FType* A, FType** factors, int tmode, IType rank, int* mode_pos, LIT* mode_masks, LIT* lidx, FType* values, int nmode, IType nnz, IType thread_count) {
    // Identify curr element and rank to work with
    IType thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    IType curr_elem = thread_id / rank; // Nonzero index
    thread_id = thread_id % rank; // Column index of result matrix

    if (curr_elem < thread_count) {
        FType value;
        IType row;
        LIT idx;
        IType mode_idx;

        while (curr_elem < nnz) {
            __syncwarp();
            value = values[curr_elem];
            row = 0, idx = lidx[curr_elem], mode_idx = 0;
            
            for (int m = 0; m < tmode; m++) {
                row = alt_pext(idx, mode_pos[m], mode_masks[m]);
                value *= factors[m][rank * row + thread_id];
            }
            mode_idx = alt_pext(idx, mode_pos[tmode], mode_masks[tmode]);
            for (int m = tmode + 1; m < nmode; m++) {
                row = alt_pext(idx, mode_pos[m], mode_masks[m]);
                value *= factors[m][rank * row + thread_id];
            }

            atomicAdd(A + rank * mode_idx + thread_id, value);
            curr_elem += thread_count;
        }
    }
}

template <typename LIT>
void mttkrp_atomic(AltoTensor<LIT>* at, FType** mats_staging_ptr, FType** mats_dev, int target_mode, IType rank, int block_size, int element_stride) {
    int blocks = element_stride * rank / block_size + 1;
    
    if (at->nmode == 3) {
        mttkrp_atomic_3d_kernel <<<blocks, block_size, 0, 0>>> (
            mats_staging_ptr[target_mode], mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nnz, element_stride);
    } else if (at->nmode == 4) {
        mttkrp_atomic_4d_kernel <<<blocks, block_size, 0, 0>>> (
            mats_staging_ptr[target_mode], mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nnz, element_stride);
    } else {
        mttkrp_atomic_kernel <<<blocks, block_size, 0, 0>>> (
            mats_staging_ptr[target_mode], mats_dev, target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nmode, at->nnz, element_stride);
    }
    
    check_cuda(cudaGetLastError(), "mttkrp_atomic_kernel launch");
}


// The entrypoint for device execution
template <typename LIT> void mttkrp_alto_dev(AltoTensor<LIT>* at_host, KruskalModel* M, int kernel, int iters, int target_mode, int block_size, IType thread_cf) 
{
    #ifndef ALT_PEXT
        printf("!ERROR! ALTERNATIVE_PEXT must be set to run GPU code\n");
        exit(1);
    #endif

    printf("--> Begin copying to GPU device\n");

    printf("Copy base ALTO struct\n");

    // Make ALTO struct on device
	AltoTensor<LIT>* AT = (AltoTensor<LIT>*) AlignedMalloc(sizeof(AltoTensor<LIT>));
	assert(AT);
    AT->nmode = at_host->nmode;
    AT->nprtn = at_host->nprtn;
    AT->nnz = at_host->nnz;
    AT->dims = make_device_copy(at_host->dims, AT->nmode, "Mode lengths");
    AT->alto_mask = at_host -> alto_mask;
    AT->mode_masks = make_device_copy(at_host->mode_masks, AT->nmode, "Mode masks");
    AT->mode_pos = make_device_copy(at_host->mode_pos, AT->nmode, "Mode positions");
    AT->alto_cr_mask = at_host->alto_cr_mask;
    AT->cr_masks = make_device_copy(at_host->cr_masks, AT->nmode, "CR masks");

    cudaMemcpyToSymbol(ALTO_MASKS, at_host->mode_masks, AT->nmode * sizeof(LIT));
    cudaMemcpyToSymbol(ALTO_POS, at_host->mode_pos, AT->nmode * sizeof(int));
    
#ifdef OPT_ALTO
    AT->prtn_id = make_device_copy(at_host->prtn_id, AT->nprtn * sizeof(LPType), "prtn_id");
    AT->prtn_mask = make_device_copy(at_host->prtn_mask, AT->nprtn * sizeof(LPType), "prtn_mask");
    AT->prtn_mode_masks = make_device_copy(at_host->prtn_mode_masks, AT->nprtn * AT->nmode * sizeof(LPType), "prtn_mode_masks");
#endif

    printf("Copy over Interval data\n");
    // Copy interval data over
    AT->prtn_ptr = make_device_copy(at_host->prtn_ptr, AT->nprtn + 1, "Partition load balancing array");
    check_cuda(cudaMalloc(&AT->prtn_intervals, AT->nprtn * AT->nmode * sizeof(Interval)), "cudaMalloc intervals");
    check_cuda(cudaMemcpy(AT->prtn_intervals, at_host->prtn_intervals, AT->nprtn * AT->nmode * sizeof(Interval), cudaMemcpyHostToDevice), "cudaMemcpy intervals");

    printf("Allocate partial matrices\n");
    // Generate partial matrices if kernel uses it
    FType** partial_matrices = NULL;
    FType** partial_matrices_dev = NULL;
    if (kernel == 3 || kernel == 5 || kernel == 7 || kernel == 8) {
        partial_matrices = create_da_mem_dev(target_mode, M->rank, at_host);
        partial_matrices_dev = make_device_copy(partial_matrices, at_host->nprtn, "partial matrices pointer");
    }

    printf("Copy over factor matrices\n");
    // Factor matrices
    FType** mats_staging_ptr = new FType*[AT->nmode];
    for (IType i = 0; i < AT->nmode; i++) {
        mats_staging_ptr[i] = make_device_copy(M->U[i], at_host->dims[i] * M->rank, "mats");
    }
    FType** mats_dev = make_device_copy(mats_staging_ptr, AT->nmode, "mats_dev");

    printf("Copy over coordinates and values\n");

    // Copy coordinate and value data over. If it crashes here then out of GPU memory (TODO: stream in if this is the case)
    AT->idx = make_device_copy(at_host->idx, AT->nnz, "Linearized coordinate indices");
    AT->vals = make_device_copy(at_host->vals, AT->nnz, "Nonzero element values");

    printf("--> Start MTTKRP mode %d (%d iters)\n", target_mode, iters);

    // Timers
    float time = 0, ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int cache_size = 4; // TODO: TUNE

    // Do MTTKRP
    for (IType i = 0; i < iters; i++) {
        // Zero out output matrix and ofibs
        //check_cuda(cudaMemset(mats_staging_ptr[target_mode], 0, sizeof(FType) * at_host->dims[target_mode] * M->rank), "cudaMemset factor matrix");
        if (kernel == 3 || kernel == 5 || kernel == 7 || kernel == 8) zero_partials(target_mode, M->rank, at_host, partial_matrices);
        check_cuda(cudaDeviceSynchronize(), "Zero factor matrix");

        cudaEventRecord(start);

        // Call relevant kernel
        if (kernel == 1) {
           mttkrp_lvl1(AT, mats_staging_ptr, mats_dev, target_mode, M->rank); 
        } else if (kernel == 2) {
           mttkrp_lvl2(AT, mats_staging_ptr, mats_dev, target_mode, M->rank, thread_cf); 
        } else if (kernel == 3) {
            mttkrp_lvl3(AT, mats_staging_ptr, mats_dev, target_mode, M->rank, at_host->dims[target_mode], partial_matrices, partial_matrices_dev, thread_cf); 
        } else if (kernel == 4){        	
            mttkrp_stash(AT, mats_staging_ptr, mats_dev, target_mode, M->rank, block_size, cache_size);
        } else if (kernel == 5) {
            mttkrp_hierarchical(AT, mats_staging_ptr, mats_dev, target_mode, at_host->dims[target_mode], M->rank, partial_matrices, partial_matrices_dev, block_size, cache_size);
        } else if (kernel == 7) {
            mttkrp_partials(AT, mats_staging_ptr, mats_dev, target_mode, at_host->dims[target_mode], M->rank, partial_matrices, partial_matrices_dev, block_size);
        } else if (kernel == 8) {
            mttkrp_partialswarp(AT, mats_staging_ptr, mats_dev, target_mode, at_host->dims[target_mode], M->rank, partial_matrices, partial_matrices_dev, block_size);
        } else if (kernel == 13) {
            mttkrp_atomic(AT, mats_staging_ptr, mats_dev, target_mode, M->rank, block_size, AT->nprtn);
        } else{
            printf("!ERROR! Unknown kernel specified");
            break;
        }

        cudaEventRecord(stop);
        check_cuda(cudaDeviceSynchronize(), "kernel execution");
        cudaEventElapsedTime(&ms, start, stop);

        /*
        if (kernel == 5 || kernel == 7) {
            cudaEventRecord(start);
            partial_matrix_reduction<<<at_host->dims[target_mode], block_size, 0, 0>>> (mats_staging_ptr[target_mode], AT->nmode, target_mode, AT->nprtn, M->rank, partial_matrices_dev, AT->prtn_intervals);
            cudaEventRecord(stop);
            check_cuda(cudaGetLastError(), "partial_matrix_reduction launch");
            check_cuda(cudaDeviceSynchronize(), "kernel execution");
            cudaEventElapsedTime(&ms, start, stop);
            printf("Exec time reduc: %0.3f ms\n", ms);
        }
        */
        
        time += ms;
    }
    printf("Total time for GPU: %0.3f (time per iter: %0.3f ms)\n", time, time / iters);

    // Copy back
    cudaMemcpy(M->U[target_mode], mats_staging_ptr[target_mode], sizeof(FType) * at_host->dims[target_mode] * M->rank, cudaMemcpyDeviceToHost);
    //PrintKruskalModel(M);
    check_cuda(cudaDeviceSynchronize(), "copy data");
    
    // Delete partial matrices
    if (partial_matrices) {
        for (IType i = 0; i < AT->nmode; i++) cudaFree(partial_matrices[i]);
        cudaFree(partial_matrices_dev);
        delete [] partial_matrices;
    }

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (IType i = 0; i < AT->nmode; i++) cudaFree(mats_staging_ptr[i]);
    cudaFree(mats_dev);
    delete [] mats_staging_ptr;
    cudaFree(AT->dims);
    cudaFree(AT->mode_masks);
    cudaFree(AT->mode_pos);
    cudaFree(AT->cr_masks);
    cudaFree(AT->prtn_ptr);
    cudaFree(AT->prtn_intervals);
    cudaFree(AT->idx);
    cudaFree(AT->vals);
}


// TODO fix this template stuff
#if ALTO_MASK_LENGTH == 64
    typedef unsigned long long LIType;
#elif ALTO_MASK_LENGTH == 128
    typedef unsigned __int128 LIType;
#else
    #pragma message("!WARNING! ALTO_MASK_LENGTH invalid. Using default 64-bit.")
    typedef unsigned long long LIType;
#endif
template void mttkrp_alto_dev(AltoTensor<LIType>* at_host, KruskalModel* M, int kernel, int iters, int target_mode, int block_size, IType thread_cf);
#endif