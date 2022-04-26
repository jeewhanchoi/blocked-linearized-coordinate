#include "alto_dev.hpp"
#include "utils.hpp"
#include <cassert>
#include <cooperative_groups.h>
#include "blco.hpp"
#include <chrono>

namespace cg = cooperative_groups;

__constant__ IType ALTO_MASKS[MAX_NUM_MODES];
__constant__  int ALTO_POS[MAX_NUM_MODES];    

template <typename LIT>
__device__ inline IType alt_pext(LIT x, int pos, IType mask, IType block_coord) {
    return ((x >> pos) & mask) | block_coord;
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
        const IType rank, const LIT* __restrict__ lidx, const FType* __restrict__ values, const IType nnz, IType* block_coords) 
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
    const IType mm0 = ALTO_MASKS[0];
    const IType mm1 = ALTO_MASKS[1];
    const IType mm2 = ALTO_MASKS[2];
    const IType bc0 = block_coords[0];
    const IType bc1 = block_coords[1];
    const IType bc2 = block_coords[2];

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
    const IType bc0 = block_coords[0];
    const IType bc1 = block_coords[1];
    const IType bc2 = block_coords[2];
    const IType bc3 = block_coords[3];

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
void mttkrp_lvl1(blco_block* at, FType** mats_staging_ptr, FType* output, FType** mats_dev, int target_mode, IType rank, cudaStream_t stream) 
{
    int nnz_block = TILE_SIZE;
    IType blocks = (at->nnz + nnz_block -1) / nnz_block;
    int smem_sz = nnz_block * (sizeof(FType) + (at->nmode+1) * sizeof(IType) + sizeof(int)) ;

    if (at->nmode == 3) {
        mttkrp_lvl1_3d_kernel<<<blocks, nnz_block, smem_sz, stream>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], 
            output, target_mode, rank, at->idx, at->vals, at->nnz, at->block_coords);
    } else if (at->nmode == 4) {
        mttkrp_lvl1_4d_kernel<<<blocks, nnz_block, smem_sz, stream>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], 
            output, target_mode, rank, at->idx, at->vals, at->nnz, at->block_coords);
    } else {
        printf("!ERROR! Only order 3 and 4 tensors are supported\n");
    }
    check_cuda(cudaGetLastError(), "mttkrp_lvl1_kernel launch. Exceeded shared mem space?");
}


// Register-based conflict resolution using tile-based execution. TODO make templated
__global__ void mttkrp_lvl1_3d_batched_kernel(FType* f0, FType* f1, FType* f2, FType* output, const int tmode, blco_block** blocks, const IType warps_per_block, const IType rank) 
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

    // Identify block
    IType blco_block_id = block.group_index().x / warps_per_block;
    blco_block* bb = blocks[blco_block_id]; 
    IType* lidx = bb->idx;
    FType* values = bb->vals;
    IType nnz = bb->nnz;
    IType* block_coords = bb->block_coords;

    // Identify block-level workload
    IType curr_elem = (block.group_index().x % warps_per_block) * block.size(); // Index of start element
    IType end_elem = min(nnz, curr_elem + block.size()); // Index of last element
    
    const int mp0 = ALTO_POS[0];
    const int mp1 = ALTO_POS[1];
    const int mp2 = ALTO_POS[2];
    const IType mm0 = ALTO_MASKS[0];
    const IType mm1 = ALTO_MASKS[1];
    const IType mm2 = ALTO_MASKS[2];
    const IType bc0 = block_coords[0];
    const IType bc1 = block_coords[1];
    const IType bc2 = block_coords[2];

    // Iterate workload
    while (curr_elem < end_elem) {
        // Threads collaborate to perform On-the-fly delinerization, sorting, and segmented scan. 
        count[tid] = 0;
        
        //const LIT idx = lidx[curr_elem+tid];
        IType idx;
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

// TODO make templated
__global__ void mttkrp_lvl1_4d_batched_kernel(FType* f0, FType* f1, FType* f2, FType* f3, FType* output, const int tmode, blco_block** blocks, const IType warps_per_block, const IType rank) 
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

    // Identify block
    IType blco_block_id = block.group_index().x / warps_per_block;
    blco_block* bb = blocks[blco_block_id]; 
    IType* lidx = bb->idx;
    FType* values = bb->vals;
    IType nnz = bb->nnz;
    IType* block_coords = bb->block_coords;

    // Identify block-level workload
    IType curr_elem = (block.group_index().x % warps_per_block) * block.size(); // Index of start element
    IType end_elem = min(nnz, curr_elem + block.size()); // Index of last element
       
    const int mp0 = ALTO_POS[0];
    const int mp1 = ALTO_POS[1];
    const int mp2 = ALTO_POS[2];
    const int mp3 = ALTO_POS[3];
    const IType mm0 = ALTO_MASKS[0];
    const IType mm1 = ALTO_MASKS[1];
    const IType mm2 = ALTO_MASKS[2];
    const IType mm3 = ALTO_MASKS[3];
    const IType bc0 = block_coords[0];
    const IType bc1 = block_coords[1];
    const IType bc2 = block_coords[2];
    const IType bc3 = block_coords[3];

    // Iterate workload
    while (curr_elem < end_elem) {
        // Threads collaborate to perform On-the-fly delinerization, sorting, and segmented scan. 
        count[tid] = 0;
        
        //const LIT idx = lidx[curr_elem+tid];
        IType idx;
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
void mttkrp_lvl1_batched(blcotensor* at, FType** mats_staging_ptr, FType* output, FType** mats_dev, int target_mode, IType rank, cudaStream_t stream) 
{
    int nnz_block = TILE_SIZE;
    IType warps_per_block = (at->max_nnz + nnz_block - 1) / nnz_block;
    IType blocks = warps_per_block * at->block_count;
    int smem_sz = nnz_block * (sizeof(FType) + (at->N+1) * sizeof(IType) + sizeof(int)) ;

    if (at->N == 3) {
        mttkrp_lvl1_3d_batched_kernel<<<blocks, nnz_block, smem_sz, stream>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], 
            output, target_mode, at->blocks_dev_ptr, warps_per_block, rank);
    } else if (at->N == 4) {
        mttkrp_lvl1_4d_batched_kernel<<<blocks, nnz_block, smem_sz, stream>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], 
            output, target_mode, at->blocks_dev_ptr, warps_per_block, rank);
    } else {
        printf("!ERROR! Only order 3 and 4 tensors are supported\n");
    }
    check_cuda(cudaGetLastError(), "mttkrp_lvl1_batched_kernel launch. Exceeded shared mem space?");
}


// Register-based conflict resolution using tile-based execution. TODO make templated
__global__ void mttkrp_lvl1_3d_batched_v2_kernel(FType* f0, FType* f1, FType* f2, FType* output, const int tmode, blco_block** blocks, IType* warp_info, const IType total_blocks, const IType rank) 
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

    // Identify block
    IType wi = warp_info[block.group_index().x];
    IType blco_block_id = wi % total_blocks;
    blco_block* bb = blocks[blco_block_id]; 
    IType* lidx = bb->idx;
    FType* values = bb->vals;
    IType nnz = bb->nnz;
    IType* block_coords = bb->block_coords;

    // Identify block-level workload
    IType curr_elem = wi / total_blocks; // Index of start element
    IType end_elem = min(nnz, curr_elem + block.size()); // Index of last element
    
    const int mp0 = ALTO_POS[0];
    const int mp1 = ALTO_POS[1];
    const int mp2 = ALTO_POS[2];
    const IType mm0 = ALTO_MASKS[0];
    const IType mm1 = ALTO_MASKS[1];
    const IType mm2 = ALTO_MASKS[2];
    const IType bc0 = block_coords[0];
    const IType bc1 = block_coords[1];
    const IType bc2 = block_coords[2];

    // Iterate workload
    while (curr_elem < end_elem) {
        // Threads collaborate to perform On-the-fly delinerization, sorting, and segmented scan. 
        count[tid] = 0;
        
        //const LIT idx = lidx[curr_elem+tid];
        IType idx;
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

// TODO make templated
__global__ void mttkrp_lvl1_4d_batched_v2_kernel(FType* f0, FType* f1, FType* f2, FType* f3, FType* output, const int tmode, blco_block** blocks, IType* warp_info, const IType total_blocks, const IType rank) 
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

    // Identify block
    IType wi = warp_info[block.group_index().x];
    IType blco_block_id = wi % total_blocks;
    blco_block* bb = blocks[blco_block_id]; 
    IType* lidx = bb->idx;
    FType* values = bb->vals;
    IType nnz = bb->nnz;
    IType* block_coords = bb->block_coords;

    // Identify block-level workload
    IType curr_elem = wi / total_blocks; // Index of start element
    IType end_elem = min(nnz, curr_elem + block.size()); // Index of last element
       
    const int mp0 = ALTO_POS[0];
    const int mp1 = ALTO_POS[1];
    const int mp2 = ALTO_POS[2];
    const int mp3 = ALTO_POS[3];
    const IType mm0 = ALTO_MASKS[0];
    const IType mm1 = ALTO_MASKS[1];
    const IType mm2 = ALTO_MASKS[2];
    const IType mm3 = ALTO_MASKS[3];
    const IType bc0 = block_coords[0];
    const IType bc1 = block_coords[1];
    const IType bc2 = block_coords[2];
    const IType bc3 = block_coords[3];

    // Iterate workload
    while (curr_elem < end_elem) {
        // Threads collaborate to perform On-the-fly delinerization, sorting, and segmented scan. 
        count[tid] = 0;
        
        //const LIT idx = lidx[curr_elem+tid];
        IType idx;
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
void mttkrp_lvl1_batched_v2(blcotensor* at, FType** mats_staging_ptr, FType* output, FType** mats_dev, int target_mode, IType rank, cudaStream_t stream) 
{
    int nnz_block = TILE_SIZE;
    //IType warps_per_block = (at->max_nnz + nnz_block - 1) / nnz_block;
    //IType blocks = warps_per_block * at->block_count;
    int smem_sz = nnz_block * (sizeof(FType) + (at->N+1) * sizeof(IType) + sizeof(int)) ;

    if (at->N == 3) {
        mttkrp_lvl1_3d_batched_v2_kernel<<<at->warp_info_length, nnz_block, smem_sz, stream>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], 
            output, target_mode, at->blocks_dev_ptr, at->warp_info_gpu, at->block_count, rank);
    } else if (at->N == 4) {
        mttkrp_lvl1_4d_batched_v2_kernel<<<at->warp_info_length, nnz_block, smem_sz, stream>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], 
            output, target_mode, at->blocks_dev_ptr, at->warp_info_gpu, at->block_count, rank);
    } else {
        printf("!ERROR! Only order 3 and 4 tensors are supported\n");
    }
    check_cuda(cudaGetLastError(), "mttkrp_lvl1_batched_v2_kernel launch. Exceeded shared mem space?");
}


// Register- and smem-based conflict resolution using tile-based execution with thread coarsening
template <typename LIT>
__global__ void mttkrp_lvl2_3d_kernel(FType* f0, FType* f1, FType* f2, FType* output, const int tmode, 
        const IType rank, const LIT* __restrict__ lidx, const FType* __restrict__ values, const IType nnz, IType THREAD_CF, IType* block_coords) 
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
    const IType bc0 = block_coords[0];
    const IType bc1 = block_coords[1];
    const IType bc2 = block_coords[2];

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
        const IType rank, const LIT* __restrict__ lidx, const FType* __restrict__ values, const IType nnz, IType THREAD_CF, IType* block_coords) 
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
    const IType bc0 = block_coords[0];
    const IType bc1 = block_coords[1];
    const IType bc2 = block_coords[2];
    const IType bc3 = block_coords[3];

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
void mttkrp_lvl2(blco_block* at, FType** mats_staging_ptr, FType* output, FType** mats_dev, int target_mode, IType rank, IType THREAD_CF, cudaStream_t stream) 
{
    int nnz_block = TILE_SIZE * THREAD_CF;
    IType blocks = (at->nnz + nnz_block -1) / nnz_block;
    int smem_sz = TILE_SIZE * (sizeof(FType) + (at->nmode+1) * sizeof(IType) + sizeof(int)) ;
    smem_sz += (rank + 1) * STASH_SIZE * sizeof(FType);

    if (at->nmode == 3) {
        mttkrp_lvl2_3d_kernel<<<blocks, TILE_SIZE, smem_sz, stream>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], 
            output, target_mode, rank, at->idx, at->vals, at->nnz, THREAD_CF, at->block_coords);
    } else if (at->nmode == 4) {
        mttkrp_lvl2_4d_kernel<<<blocks, TILE_SIZE, smem_sz, stream>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], 
            output, target_mode, rank, at->idx, at->vals, at->nnz, THREAD_CF, at->block_coords);
    } else {
        printf("!ERROR! Only order 3 and 4 tensors are supported\n");
    }
    check_cuda(cudaGetLastError(), "mttkrp_lvl2_kernel launch. Exceeded shared mem space?");
}

// Register-, smem- and gmem-based conflict resolution using tile-based execution with thread coarsening
template <typename LIT>
__global__ void mttkrp_lvl3_3d_kernel(FType* f0, FType* f1, FType* f2, FType* output, const int tmode, 
        const IType rank, const LIT* __restrict__ lidx, const FType* __restrict__ values, const IType nnz,
		FType** ofibs, IType THREAD_CF, IType* block_coords, IType nprtn) 
{
    auto grid = cg::this_grid();
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
    output = ofibs[block.group_index().x % nprtn];
    
    // Identify block-level workload
    //IType curr_elem = prtn_ptr[block.group_index().x] + block.group_index().y * block.size() * THREAD_CF; // Index of start element
    //IType end_elem = min(prtn_ptr[block.group_index().x + 1], curr_elem + block.size() * THREAD_CF); // Index of last element    
    IType curr_elem = ((block.group_index().x % nprtn) * (grid.group_dim().x / nprtn) + (block.group_index().x / nprtn)) * block.size() * THREAD_CF; // Index of start element
    IType end_elem = min(nnz, curr_elem + block.size() * THREAD_CF); // Index of last element
    
    const int mp0 = ALTO_POS[0];
    const int mp1 = ALTO_POS[1];
    const int mp2 = ALTO_POS[2];
    const LIT mm0 = ALTO_MASKS[0];
    const LIT mm1 = ALTO_MASKS[1];
    const LIT mm2 = ALTO_MASKS[2];
    const IType bc0 = block_coords[0];
    const IType bc1 = block_coords[1];
    const IType bc2 = block_coords[2];

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
__global__ void mttkrp_lvl3_4d_kernel(FType* f0, FType* f1, FType* f2, FType* f3, FType* output, const int tmode, 
        const IType rank, const LIT* __restrict__ lidx, const FType* __restrict__ values, const IType nnz,
		FType** ofibs, IType THREAD_CF, IType* block_coords, IType nprtn) 
{
    auto grid = cg::this_grid();
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

    output = ofibs[block.group_index().x % nprtn];
    
    // Identify block-level workload
    //IType curr_elem = prtn_ptr[block.group_index().x] + block.group_index().y * block.size() * THREAD_CF; // Index of start element
    //IType end_elem = min(prtn_ptr[block.group_index().x + 1], curr_elem + block.size() * THREAD_CF); // Index of last element    
    IType curr_elem = ((block.group_index().x % nprtn) * (grid.group_dim().x / nprtn) + (block.group_index().x / nprtn)) * block.size() * THREAD_CF; // Index of start element
    IType end_elem = min(nnz, curr_elem + block.size() * THREAD_CF); // Index of last element
       
    const int mp0 = ALTO_POS[0];
    const int mp1 = ALTO_POS[1];
    const int mp2 = ALTO_POS[2];
    const int mp3 = ALTO_POS[3];
    const LIT mm0 = ALTO_MASKS[0];
    const LIT mm1 = ALTO_MASKS[1];
    const LIT mm2 = ALTO_MASKS[2];
    const LIT mm3 = ALTO_MASKS[3];
    const IType bc0 = block_coords[0];
    const IType bc1 = block_coords[1];
    const IType bc2 = block_coords[2];
    const IType bc3 = block_coords[3];

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

// Thread block per output matrix. Pull based reduction
__global__ void partial_matrix_reduction(FType* output, int nmodes, int target_mode, int nprtn, IType rank, FType** partials) {
    auto block = cg::this_thread_block();
    int block_rank = block.thread_rank();
    IType row = block.group_index().x;

    if (block_rank < rank) {
        // Iterate partial matrices
        for (IType i = 0; i < nprtn; i++) {
            output[row * rank + block_rank] += partials[i][row * rank + block_rank];
        }
    }
}

// Register-, smem- and gmem-based conflict resolution using tile-based execution with thread coarsening
template <typename LIT>
void mttkrp_lvl3(blco_block* at, FType** mats_staging_ptr, FType* output, FType** mats_dev, int target_mode, IType rank, 
		IType mode_length, int nprtn, IType THREAD_CF, cudaStream_t stream) 
{
    int nnz_block = TILE_SIZE * THREAD_CF;
    IType nnz_ptrn = (at->nnz + nprtn - 1) / nprtn;
    IType blocks = (nnz_ptrn + nnz_block - 1) / nnz_block;
    int grid_dim = nprtn * blocks;
    int smem_sz = TILE_SIZE * (sizeof(FType) + (at->nmode+1) * sizeof(IType) + sizeof(int)) ;
    smem_sz += (rank + 1) * STASH_SIZE * sizeof(FType);

    //printf("Lvl3: grid_dim %d TILE_SIZE %d smem_sz %d\n", grid_dim, TILE_SIZE, smem_sz);

    if (at->nmode == 3) {
        mttkrp_lvl3_3d_kernel<<<grid_dim, TILE_SIZE, smem_sz, stream>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], 
            output, target_mode, rank, at->idx, at->vals, at->nnz,
			at->pmatrices, THREAD_CF, at->block_coords, nprtn);
    } else if (at->nmode == 4) {
        mttkrp_lvl3_4d_kernel<<<grid_dim, TILE_SIZE, smem_sz, stream>>>(mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], 
            output, target_mode, rank, at->idx, at->vals, at->nnz,
			at->pmatrices, THREAD_CF, at->block_coords, nprtn);
    } else {
        printf("!ERROR! Only order 3 and 4 tensors are supported\n");
    }
    check_cuda(cudaGetLastError(), "mttkrp_lvl3_kernel launch. Exceeded shared mem space?");
}


void zero_partials(int target_mode, IType rank, int nprtn, blcotensor* at, FType** ofibs) {
	if (ofibs) {
        for (IType p = 0; p < nprtn; p++) {
            check_cuda(cudaMemset(ofibs[p], 0,  sizeof(FType) * rank * at->modes[target_mode]), "cudaMemset partial matrix");
        }
    }
}

void zero_partials_Async(int target_mode, IType rank, int nprtn, blcotensor* at, FType** ofibs, cudaStream_t stream) {
	if (ofibs) {
        for (IType p = 0; p < nprtn; p++) {
            check_cuda(cudaMemsetAsync(ofibs[p], 0,  sizeof(FType) * rank * at->modes[target_mode], stream), "cudaMemset partial matrix");
        }
    }
}

// Make GPU partial matrices
FType** create_da_mem_dev(int kernel, int target_mode, IType rank, int nprtn, blcotensor* at) {
    int const nmode = at->N;

    FType** ofibs = new FType*[nprtn];
    double total_storage = 0.0;
    IType max_num_fibs = 0;
    
    if (target_mode == -1) {
        //allocate enough da_mem for all short modes
        for (int n = 0; n < nmode; ++n) {
            if (kernel == 3 || at->modes[n] <= LVL3_MAX_MODE_LENGTH) max_num_fibs = std::max(max_num_fibs, at->modes[n]);
        }
    } else {
        max_num_fibs = at->modes[target_mode];
    }
	
    //#pragma omp parallel for reduction(+: total_storage)
    if (max_num_fibs > 0) {
        for (IType p = 0; p < nprtn; p++) {
            IType alloc = max_num_fibs;
            alloc *= rank * sizeof(FType);
            check_cuda(cudaMalloc(&ofibs[p], alloc), "cudaMalloc partial matrix");
            total_storage += ((double) alloc) / (1024.0*1024.0);
        }
    }

    //printf("Total storage ofibs: %f MB\n", total_storage);
    return ofibs;
}


// mode_masks are bit masks, mode_pos is shift amount
template <typename LIT>
__global__ void mttkrp_atomic_3d_kernel(FType* A, FType* f0, FType* f1, FType* f2, IType tmode, IType rank, int* mode_pos, LIT* mode_masks, LIT* lidx, FType* values, IType nnz, IType thread_count, IType* block_coords) {
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
    const IType bc0 = block_coords[0];
    const IType bc1 = block_coords[1];
    const IType bc2 = block_coords[2];

    if (curr_elem < thread_count) {
        while (curr_elem < nnz) {
            __syncwarp();
            idx = lidx[curr_elem];
            value = values[curr_elem];

            x = alt_pext(idx, mp0, mm0, bc0);
            y = alt_pext(idx, mp1, mm1, bc1);
            z = alt_pext(idx, mp2, mm2, bc2);

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
__global__ void mttkrp_atomic_4d_kernel(FType* A, FType* f0, FType* f1, FType* f2, FType* f3, IType tmode, IType rank, int* mode_pos, LIT* mode_masks, LIT* lidx, FType* values, IType nnz, IType thread_count, IType* block_coords) {
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
    const IType bc0 = block_coords[0];
    const IType bc1 = block_coords[1];
    const IType bc2 = block_coords[2];
    const IType bc3 = block_coords[3];

    FType value;
    IType x, y, z, w;
    LIT idx;
    FType* location;

    if (curr_elem < thread_count) {
        while (curr_elem < nnz) {
            __syncwarp();
            idx = lidx[curr_elem];
            value = values[curr_elem];

            x = alt_pext(idx, mp0, mm0, bc0);
            y = alt_pext(idx, mp1, mm1, bc1);
            z = alt_pext(idx, mp2, mm2, bc2);
            w = alt_pext(idx, mp3, mm3, bc3);

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
__global__ void mttkrp_atomic_kernel(FType* A, FType** factors, int tmode, IType rank, int* mode_pos, LIT* mode_masks, LIT* lidx, FType* values, int nmode, IType nnz, IType thread_count, IType* block_coords) {
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
                row = alt_pext(idx, mode_pos[m], mode_masks[m], block_coords[m]);
                value *= factors[m][rank * row + thread_id];
            }
            mode_idx = alt_pext(idx, mode_pos[tmode], mode_masks[tmode], block_coords[tmode]);
            for (int m = tmode + 1; m < nmode; m++) {
                row = alt_pext(idx, mode_pos[m], mode_masks[m], block_coords[m]);
                value *= factors[m][rank * row + thread_id];
            }

            atomicAdd(A + rank * mode_idx + thread_id, value);
            curr_elem += thread_count;
        }
    }
}

template <typename LIT>
void mttkrp_atomic(blco_block* at, FType** mats_staging_ptr, FType* output, FType** mats_dev, int target_mode, IType rank, int block_size, int element_stride, cudaStream_t stream) {
    int blocks = element_stride * rank / block_size + 1;
    
    if (at->nmode == 3) {
        mttkrp_atomic_3d_kernel <<<blocks, block_size, 0, stream>>> (
            output, mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nnz, element_stride, at->block_coords);
    } else if (at->nmode == 4) {
        mttkrp_atomic_4d_kernel <<<blocks, block_size, 0, stream>>> (
            output, mats_staging_ptr[0], mats_staging_ptr[1], mats_staging_ptr[2], mats_staging_ptr[3], target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nnz, element_stride, at->block_coords);
    } else {
        mttkrp_atomic_kernel <<<blocks, block_size, 0, stream>>> (
            output, mats_dev, target_mode, rank, at->mode_pos, at->mode_masks, at->idx, at->vals, at->nmode, at->nnz, element_stride, at->block_coords);
    }
    
    check_cuda(cudaGetLastError(), "mttkrp_atomic_kernel launch");
}

// TODO make the __constant__ masks extern so we don't have to do this hack
void send_masks_over(blcotensor* at_host) {
    cudaMemcpyToSymbol(ALTO_MASKS, at_host->mode_masks, at_host->N * sizeof(IType)); // NOT "LIT" datatype
    cudaMemcpyToSymbol(ALTO_POS, at_host->mode_pos, at_host->N * sizeof(int));
}


// The entrypoint for MTTKRP benchmark
template <typename LIT> 
void mttkrp_alto_dev(blcotensor* at_host, KruskalModel* M, int kernel, int iters, int target_mode, IType thread_cf, bool stream_data, bool do_batching, int nprtn) 
{
    printf("\nCopying to GPU device\n");
    printf("--> Init BLCO on GPU\n");
    //printf("--> nprtn (line segments for lvl3) = %d\n", nprtn);

    // Initiate BLCO on GPU
    IType tmode_length = at_host->modes[target_mode];
    if (kernel == 3 && tmode_length > LVL3_MAX_MODE_LENGTH) printf("Warning: using lvl3 with excessive mode lengths. OOM errors possible\n");
    if (kernel == 11 && stream_data) printf("Warning: data streaming not supported with kernel 11\n");
    if (kernel == 12 && stream_data) printf("Warning: data streaming not supported with kernel 12\n");
    IType num_streams = NUM_STREAMS; 
    IType max_block_size = at_host->max_block_size;
    if (!stream_data) {
        num_streams = at_host->block_count;
        max_block_size = 0;
    }
    blcotensor* at_dev = gen_blcotensor_device(at_host, num_streams, max_block_size, do_batching);
    if (!stream_data) {
        printf("--> Send over BLCO tensor\n");
        send_blcotensor_over(at_dev->streams[0], at_dev, at_host);
    }
    send_masks_over(at_host);

    // Generate partial matrices if kernel uses it
    FType** partial_matrices = NULL;
    FType** partial_matrices_dev = NULL;
    if (kernel == 3 || (kernel == 10 && tmode_length <= LVL3_MAX_MODE_LENGTH)) {
        printf("--> Allocate partial matrices\n");
        for (int i = 0; i < num_streams; i++) {
            at_dev->blocks[i]->pmatrices_staging_ptr = create_da_mem_dev(kernel, target_mode, M->rank, nprtn, at_host);
            at_dev->blocks[i]->pmatrices = make_device_copy(at_dev->blocks[i]->pmatrices_staging_ptr, nprtn, "partial matrices pointer");
        }
        if (num_streams > 1) {
            partial_matrices = create_da_mem_dev(kernel, target_mode, M->rank, num_streams, at_host);
            partial_matrices_dev = make_device_copy(partial_matrices, num_streams, "partial matrices pointer");
        } 
    }
    
    printf("--> Copy over Kruskal model (factor matrices)\n");
    // Factor matrices
    KruskalModel* M_dev = make_device_copy(M);
    FType** mats_staging_ptr = M_dev->U;
    FType** mats_dev = M_dev->U_dev;

    // Do MTTKRP (both streamed and in-memory)
    
    // Use events in stream-0 for accurate timing.
    // stream-0 is always synchronous (implicit cudaDeviceSynchronize() between calls),
    // i.e, it forces the host to wait until all issued CUDA/device calls are complete.
    float etime = 0, ms;
    cudaEvent_t start_event, stop_event;
    check_cuda(cudaEventCreate(&start_event), "cudaEventCreate");
    check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate");
    
    printf("--> Start MTTKRP mode %d (%d iters)\n", target_mode, iters);
    for (IType i = 0; i < iters; i++) {
        cudaEventRecord(start_event, 0);
        
        mttkrp_alto_dev_onemode<LIT>(at_host, at_dev, M, mats_staging_ptr[target_mode], kernel, target_mode, thread_cf, stream_data, nprtn, mats_staging_ptr, mats_dev, partial_matrices, partial_matrices_dev);

        check_cuda(cudaEventRecord(stop_event, 0), "cudaEventRecord");
        check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize");
        check_cuda(cudaEventElapsedTime(&ms, start_event, stop_event), "cudaEventElapsedTime");
        etime += ms;
    }

    printf("Total time for MTTKRP: %0.3f (time per iter: %0.3f ms)\n", etime, etime / iters);

    // Copy back
    cudaMemcpy(M->U[target_mode], mats_staging_ptr[target_mode], sizeof(FType) * tmode_length * M->rank, cudaMemcpyDeviceToHost);
    //PrintKruskalModel(M);
    check_cuda(cudaDeviceSynchronize(), "copy mttkrp result back to host");
    
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

    // Clean up
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    delete_blcotensor_device(at_dev);
    destroy_kruskal_model_dev(M_dev);
}

// Called by both MTTKRP benchmark and CP-ALS routine
template <typename LIT> 
void mttkrp_alto_dev_onemode(blcotensor* at_host, blcotensor* at_dev, KruskalModel* M, FType* output, int kernel, int target_mode, IType thread_cf, bool stream_data, int nprtn, 
    FType** mats_staging_ptr, FType** mats_dev, FType** partial_matrices, FType** partial_matrices_dev) {

    IType tmode_length = at_host->modes[target_mode];
    IType num_streams = at_dev->block_count;

    // Zero out ofibs
    if (kernel == 3 || (kernel == 10 && tmode_length <= LVL3_MAX_MODE_LENGTH)) {
        for (int i = 0; i < num_streams; i++) {
            zero_partials_Async(target_mode, M->rank, nprtn, at_host, at_dev->blocks[i]->pmatrices_staging_ptr, at_dev->streams[i]);
        }
        if (num_streams > 1) zero_partials(target_mode, M->rank, num_streams, at_host, partial_matrices);
        check_cuda(cudaDeviceSynchronize(), "Zero partial matrices");
    }
    
    IType stream_id = at_host->block_count - 1;
    if (kernel == 11) { // Level 1 batched, assume no streaming
        mttkrp_lvl1_batched<LIT>(at_dev, mats_staging_ptr, output, mats_dev, target_mode, M->rank, at_dev->streams[0]); 
        check_cuda(cudaGetLastError(), "mttkrp_blco launch");
    } else if (kernel == 12 || (kernel == 10 && tmode_length > LVL3_MAX_MODE_LENGTH && at_dev->warp_info_gpu)) {
        mttkrp_lvl1_batched_v2<LIT>(at_dev, mats_staging_ptr, output, mats_dev, target_mode, M->rank, at_dev->streams[0]); 
        check_cuda(cudaGetLastError(), "mttkrp_blco launch");
    } else {
    for (IType b = 0; b < at_host->block_count; b++) {
        // Determine stream and GPU block
        stream_id = (stream_id + 1) % at_dev->block_count;
        cudaStream_t stream = at_dev->streams[stream_id];
        //cudaEvent_t stream_signal = at_dev->events[stream_id];
        blco_block* blk = at_dev->blocks[stream_id];

        // Send CPU block over
        if (stream_data) send_block_over(stream, blk, at_host->blocks[b]);

        // Invoke relevent kernel
        if (kernel == 1) { // Level 1
            mttkrp_lvl1<LIT>(blk, mats_staging_ptr, output, mats_dev, target_mode, M->rank, stream); 
        } else if (kernel == 2) { // Level 2
            mttkrp_lvl2<LIT>(blk, mats_staging_ptr, output, mats_dev, target_mode, M->rank, thread_cf, stream); 
        } else if (kernel == 3) { // Level 3
            mttkrp_lvl3<LIT>(blk, mats_staging_ptr, output, mats_dev, target_mode, M->rank, tmode_length, nprtn, thread_cf, stream); 
        } else if (kernel == 10) { // AUTO
            if (tmode_length <= LVL3_MAX_MODE_LENGTH) {
                mttkrp_lvl3<LIT>(blk, mats_staging_ptr, output, mats_dev, target_mode, M->rank, tmode_length, nprtn, thread_cf, stream); 
            } else {
                mttkrp_lvl1<LIT>(blk, mats_staging_ptr, output, mats_dev, target_mode, M->rank, stream); 
            }
        } else if (kernel == 13) { // Atomic
            mttkrp_atomic<LIT>(blk, mats_staging_ptr, output, mats_dev, target_mode, M->rank, 128, 32768, stream);
        } else{
            printf("!ERROR! Unknown kernel specified");
            break;
        }
        //check_cuda(cudaEventRecord(stream_signal, stream), "cudaEventRecord");
        // Control stream
        //check_cuda(cudaStreamWaitEvent(at_dev->streams[num_streams], stream_signal, 0), "cudaStreamWaitEvent");
        check_cuda(cudaGetLastError(), "mttkrp_blco launch");
    }
    }
    
    // Merge partial results, if necessary.
    if (kernel == 3 || (kernel == 10 && tmode_length <= LVL3_MAX_MODE_LENGTH)) {
        IType tb_size = ((M->rank + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE; 
        if (num_streams > 1) {
            // Enqueue partial_matrix_reduction for each stream.
            for (int i = 0; i < num_streams; i++) {
                blco_block* blk = at_dev->blocks[i];
                partial_matrix_reduction<<<tmode_length, tb_size, 0, at_dev->streams[i]>>> (partial_matrices[i], at_host->N, target_mode, nprtn, M->rank, blk->pmatrices);
                //check_cuda(cudaGetLastError(), "partial_matrix_reduction launch");
            }
            // Final synchronous partial_matrix_reduction for.
            partial_matrix_reduction<<<tmode_length, tb_size, 0, 0>>> (output, at_host->N, target_mode, num_streams, M->rank, partial_matrices_dev);   	
        } else { //one stream
            blco_block* blk = at_dev->blocks[0];
            partial_matrix_reduction<<<tmode_length, tb_size, 0, 0>>> (output, at_host->N, target_mode, nprtn, M->rank, blk->pmatrices);   	
        }
    }
}

// TODO fix this template stuff
template void mttkrp_alto_dev<IType>(blcotensor* at_host, KruskalModel* M, int kernel, int iters, int target_mode, IType thread_cf, bool stream_data, bool do_batching, int nprtn);

template void mttkrp_alto_dev_onemode<IType>(blcotensor* at_host, blcotensor* at_dev, KruskalModel* M, FType* output, int kernel, int target_mode, IType thread_cf, bool stream_data, int nprtn, 
    FType** mats_staging_ptr, FType** mats_dev, FType** partial_matrices, FType** partial_matrices_dev);
