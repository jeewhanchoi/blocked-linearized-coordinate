#ifndef ALTO_DEV_HPP_
#define ALTO_DEV_HPP_

#include "sptensor.hpp"
#include "kruskal_model.hpp"
#include "common.hpp"
#include "alto.hpp"
#include "blco.hpp"

#ifndef ALT_PEXT
#error ALTERNATIVE_PEXT must be set to run GPU code
#endif


#define INVALID_ID	((IType) -1)

// TODO make the __constant__ masks extern so we don't have to do this hack
void send_masks_over(blcotensor* at_host);

FType** create_da_mem_dev(int kernel, int target_mode, IType rank, int nprtn, blcotensor* at);

// MTTKRP benchmark driver function
template <typename LIT> void mttkrp_alto_dev(blcotensor* at_host, KruskalModel* M, int kernel, int iters, int target_mode, IType thread_cf, bool stream_data, bool do_batching, int nprtn);

// Single MTTKRP execution, used by both benchmark MTTKRP and CPD driver
template <typename LIT> void mttkrp_alto_dev_onemode(blcotensor* at_host, blcotensor* at_dev, KruskalModel* M, FType* output, int kernel, int target_mode, IType thread_cf, bool stream_data, int nprtn, 
    FType** mats_staging_ptr, FType** mats_dev, FType** partial_matrices, FType** partial_matrices_dev);

#endif
