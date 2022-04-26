#ifndef KRUSKAL_MODEL_HPP_
#define KRUSKAL_MODEL_HPP_


#include "common.hpp"


typedef struct KruskalModel_
{
    int mode;
    IType *dims;
    IType rank;
    FType **U;     // Kruskal factors, I_n * rank
    FType *lambda; // rank * 1
    FType** U_dev; // Kruskal factors on device
} KruskalModel;

typedef enum
{
    MAT_NORM_2,
    MAT_NORM_MAX
} mat_norm_type;

void CreateKruskalModel(int mode, IType *dim, IType rank, KruskalModel **M_);

void KruskalModelNormalize(KruskalModel *M);

void KruskalModelNorm(KruskalModel* M,
                         IType mode, 
                         mat_norm_type which,
                         FType ** scratchpad);

void KruskalModelRandomInit(KruskalModel *M, unsigned int seed);

void ExportKruskalModel(KruskalModel *M, const char *file_path);

void DestroyKruskalModel(KruskalModel *M);

void PrintKruskalModel(KruskalModel *M);

void RedistributeLambda (KruskalModel *M, int n);

double KruskalTensorFit();

// Replicates the Kruskal model on the device
KruskalModel* make_device_copy(KruskalModel* M_host);

// Destroys a Kruskal model stored on the device
void destroy_kruskal_model_dev(KruskalModel* M_dev);

#endif // KRUSKAL_MODEL_HPP_
