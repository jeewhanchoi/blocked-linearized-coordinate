#include "common.hpp"
#include "utils.hpp"
#include "kruskal_model.hpp"


KruskalModel* make_device_copy(KruskalModel* M_host) {
    KruskalModel* M_dev = new KruskalModel;
    M_dev->mode = M_host->mode;
    M_dev->rank = M_host->rank;
    M_dev->U = new FType*[M_host->mode]; // Dev staging pointer
    M_dev->dims = make_device_copy(M_host->dims, M_host->mode, "kruskal dims");
    M_dev->lambda = make_device_copy(M_host->lambda, M_host->rank, "kruskal lambda");
    for (IType i = 0; i < M_host->mode; i++) {
        M_dev->U[i] = make_device_copy(M_host->U[i], M_host->dims[i] * M_host->rank, "kruskal mats dev staging");
    }
    M_dev->U_dev = make_device_copy(M_dev->U, M_host->mode, "kruskal mats dev");
    return M_dev;
}


void destroy_kruskal_model_dev(KruskalModel* M_dev) {

    for (IType i = 0; i < M_dev->mode; i++) cudaFree(M_dev->U[i]);
    cudaFree(M_dev->U_dev);
    cudaFree(M_dev->lambda);
    delete[] M_dev->U;
    cudaFree(M_dev->dims);
    delete M_dev;
}
