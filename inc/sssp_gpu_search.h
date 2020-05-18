#ifndef SSSP_GPU_Search_H
#define SSSP_GPU_Search_H

#include "sssp.h"

#include "common.cuh"
#include "alg.cuh"

class SSSP_GPU_Search : public SSSP {
public:
    /**
     * SSSP instance for standard CUDA computation.
     * @param graph graph that is used for computations.
     */
    explicit SSSP_GPU_Search(std::shared_ptr<Graph> graph);
    std::shared_ptr<Paths> compute(int source_node) override;
};

#endif /* SSSP_GPU_Search_H */
