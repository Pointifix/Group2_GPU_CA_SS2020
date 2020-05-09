#ifndef SSP_Pinned_Memory_H
#define SSP_Pinned_Memory_H

#include "sssp.h"

class SSSP_Pinned_Memory : public SSSP {
public:
    /**
     * SSSP instance using pinned memory for CUDA computation.
     * @param graph graph that is used for computations.
     */
    explicit SSSP_Pinned_Memory(std::shared_ptr<Graph> graph);
    std::vector<std::vector<int>> compute(int source_node) override;
};

#endif /* SSP_Pinned_Memory_H */
