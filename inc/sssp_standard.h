#ifndef SSSP_Standard_H
#define SSSP_Standard_H

#include "sssp.h"

#include "common.cuh"
#include "alg.cuh"

#include <iostream>
#include <limits>
#include <algorithm>

class SSSP_Standard : public SSSP {
public:

    enum MemoryType {
        NORMAL,
        PINNED,
        ZERO_COPY
    };

    enum SearchType {
        CPU,
        GPU
    };

    /**
     * SSSP instance for standard CUDA computation.
     * @param graph graph that is used for computations.
     */
    explicit SSSP_Standard(std::shared_ptr<Graph> graph, MemoryType memType, SearchType searchType);
    std::shared_ptr<Paths> compute(int source_node) override;

private:
    MemoryType m_memType;
    SearchType m_searchType;

};

#endif /* SSSP_Standard_H */
