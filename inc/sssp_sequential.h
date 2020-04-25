#ifndef SSSP_Sequential_H
#define SSSP_Sequential_H

#include "sssp.h"
#include <algorithm>

class SSSP_Sequential : public SSSP {
public:
    /**
     * SSSP instance for sequential computation the CPU.
     * @param graph graph that is used for computations.
     */
    explicit SSSP_Sequential(std::shared_ptr<Graph> graph);
    std::vector<std::vector<int>> compute(int source_node) override;
};

#endif /* SSSP_Sequential_H */
