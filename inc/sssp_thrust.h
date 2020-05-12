#ifndef SSSP_Thrust_H
#define SSSP_Thrust_H

#include "sssp.h"
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "common.cuh"
#include "alg.cuh"
#include "graph.h"

class SSSP_Thrust : public SSSP {
public:
    /**
     * Instance of a SSSP for computation using thrust.
     * @param graph graph that is used for computations.
     */
    explicit SSSP_Thrust(std::shared_ptr<Graph> graph);
    std::shared_ptr<Paths> compute(int source_node) override;
};

#endif /* SSSP_Thrust_H */