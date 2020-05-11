#pragma once

#include "catch.hpp"

#include "graph_generator.h"
#include "graph_io.h"

#include "sssp.h"
#include "sssp_sequential.h"
#include "sssp_standard.h"
#include "sssp_thrust.h"
#include "sssp_pinned_memory.h"
#include "sssp_zero_copy_memory.h"

#include <cstdlib>

using namespace graphgen;
using namespace graphio;

TEST_CASE("Test SSSP") {
    SECTION("Small graph") {
        std::shared_ptr<Graph> g = generateConnectedGraph(25, calculateDensity(50, 25, false), false);

        srand(time(nullptr));
        int random_source = rand() % 25;

        SSSP_Sequential sequential(g);
        std::shared_ptr<Paths> paths_sequential = sequential.compute(random_source);

        SSSP_Standard standard(g);
        std::shared_ptr<Paths> paths_standard = standard.compute(random_source);

        SSSP_Thrust thrust(g);
        std::shared_ptr<Paths> paths_thrust = thrust.compute(random_source);

        SSSP_Pinned_Memory pinnedMemory(g);
        std::shared_ptr<Paths> paths_pinned = pinnedMemory.compute(random_source);

        SSSP_Zero_Copy_Memory zeroCopyMemory(g);
        std::shared_ptr<Paths> paths_zero_copy = zeroCopyMemory.compute(random_source);

        REQUIRE(paths_sequential->costs == paths_standard->costs);
        REQUIRE(paths_sequential->costs == paths_thrust->costs);
        REQUIRE(paths_sequential->costs == paths_pinned->costs);
        REQUIRE(paths_sequential->costs == paths_zero_copy->costs);
    }
    SECTION("Large graph") {
        std::shared_ptr<Graph> g = generateConnectedGraph(2'000, calculateDensity(10'000, 2'000, false), false);

        srand(time(nullptr));
        int random_source = rand() % 2'000;

        SSSP_Sequential sequential(g);
        std::shared_ptr<Paths> paths_sequential = sequential.compute(random_source);

        SSSP_Standard standard(g);
        std::shared_ptr<Paths> paths_standard = standard.compute(random_source);

        SSSP_Thrust thrust(g);
        std::shared_ptr<Paths> paths_thrust = thrust.compute(random_source);

        SSSP_Pinned_Memory pinnedMemory(g);
        std::shared_ptr<Paths> paths_pinned = pinnedMemory.compute(random_source);

        SSSP_Zero_Copy_Memory zeroCopyMemory(g);
        std::shared_ptr<Paths> paths_zero_copy = zeroCopyMemory.compute(random_source);

        REQUIRE(paths_sequential->costs == paths_standard->costs);
        REQUIRE(paths_sequential->costs == paths_thrust->costs);
        REQUIRE(paths_sequential->costs == paths_pinned->costs);
        REQUIRE(paths_sequential->costs == paths_zero_copy->costs);
    }
}