#pragma once

#include "catch.hpp"

#include "../src/graph_generator.h"

float calculateDensity(int e, int v, bool directed) {
    return (float)(directed ? 1 : 2) * (float)e / ((float)v * ((float)v - 1));
}

TEST_CASE("Graph Generator simple constraints", "[graph][graph_generator]") {
    REQUIRE(!generateConnectedGraph(-1, 1));
    REQUIRE(!generateConnectedGraph(0, 1.0001));
    REQUIRE(!generateConnectedGraph(1, -0.0001));
    REQUIRE(generateConnectedGraph(2, 0));
    REQUIRE(generateConnectedGraph(3, 0.5));
    REQUIRE(generateConnectedGraph(4, 1));
    REQUIRE(generateConnectedGraph(5, 0.25, true));
    REQUIRE(generateConnectedGraph(6, 0.75, false));
}