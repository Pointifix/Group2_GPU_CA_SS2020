#pragma once

#include "catch.hpp"

#include "graph_generator.h"

#include <cstdlib>

using namespace graphgen;

constexpr float den(int e, int v, bool directed) {
    return (float)(directed ? 1 : 2) * (float)e / ((float)v * ((float)v - 1));
}

TEST_CASE("Graph Generator simple constraints") {
    CHECK_FALSE(generateConnectedGraph(-1, 1));
    CHECK_FALSE(generateConnectedGraph(0, 1.0001));
    CHECK_FALSE(generateConnectedGraph(1, -0.0001));
    CHECK(generateConnectedGraph(2, 0));
    CHECK(generateConnectedGraph(3, 0.5));
    CHECK(generateConnectedGraph(4, 1));
    CHECK(generateConnectedGraph(5, 0.25, true));
    CHECK(generateConnectedGraph(6, 0.75, false));
}

TEST_CASE("Graph Generator simple graph") {
    std::shared_ptr<Graph> g;

    g = generateConnectedGraph(0, 0);
    CHECK(g->edges.empty());
    CHECK(g->destinations.empty());
    CHECK(g->weights.empty());

    g = generateConnectedGraph(1, 0);
    CHECK(g->edges.size() == 1);
    CHECK(g->destinations.empty());
    CHECK(g->weights.empty());

    g = generateConnectedGraph(1000, 0);
    CHECK(g->edges.size() == 1000);
    CHECK(g->destinations.empty());
    CHECK(g->weights.empty());

    g = generateConnectedGraph(10, den(5, 10, true), true);
    CHECK(g->edges.size() == 10);
    CHECK(g->destinations.size() == 5);
    CHECK(g->weights.size() == 5);

    g = generateConnectedGraph(10, den(5, 10, false), false);
    CHECK(g->edges.size() == 10);
    CHECK(g->destinations.size() == 5);
    CHECK(g->weights.size() == 5);

    g = generateConnectedGraph(1000, den(700, 1000, true), true);
    CHECK(g->edges.size() == 1000);
    CHECK(g->destinations.size() == 700);
    CHECK(g->weights.size() == 700);

    g = generateConnectedGraph(1000, den(300, 1000, false), false);
    CHECK(g->edges.size() == 1000);
    CHECK(g->destinations.size() == 300);
    CHECK(g->weights.size() == 300);
}

TEST_CASE("Graph Generator weak connectedness") {

}