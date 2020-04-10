#pragma once
#include <memory>
#include <string>

#include "graph.h"

/**
 * Saves the given graph into the given file.
 * @param filename Name of the file.
 * @param graph Graph.
 */
void writeGraph(const std::string& filename, const std::shared_ptr<Graph>& graph)
{
    // TODO
}

/**
 * Reads a graph from the given file.
 * @param filename Name of the file.
 * @return a Graph.
 */
std::shared_ptr<Graph> readGraph(const std::string& filename)
{
    // TODO
    return nullptr;
}