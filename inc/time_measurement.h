#ifndef GROUP2_GPU_CA_SS2020_TIME_MEASUREMENT_H
#define GROUP2_GPU_CA_SS2020_TIME_MEASUREMENT_H

#include <iostream>
#include <chrono>
#include <map>

namespace time_measurement
{
    std::map<std::string, std::vector<std::pair<std::chrono::time_point<std::chrono::steady_clock>, std::chrono::time_point<std::chrono::steady_clock>>>> time_measurements;

    void startMeasurement(std::string name)
    {
        std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now();
        time_measurements[name].push_back(std::make_pair(now, now));
    }

    void endMeasurement(std::string name)
    {
        time_measurements[name].back().second = std::chrono::steady_clock::now();
    }

    void printMeasurements()
    {
        std::cout << "----------------------------------------" << std::endl;
        for (auto const& time_point_vector : time_measurements)
        {
            std::string name = time_point_vector.first + ":";
            name.append(std::string( 20 - name.length(), ' ' ) + "\t");
            for (auto const& time_point : time_point_vector.second)
            {
                name.append(std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(time_point.second - time_point.first).count()) + " ms\t");
            }
            std::cout << name << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }
}

#endif //GROUP2_GPU_CA_SS2020_TIME_MEASUREMENT_H
