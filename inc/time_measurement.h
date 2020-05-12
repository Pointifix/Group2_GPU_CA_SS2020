#ifndef GROUP2_GPU_CA_SS2020_TIME_MEASUREMENT_H
#define GROUP2_GPU_CA_SS2020_TIME_MEASUREMENT_H

#include <iostream>
#include <chrono>
#include <map>

namespace time_measurement
{

    std::map<std::string, std::pair<std::chrono::time_point<std::chrono::steady_clock>, std::chrono::time_point<std::chrono::steady_clock>>> time_measurements;

    void startMeasurement(std::string name)
    {
        std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now();
        time_measurements[name] = std::make_pair(now, now);
    }

    void endMeasurement(std::string name)
    {
        time_measurements[name].second = std::chrono::steady_clock::now();
    }

    void printMeasurements()
    {
        std::cout << "----------------------------------------" << std::endl;
        for (auto const& time_point : time_measurements)
        {
            std::string name = time_point.first + ":";

            name.append(std::string( 30 - name.length(), ' ' ));

            std::cout << name
            << std::chrono::duration_cast<std::chrono::milliseconds>(time_point.second.second - time_point.second.first).count()
            << " ms" << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }
}

#endif //GROUP2_GPU_CA_SS2020_TIME_MEASUREMENT_H
