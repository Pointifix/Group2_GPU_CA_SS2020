//
// Created by Lucas on 01/04/2020.
//

#include "test.h"

#include <string>

std::string getOsName() {
#ifdef _WIN32
    return "Windows 32-bit";
#elif _WIN64
    return "Windows 64-bit";
#elif __APPLE__ || __MACH__
    return "Mac OSX";
#elif __linux__
    return "Linux";
#elif __FreeBSD__
    return "FreeBSD";
    #elif __unix || __unix__
    return "Unix";
    #else
    return "Other";
#endif
}

int main(int argc, char * argv[]) {
    printf("Hello! Your OS is %s", getOsName().c_str());
}