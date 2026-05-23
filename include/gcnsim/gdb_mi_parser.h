#pragma once
#include <string>
#include <vector>

namespace gcnsim {

struct GdbMiCommand {
    std::string token;
    std::string command;
    std::vector<std::string> args;
    bool is_mi = false; // true if starts with '-'
};

GdbMiCommand parse_gdb_mi(const std::string& line);

} // namespace gcnsim
