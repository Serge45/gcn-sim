#include "gcnsim/gdb_mi_parser.h"
#include <cctype>

namespace gcnsim {

GdbMiCommand parse_gdb_mi(const std::string& line) {
    GdbMiCommand cmd;
    size_t i = 0;

    // Skip initial whitespace
    while (i < line.size() && std::isspace(line[i])) {
        i++;
    }

    // Extract token (digits)
    while (i < line.size() && std::isdigit(line[i])) {
        cmd.token += line[i];
        i++;
    }

    // Skip whitespace after token
    while (i < line.size() && std::isspace(line[i])) {
        i++;
    }

    if (i >= line.size()) {
        return cmd;
    }

    // Check if MI command (starts with '-')
    if (line[i] == '-') {
        cmd.is_mi = true;
        // Parse command name
        while (i < line.size() && !std::isspace(line[i])) {
            cmd.command += line[i];
            i++;
        }
    } else {
        cmd.is_mi = false;
        // Parse raw command name (CLI command)
        while (i < line.size() && !std::isspace(line[i])) {
            cmd.command += line[i];
            i++;
        }
    }

    // Parse arguments
    while (i < line.size()) {
        // Skip whitespace
        while (i < line.size() && std::isspace(line[i])) {
            i++;
        }
        if (i >= line.size()) break;

        std::string arg;
        if (line[i] == '"') {
            i++; // Skip open quote
            while (i < line.size()) {
                if (line[i] == '\\' && i + 1 < line.size()) {
                    arg += line[i + 1];
                    i += 2;
                } else if (line[i] == '"') {
                    i++; // Skip close quote
                    break;
                } else {
                    arg += line[i];
                    i++;
                }
            }
        } else {
            while (i < line.size() && !std::isspace(line[i])) {
                arg += line[i];
                i++;
            }
        }
        cmd.args.push_back(arg);
    }

    return cmd;
}

} // namespace gcnsim
