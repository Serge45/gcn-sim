#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace gcnsim {

struct LineEntry {
    uint64_t address;
    std::string file;
    uint32_t line;
};

class DwarfParser {
public:
    DwarfParser() = default;
    bool load_from_elf(const std::vector<uint8_t>& elf_data);
    bool get_source_location(uint64_t address, std::string& out_file, uint32_t& out_line) const;
    const std::vector<LineEntry>& get_entries() const { return entries_; }

private:
    std::vector<LineEntry> entries_;
};

} // namespace gcnsim
