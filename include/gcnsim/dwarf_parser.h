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
    bool load_from_elf(const uint8_t* data, size_t size);
    bool get_source_location(uint64_t address, std::string& out_file, uint32_t& out_line) const;
    bool get_address_from_location(const std::string& file, uint32_t line, uint64_t& out_address) const;
    const std::vector<LineEntry>& get_entries() const { return entries_; }

private:
    std::vector<LineEntry> entries_;
    const uint8_t* debug_line_str_ptr_ = nullptr;
    size_t debug_line_str_size_ = 0;
    const uint8_t* debug_str_ptr_ = nullptr;
    size_t debug_str_size_ = 0;
};

} // namespace gcnsim
