#include "gcnsim/dwarf_parser.h"
#include <elf.h>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

namespace gcnsim {

// DWARF constants
static constexpr uint8_t DW_LNE_end_sequence = 1;
static constexpr uint8_t DW_LNE_set_address = 2;
static constexpr uint8_t DW_LNE_define_file = 3;
static constexpr uint8_t DW_LNE_set_discriminator = 4;

static constexpr uint8_t DW_LNS_copy = 1;
static constexpr uint8_t DW_LNS_advance_pc = 2;
static constexpr uint8_t DW_LNS_advance_line = 3;
static constexpr uint8_t DW_LNS_set_file = 4;
static constexpr uint8_t DW_LNS_set_column = 5;
static constexpr uint8_t DW_LNS_negate_stmt = 6;
static constexpr uint8_t DW_LNS_set_basic_block = 7;
static constexpr uint8_t DW_LNS_const_add_pc = 8;
static constexpr uint8_t DW_LNS_fixed_advance_pc = 9;
static constexpr uint8_t DW_LNS_set_prologue_end = 10;
static constexpr uint8_t DW_LNS_set_epilogue_begin = 11;
static constexpr uint8_t DW_LNS_set_isa = 12;

// Utility functions for decoding
static inline uint16_t read_uint16(const uint8_t*& ptr) {
    uint16_t val = ptr[0] | (ptr[1] << 8);
    ptr += 2;
    return val;
}

static inline uint32_t read_uint32(const uint8_t*& ptr) {
    uint32_t val = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
    ptr += 4;
    return val;
}

static inline uint64_t read_uint64(const uint8_t*& ptr) {
    uint64_t val = 0;
    for (int i = 0; i < 8; ++i) {
        val |= uint64_t(ptr[i]) << (i * 8);
    }
    ptr += 8;
    return val;
}

static inline uint64_t read_uleb128(const uint8_t*& ptr, const uint8_t* end) {
    uint64_t result = 0;
    int shift = 0;
    while (ptr < end) {
        uint8_t byte = *ptr++;
        result |= uint64_t(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0)
            break;
        shift += 7;
    }
    return result;
}

static inline int64_t read_sleb128(const uint8_t*& ptr, const uint8_t* end) {
    int64_t result = 0;
    int shift = 0;
    uint8_t byte = 0;
    while (ptr < end) {
        byte = *ptr++;
        result |= int64_t(byte & 0x7F) << shift;
        shift += 7;
        if ((byte & 0x80) == 0)
            break;
    }
    if ((shift < 64) && (byte & 0x40)) {
        result |= -(1ULL << shift);
    }
    return result;
}

bool DwarfParser::load_from_elf(const std::vector<uint8_t>& elf_data) {
    return load_from_elf(elf_data.data(), elf_data.size());
}

bool DwarfParser::load_from_elf(const uint8_t* data, size_t data_size) {
    if (!data) return false;
    if (data_size < sizeof(Elf64_Ehdr)) return false;
    const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(data);
    if (ehdr->e_ident[0] != 0x7f || ehdr->e_ident[1] != 'E' ||
        ehdr->e_ident[2] != 'L' || ehdr->e_ident[3] != 'F') {
        return false;
    }

    uint64_t shoff = ehdr->e_shoff;
    uint16_t shentsize = ehdr->e_shentsize;
    uint16_t shnum = ehdr->e_shnum;
    uint16_t shstrndx = ehdr->e_shstrndx;

    if (shoff == 0 || shnum == 0 || shoff + (uint64_t)shentsize * shnum > data_size) {
        return false;
    }

    auto get_shdr = [&](int idx) -> const Elf64_Shdr* {
        if (idx < 0 || idx >= shnum) return nullptr;
        return reinterpret_cast<const Elf64_Shdr*>(data + shoff + idx * shentsize);
    };

    const Elf64_Shdr* shstrtab_shdr = get_shdr(shstrndx);
    if (!shstrtab_shdr || shstrtab_shdr->sh_offset + shstrtab_shdr->sh_size > data_size) {
        return false;
    }
    const char* shstrtab = reinterpret_cast<const char*>(data + shstrtab_shdr->sh_offset);

    // Find relevant debug sections
    const Elf64_Shdr* debug_line_shdr = nullptr;
    const Elf64_Shdr* debug_line_str_shdr = nullptr;
    const Elf64_Shdr* debug_str_shdr = nullptr;
    for (int i = 0; i < shnum; ++i) {
        const Elf64_Shdr* sh = get_shdr(i);
        if (sh && sh->sh_name < shstrtab_shdr->sh_size) {
            std::string sname(shstrtab + sh->sh_name);
            if (sname == ".debug_line") {
                debug_line_shdr = sh;
            } else if (sname == ".debug_line_str") {
                debug_line_str_shdr = sh;
            } else if (sname == ".debug_str") {
                debug_str_shdr = sh;
            }
        }
    }

    if (!debug_line_shdr) return false;

    if (debug_line_str_shdr) {
        debug_line_str_ptr_ = data + debug_line_str_shdr->sh_offset;
        debug_line_str_size_ = debug_line_str_shdr->sh_size;
    }
    if (debug_str_shdr) {
        debug_str_ptr_ = data + debug_str_shdr->sh_offset;
        debug_str_size_ = debug_str_shdr->sh_size;
    }

    const uint8_t* debug_line_ptr = data + debug_line_shdr->sh_offset;
    const uint8_t* debug_line_end = debug_line_ptr + debug_line_shdr->sh_size;
    const uint8_t* ptr = debug_line_ptr;

    while (ptr < debug_line_end) {
        const uint8_t* unit_start = ptr;
        uint64_t unit_length = read_uint32(ptr);
        bool is_dwarf64 = false;
        if (unit_length == 0xffffffff) {
            unit_length = read_uint64(ptr);
            is_dwarf64 = true;
        }
        if (ptr + unit_length > debug_line_end) break;
        const uint8_t* unit_end = ptr + unit_length;

        uint16_t version = read_uint16(ptr);
        uint8_t address_size = 8;
        uint8_t segment_selector_size = 0;
        if (version >= 5) {
            address_size = *ptr++;
            segment_selector_size = *ptr++;
        }
        uint64_t header_length = is_dwarf64 ? read_uint64(ptr) : read_uint32(ptr);
        const uint8_t* header_end = ptr + header_length;

        uint8_t min_insn_len = *ptr++;
        uint8_t max_ops_per_insn = 1;
        if (version >= 4) max_ops_per_insn = *ptr++;
        uint8_t default_is_stmt = *ptr++;
        int8_t line_base = (int8_t)*ptr++;
        uint8_t line_range = *ptr++;
        uint8_t opcode_base = *ptr++;

        std::vector<uint8_t> std_opcode_lengths(opcode_base - 1);
        for (int i = 1; i < opcode_base; ++i) std_opcode_lengths[i - 1] = *ptr++;

        std::vector<std::string> include_dirs;
        struct FileEntry { std::string name; uint64_t dir_idx; };
        std::vector<FileEntry> files;

        auto resolve_string = [&](uint64_t form) -> std::string {
            if (form == 0x08) { // DW_FORM_string
                std::string s((const char*)ptr);
                ptr += s.size() + 1;
                return s;
            } else if (form == 0x1f) { // DW_FORM_line_strp
                uint32_t off = read_uint32(ptr);
                if (debug_line_str_ptr_ && off < debug_line_str_size_)
                    return std::string((const char*)debug_line_str_ptr_ + off);
            } else if (form == 0x0e) { // DW_FORM_strp
                uint32_t off = read_uint32(ptr);
                if (debug_str_ptr_ && off < debug_str_size_)
                    return std::string((const char*)debug_str_ptr_ + off);
            } else if (form == 0x11 || form == 0x1a || form == 0x25 || form == 0x26) {
                ptr += (address_size == 4) ? 4 : 8;
            } else if (form == 0x13 || form == 0x18 || form == 0x0d || form == 0x0f || form == 0x0b) {
                read_uleb128(ptr, header_end);
            } else if (form == 0x19) { // block
                uint64_t len = read_uleb128(ptr, header_end);
                ptr += len;
            } else {
                read_uleb128(ptr, header_end);
            }
            return "";
        };

        if (version < 5) {
            include_dirs.push_back(".");
            while (ptr < header_end && *ptr != 0) {
                std::string dir((const char*)ptr);
                ptr += dir.size() + 1;
                include_dirs.push_back(dir);
            }
            if (ptr < header_end) ptr++;
            while (ptr < header_end && *ptr != 0) {
                std::string fname((const char*)ptr);
                ptr += fname.size() + 1;
                uint64_t dir_idx = read_uleb128(ptr, header_end);
                read_uleb128(ptr, header_end);
                read_uleb128(ptr, header_end);
                files.push_back({fname, dir_idx});
            }
            if (ptr < header_end) ptr++;
        } else {
            // DWARF 5 Directory Table
            uint8_t format_count = *ptr++;
            struct Format { uint64_t type; uint64_t form; };
            std::vector<Format> formats;
            for (int i = 0; i < format_count; i++) formats.push_back({read_uleb128(ptr, header_end), read_uleb128(ptr, header_end)});
            uint64_t dir_count = read_uleb128(ptr, header_end);
            for (uint64_t i = 0; i < dir_count; i++) {
                std::string dname = "";
                for (const auto& fmt : formats) {
                    std::string val = resolve_string(fmt.form);
                    if (fmt.type == 1) dname = val;
                }
                include_dirs.push_back(dname);
            }

            // DWARF 5 File Table
            format_count = *ptr++;
            formats.clear();
            for (int i = 0; i < format_count; i++) formats.push_back({read_uleb128(ptr, header_end), read_uleb128(ptr, header_end)});
            uint64_t file_count = read_uleb128(ptr, header_end);
            for (uint64_t i = 0; i < file_count; i++) {
                std::string fname = "";
                uint64_t d_idx = 0;
                for (const auto& fmt : formats) {
                    if (fmt.type == 1) fname = resolve_string(fmt.form);
                    else if (fmt.type == 2) {
                        if (fmt.form == 0x0f) d_idx = read_uleb128(ptr, header_end);
                        else if (fmt.form == 0x0d) d_idx = read_uint32(ptr);
                        else if (fmt.form == 0x05) d_idx = read_uint16(ptr);
                        else if (fmt.form == 0x13) d_idx = *ptr++;
                        else resolve_string(fmt.form);
                    } else resolve_string(fmt.form);
                }
                files.push_back({fname, d_idx});
            }
            ptr = header_end;
        }

        uint64_t address = 0;
        uint64_t file = (version < 5) ? 1 : 0;
        int64_t line = 1;
        uint64_t column = 0;
        bool is_stmt = (default_is_stmt != 0);
        bool basic_block = false;
        bool prologue_end = false;
        bool epilogue_begin = false;
        uint64_t isa = 0;
        uint64_t discriminator = 0;

        auto add_entry = [&]() {
            if (address == 0) return;
            std::string fpath = "";
            bool found = false;
            size_t f_idx = (version < 5) ? (file > 0 ? file - 1 : 0) : file;
            if (f_idx < files.size()) {
                std::string fname = files[f_idx].name;
                uint64_t d_idx = files[f_idx].dir_idx;
                std::string raw_path;
                if (!fname.empty() && fname[0] == '/') {
                    raw_path = fname;
                } else if (d_idx < include_dirs.size()) {
                    std::string dir = include_dirs[d_idx];
                    if (!dir.empty() && dir != ".") {
                        if (dir.back() == '/') raw_path = dir + fname;
                        else raw_path = dir + "/" + fname;
                    } else raw_path = fname;
                } else raw_path = fname;
                
                try {
                    if (!raw_path.empty() && std::filesystem::exists(raw_path)) {
                        fpath = std::filesystem::canonical(raw_path).string();
                    } else {
                        fpath = raw_path;
                    }
                } catch(...) { fpath = raw_path; }
            }
            entries_.push_back({address, fpath, (uint32_t)line});
        };

        while (ptr < unit_end) {
            uint8_t opcode = *ptr++;
            if (opcode == 0) {
                uint64_t ext_len = read_uleb128(ptr, unit_end);
                const uint8_t* ext_end = ptr + ext_len;
                uint8_t ext_opcode = *ptr++;
                if (ext_opcode == DW_LNE_end_sequence) {
                    add_entry();
                    address = 0; file = (version < 5) ? 1 : 0; line = 1; column = 0;
                    is_stmt = (default_is_stmt != 0);
                    basic_block = false; prologue_end = false; epilogue_begin = false;
                    isa = 0; discriminator = 0;
                } else if (ext_opcode == DW_LNE_set_address) {
                    address = read_uint64(ptr);
                } else if (ext_opcode == DW_LNE_define_file) {
                    std::string fname((const char*)ptr);
                    ptr += fname.size() + 1;
                    uint64_t dir_idx = read_uleb128(ptr, ext_end);
                    read_uleb128(ptr, ext_end);
                    read_uleb128(ptr, ext_end);
                    files.push_back({fname, dir_idx});
                } else if (ext_opcode == DW_LNE_set_discriminator) {
                    discriminator = read_uleb128(ptr, ext_end);
                }
                ptr = ext_end;
            } else if (opcode < opcode_base) {
                switch (opcode) {
                    case DW_LNS_copy:
                        add_entry();
                        basic_block = false; prologue_end = false; epilogue_begin = false; discriminator = 0;
                        break;
                    case DW_LNS_advance_pc:
                        address += read_uleb128(ptr, unit_end) * min_insn_len;
                        break;
                    case DW_LNS_advance_line:
                        line += read_sleb128(ptr, unit_end);
                        break;
                    case DW_LNS_set_file:
                        file = read_uleb128(ptr, unit_end);
                        break;
                    case DW_LNS_set_column:
                        column = read_uleb128(ptr, unit_end);
                        break;
                    case DW_LNS_negate_stmt:
                        is_stmt = !is_stmt;
                        break;
                    case DW_LNS_set_basic_block:
                        basic_block = true;
                        break;
                    case DW_LNS_const_add_pc:
                        address += ((255 - opcode_base) / line_range) * min_insn_len;
                        break;
                    case DW_LNS_fixed_advance_pc:
                        address += read_uint16(ptr);
                        break;
                    case DW_LNS_set_prologue_end:
                        prologue_end = true;
                        break;
                    case DW_LNS_set_epilogue_begin:
                        epilogue_begin = true;
                        break;
                    case DW_LNS_set_isa:
                        isa = read_uleb128(ptr, unit_end);
                        break;
                    default: {
                        if (opcode - 1 < std_opcode_lengths.size()) {
                            int num_args = std_opcode_lengths[opcode - 1];
                            for (int a = 0; a < num_args; ++a) read_uleb128(ptr, unit_end);
                        }
                        break;
                    }
                }
            } else {
                uint8_t adj_opcode = opcode - opcode_base;
                address += (adj_opcode / line_range) * min_insn_len;
                line += line_base + (adj_opcode % line_range);
                add_entry();
                basic_block = false; prologue_end = false; epilogue_begin = false; discriminator = 0;
            }
        }
        ptr = unit_end;
    }

    std::sort(entries_.begin(), entries_.end(), [](const LineEntry& a, const LineEntry& b) {
        return a.address < b.address;
    });

    return !entries_.empty();
}

bool DwarfParser::get_source_location(uint64_t address, std::string& out_file, uint32_t& out_line) const {
    if (entries_.empty()) return false;
    auto it = std::upper_bound(entries_.begin(), entries_.end(), address,
                               [](uint64_t addr, const LineEntry& entry) {
                                   return addr < entry.address;
                               });
    if (it == entries_.begin()) return false;
    const auto& entry = *(it - 1);
    out_file = entry.file;
    out_line = entry.line;
    return true;
}

bool DwarfParser::get_address_from_location(const std::string& file, uint32_t line, uint64_t& out_address) const {
    if (entries_.empty()) return false;
    
    // We want to find the first address mapping to the specified file and line.
    // The input file might be a relative path or basename, so we perform robust matching.
    std::string target = file;
    // Normalize target separators to forward slashes for matching
    std::replace(target.begin(), target.end(), '\\', '/');
    
    for (const auto& entry : entries_) {
        if (entry.line == line) {
            std::string entry_file = entry.file;
            std::replace(entry_file.begin(), entry_file.end(), '\\', '/');
            
            // Exact match
            if (entry_file == target) {
                out_address = entry.address;
                return true;
            }
            
            // Suffix match (e.g. entry is /absolute/path/file.hip, target is file.hip)
            if (entry_file.size() >= target.size() + 1) {
                if (entry_file.compare(entry_file.size() - target.size(), target.size(), target) == 0 &&
                    entry_file[entry_file.size() - target.size() - 1] == '/') {
                    out_address = entry.address;
                    return true;
                }
            }
            
            // Reverse suffix match (e.g. entry is tools/vector_add/vector_add.hip, target is /absolute/path/tools/vector_add/vector_add.hip)
            if (target.size() >= entry_file.size() + 1) {
                if (target.compare(target.size() - entry_file.size(), entry_file.size(), entry_file) == 0 &&
                    target[target.size() - entry_file.size() - 1] == '/') {
                    out_address = entry.address;
                    return true;
                }
            }
            
            // Also handle simple case if target is just a filename and entry_file has it
            size_t last_slash = entry_file.find_last_of('/');
            std::string entry_basename = (last_slash == std::string::npos) ? entry_file : entry_file.substr(last_slash + 1);
            size_t target_slash = target.find_last_of('/');
            std::string target_basename = (target_slash == std::string::npos) ? target : target.substr(target_slash + 1);
            if (entry_basename == target_basename) {
                out_address = entry.address;
                return true;
            }
        }
    }
    return false;
}

} // namespace gcnsim
