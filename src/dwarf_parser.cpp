#include "gcnsim/dwarf_parser.h"
#include <elf.h>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>

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
    if (elf_data.size() < sizeof(Elf64_Ehdr)) {
        return false;
    }
    const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(elf_data.data());
    if (ehdr->e_ident[0] != 0x7f || ehdr->e_ident[1] != 'E' ||
        ehdr->e_ident[2] != 'L' || ehdr->e_ident[3] != 'F') {
        return false;
    }

    uint64_t shoff = ehdr->e_shoff;
    uint16_t shentsize = ehdr->e_shentsize;
    uint16_t shnum = ehdr->e_shnum;
    uint16_t shstrndx = ehdr->e_shstrndx;

    if (shoff == 0 || shnum == 0 || shoff + (uint64_t)shentsize * shnum > elf_data.size()) {
        return false;
    }

    auto get_shdr = [&](int idx) -> const Elf64_Shdr* {
        if (idx < 0 || idx >= shnum) return nullptr;
        return reinterpret_cast<const Elf64_Shdr*>(elf_data.data() + shoff + idx * shentsize);
    };

    const Elf64_Shdr* shstrtab_shdr = get_shdr(shstrndx);
    if (!shstrtab_shdr || shstrtab_shdr->sh_offset + shstrtab_shdr->sh_size > elf_data.size()) {
        return false;
    }
    const char* shstrtab = reinterpret_cast<const char*>(elf_data.data() + shstrtab_shdr->sh_offset);

    // Find .debug_line section
    const Elf64_Shdr* debug_line_shdr = nullptr;
    for (int i = 0; i < shnum; ++i) {
        const Elf64_Shdr* sh = get_shdr(i);
        if (sh && sh->sh_name < shstrtab_shdr->sh_size) {
            std::string sname(shstrtab + sh->sh_name);
            if (sname == ".debug_line") {
                debug_line_shdr = sh;
                break;
            }
        }
    }

    if (!debug_line_shdr || debug_line_shdr->sh_offset + debug_line_shdr->sh_size > elf_data.size()) {
        return false; // No debug line info
    }

    const uint8_t* debug_line_ptr = elf_data.data() + debug_line_shdr->sh_offset;
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
        if (ptr + unit_length > debug_line_end) {
            break;
        }
        const uint8_t* unit_end = ptr + unit_length;

        uint16_t version = read_uint16(ptr);
        uint64_t header_length = is_dwarf64 ? read_uint64(ptr) : read_uint32(ptr);
        const uint8_t* header_end = ptr + header_length;

        uint8_t min_insn_len = *ptr++;
        uint8_t max_ops_per_insn = 1;
        if (version >= 4) {
            max_ops_per_insn = *ptr++;
        }
        uint8_t default_is_stmt = *ptr++;
        int8_t line_base = (int8_t)*ptr++;
        uint8_t line_range = *ptr++;
        uint8_t opcode_base = *ptr++;

        std::vector<uint8_t> std_opcode_lengths(opcode_base - 1);
        for (int i = 1; i < opcode_base; ++i) {
            std_opcode_lengths[i - 1] = *ptr++;
        }

        // include directories
        std::vector<std::string> include_dirs;
        while (ptr < header_end && *ptr != 0) {
            std::string dir((const char*)ptr);
            ptr += dir.size() + 1;
            include_dirs.push_back(dir);
        }
        if (ptr < header_end) ptr++;

        // file names
        struct FileEntry {
            std::string name;
            uint64_t dir_idx;
        };
        std::vector<FileEntry> files;
        while (ptr < header_end && *ptr != 0) {
            std::string fname((const char*)ptr);
            ptr += fname.size() + 1;
            uint64_t dir_idx = read_uleb128(ptr, header_end);
            read_uleb128(ptr, header_end); // skip mod time
            read_uleb128(ptr, header_end); // skip size
            files.push_back({fname, dir_idx});
        }
        if (ptr < header_end) ptr++;

        // Line program loop
        ptr = header_end;
        uint64_t address = 0;
        uint64_t file = 1;
        uint64_t line = 1;
        uint64_t column = 0;
        bool is_stmt = (default_is_stmt != 0);
        bool basic_block = false;
        bool prologue_end = false;
        bool epilogue_begin = false;
        uint64_t isa = 0;
        uint64_t discriminator = 0;

        while (ptr < unit_end) {
            uint8_t opcode = *ptr++;
            if (opcode == 0) {
                // Extended opcode
                uint64_t ext_len = read_uleb128(ptr, unit_end);
                const uint8_t* ext_end = ptr + ext_len;
                uint8_t ext_opcode = *ptr++;
                if (ext_opcode == DW_LNE_end_sequence) {
                    if (address != 0) {
                        std::string fpath = "";
                        if (file > 0 && file <= files.size()) {
                            fpath = files[file - 1].name;
                        }
                        entries_.push_back({address, fpath, (uint32_t)line});
                    }
                    address = 0;
                    file = 1;
                    line = 1;
                    column = 0;
                    is_stmt = (default_is_stmt != 0);
                    basic_block = false;
                    prologue_end = false;
                    epilogue_begin = false;
                    isa = 0;
                    discriminator = 0;
                } else if (ext_opcode == DW_LNE_set_address) {
                    uint64_t addr = 0;
                    for (int i = 0; i < 8; ++i) {
                        addr |= uint64_t(*ptr++) << (i * 8);
                    }
                    address = addr;
                } else if (ext_opcode == DW_LNE_define_file) {
                    std::string fname((const char*)ptr);
                    ptr += fname.size() + 1;
                    uint64_t dir_idx = read_uleb128(ptr, ext_end);
                    read_uleb128(ptr, ext_end);
                    read_uleb128(ptr, ext_end);
                    files.push_back({fname, dir_idx});
                } else if (ext_opcode == DW_LNE_set_discriminator) {
                    discriminator = read_uleb128(ptr, ext_end);
                } else {
                    ptr = ext_end;
                }
                ptr = ext_end;
            } else if (opcode < opcode_base) {
                // Standard opcode
                switch (opcode) {
                    case DW_LNS_copy:
                        if (address != 0) {
                            std::string fpath = "";
                            if (file > 0 && file <= files.size()) {
                                fpath = files[file - 1].name;
                            }
                            entries_.push_back({address, fpath, (uint32_t)line});
                        }
                        basic_block = false;
                        prologue_end = false;
                        epilogue_begin = false;
                        discriminator = 0;
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
                    case DW_LNS_fixed_advance_pc: {
                        uint16_t adv = ptr[0] | (ptr[1] << 8);
                        ptr += 2;
                        address += adv;
                        break;
                    }
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
                        int num_args = std_opcode_lengths[opcode - 1];
                        for (int a = 0; a < num_args; ++a) {
                            read_uleb128(ptr, unit_end);
                        }
                        break;
                    }
                }
            } else {
                // Special opcode
                uint8_t adj_opcode = opcode - opcode_base;
                int line_inc = line_base + (adj_opcode % line_range);
                int pc_inc = (adj_opcode / line_range) * min_insn_len;
                line += line_inc;
                address += pc_inc;

                if (address != 0) {
                    std::string fpath = "";
                    if (file > 0 && file <= files.size()) {
                        fpath = files[file - 1].name;
                    }
                    entries_.push_back({address, fpath, (uint32_t)line});
                }
                basic_block = false;
                prologue_end = false;
                epilogue_begin = false;
                discriminator = 0;
            }
        }
        ptr = unit_end;
    }

    // Sort entries by address for quick binary searching
    std::sort(entries_.begin(), entries_.end(), [](const LineEntry& a, const LineEntry& b) {
        return a.address < b.address;
    });

    return !entries_.empty();
}

bool DwarfParser::get_source_location(uint64_t address, std::string& out_file, uint32_t& out_line) const {
    if (entries_.empty()) return false;

    // Binary search for the closest entry with address <= requested_address
    auto it = std::upper_bound(entries_.begin(), entries_.end(), address,
                               [](uint64_t addr, const LineEntry& entry) {
                                   return addr < entry.address;
                               });

    if (it == entries_.begin()) {
        // address is before the first entry
        return false;
    }

    // The upper_bound returns the first entry strictly greater than the address,
    // so the matching entry is the one right before it.
    const auto& entry = *(it - 1);
    out_file = entry.file;
    out_line = entry.line;
    return true;
}

} // namespace gcnsim
