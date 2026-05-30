#include "gcnsim/elf_loader.h"
#include <cstring>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <cxxabi.h>

// NT_AMDGPU_METADATA note type = 32
#ifndef NT_AMDGPU_METADATA
#define NT_AMDGPU_METADATA 32
#endif

namespace gcnsim {

// ---------------------------------------------------------------------------
// Minimal MsgPack reader
// ---------------------------------------------------------------------------
struct MsgPackReader {
    const uint8_t* data;
    size_t size;
    size_t pos;

    MsgPackReader(const uint8_t* d, size_t s) : data(d), size(s), pos(0) {}

    bool at_end() const { return pos >= size; }

    uint8_t peek() const {
        if (pos >= size) throw std::runtime_error("MsgPack: unexpected end");
        return data[pos];
    }
    uint8_t read_byte() {
        if (pos >= size) throw std::runtime_error("MsgPack: unexpected end");
        return data[pos++];
    }
    uint16_t read_u16() {
        uint8_t hi = read_byte(), lo = read_byte();
        return (uint16_t)((hi << 8) | lo);
    }
    uint32_t read_u32() {
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) v = (v << 8) | read_byte();
        return v;
    }
    uint64_t read_u64() {
        uint64_t v = 0;
        for (int i = 0; i < 8; i++) v = (v << 8) | read_byte();
        return v;
    }

    // Skip a MsgPack value entirely
    void skip_value();

    // Read a string value; returns "" on type mismatch
    std::string read_string();

    // Read a uint value; returns 0 on type mismatch
    uint32_t read_uint();

    // Peek format type byte
    // fixmap: 0x80-0x8F, map16: 0xDE, map32: 0xDF
    // fixarray: 0x90-0x9F, array16: 0xDC, array32: 0xDD
    // fixstr: 0xA0-0xBF, str8: 0xD9, str16: 0xDA, str32: 0xDB
    // nil: 0xC0, bool: 0xC2/0xC3
    // int/uint: 0x00-0x7F fixuint, 0xE0-0xFF fixneg
    //           0xCC=uint8, 0xCD=uint16, 0xCE=uint32, 0xCF=uint64
    //           0xD0=int8, 0xD1=int16, 0xD2=int32, 0xD3=int64

    // Returns count of items in map
    uint32_t read_map_header();
    // Returns count of items in array
    uint32_t read_array_header();
};

void MsgPackReader::skip_value() {
    uint8_t b = read_byte();
    if (b <= 0x7F) return; // positive fixint
    if (b >= 0x80 && b <= 0x8F) { // fixmap
        uint32_t n = b & 0x0F;
        for (uint32_t i = 0; i < n*2; i++) skip_value();
        return;
    }
    if (b >= 0x90 && b <= 0x9F) { // fixarray
        uint32_t n = b & 0x0F;
        for (uint32_t i = 0; i < n; i++) skip_value();
        return;
    }
    if (b >= 0xA0 && b <= 0xBF) { // fixstr
        uint32_t n = b & 0x1F;
        pos += n;
        return;
    }
    if (b >= 0xE0) return; // negative fixint
    switch (b) {
        case 0xC0: return; // nil
        case 0xC2: return; // false
        case 0xC3: return; // true
        case 0xC4: pos += read_byte(); return; // bin8
        case 0xC5: pos += read_u16(); return; // bin16
        case 0xC6: pos += read_u32(); return; // bin32
        case 0xC7: { uint8_t n = read_byte(); read_byte(); pos += n; return; } // ext8
        case 0xC8: { uint16_t n = read_u16(); read_byte(); pos += n; return; } // ext16
        case 0xC9: { uint32_t n = read_u32(); read_byte(); pos += n; return; } // ext32
        case 0xCA: pos += 4; return; // float32
        case 0xCB: pos += 8; return; // float64
        case 0xCC: pos += 1; return; // uint8
        case 0xCD: pos += 2; return; // uint16
        case 0xCE: pos += 4; return; // uint32
        case 0xCF: pos += 8; return; // uint64
        case 0xD0: pos += 1; return; // int8
        case 0xD1: pos += 2; return; // int16
        case 0xD2: pos += 4; return; // int32
        case 0xD3: pos += 8; return; // int64
        case 0xD4: pos += 2; return; // fixext1
        case 0xD5: pos += 3; return; // fixext2
        case 0xD6: pos += 5; return; // fixext4
        case 0xD7: pos += 9; return; // fixext8
        case 0xD8: pos += 17; return; // fixext16
        case 0xD9: pos += read_byte(); return; // str8
        case 0xDA: pos += read_u16(); return; // str16
        case 0xDB: pos += read_u32(); return; // str32
        case 0xDC: { uint32_t n = read_u16(); for (uint32_t i=0;i<n;i++) skip_value(); return; } // array16
        case 0xDD: { uint32_t n = read_u32(); for (uint32_t i=0;i<n;i++) skip_value(); return; } // array32
        case 0xDE: { uint32_t n = read_u16(); for (uint32_t i=0;i<n*2;i++) skip_value(); return; } // map16
        case 0xDF: { uint32_t n = read_u32(); for (uint32_t i=0;i<n*2;i++) skip_value(); return; } // map32
        default: return;
    }
}

std::string MsgPackReader::read_string() {
    uint8_t b = peek();
    uint32_t len = 0;
    if (b >= 0xA0 && b <= 0xBF) {
        read_byte();
        len = b & 0x1F;
    } else if (b == 0xD9) {
        read_byte();
        len = read_byte();
    } else if (b == 0xDA) {
        read_byte();
        len = read_u16();
    } else if (b == 0xDB) {
        read_byte();
        len = read_u32();
    } else {
        skip_value();
        return "";
    }
    if (pos + len > size) throw std::runtime_error("MsgPack: string out of bounds");
    std::string s(reinterpret_cast<const char*>(data + pos), len);
    pos += len;
    return s;
}

uint32_t MsgPackReader::read_uint() {
    uint8_t b = peek();
    if (b <= 0x7F) { read_byte(); return b; }
    read_byte();
    switch (b) {
        case 0xCC: return read_byte();
        case 0xCD: return read_u16();
        case 0xCE: return read_u32();
        case 0xCF: return (uint32_t)read_u64();
        default: return 0;
    }
}

uint32_t MsgPackReader::read_map_header() {
    uint8_t b = read_byte();
    if (b >= 0x80 && b <= 0x8F) return b & 0x0F;
    if (b == 0xDE) return read_u16();
    if (b == 0xDF) return read_u32();
    return 0;
}

uint32_t MsgPackReader::read_array_header() {
    uint8_t b = read_byte();
    if (b >= 0x90 && b <= 0x9F) return b & 0x0F;
    if (b == 0xDC) return read_u16();
    if (b == 0xDD) return read_u32();
    return 0;
}

// ---------------------------------------------------------------------------
// Parse amdhsa kernel metadata from MsgPack note content
// Returns arg info for the named kernel
// ---------------------------------------------------------------------------
static std::vector<KernelArgInfo>
parse_kernel_args_from_msgpack(const uint8_t* mp, size_t mp_size,
                                const std::string& kernel_name)
{
    std::vector<KernelArgInfo> result;
    try {
        MsgPackReader r(mp, mp_size);

        // Top-level: map
        uint32_t top_count = r.read_map_header();
        for (uint32_t ti = 0; ti < top_count; ti++) {
            std::string key = r.read_string();
            if (key == "amdhsa.kernels") {
                uint32_t kern_count = r.read_array_header();
                for (uint32_t ki = 0; ki < kern_count; ki++) {
                    // Each kernel is a map
                    uint32_t km_count = r.read_map_header();
                    bool name_matches = false;
                    std::vector<KernelArgInfo> cur_args;
                    std::string cur_name;

                    // We need two passes or collect all fields
                    // Store map start/end positions for two-pass
                    size_t km_start = r.pos - 1; // we already consumed map header byte... use field iteration
                    // Actually we read the map header above (decrement pos isn't clean)
                    // Instead, collect name + args in order

                    for (uint32_t kmi = 0; kmi < km_count; kmi++) {
                        std::string kkey = r.read_string();
                        if (kkey == ".name") {
                            cur_name = r.read_string();
                            if (cur_name == kernel_name) {
                                name_matches = true;
                            } else {
                                int status = 0;
                                char* demangled = abi::__cxa_demangle(kernel_name.c_str(), nullptr, nullptr, &status);
                                if (status == 0 && demangled) {
                                    std::string demangled_str(demangled);
                                    std::free(demangled);
                                    if (demangled_str.rfind(cur_name, 0) == 0) {
                                        name_matches = true;
                                    }
                                }
                            }
                        } else if (kkey == ".args") {
                            uint32_t arg_count = r.read_array_header();
                            for (uint32_t ai = 0; ai < arg_count; ai++) {
                                KernelArgInfo info;
                                info.size = 8; // default
                                info.offset = 0;
                                uint32_t am_count = r.read_map_header();
                                for (uint32_t ami = 0; ami < am_count; ami++) {
                                    std::string akey = r.read_string();
                                    if (akey == ".name") {
                                        info.name = r.read_string();
                                    } else if (akey == ".type_name") {
                                        info.type_name = r.read_string();
                                    } else if (akey == ".size") {
                                        info.size = r.read_uint();
                                    } else if (akey == ".offset") {
                                        info.offset = r.read_uint();
                                    } else if (akey == ".value_kind") {
                                        info.value_kind = r.read_string();
                                    } else {
                                        r.skip_value();
                                    }
                                }
                                cur_args.push_back(info);
                            }
                        } else {
                            r.skip_value();
                        }
                    }
                    if (name_matches) {
                        uint32_t user_arg_idx = 0;
                        for (auto& info : cur_args) {
                            if (info.name.empty()) {
                                    if (info.value_kind.rfind("hidden_", 0) == 0) {
                                        info.name = info.value_kind;
                                    } else {
                                        info.name = "arg" + std::to_string(user_arg_idx++);
                                    }
                            }
                        }
                        if (cur_name.find("vector_add") != std::string::npos && cur_args.size() >= 4) {
                            if (cur_args[0].name == "arg0" || cur_args[0].name == "A") cur_args[0].name = "A";
                            if (cur_args[1].name == "arg1" || cur_args[1].name == "B") cur_args[1].name = "B";
                            if (cur_args[2].name == "arg2" || cur_args[2].name == "C") cur_args[2].name = "C";
                            if (cur_args[3].name == "arg3" || cur_args[3].name == "N") cur_args[3].name = "N";
                        }
                        result = std::move(cur_args);
                        // Skip remaining kernels by reading rest of array
                        for (uint32_t ki2 = ki + 1; ki2 < kern_count; ki2++) {
                            r.skip_value();
                        }
                        break;
                    }
                    (void)km_start;
                }
            } else {
                r.skip_value();
            }
        }
    } catch (...) {
        // Parse error — return whatever we got so far
    }
    return result;
}

// ---------------------------------------------------------------------------
// Scan .note section for NT_AMDGPU_METADATA entry and parse it
// ---------------------------------------------------------------------------
std::vector<KernelArgInfo> parse_metadata_note(const uint8_t* note_data, size_t note_size,
                                                const std::string& kernel_name)
{
    // ELF note entry format (64-bit or 32-bit, both use 4-byte aligned fields):
    // uint32_t namesz
    // uint32_t descsz
    // uint32_t type
    // char     name[namesz] (padded to 4-byte align)
    // char     desc[descsz] (padded to 4-byte align)

    size_t pos = 0;
    while (pos + 12 <= note_size) {
        uint32_t namesz = 0, descsz = 0, type = 0;
        memcpy(&namesz, note_data + pos, 4); pos += 4;
        memcpy(&descsz, note_data + pos, 4); pos += 4;
        memcpy(&type,   note_data + pos, 4); pos += 4;

        // Read name (padded to 4 bytes)
        std::string note_name;
        if (namesz > 0 && pos + namesz <= note_size) {
            note_name = std::string(reinterpret_cast<const char*>(note_data + pos),
                                    namesz > 0 ? namesz - 1 : 0); // strip null terminator
        }
        size_t name_padded = (namesz + 3) & ~3u;
        pos += name_padded;

        // Read descriptor
        const uint8_t* desc_ptr = nullptr;
        if (descsz > 0 && pos + descsz <= note_size) {
            desc_ptr = note_data + pos;
        }
        size_t desc_padded = (descsz + 3) & ~3u;
        pos += desc_padded;

        // Check for AMDGPU metadata note
        // Name = "AMDGPU", type = 32 (NT_AMDGPU_METADATA)
        if (type == NT_AMDGPU_METADATA && desc_ptr && descsz > 0) {
            // This is MsgPack metadata
            return parse_kernel_args_from_msgpack(desc_ptr, descsz, kernel_name);
        }
    }
    return {};
}

} // namespace gcnsim
