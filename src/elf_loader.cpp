#include "gcnsim/elf_loader.h"
#include "gcnsim/kernel_descriptor.h"

#include <elf.h>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <optional>
#include <algorithm>
#include <cstdint>

// Pull in AMD-specific constants without including the full AMD headers in a way that
// would conflict with system elf.h.
#ifndef EM_AMDGPU
#define EM_AMDGPU 224
#endif
#ifndef ELFOSABI_AMDGPU_HSA
#define ELFOSABI_AMDGPU_HSA 64
#endif
// STT_LOOS is defined in elf.h (10)
#ifndef STT_AMDGPU_HSA_KERNEL
#define STT_AMDGPU_HSA_KERNEL (STT_LOOS + 0)  // = 10
#endif

// AMDGPU metadata note type (COV3/V4 MsgPack metadata in .note section)
#ifndef NT_AMDGPU_METADATA
#define NT_AMDGPU_METADATA 32
#endif

namespace gcnsim {

// Forward declaration from metadata_parser.cpp
std::vector<KernelArgInfo> parse_metadata_note(const uint8_t* note_data, size_t note_size,
                                                const std::string& kernel_name);


std::optional<HsacoModule> load_hsaco(const std::string& path, std::string& error_msg) {
    // Read file
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error_msg = "Cannot open file: " + path;
        return std::nullopt;
    }
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz <= 0) {
        error_msg = "Empty or unreadable file: " + path;
        return std::nullopt;
    }

    HsacoModule mod;
    mod.elf_data.resize((size_t)sz);
    f.read(reinterpret_cast<char*>(mod.elf_data.data()), sz);
    if (!f) {
        error_msg = "Read error: " + path;
        return std::nullopt;
    }

    const uint8_t* data = mod.elf_data.data();
    const size_t   data_size = mod.elf_data.size();

    // Validate ELF magic
    if (data_size < sizeof(Elf64_Ehdr)) {
        error_msg = "File too small to be ELF";
        return std::nullopt;
    }
    if (data[0] != 0x7f || data[1] != 'E' || data[2] != 'L' || data[3] != 'F') {
        error_msg = "Not an ELF file";
        return std::nullopt;
    }
    if (data[EI_CLASS] != ELFCLASS64) {
        error_msg = "Not a 64-bit ELF";
        return std::nullopt;
    }

    const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(data);

    // Validate AMDGPU
    if (ehdr->e_machine != EM_AMDGPU) {
        error_msg = "Not an AMDGPU ELF (e_machine != EM_AMDGPU)";
        return std::nullopt;
    }
    if (data[EI_OSABI] != ELFOSABI_AMDGPU_HSA) {
        error_msg = "Not an AMDGPU HSA ELF";
        return std::nullopt;
    }

    // Determine COV version from EI_ABIVERSION
    // ELFABIVERSION_AMDGPU_HSA_V2=0, V3=1, V4=2, V5=3
    int abi_ver = data[EI_ABIVERSION];
    if (abi_ver == 0)      mod.cov_version = 2;  // COV2 (legacy)
    else if (abi_ver == 1) mod.cov_version = 3;
    else if (abi_ver == 2) mod.cov_version = 4;
    else if (abi_ver == 3) mod.cov_version = 5;
    else if (abi_ver == 4) mod.cov_version = 6;
    else                   mod.cov_version = abi_ver + 2;

    // Get section header table
    uint64_t shoff = ehdr->e_shoff;
    uint16_t shentsize = ehdr->e_shentsize;
    uint16_t shnum = ehdr->e_shnum;
    uint16_t shstrndx = ehdr->e_shstrndx;

    if (shoff == 0 || shnum == 0) {
        error_msg = "No section headers";
        return std::nullopt;
    }
    if (shoff + (uint64_t)shentsize * shnum > data_size) {
        error_msg = "Section header table out of bounds";
        return std::nullopt;
    }

    auto get_shdr = [&](int idx) -> const Elf64_Shdr* {
        if (idx < 0 || idx >= shnum) return nullptr;
        return reinterpret_cast<const Elf64_Shdr*>(data + shoff + idx * shentsize);
    };

    auto get_section_data = [&](const Elf64_Shdr* shdr) -> const uint8_t* {
        if (!shdr || shdr->sh_type == SHT_NOBITS || shdr->sh_offset + shdr->sh_size > data_size)
            return nullptr;
        return data + shdr->sh_offset;
    };

    // Load section name string table
    const Elf64_Shdr* shstrtab_shdr = get_shdr(shstrndx);
    const char* shstrtab = shstrtab_shdr ?
        reinterpret_cast<const char*>(get_section_data(shstrtab_shdr)) : nullptr;

    auto get_section_name = [&](const Elf64_Shdr* shdr) -> std::string {
        if (!shstrtab || !shdr) return "";
        uint32_t name_off = shdr->sh_name;
        if (shstrtab_shdr && name_off < shstrtab_shdr->sh_size)
            return std::string(shstrtab + name_off);
        return "";
    };

    // Find relevant sections
    const Elf64_Shdr* symtab_shdr  = nullptr;
    const Elf64_Shdr* dynsym_shdr  = nullptr;
    const Elf64_Shdr* strtab_shdr  = nullptr;
    const Elf64_Shdr* dynstr_shdr  = nullptr;
    const Elf64_Shdr* note_shdr    = nullptr;

    for (int i = 0; i < shnum; i++) {
        const Elf64_Shdr* shdr = get_shdr(i);
        if (!shdr) continue;
        std::string name = get_section_name(shdr);

        if (name == ".symtab")       symtab_shdr  = shdr;
        else if (name == ".dynsym")  dynsym_shdr  = shdr;
        else if (name == ".strtab")  strtab_shdr  = shdr;
        else if (name == ".dynstr")  dynstr_shdr  = shdr;
        else if (shdr->sh_type == SHT_NOTE) note_shdr = shdr;
    }

    // Prefer .dynsym over .symtab for kernel symbols (typical in HSACO)
    const Elf64_Shdr* sym_shdr = dynsym_shdr ? dynsym_shdr : symtab_shdr;
    const Elf64_Shdr* str_shdr = dynstr_shdr ? dynstr_shdr : strtab_shdr;

    if (!sym_shdr) {
        error_msg = "No symbol table found";
        return std::nullopt;
    }

    const uint8_t* sym_data = get_section_data(sym_shdr);
    const char*    str_data = str_shdr ?
        reinterpret_cast<const char*>(get_section_data(str_shdr)) : nullptr;

    if (!sym_data) {
        error_msg = "Symbol table data not accessible";
        return std::nullopt;
    }

    // Parse metadata from .note section for arg info
    std::vector<uint8_t> note_buf;
    if (note_shdr) {
        const uint8_t* nd = get_section_data(note_shdr);
        if (nd) {
            note_buf.assign(nd, nd + note_shdr->sh_size);
        }
    }

    // Walk symbols, find kernel descriptors
    size_t sym_count = sym_shdr->sh_size / sizeof(Elf64_Sym);
    const Elf64_Sym* syms = reinterpret_cast<const Elf64_Sym*>(sym_data);

    for (size_t i = 0; i < sym_count; i++) {
        const Elf64_Sym& sym = syms[i];

        // Get symbol name
        std::string sym_name;
        size_t str_size = str_shdr ? str_shdr->sh_size : 0;
        if (str_data && sym.st_name < str_size) {
            sym_name = std::string(str_data + sym.st_name);
        }

        // Kernel descriptor symbols end with ".kd"
        // In COV3/COV4 the .kd symbol may have type STT_AMDGPU_HSA_KERNEL (10) or STT_OBJECT (1)
        // So we search by name suffix, not type.
        if (sym_name.size() < 3) continue;
        if (sym_name.substr(sym_name.size() - 3) != ".kd") continue;
        if (sym.st_shndx == SHN_UNDEF) continue;

        std::string kernel_name = sym_name.substr(0, sym_name.size() - 3);

        // Find the section this symbol lives in
        uint16_t shndx = sym.st_shndx;
        if (shndx == SHN_UNDEF || shndx >= shnum) continue;

        const Elf64_Shdr* sym_sec = get_shdr(shndx);
        if (!sym_sec) continue;
        const uint8_t* sec_data = get_section_data(sym_sec);
        if (!sec_data) continue;

        // Symbol value is VMA; offset within section = sym.st_value - sym_sec->sh_addr
        if (sym.st_value < sym_sec->sh_addr) continue;
        uint64_t sym_offset = sym.st_value - sym_sec->sh_addr;
        if (sym_offset + sizeof(KernelDescriptor) > sym_sec->sh_size) continue;

        const KernelDescriptor* kd = reinterpret_cast<const KernelDescriptor*>(sec_data + sym_offset);

        // Compute code_ptr: kd VMA + kernel_code_entry_byte_offset gives code VMA
        // Then find the section that contains that VMA and convert to file offset
        int64_t entry_offset = kd->kernel_code_entry_byte_offset;

        // kd VMA = sym.st_value
        uint64_t kd_vma = sym.st_value;
        uint64_t code_vma = (uint64_t)((int64_t)kd_vma + entry_offset);

        // Find which section contains code_vma and get file offset
        const uint8_t* code_ptr = nullptr;
        size_t         code_size_from_sec = 0;
        for (int si = 0; si < shnum; si++) {
            const Elf64_Shdr* sh = get_shdr(si);
            if (!sh || sh->sh_type == SHT_NOBITS || sh->sh_size == 0) continue;
            if (sh->sh_addr == 0) continue;
            uint64_t sec_vma_start = sh->sh_addr;
            uint64_t sec_vma_end   = sh->sh_addr + sh->sh_size;
            if (code_vma >= sec_vma_start && code_vma < sec_vma_end) {
                uint64_t code_file_off = sh->sh_offset + (code_vma - sec_vma_start);
                if (code_file_off < data_size) {
                    code_ptr = data + code_file_off;
                    code_size_from_sec = (size_t)(sec_vma_end - code_vma);
                }
                break;
            }
        }
        if (!code_ptr) continue;

        size_t code_size = code_size_from_sec;

        // Parse arg info from metadata
        std::vector<KernelArgInfo> args;
        if (!note_buf.empty()) {
            args = parse_metadata_note(note_buf.data(), note_buf.size(), kernel_name);
        }

        KernelInfo ki;
        ki.name       = kernel_name;
        ki.descriptor = kd;
        ki.code_ptr   = code_ptr;
        ki.code_size  = code_size;
        ki.args       = std::move(args);

        mod.kernels.push_back(std::move(ki));
    }

    return mod;
}

} // namespace gcnsim
