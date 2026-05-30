#include <gtest/gtest.h>
#include "gcnsim/gdb_mi_parser.h"
#include "gcnsim/dwarf_parser.h"
#include "gcnsim/debugger_core.h"
#include "gcnsim/elf_loader.h"
#include <fstream>
#include <vector>
#include <cstring>

namespace gcnsim {

TEST(GdbMiParserTest, SimpleMiCommand) {
    auto cmd = parse_gdb_mi("123-exec-run");
    EXPECT_EQ(cmd.token, "123");
    EXPECT_EQ(cmd.command, "-exec-run");
    EXPECT_TRUE(cmd.is_mi);
    EXPECT_TRUE(cmd.args.empty());
}

TEST(GdbMiParserTest, MiCommandWithArgs) {
    auto cmd = parse_gdb_mi("4-break-insert -f \"vector_add.hip:24\"");
    EXPECT_EQ(cmd.token, "4");
    EXPECT_EQ(cmd.command, "-break-insert");
    EXPECT_TRUE(cmd.is_mi);
    ASSERT_EQ(cmd.args.size(), 2);
    EXPECT_EQ(cmd.args[0], "-f");
    EXPECT_EQ(cmd.args[1], "vector_add.hip:24");
}

TEST(GdbMiParserTest, CliCommand) {
    auto cmd = parse_gdb_mi("info registers");
    EXPECT_EQ(cmd.token, "");
    EXPECT_EQ(cmd.command, "info");
    EXPECT_FALSE(cmd.is_mi);
    ASSERT_EQ(cmd.args.size(), 1);
    EXPECT_EQ(cmd.args[0], "registers");
}

TEST(GdbMiParserTest, CliCommandWithToken) {
    auto cmd = parse_gdb_mi("99print $s0");
    EXPECT_EQ(cmd.token, "99");
    EXPECT_EQ(cmd.command, "print");
    EXPECT_FALSE(cmd.is_mi);
    ASSERT_EQ(cmd.args.size(), 1);
    EXPECT_EQ(cmd.args[0], "$s0");
}

TEST(DwarfParserTest, EmptyParser) {
    DwarfParser parser;
    std::string file;
    uint32_t line = 0;
    EXPECT_FALSE(parser.get_source_location(0x1000, file, line));
}

TEST(DwarfParserTest, AddressFromLocationMapping) {
    // Try to load the vector_add hsaco from build/tools/vector_add
    std::string path = "build/tools/vector_add/vector_add.hsaco";
    std::string err;
    auto hsaco = load_hsaco(path, err);
    if (!hsaco) {
        // Fallback if running from a different cwd
        path = "../build/tools/vector_add/vector_add.hsaco";
        hsaco = load_hsaco(path, err);
    }
    
    if (hsaco) {
        DwarfParser parser;
        ASSERT_TRUE(parser.load_from_elf(hsaco->elf_data));
        
        // Find a valid entry to test mapping in both directions
        const auto& entries = parser.get_entries();
        if (!entries.empty()) {
            const auto& test_entry = entries[0];
            uint64_t addr = 0;
            // Test exact match
            EXPECT_TRUE(parser.get_address_from_location(test_entry.file, test_entry.line, addr));
            EXPECT_EQ(addr, test_entry.address);
            
            // Test basename match
            size_t slash = test_entry.file.find_last_of("/\\");
            std::string basename = (slash == std::string::npos) ? test_entry.file : test_entry.file.substr(slash + 1);
            EXPECT_TRUE(parser.get_address_from_location(basename, test_entry.line, addr));
            EXPECT_EQ(addr, test_entry.address);

            // Test reverse suffix match (query has extra absolute prefix)
            std::string abs_query = "/absolute/prefix/path/to/project/" + test_entry.file;
            EXPECT_TRUE(parser.get_address_from_location(abs_query, test_entry.line, addr));
            EXPECT_EQ(addr, test_entry.address);
        }
    }
}

TEST(DebuggerTest, ExpressionEvaluation) {
    Debugger db;
    WavefrontState wave;
    std::memset(&wave, 0, sizeof(WavefrontState));
    
    // Initialize registers with distinct values
    wave.sgpr[0] = 0xdeadbeef;
    wave.sgpr[1] = 12345;
    wave.vgpr[0][0] = 0xabcdef01;
    wave.vgpr[0][5] = 0x11223344;
    wave.agpr[2][0] = 0x778899aa;
    wave.agpr[2][10] = 0x55667788;
    
    wave.exec = 0x0f0f0f0f0f0f0f0fULL;
    wave.vcc = 0x123456789abcdef0ULL;
    wave.scc = 1;
    wave.m0 = 0x200;
    wave.pc = 0x40;
    
    db.active_lane = 0;
    db.kernel.code_vma = 0x1000;
    
    // Test scalar registers
    EXPECT_EQ(db.evaluate_var_expr(wave, "$s0"), "0xdeadbeef");
    EXPECT_EQ(db.evaluate_var_expr(wave, "s1"), "0x3039"); // 12345 in hex
    
    // Test vector GPRs (default lane, specific lane, all lanes)
    EXPECT_EQ(db.evaluate_var_expr(wave, "$v0"), "0xabcdef01");
    EXPECT_EQ(db.evaluate_var_expr(wave, "$v0[5]"), "0x11223344");
    
    db.active_lane = 5;
    EXPECT_EQ(db.evaluate_var_expr(wave, "$v0"), "0x11223344");
    
    // Test accumulator GPRs
    db.active_lane = 0;
    EXPECT_EQ(db.evaluate_var_expr(wave, "$a2"), "0x778899aa");
    EXPECT_EQ(db.evaluate_var_expr(wave, "$a2[10]"), "0x55667788");
    
    // Test special registers
    EXPECT_EQ(db.evaluate_var_expr(wave, "$pc"), "0x1040"); // absolute PC
    EXPECT_EQ(db.evaluate_var_expr(wave, "$pc_offset"), "0x40");
    EXPECT_EQ(db.evaluate_var_expr(wave, "$exec"), "0xf0f0f0f0f0f0f0f");
    EXPECT_EQ(db.evaluate_var_expr(wave, "$vcc"), "0x123456789abcdef0");
    EXPECT_EQ(db.evaluate_var_expr(wave, "$scc"), "1");
    EXPECT_EQ(db.evaluate_var_expr(wave, "$m0"), "0x200");
    EXPECT_EQ(db.evaluate_var_expr(wave, "$lane"), "0");
    
    // Test kernel arguments
    KernelArgInfo arg1;
    arg1.name = "N";
    arg1.type_name = "int";
    arg1.size = 4;
    arg1.offset = 0;
    arg1.value_kind = "by_value";
    
    KernelArgInfo arg2;
    arg2.name = "val";
    arg2.type_name = "float";
    arg2.size = 4;
    arg2.offset = 4;
    arg2.value_kind = "by_value";
    
    db.kernel.args = { arg1, arg2 };
    
    uint8_t kernarg_buf[8];
    int32_t n_val = 256;
    float val_val = 3.14f;
    std::memcpy(kernarg_buf, &n_val, 4);
    std::memcpy(kernarg_buf + 4, &val_val, 4);
    wave.kernarg_ptr = kernarg_buf;
    
    EXPECT_EQ(db.evaluate_var_expr(wave, "N"), "256");
    // std::to_string(3.14f) might have some trailing decimals depending on platform,
    // so we just check it starts with "3.14"
    std::string val_str = db.evaluate_var_expr(wave, "val");
    EXPECT_EQ(val_str.substr(0, 4), "3.14");
}

} // namespace gcnsim
