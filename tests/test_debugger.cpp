#include <gtest/gtest.h>
#include "gcnsim/gdb_mi_parser.h"
#include "gcnsim/dwarf_parser.h"

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

} // namespace gcnsim
