#pragma once
#include "gcnsim/elf_loader.h"
#include "gcnsim/execution_engine.h"
#include "gcnsim/dwarf_parser.h"
#include "gcnsim/gdb_mi_parser.h"
#include "gcnsim/wavefront.h"
#include <string>
#include <vector>
#include <map>

namespace gcnsim {

struct Breakpoint {
    int id;
    uint64_t address;
    std::string file;
    uint32_t line;
    std::string func;
};

class Debugger {
public:
    HsacoModule module;
    KernelInfo kernel;
    LaunchConfig cfg;
    DwarfParser dwarf;
    std::string hsaco_path;

    bool mi_mode = false;
    bool running = false;
    bool exited = false;

    std::vector<Breakpoint> breakpoints;
    int next_bkpt_id = 1;

    // List of register names for GDB/MI index mapping
    const std::vector<std::string> reg_names;

    // Active wavefronts for the current workgroup
    std::vector<WavefrontState*> waves;
    uint32_t active_wave_idx = 0;
    uint32_t active_lane = 0;

    std::vector<std::string> source_dirs;
    std::string cwd;

    bool single_stepping = false;
    bool step_requested = false;
    bool skipping_headers = false;

    // Line stepping state
    bool stepping_line = false;
    uint32_t step_line = 0;
    std::string step_file;

    // Variable objects map
    std::map<std::string, std::string> var_objects;

    int output_fd = 1; // default to stdout
    FILE* mi_log_file = nullptr;

    Debugger();
    
    void set_output_fd(int fd) { output_fd = fd; }
    void mi_printf(const char* fmt, ...);

    // Core debugger loop for integration
    void debug_hook_impl(WavefrontState& wave);
    void handle_command(const GdbMiCommand& cmd);
    void handle_cli_command(const std::string& cli_cmd);
    void run_repl();
    void run_mi();

    std::string serialize_state() const;
    void deserialize_state(const std::string& state);

    // Helper functions
    void print_mi_breakpoint(const std::string& token, const Breakpoint& bp, bool is_notification);
    std::string evaluate_expr(const WavefrontState& wave, const std::string& expr);
    std::string evaluate_var_expr(const WavefrontState& wave, const std::string& expr) const;
    std::string get_frame_info(const WavefrontState& wave, const std::string& func_name);
    int get_thread_id(const WavefrontState& wave);
    void print_source_context(const std::string& filepath, uint32_t line_num);
    uint64_t evaluate_address(const std::string& expr);

private:
    static std::vector<std::string> init_reg_names();
};

// Global accessor for hooks
void set_global_debugger(Debugger* db);
void global_debug_hook(WavefrontState& wave);

} // namespace gcnsim
