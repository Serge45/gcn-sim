#include "gcnsim/elf_loader.h"
#include "gcnsim/execution_engine.h"
#include "gcnsim/dwarf_parser.h"
#include "gcnsim/gdb_mi_parser.h"
#include "gcnsim/wavefront.h"
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

using namespace gcnsim;

#include "gcnsim/isa_decoder.h"

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

    bool mi_mode = false;
    bool running = false;
    bool exited = false;

    std::vector<Breakpoint> breakpoints;
    int next_bkpt_id = 1;

    // Active wavefronts for the current workgroup
    std::vector<WavefrontState*> waves;
    uint32_t active_wave_idx = 0;

    bool single_stepping = false;
    bool step_requested = false;

    // Command-line simulated variables
    std::vector<uint8_t> mock_a;
    std::vector<uint8_t> mock_b;
    std::vector<uint8_t> mock_c;

    void run_repl();
    void run_mi();
    void debug_hook_impl(WavefrontState& wave);
    void handle_command(const GdbMiCommand& cmd);
};

static Debugger* g_debugger = nullptr;

static void global_debug_hook(WavefrontState& wave) {
    if (g_debugger) {
        g_debugger->debug_hook_impl(wave);
    }
}

static bool addr_is_valid(uint64_t addr) {
    if (addr < 4096) return false;
    void* page = (void*)(addr & ~(uint64_t)4095);
    return msync(page, 1, MS_ASYNC) == 0;
}

std::string get_frame_info(const WavefrontState& wave, const DwarfParser& dwarf, const std::string& func_name) {
    std::string file;
    uint32_t line = 0;
    char pc_hex[32];
    sprintf(pc_hex, "0x%llx", (unsigned long long)wave.pc);

    std::string frame = "frame={addr=\"" + std::string(pc_hex) + "\",func=\"" + func_name + "\"";
    if (dwarf.get_source_location(wave.pc, file, line)) {
        frame += ",file=\"" + file + "\",fullname=\"" + file + "\",line=\"" + std::to_string(line) + "\"";
    }
    frame += "}";
    return frame;
}

int get_thread_id(const WavefrontState& wave) {
    return wave.trace_wave_id + 1;
}

std::string evaluate_expr(const WavefrontState& wave, const std::string& expr) {
    std::string name = expr;
    if (!name.empty() && name[0] == '$') {
        name = name.substr(1);
    }

    // Check SGPRs
    if (name[0] == 's') {
        if (name == "scc") {
            return std::to_string(wave.scc);
        }
        try {
            int idx = std::stoi(name.substr(1));
            if (idx >= 0 && idx < NUM_SGPR) {
                char val_str[32];
                sprintf(val_str, "0x%x", wave.sgpr[idx]);
                return std::string(val_str);
            }
        } catch (...) {}
    }

    // Check VGPRs
    if (name[0] == 'v') {
        if (name == "vcc") {
            char val_str[32];
            sprintf(val_str, "0x%llx", (unsigned long long)wave.vcc);
            return std::string(val_str);
        }
        try {
            int idx = std::stoi(name.substr(1));
            if (idx >= 0 && idx < NUM_VGPR) {
                std::string res = "[";
                for (int lane = 0; lane < 64; ++lane) {
                    char val_str[32];
                    sprintf(val_str, "0x%x", wave.vgpr[idx][lane]);
                    res += std::string(val_str);
                    if (lane < 63) res += ", ";
                }
                res += "]";
                return res;
            }
        } catch (...) {}
    }

    // Check AGPRs
    if (name[0] == 'a') {
        try {
            int idx = std::stoi(name.substr(1));
            if (idx >= 0 && idx < NUM_AGPR) {
                std::string res = "[";
                for (int lane = 0; lane < 64; ++lane) {
                    char val_str[32];
                    sprintf(val_str, "0x%x", wave.agpr[idx][lane]);
                    res += std::string(val_str);
                    if (lane < 63) res += ", ";
                }
                res += "]";
                return res;
            }
        } catch (...) {}
    }

    if (name == "exec") {
        char val_str[32];
        sprintf(val_str, "0x%llx", (unsigned long long)wave.exec);
        return std::string(val_str);
    }
    if (name == "pc") {
        char val_str[32];
        sprintf(val_str, "0x%llx", (unsigned long long)wave.pc);
        return std::string(val_str);
    }
    if (name == "m0") {
        char val_str[32];
        sprintf(val_str, "0x%x", wave.m0);
        return std::string(val_str);
    }

    return "unknown register";
}

void print_source_context(const std::string& filepath, uint32_t line_num) {
    std::ifstream file(filepath);
    if (!file) {
        std::cout << "Could not open source file: " << filepath << "\n";
        return;
    }
    std::string line;
    uint32_t curr = 1;
    uint32_t start = (line_num > 5) ? line_num - 5 : 1;
    uint32_t end = line_num + 5;
    while (std::getline(file, line)) {
        if (curr >= start && curr <= end) {
            if (curr == line_num) {
                std::cout << "=> " << curr << "\t" << line << "\n";
            } else {
                std::cout << "   " << curr << "\t" << line << "\n";
            }
        }
        curr++;
    }
}

void Debugger::debug_hook_impl(WavefrontState& wave) {
    // Collect active waves for the current workgroup
    if (waves.empty() || waves[0]->workgroup_id[0] != wave.workgroup_id[0] ||
        waves[0]->workgroup_id[1] != wave.workgroup_id[1] ||
        waves[0]->workgroup_id[2] != wave.workgroup_id[2]) {
        waves.clear();
        active_wave_idx = 0;
    }
    bool found = false;
    for (size_t i = 0; i < waves.size(); ++i) {
        if (waves[i] == &wave) {
            found = true;
            break;
        }
    }
    if (!found) {
        waves.push_back(&wave);
    }

    // Find if we hit a breakpoint
    bool is_bkpt = false;
    int bkpt_id = -1;
    for (const auto& bp : breakpoints) {
        if (bp.address == wave.pc) {
            is_bkpt = true;
            bkpt_id = bp.id;
            break;
        }
    }

    // Should we pause?
    if (is_bkpt || single_stepping) {
        // Set this wave as the active inspected wave
        for (size_t i = 0; i < waves.size(); ++i) {
            if (waves[i] == &wave) {
                active_wave_idx = i;
                break;
            }
        }

        if (mi_mode) {
            std::string stop_reason = is_bkpt ? "breakpoint-hit" : "end-stepping-range";
            std::string out = "*stopped,reason=\"" + stop_reason + "\"";
            if (is_bkpt) {
                out += ",bkptno=\"" + std::to_string(bkpt_id) + "\"";
            }
            out += "," + get_frame_info(wave, dwarf, kernel.name) +
                   ",thread-id=\"" + std::to_string(get_thread_id(wave)) + "\",stopped-threads=\"all\"";
            std::cout << out << "\n(gdb)\n" << std::flush;

            // Wait for user input to step/continue
            step_requested = false;
            std::string line;
            while (!step_requested && !exited && std::getline(std::cin, line)) {
                GdbMiCommand cmd = parse_gdb_mi(line);
                handle_command(cmd);
            }
        } else {
            // Human-readable REPL mode
            std::string file;
            uint32_t line_num = 0;
            std::cout << "\n========================================\n";
            if (is_bkpt) {
                std::cout << "Breakpoint " << bkpt_id << " hit at PC 0x" << std::hex << wave.pc << std::dec << "\n";
            } else {
                std::cout << "Stopped at PC 0x" << std::hex << wave.pc << std::dec << "\n";
            }
            std::cout << "Wavefront ID: " << wave.trace_wave_id << ", Workgroup: (" 
                      << wave.workgroup_id[0] << ", " << wave.workgroup_id[1] << ", " << wave.workgroup_id[2] << ")\n";

            if (dwarf.get_source_location(wave.pc, file, line_num)) {
                std::cout << "Source: " << file << ":" << line_num << "\n";
                print_source_context(file, line_num);
            } else {
                Instruction inst = decode(wave.kernel_code_ptr + wave.pc);
                std::cout << "Assembly: " << opcode_name(inst.opcode) << "\n";
            }

            // Command loop
            step_requested = false;
            std::string cmd_line;
            while (!step_requested && !exited) {
                std::cout << "(gcndb) " << std::flush;
                if (!std::getline(std::cin, cmd_line)) {
                    exited = true;
                    break;
                }
                
                std::stringstream ss(cmd_line);
                std::string cmd;
                ss >> cmd;
                if (cmd.empty()) continue;

                if (cmd == "continue" || cmd == "c") {
                    single_stepping = false;
                    step_requested = true;
                } else if (cmd == "step" || cmd == "s" || cmd == "stepi" || cmd == "si") {
                    single_stepping = true;
                    step_requested = true;
                } else if (cmd == "print" || cmd == "p") {
                    std::string expr;
                    ss >> expr;
                    if (expr.empty()) {
                        std::cout << "Usage: print <register>\n";
                    } else {
                        std::cout << expr << " = " << evaluate_expr(wave, expr) << "\n";
                    }
                } else if (cmd == "info" || cmd == "i") {
                    std::string sub;
                    ss >> sub;
                    if (sub == "registers" || sub == "r") {
                        for (int r = 0; r < 8; ++r) {
                            std::cout << "s" << r << ": " << evaluate_expr(wave, "s" + std::to_string(r)) << "\t";
                            if (r % 4 == 3) std::cout << "\n";
                        }
                        std::cout << "exec: " << evaluate_expr(wave, "exec") 
                                  << "\tvcc: " << evaluate_expr(wave, "vcc") 
                                  << "\tscc: " << evaluate_expr(wave, "scc") << "\n";
                    } else if (sub == "waves" || sub == "w") {
                        std::cout << "Active Wavefronts in current workgroup:\n";
                        for (size_t w_idx = 0; w_idx < waves.size(); ++w_idx) {
                            char mark = (w_idx == active_wave_idx) ? '*' : ' ';
                            std::cout << mark << " Thread " << (w_idx + 1) << " (Wave " << waves[w_idx]->trace_wave_id 
                                      << ") PC=0x" << std::hex << waves[w_idx]->pc << std::dec << "\n";
                        }
                    } else {
                        std::cout << "Usage: info registers | info waves\n";
                    }
                } else if (cmd == "disassemble" || cmd == "disas") {
                    uint64_t dis_pc = wave.pc;
                    for (int step = 0; step < 5; ++step) {
                        if (dis_pc >= wave.kernel_code_size) break;
                        Instruction inst = decode(wave.kernel_code_ptr + dis_pc);
                        std::cout << "  0x" << std::hex << dis_pc << std::dec << ": " << opcode_name(inst.opcode) << "\n";
                        dis_pc += inst.size_bytes;
                    }
                } else if (cmd == "thread" || cmd == "t") {
                    int w_id = -1;
                    if (ss >> w_id) {
                        if (w_id > 0 && w_id <= (int)waves.size()) {
                            active_wave_idx = w_id - 1;
                            std::cout << "Switched to Thread " << w_id << "\n";
                        } else {
                            std::cout << "Invalid Thread ID\n";
                        }
                    } else {
                        std::cout << "Current Thread ID: " << (active_wave_idx + 1) << "\n";
                    }
                } else if (cmd == "quit" || cmd == "q") {
                    exited = true;
                    step_requested = true;
                    exit(0);
                } else {
                    std::cout << "Unknown command: " << cmd << "\n";
                }
            }
        }
    }
}

void Debugger::handle_command(const GdbMiCommand& cmd) {
    std::string token = cmd.token;

    if (cmd.command == "-gdb-set" || cmd.command == "-gdb-show") {
        std::cout << token << "^done\n(gdb)\n" << std::flush;
    } else if (cmd.command == "-list-features") {
        std::cout << token << "^done,features=[\"breakpoint-notifications\",\"thread-info\"]\n(gdb)\n" << std::flush;
    } else if (cmd.command == "-file-exec-and-symbols") {
        std::cout << token << "^done\n(gdb)\n" << std::flush;
    } else if (cmd.command == "-exec-arguments") {
        std::cout << token << "^done\n(gdb)\n" << std::flush;
    } else if (cmd.command == "-break-insert") {
        std::string loc;
        for (size_t i = 0; i < cmd.args.size(); ++i) {
            if (cmd.args[i] == "-f") {
                if (i + 1 < cmd.args.size()) loc = cmd.args[i + 1];
            } else if (loc.empty() && cmd.args[i][0] != '-') {
                loc = cmd.args[i];
            }
        }

        uint64_t addr = 0;
        bool found = false;
        std::string file = "";
        uint32_t line = 0;

        if (!loc.empty() && loc[0] == '*') {
            addr = std::stoull(loc.substr(1), nullptr, 16);
            found = true;
        } else if (!loc.empty() && loc.find(':') != std::string::npos) {
            size_t colon = loc.find(':');
            file = loc.substr(0, colon);
            line = std::stoi(loc.substr(colon + 1));
            for (const auto& entry : dwarf.get_entries()) {
                if (entry.file.size() >= file.size() &&
                    entry.file.compare(entry.file.size() - file.size(), file.size(), file) == 0 &&
                    entry.line == line) {
                    addr = entry.address;
                    found = true;
                    break;
                }
            }
        } else if (!loc.empty()) {
            // Treat as function name, default to entry offset 256
            addr = 256;
            found = true;
        }

        if (found) {
            Breakpoint bp{next_bkpt_id++, addr, file, line, kernel.name};
            breakpoints.push_back(bp);
            char addr_hex[32];
            sprintf(addr_hex, "0x%llx", (unsigned long long)addr);

            std::cout << token << "^done,bkpt={number=\"" << bp.id 
                      << "\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"" << addr_hex 
                      << "\",func=\"" << kernel.name << "\",file=\"" << file 
                      << "\",fullname=\"" << file << "\",line=\"" << line << "\",times=\"0\"}\n(gdb)\n" << std::flush;
        } else {
            std::cout << token << "^error,msg=\"Location not found\"\n(gdb)\n" << std::flush;
        }
    } else if (cmd.command == "-break-delete") {
        if (!cmd.args.empty()) {
            int bp_id = std::stoi(cmd.args[0]);
            auto it = std::remove_if(breakpoints.begin(), breakpoints.end(), [bp_id](const Breakpoint& b) {
                return b.id == bp_id;
            });
            breakpoints.erase(it, breakpoints.end());
        }
        std::cout << token << "^done\n(gdb)\n" << std::flush;
    } else if (cmd.command == "-exec-run") {
        std::cout << token << "^running\n" << std::flush;
        running = true;
        
        // Start simulation engine asynchronously
        std::string err_msg;
        gcnsim::launch_kernel(kernel, cfg, err_msg);
        
        std::cout << token << "^done\n(gdb)\n" << std::flush;
    } else if (cmd.command == "-exec-continue") {
        std::cout << token << "^running\n" << std::flush;
        single_stepping = false;
        step_requested = true;
    } else if (cmd.command == "-exec-step" || cmd.command == "-exec-next") {
        std::cout << token << "^running\n" << std::flush;
        single_stepping = true;
        step_requested = true;
    } else if (cmd.command == "-exec-step-instruction" || cmd.command == "-exec-next-instruction") {
        std::cout << token << "^running\n" << std::flush;
        single_stepping = true;
        step_requested = true;
    } else if (cmd.command == "-thread-info") {
        std::string out = token + "^done,threads=[";
        for (size_t i = 0; i < waves.size(); ++i) {
            char tid[16];
            sprintf(tid, "%u", (unsigned int)(i + 1));
            out += "{id=\"" + std::string(tid) + "\",target-id=\"Wave " + std::to_string(waves[i]->trace_wave_id) + "\",";
            out += get_frame_info(*waves[i], dwarf, kernel.name) + ",state=\"stopped\"}";
            if (i + 1 < waves.size()) out += ",";
        }
        out += "],current-thread-id=\"" + std::to_string(active_wave_idx + 1) + "\"";
        std::cout << out << "\n(gdb)\n" << std::flush;
    } else if (cmd.command == "-thread-select") {
        if (!cmd.args.empty()) {
            int tid = std::stoi(cmd.args[0]);
            if (tid > 0 && tid <= (int)waves.size()) {
                active_wave_idx = tid - 1;
                std::cout << token << "^done,new-thread-id=\"" << tid << "\"," 
                          << get_frame_info(*waves[active_wave_idx], dwarf, kernel.name) << "\n(gdb)\n" << std::flush;
                return;
            }
        }
        std::cout << token << "^error,msg=\"Invalid thread-id\"\n(gdb)\n" << std::flush;
    } else if (cmd.command == "-stack-list-frames") {
        if (active_wave_idx < waves.size()) {
            std::cout << token << "^done,stack=[frame={level=\"0\"," 
                      << get_frame_info(*waves[active_wave_idx], dwarf, kernel.name).substr(6) << "]\n(gdb)\n" << std::flush;
        } else {
            std::cout << token << "^error,msg=\"No active thread\"\n(gdb)\n" << std::flush;
        }
    } else if (cmd.command == "-data-evaluate-expression") {
        if (active_wave_idx < waves.size() && !cmd.args.empty()) {
            std::string val = evaluate_expr(*waves[active_wave_idx], cmd.args[0]);
            std::cout << token << "^done,value=\"" << val << "\"\n(gdb)\n" << std::flush;
        } else {
            std::cout << token << "^error,msg=\"No expression or thread\"\n(gdb)\n" << std::flush;
        }
    } else if (cmd.command == "-data-read-memory") {
        if (cmd.args.size() >= 3) {
            uint64_t addr = std::stoull(cmd.args[0], nullptr, 0);
            size_t bytes = std::stoul(cmd.args[2]);
            if (addr_is_valid(addr)) {
                std::vector<uint8_t> buf(bytes);
                std::memcpy(buf.data(), (void*)addr, bytes);
                char addr_hex[32];
                sprintf(addr_hex, "0x%llx", (unsigned long long)addr);

                std::string res = token + "^done,addr=\"" + std::string(addr_hex) + "\",nr-bytes=\"" + std::to_string(bytes) + "\",total-bytes=\"" + std::to_string(bytes) + "\",memory=[{addr=\"" + std::string(addr_hex) + "\",data=[";
                for (size_t i = 0; i < bytes; ++i) {
                    char byte_hex[16];
                    sprintf(byte_hex, "\"0x%02x\"", buf[i]);
                    res += std::string(byte_hex);
                    if (i + 1 < bytes) res += ",";
                }
                res += "]}]";
                std::cout << res << "\n(gdb)\n" << std::flush;
            } else {
                std::cout << token << "^error,msg=\"Invalid memory address\"\n(gdb)\n" << std::flush;
            }
        } else {
            std::cout << token << "^error,msg=\"Invalid memory arguments\"\n(gdb)\n" << std::flush;
        }
    } else if (cmd.command == "-gdb-exit") {
        exited = true;
        std::cout << token << "^exit\n" << std::flush;
        exit(0);
    } else {
        // Fallback for CLI commands typed in VS Code console
        if (cmd.command == "info" && cmd.args.size() >= 1 && cmd.args[0] == "registers") {
            std::cout << token << "~\"Registers:\\n\"" << std::flush;
            if (active_wave_idx < waves.size()) {
                const auto& wave = *waves[active_wave_idx];
                for (int r = 0; r < 8; ++r) {
                    std::cout << "~\"s" << r << ": " << evaluate_expr(wave, "s" + std::to_string(r)) << "\\t\"" << std::flush;
                    if (r % 4 == 3) std::cout << "~\"\\n\"" << std::flush;
                }
            }
            std::cout << token << "^done\n(gdb)\n" << std::flush;
        } else {
            std::cout << token << "^done\n(gdb)\n" << std::flush;
        }
    }
}

void Debugger::run_mi() {
    std::cout << "=thread-group-added,id=\"i1\"\n(gdb)\n" << std::flush;
    std::string line;
    while (!exited && std::getline(std::cin, line)) {
        GdbMiCommand cmd = parse_gdb_mi(line);
        handle_command(cmd);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: gcndb <kernel.hsaco> [options]\n";
        std::cerr << "Options:\n";
        std::cerr << "  --interpreter=mi     Enable GDB/MI compatibility mode\n";
        return 1;
    }

    std::string hsaco_path = argv[1];
    bool interpreter_mi = false;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--interpreter=mi") == 0 || std::strcmp(argv[i], "--interpreter=mi2") == 0) {
            interpreter_mi = true;
        }
    }

    std::string error_msg;
    auto hsaco = load_hsaco(hsaco_path, error_msg);
    if (!hsaco) {
        std::cerr << "Error loading HSACO file: " << error_msg << "\n";
        return 1;
    }

    if (hsaco->kernels.empty()) {
        std::cerr << "No kernels found in HSACO file\n";
        return 1;
    }

    Debugger db;
    db.module = std::move(*hsaco);
    db.kernel = db.module.kernels[0]; // Debug first kernel by default
    db.mi_mode = interpreter_mi;
    g_debugger = &db;

    // Load DWARF debug info
    db.dwarf.load_from_elf(db.module.elf_data);

    // Setup launch config with default small grid/block for testing
    db.cfg.grid_x = 256;
    db.cfg.grid_y = 1;
    db.cfg.grid_z = 1;
    db.cfg.block_x = 256;
    db.cfg.block_y = 1;
    db.cfg.block_z = 1;

    // Mock small buffer allocations for kernargs (A, B, C)
    db.mock_a.resize(1024 * sizeof(float), 0);
    db.mock_b.resize(1024 * sizeof(float), 0);
    db.mock_c.resize(1024 * sizeof(float), 0);
    
    // Fill mockup buffers
    float* fA = (float*)db.mock_a.data();
    float* fB = (float*)db.mock_b.data();
    for (int i = 0; i < 1024; ++i) {
        fA[i] = (float)i;
        fB[i] = (float)(i * 2);
    }

    // Set up mock kernel arguments
    struct MockArgs {
        float* pA;
        float* pB;
        float* pC;
        int n;
    } args;
    args.pA = (float*)db.mock_a.data();
    args.pB = (float*)db.mock_b.data();
    args.pC = (float*)db.mock_c.data();
    args.n = 1024;

    db.cfg.kernarg = &args;
    db.cfg.kernarg_size = sizeof(args);
    db.cfg.debug_hook = global_debug_hook;

    if (db.mi_mode) {
        db.run_mi();
    } else {
        std::cout << "gcndb — GCN GPU ISA debugger\n";
        std::cout << "Loaded kernel: " << db.kernel.name << " from " << hsaco_path << "\n";
        std::cout << "DWARF Line info status: " << (db.dwarf.get_entries().empty() ? "No DWARF line info found (recompile with -g)" : "DWARF Line table loaded successfully") << "\n";
        std::cout << "Type 'run' or 'r' to start simulation debugging\n";

        std::string line;
        while (!db.exited) {
            std::cout << "(gcndb) " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line == "run" || line == "r") {
                db.single_stepping = false;
                std::string err_msg;
                gcnsim::launch_kernel(db.kernel, db.cfg, err_msg);
                std::cout << "Simulation finished.\n";
            } else if (line == "quit" || line == "q") {
                break;
            } else if (line == "help" || line == "h") {
                std::cout << "Available commands:\n";
                std::cout << "  run / r          Start the simulation kernel\n";
                std::cout << "  quit / q         Exit debugger\n";
            } else {
                std::cout << "Use 'run' to start simulation first.\n";
            }
        }
    }

    return 0;
}
