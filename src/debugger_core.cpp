#include "gcnsim/debugger_core.h"
#include "gcnsim/isa_decoder.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <stdarg.h>
#include <filesystem>

namespace gcnsim {

Debugger* g_debugger = nullptr;

void set_global_debugger(Debugger* db) {
    g_debugger = db;
}

void global_debug_hook(WavefrontState& wave) {
    if (g_debugger) {
        g_debugger->debug_hook_impl(wave);
    }
}

static bool addr_is_valid(uint64_t addr) {
    if (addr < 4096) return false;
    void* page = (void*)(addr & ~(uint64_t)4095);
    return msync(page, 1, MS_ASYNC) == 0;
}

std::vector<std::string> Debugger::init_reg_names() {
    std::vector<std::string> names;
    for (int i = 0; i < 106; ++i) names.push_back("s" + std::to_string(i));
    names.push_back("vcc_lo"); names.push_back("vcc_hi");
    names.push_back("exec_lo"); names.push_back("exec_hi");
    for (int i = 0; i < 256; ++i) names.push_back("v" + std::to_string(i));
    for (int i = 0; i < 256; ++i) names.push_back("a" + std::to_string(i));
    names.push_back("pc"); names.push_back("exec"); names.push_back("vcc"); 
    names.push_back("scc"); names.push_back("m0");
    return names;
}

Debugger::Debugger() : reg_names(init_reg_names()) {
    char buf[1024];
    if (getcwd(buf, sizeof(buf))) cwd = buf;
    source_dirs.push_back("$cdir");
    source_dirs.push_back("$cwd");
    
    mi_log_file = fopen("/tmp/gcnsim_mi.log", "a");
    if (mi_log_file) {
        fprintf(mi_log_file, "--- Debugger Started (PID %d) ---\n", (int)getpid());
        fflush(mi_log_file);
    }
}

void Debugger::mi_printf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    if (mi_log_file) {
        va_list args2;
        va_copy(args2, args);
        vfprintf(mi_log_file, fmt, args2);
        fprintf(mi_log_file, "(gdb) \n");
        fflush(mi_log_file);
        va_end(args2);
    }
    vdprintf(output_fd, fmt, args);
    va_end(args);
    dprintf(output_fd, "(gdb) \n");
}

std::string Debugger::get_frame_info(const WavefrontState& wave, const std::string& func_name) {
    std::string file;
    uint32_t line = 0;
    uint64_t absolute_pc = kernel.code_vma + wave.pc;
    char pc_hex[64];
    sprintf(pc_hex, "0x%016llx", (unsigned long long)absolute_pc);

    std::string frame = "frame={level=\"0\",addr=\"" + std::string(pc_hex) + "\",func=\"" + func_name + "\"";
    if (dwarf.get_source_location(absolute_pc, file, line)) {
        frame += ",file=\"" + file + "\",fullname=\"" + file + "\",line=\"" + std::to_string(line) + "\"";
    }
    frame += "}";
    return frame;
}

int Debugger::get_thread_id(const WavefrontState& wave) {
    return wave.trace_wave_id + 1;
}

std::string Debugger::evaluate_expr(const WavefrontState& wave, const std::string& expr) {
    return evaluate_var_expr(wave, expr);
}

std::string Debugger::evaluate_var_expr(const WavefrontState& wave, const std::string& expr) const {
    std::string name = expr;
    // Strip leading '$' if present
    if (!name.empty() && name[0] == '$') {
        name = name.substr(1);
    }
    
    // Check if it's a bracket query like v0[5] or v0[all]
    size_t bracket = name.find('[');
    std::string base_name = (bracket == std::string::npos) ? name : name.substr(0, bracket);
    std::string index_str = (bracket == std::string::npos) ? "" : name.substr(bracket + 1, name.size() - bracket - 2);

    // Scalar GPRs: s0..s105
    if (base_name[0] == 's' && base_name != "scc") {
        try {
            int idx = std::stoi(base_name.substr(1));
            if (idx >= 0 && idx < NUM_SGPR) {
                char buf[32];
                sprintf(buf, "0x%x", wave.sgpr[idx]);
                return buf;
            }
        } catch (...) {}
    }

    // Vector GPRs: v0..v255
    if (base_name[0] == 'v' && base_name != "vcc") {
        try {
            int reg_idx = std::stoi(base_name.substr(1));
            if (reg_idx >= 0 && reg_idx < NUM_VGPR) {
                if (index_str == "all") {
                    std::string s = "[";
                    for (int l = 0; l < WAVE_SIZE; ++l) {
                        char buf[32];
                        sprintf(buf, "0x%x", wave.vgpr[reg_idx][l]);
                        s += buf;
                        if (l < WAVE_SIZE - 1) s += ", ";
                    }
                    s += "]";
                    return s;
                } else if (!index_str.empty()) {
                    int lane = std::stoi(index_str);
                    if (lane >= 0 && lane < WAVE_SIZE) {
                        char buf[32];
                        sprintf(buf, "0x%x", wave.vgpr[reg_idx][lane]);
                        return buf;
                    }
                } else {
                    char buf[32];
                    sprintf(buf, "0x%x", wave.vgpr[reg_idx][active_lane]);
                    return buf;
                }
            }
        } catch (...) {}
    }

    // Accumulator GPRs: a0..a255
    if (base_name[0] == 'a') {
        try {
            int reg_idx = std::stoi(base_name.substr(1));
            if (reg_idx >= 0 && reg_idx < NUM_AGPR) {
                if (index_str == "all") {
                    std::string s = "[";
                    for (int l = 0; l < WAVE_SIZE; ++l) {
                        char buf[32];
                        sprintf(buf, "0x%x", wave.agpr[reg_idx][l]);
                        s += buf;
                        if (l < WAVE_SIZE - 1) s += ", ";
                    }
                    s += "]";
                    return s;
                } else if (!index_str.empty()) {
                    int lane = std::stoi(index_str);
                    if (lane >= 0 && lane < WAVE_SIZE) {
                        char buf[32];
                        sprintf(buf, "0x%x", wave.agpr[reg_idx][lane]);
                        return buf;
                    }
                } else {
                    char buf[32];
                    sprintf(buf, "0x%x", wave.agpr[reg_idx][active_lane]);
                    return buf;
                }
            }
        } catch (...) {}
    }

    // Special registers
    if (base_name == "pc") {
        char buf[32];
        sprintf(buf, "0x%llx", (unsigned long long)(kernel.code_vma + wave.pc));
        return buf;
    }
    if (base_name == "pc_offset") {
        char buf[32];
        sprintf(buf, "0x%llx", (unsigned long long)wave.pc);
        return buf;
    }
    if (base_name == "exec") {
        char buf[32];
        sprintf(buf, "0x%llx", (unsigned long long)wave.exec);
        return buf;
    }
    if (base_name == "exec_lo") {
        char buf[32];
        sprintf(buf, "0x%x", (uint32_t)(wave.exec & 0xffffffff));
        return buf;
    }
    if (base_name == "exec_hi") {
        char buf[32];
        sprintf(buf, "0x%x", (uint32_t)(wave.exec >> 32));
        return buf;
    }
    if (base_name == "vcc") {
        char buf[32];
        sprintf(buf, "0x%llx", (unsigned long long)wave.vcc);
        return buf;
    }
    if (base_name == "vcc_lo") {
        char buf[32];
        sprintf(buf, "0x%x", (uint32_t)(wave.vcc & 0xffffffff));
        return buf;
    }
    if (base_name == "vcc_hi") {
        char buf[32];
        sprintf(buf, "0x%x", (uint32_t)(wave.vcc >> 32));
        return buf;
    }
    if (base_name == "scc") return std::to_string(wave.scc);
    if (base_name == "m0") {
        char buf[32];
        sprintf(buf, "0x%x", wave.m0);
        return buf;
    }
    if (base_name == "lane") return std::to_string(active_lane);

    // Kernel arguments
    for (const auto& arg : kernel.args) {
        if (arg.name == base_name) {
            if (!wave.kernarg_ptr) return "N/A (no kernarg)";
            const uint8_t* ptr = wave.kernarg_ptr + arg.offset;
            
            if (arg.value_kind == "global_buffer" || arg.size == 8) {
                uint64_t val = 0;
                std::memcpy(&val, ptr, 8);
                char buf[32];
                sprintf(buf, "0x%llx", (unsigned long long)val);
                return buf;
            } else if (arg.size == 4) {
                if (arg.type_name.find("float") != std::string::npos) {
                    float fval = 0.0f;
                    std::memcpy(&fval, ptr, 4);
                    return std::to_string(fval);
                } else {
                    int32_t ival = 0;
                    std::memcpy(&ival, ptr, 4);
                    return std::to_string(ival);
                }
            } else {
                std::string res = "0x";
                for (uint32_t b = 0; b < arg.size; ++b) {
                    char buf[8];
                    sprintf(buf, "%02x", ptr[b]);
                    res += buf;
                }
                return res;
            }
        }
    }

    return "unknown";
}

void Debugger::print_source_context(const std::string& filepath, uint32_t line_num) {
    std::ifstream file(filepath);
    if (!file) return;
    std::string line;
    uint32_t curr = 1;
    uint32_t start = (line_num > 5) ? line_num - 5 : 1;
    uint32_t end = line_num + 5;
    while (std::getline(file, line)) {
        if (curr >= start && curr <= end) {
            if (curr == line_num) std::cout << "=> " << curr << "\t" << line << "\n";
            else std::cout << "   " << curr << "\t" << line << "\n";
        }
        curr++;
    }
}

void Debugger::debug_hook_impl(WavefrontState& wave) {
    if (waves.empty() || waves[0]->workgroup_id[0] != wave.workgroup_id[0]) {
        waves.clear(); active_wave_idx = 0;
    }
    bool found = false;
    for (auto* w : waves) if (w == &wave) { found = true; break; }
    if (!found) waves.push_back(&wave);

    bool is_bkpt = false;
    int bkpt_id = -1;
    for (const auto& bp : breakpoints) {
        if (bp.address == wave.pc) { is_bkpt = true; bkpt_id = bp.id; break; }
    }

    if (stepping_line && !is_bkpt) {
        std::string curr_file;
        uint32_t curr_line = 0;
        uint64_t absolute_pc = kernel.code_vma + wave.pc;
        if (dwarf.get_source_location(absolute_pc, curr_file, curr_line)) {
            if (curr_file == step_file && curr_line == step_line) {
                step_requested = true;
                single_stepping = true;
                return;
            }
        }
        stepping_line = false;
        single_stepping = true;
    }

    if (is_bkpt || single_stepping) {
        stepping_line = false;
        for (size_t i = 0; i < waves.size(); ++i) if (waves[i] == &wave) { active_wave_idx = i; break; }

        std::string file;
        uint32_t line_num = 0;
        uint64_t absolute_pc = kernel.code_vma + wave.pc;
        bool is_header = false;
        if (dwarf.get_source_location(absolute_pc, file, line_num)) {
            if (file.find("hip_runtime") != std::string::npos || file.find("/opt/rocm") == 0) is_header = true;
        }

        if (wave.pc == 0) skipping_headers = true;
        if (skipping_headers && is_header && !is_bkpt) {
            single_stepping = true; return;
        }
        skipping_headers = false;

        if (mi_mode) {
            static std::string last_kernel_loaded;
            if (last_kernel_loaded != kernel.name && !hsaco_path.empty()) {
                mi_printf("~\"add-symbol-file -readnow %s 0x%llx\\n\"\n", hsaco_path.c_str(), (unsigned long long)kernel.code_vma);
                last_kernel_loaded = kernel.name;
            }
            std::string stop_reason = is_bkpt ? "breakpoint-hit" : "end-stepping-range";
            std::string stop_rec = "*stopped,reason=\"" + stop_reason + "\"";
            if (is_bkpt) stop_rec += ",bkptno=\"" + std::to_string(bkpt_id) + "\"";
            stop_rec += "," + get_frame_info(wave, kernel.name) + ",thread-id=\"" + std::to_string(get_thread_id(wave)) + "\",stopped-threads=\"all\"";
            mi_printf("%s\n", stop_rec.c_str());

            step_requested = false;
            std::string line;
            if (mi_log_file) {
                fprintf(mi_log_file, "Waiting for stdin in debug_hook_impl (cin.eof=%d, cin.fail=%d)...\n", std::cin.eof(), std::cin.fail());
                fflush(mi_log_file);
            }
            while (!step_requested && !exited && std::getline(std::cin, line)) {
                if (mi_log_file) {
                    fprintf(mi_log_file, "READ: %s\n", line.c_str());
                    fflush(mi_log_file);
                }
                handle_command(parse_gdb_mi(line));
            }
            if (!step_requested && !exited) {
                if (mi_log_file) {
                    fprintf(mi_log_file, "stdin loop exited! cin.eof=%d, cin.fail=%d, cin.bad=%d, errno=%d (%s)\n",
                            std::cin.eof(), std::cin.fail(), std::cin.bad(), errno, strerror(errno));
                    fflush(mi_log_file);
                }
            }
        } else {
            if (!file.empty()) {
                std::cout << "\032\032" << file << ":" << line_num << ":0:beg:0x" << std::hex << absolute_pc << std::dec << "\n";
                print_source_context(file, line_num);
            } else {
                Instruction inst = decode(wave.kernel_code_ptr + wave.pc);
                std::cout << "Stopped at PC 0x" << std::hex << wave.pc << " (" << opcode_name(inst.opcode) << ")" << std::dec << "\n";
            }
            step_requested = false;
            std::string cmd_line;
            while (!step_requested && !exited) {
                std::cout << "(gdb) " << std::flush;
                if (!std::getline(std::cin, cmd_line)) break;
                handle_cli_command(cmd_line);
            }
        }
    }
}

void Debugger::handle_command(const GdbMiCommand& cmd) {
    std::string token = cmd.token;
    WavefrontState* wave = (waves.empty() ? nullptr : waves[active_wave_idx]);

    if (cmd.command == "-gdb-set" || cmd.command == "-gdb-show") {
        mi_printf("%s^done\n", token.c_str());
    }
    else if (cmd.command == "-list-features") {
        mi_printf("%s^done,features=[\"breakpoint-notifications\",\"thread-info\",\"var-objects\"]\n", token.c_str());
    }
    else if (cmd.command == "-file-exec-and-symbols") {
        mi_printf("%s^done\n", token.c_str());
    }
    else if (cmd.command == "-exec-arguments") {
        mi_printf("%s^done\n", token.c_str());
    }
    else if (cmd.command == "-list-thread-groups") {
        mi_printf("%s^done,groups=[{id=\"i1\",type=\"process\",executable=\"%s\"}]\n", token.c_str(), kernel.name.c_str());
    }
    else if (cmd.command == "-break-insert") {
        std::string loc;
        for (const auto& arg : cmd.args) {
            if (arg != "-f" && arg != "-d") loc = arg;
        }

        uint64_t addr = (uint64_t)-1;
        std::string file = "";
        uint32_t line = 0;
        std::string func = "";

        if (!loc.empty()) {
            if (loc[0] == '*') {
                addr = std::stoull(loc.substr(1), nullptr, 0);
                if (kernel.code_vma != 0 && addr >= kernel.code_vma) addr -= kernel.code_vma;
            } else if (loc.find(':') != std::string::npos) {
                size_t colon = loc.find(':');
                file = loc.substr(0, colon);
                line = std::stoul(loc.substr(colon + 1));
                uint64_t abs_addr = 0;
                if (dwarf.get_address_from_location(file, line, abs_addr)) {
                    addr = abs_addr;
                    if (addr >= kernel.code_vma) addr -= kernel.code_vma;
                }
            } else {
                func = loc;
                if (kernel.code_vma != 0 && kernel.name.find(loc) != std::string::npos) {
                    addr = 0;
                }
            }
        }

        Breakpoint bp{next_bkpt_id++, addr, file, line, func.empty() ? kernel.name : func};
        breakpoints.push_back(bp);

        print_mi_breakpoint(token, bp, false);
    }
    else if (cmd.command == "-break-delete") {
        bool deleted = false;
        if (!cmd.args.empty()) {
            int bp_id = std::stoi(cmd.args[0]);
            auto it = std::remove_if(breakpoints.begin(), breakpoints.end(), [bp_id](const Breakpoint& bp) { return bp.id == bp_id; });
            if (it != breakpoints.end()) {
                breakpoints.erase(it, breakpoints.end());
                deleted = true;
            }
        }
        if (deleted) {
            mi_printf("%s^done\n", token.c_str());
        } else {
            mi_printf("%s^error,msg=\"Breakpoint not found\"\n", token.c_str());
        }
    }
    else if (cmd.command == "-exec-run" || cmd.command == "run" || cmd.command == "r") {
        if (running) {
            step_requested = true;
            single_stepping = false;
            mi_printf("%s^running\n", token.c_str());
        } else {
            mi_printf("%s^running\n", token.c_str());
            running = true;
            std::string err;
            gcnsim::launch_kernel(kernel, cfg, err);
            exit(0);
        }
    }
    else if (cmd.command == "-exec-continue") {
        single_stepping = false;
        step_requested = true;
        mi_printf("%s^running\n", token.c_str());
    }
    else if (cmd.command == "-exec-step" || cmd.command == "-exec-next") {
        if (wave) {
            std::string file;
            uint32_t line_num = 0;
            uint64_t absolute_pc = kernel.code_vma + wave->pc;
            if (dwarf.get_source_location(absolute_pc, file, line_num)) {
                stepping_line = true;
                step_line = line_num;
                step_file = file;
            }
        }
        single_stepping = true;
        step_requested = true;
        mi_printf("%s^running\n", token.c_str());
    }
    else if (cmd.command == "-exec-step-instruction" || cmd.command == "-exec-next-instruction") {
        stepping_line = false;
        single_stepping = true;
        step_requested = true;
        mi_printf("%s^running\n", token.c_str());
    }
    else if (cmd.command == "-thread-info") {
        std::string res = token + "^done,threads=[";
        for (size_t i = 0; i < waves.size(); ++i) {
            if (i > 0) res += ",";
            WavefrontState* w = waves[i];
            char tid[16]; sprintf(tid, "%d", get_thread_id(*w));
            char pc_hex[64]; sprintf(pc_hex, "0x%016llx", (unsigned long long)(kernel.code_vma + w->pc));
            
            res += "{id=\"" + std::string(tid) + "\",target-id=\"Wave " + std::to_string(w->trace_wave_id) + " (WG " + std::to_string(w->workgroup_id[0]) + "," + std::to_string(w->workgroup_id[1]) + "," + std::to_string(w->workgroup_id[2]) + ")\"";
            
            std::string file;
            uint32_t line = 0;
            res += ",frame={level=\"0\",addr=\"" + std::string(pc_hex) + "\",func=\"" + kernel.name + "\"";
            if (dwarf.get_source_location(kernel.code_vma + w->pc, file, line)) {
                res += ",file=\"" + file + "\",fullname=\"" + file + "\",line=\"" + std::to_string(line) + "\"";
            }
            res += "}}";
        }
        res += "]";
        char curr_tid[16];
        sprintf(curr_tid, "%d", wave ? get_thread_id(*wave) : 1);
        res += ",current-thread-id=\"" + std::string(curr_tid) + "\"";
        mi_printf("%s\n", res.c_str());
    }
    else if (cmd.command == "-thread-list-ids") {
        std::string res = token + "^done,thread-ids=[";
        for (size_t i = 0; i < waves.size(); ++i) {
            if (i > 0) res += ",";
            res += "\"" + std::to_string(get_thread_id(*waves[i])) + "\"";
        }
        res += "]";
        char curr_tid[16];
        sprintf(curr_tid, "%d", wave ? get_thread_id(*wave) : 1);
        res += ",current-thread-id=\"" + std::string(curr_tid) + "\"";
        mi_printf("%s\n", res.c_str());
    }
    else if (cmd.command == "-thread-select") {
        if (!cmd.args.empty()) {
            int tid = std::stoi(cmd.args[0]);
            bool found = false;
            for (size_t i = 0; i < waves.size(); ++i) {
                if (get_thread_id(*waves[i]) == tid) {
                    active_wave_idx = i;
                    found = true;
                    break;
                }
            }
            wave = waves[active_wave_idx];
            if (found && wave) {
                mi_printf("%s^done,new-thread-id=\"%d\",frame=%s\n", token.c_str(), tid, get_frame_info(*wave, kernel.name).c_str());
            } else {
                mi_printf("%s^error,msg=\"Thread ID not found\"\n", token.c_str());
            }
        } else {
            mi_printf("%s^error,msg=\"Missing thread ID\"\n", token.c_str());
        }
    }
    else if (cmd.command == "-stack-list-frames") {
        if (wave) {
            std::string frame = get_frame_info(*wave, kernel.name);
            mi_printf("%s^done,stack=[%s]\n", token.c_str(), frame.c_str());
        } else {
            mi_printf("%s^done,stack=[]\n", token.c_str());
        }
    }
    else if (cmd.command == "-stack-info-depth") {
        mi_printf("%s^done,depth=\"1\"\n", token.c_str());
    }
    else if (cmd.command == "-stack-list-arguments") {
        std::string res = token + "^done,stack-args=[frame={level=\"0\",args=[";
        if (wave) {
            for (size_t i = 0; i < kernel.args.size(); ++i) {
                if (i > 0) res += ",";
                res += "{name=\"" + kernel.args[i].name + "\",value=\"" + evaluate_var_expr(*wave, kernel.args[i].name) + "\"}";
            }
        }
        res += "]}]";
        mi_printf("%s\n", res.c_str());
    }
    else if (cmd.command == "-stack-list-variables" || cmd.command == "-stack-list-locals") {
        std::string res = token + "^done,variables=[";
        if (wave) {
            for (size_t i = 0; i < kernel.args.size(); ++i) {
                if (i > 0) res += ",";
                res += "{name=\"" + kernel.args[i].name + "\",value=\"" + evaluate_var_expr(*wave, kernel.args[i].name) + "\"}";
            }
        }
        res += "]";
        mi_printf("%s\n", res.c_str());
    }
    else if (cmd.command == "-data-list-register-names") {
        std::string res = token + "^done,register-names=[";
        for (size_t i = 0; i < reg_names.size(); ++i) {
            if (i > 0) res += ",";
            res += "\"" + reg_names[i] + "\"";
        }
        res += "]";
        mi_printf("%s\n", res.c_str());
    }
    else if (cmd.command == "-data-list-register-values") {
        std::string format = cmd.args.empty() ? "x" : cmd.args[0];
        std::vector<int> regnums;
        for (size_t i = 1; i < cmd.args.size(); ++i) {
            regnums.push_back(std::stoi(cmd.args[i]));
        }
        if (regnums.empty()) {
            for (size_t i = 0; i < reg_names.size(); ++i) regnums.push_back(i);
        }

        std::string res = token + "^done,register-values=[";
        if (wave) {
            for (size_t i = 0; i < regnums.size(); ++i) {
                int reg_num = regnums[i];
                if (reg_num >= 0 && reg_num < (int)reg_names.size()) {
                    if (i > 0) res += ",";
                    res += "{number=\"" + std::to_string(reg_num) + "\",value=\"" + evaluate_var_expr(*wave, reg_names[reg_num]) + "\"}";
                }
            }
        }
        res += "]";
        mi_printf("%s\n", res.c_str());
    }
    else if (cmd.command == "-data-read-memory-bytes" || cmd.command == "-data-read-memory") {
        uint64_t address = 0;
        uint64_t count = 0;
        long long offset = 0;
        
        std::vector<std::string> args = cmd.args;
        if (!args.empty() && args[0] == "-o") {
            offset = std::stoll(args[1], nullptr, 0);
            args.erase(args.begin(), args.begin() + 2);
        }
        if (args.size() >= 2) {
            address = std::stoull(args[0], nullptr, 0) + offset;
            count = std::stoull(args[1], nullptr, 0);
        }

        bool valid = true;
        for (uint64_t a = address; a < address + count; a += 4096) {
            if (!addr_is_valid(a)) { valid = false; break; }
        }
        if (address + count > address && !addr_is_valid(address + count - 1)) {
            valid = false;
        }

        if (valid && count > 0) {
            std::string hex_content;
            uint8_t* ptr = reinterpret_cast<uint8_t*>(address);
            for (uint64_t i = 0; i < count; ++i) {
                char buf[8];
                sprintf(buf, "%02x", ptr[i]);
                hex_content += buf;
            }
            char begin_hex[32], end_hex[32], offset_hex[32];
            sprintf(begin_hex, "0x%llx", (unsigned long long)address);
            sprintf(end_hex, "0x%llx", (unsigned long long)(address + count));
            sprintf(offset_hex, "0x%llx", (unsigned long long)offset);

            mi_printf("%s^done,memory=[{begin=\"%s\",offset=\"%s\",end=\"%s\",contents=\"%s\"}]\n",
                      token.c_str(), begin_hex, offset_hex, end_hex, hex_content.c_str());
        } else {
            mi_printf("%s^error,msg=\"Cannot access memory at specified address\"\n", token.c_str());
        }
    }
    else if (cmd.command == "-data-write-memory-bytes") {
        uint64_t address = 0;
        std::string contents;
        if (cmd.args.size() >= 2) {
            address = std::stoull(cmd.args[0], nullptr, 0);
            contents = cmd.args[1];
        }

        uint64_t count = contents.size() / 2;
        bool valid = true;
        for (uint64_t a = address; a < address + count; a += 4096) {
            if (!addr_is_valid(a)) { valid = false; break; }
        }
        if (address + count > address && !addr_is_valid(address + count - 1)) {
            valid = false;
        }

        if (valid && count > 0) {
            uint8_t* ptr = reinterpret_cast<uint8_t*>(address);
            for (uint64_t i = 0; i < count; ++i) {
                std::string byte_str = contents.substr(i * 2, 2);
                ptr[i] = (uint8_t)std::stoul(byte_str, nullptr, 16);
            }
            mi_printf("%s^done\n", token.c_str());
        } else {
            mi_printf("%s^error,msg=\"Cannot write memory at specified address\"\n", token.c_str());
        }
    }
    else if (cmd.command == "-var-create") {
        std::string name = (cmd.args.size() > 0) ? cmd.args[0] : "-";
        std::string frame = (cmd.args.size() > 1) ? cmd.args[1] : "*";
        std::string expr = (cmd.args.size() > 2) ? cmd.args[2] : "";

        if (name == "-") {
            static int var_counter = 1;
            name = "var_" + std::to_string(var_counter++);
        }

        var_objects[name] = expr;
        std::string val = wave ? evaluate_var_expr(*wave, expr) : "N/A";
        
        mi_printf("%s^done,name=\"%s\",numchild=\"0\",value=\"%s\",type=\"expression\",thread-id=\"%d\",has_more=\"0\"\n",
                  token.c_str(), name.c_str(), val.c_str(), wave ? get_thread_id(*wave) : 1);
    }
    else if (cmd.command == "-var-update") {
        std::string name = cmd.args.empty() ? "*" : cmd.args[0];
        std::string res = token + "^done,changelist=[";
        
        std::vector<std::string> targets;
        if (name == "*") {
            for (const auto& pair : var_objects) targets.push_back(pair.first);
        } else {
            targets.push_back(name);
        }

        if (wave) {
            for (size_t i = 0; i < targets.size(); ++i) {
                std::string var_name = targets[i];
                if (var_objects.count(var_name)) {
                    if (i > 0) res += ",";
                    std::string val = evaluate_var_expr(*wave, var_objects[var_name]);
                    res += "{name=\"" + var_name + "\",value=\"" + val + "\",in_scope=\"true\",type_changed=\"false\",has_more=\"0\"}";
                }
            }
        }
        res += "]";
        mi_printf("%s\n", res.c_str());
    }
    else if (cmd.command == "-var-delete") {
        if (!cmd.args.empty()) {
            var_objects.erase(cmd.args[0]);
        }
        mi_printf("%s^done\n", token.c_str());
    }
    else if (cmd.command == "-var-evaluate-expression") {
        std::string name = cmd.args.empty() ? "" : cmd.args[0];
        std::string val = "N/A";
        if (var_objects.count(name) && wave) {
            val = evaluate_var_expr(*wave, var_objects[name]);
        }
        mi_printf("%s^done,value=\"%s\"\n", token.c_str(), val.c_str());
    }
    else if (cmd.command == "-data-disassemble") {
        uint64_t start = 0, end = 0;
        for (size_t i = 0; i < cmd.args.size(); ++i) {
            if (cmd.args[i] == "-s" && i+1 < cmd.args.size()) {
                start = std::stoull(cmd.args[i+1], nullptr, 0); if (start >= kernel.code_vma) start -= kernel.code_vma;
            } else if (cmd.args[i] == "-e" && i+1 < cmd.args.size()) {
                end = std::stoull(cmd.args[i+1], nullptr, 0); if (end >= kernel.code_vma) end -= kernel.code_vma;
            }
        }
        if (end == 0) end = start + 64;
        std::string res = token + "^done,asm_insns=[";
        for (uint64_t cur = start; cur < end && cur < kernel.code_size; ) {
            Instruction inst = decode(kernel.code_ptr + cur);
            if (cur > start) res += ",";
            char addr_hex[32]; sprintf(addr_hex, "0x%llx", (unsigned long long)(kernel.code_vma + cur));
            res += "{address=\"" + std::string(addr_hex) + "\",inst=\"" + opcode_name(inst.opcode) + "\"}";
            cur += inst.size_bytes;
        }
        res += "]"; mi_printf("%s\n", res.c_str());
    }
    else if (cmd.command == "-interpreter-exec") {
        if (cmd.args.size() >= 2 && cmd.args[0] == "console") {
            handle_cli_command(cmd.args[1]);
            mi_printf("%s^done\n", token.c_str());
        } else {
            mi_printf("%s^done\n", token.c_str());
        }
    }
    else if (cmd.command == "-gdb-exit") {
        exited = true;
        mi_printf("%s^exit\n", token.c_str());
        exit(0);
    }
    else {
        mi_printf("%s^done\n", token.c_str());
    }
}

void Debugger::handle_cli_command(const std::string& cli_cmd) {
    std::stringstream ss(cli_cmd);
    std::string cmd;
    ss >> cmd;
    if (cmd.empty()) return;

    WavefrontState* wave = (waves.empty() ? nullptr : waves[active_wave_idx]);

    if (cmd == "c" || cmd == "continue") {
        single_stepping = false;
        step_requested = true;
        stepping_line = false;
        if (mi_mode) mi_printf("~\"Continuing.\\n\"\n");
        else std::cout << "Continuing.\n";
    }
    else if (cmd == "s" || cmd == "step") {
        if (wave) {
            std::string file;
            uint32_t line_num = 0;
            uint64_t absolute_pc = kernel.code_vma + wave->pc;
            if (dwarf.get_source_location(absolute_pc, file, line_num)) {
                stepping_line = true;
                step_line = line_num;
                step_file = file;
            }
        }
        single_stepping = true;
        step_requested = true;
    }
    else if (cmd == "si" || cmd == "stepi" || cmd == "step-instruction" || cmd == "ni" || cmd == "nexti") {
        single_stepping = true;
        step_requested = true;
        stepping_line = false;
    }
    else if (cmd == "next" || cmd == "n") {
        if (wave) {
            std::string file;
            uint32_t line_num = 0;
            uint64_t absolute_pc = kernel.code_vma + wave->pc;
            if (dwarf.get_source_location(absolute_pc, file, line_num)) {
                stepping_line = true;
                step_line = line_num;
                step_file = file;
            }
        }
        single_stepping = true;
        step_requested = true;
    }
    else if (cmd == "p" || cmd == "print") {
        std::string expr;
        std::getline(ss, expr);
        size_t first = expr.find_first_not_of(" \t");
        if (first != std::string::npos) expr = expr.substr(first);
        
        if (wave) {
            std::string val = evaluate_var_expr(*wave, expr);
            if (mi_mode) mi_printf("~\"$1 = %s\\n\"\n", val.c_str());
            else std::cout << val << "\n";
        } else {
            if (mi_mode) mi_printf("~\"No active wavefront.\\n\"\n");
            else std::cout << "No active wavefront.\n";
        }
    }
    else if (cmd == "set" || cmd == "$lane") {
        std::string sub;
        ss >> sub;
        if (sub == "lane" || cmd == "$lane") {
            if (cmd == "$lane") {
                std::string eq; ss >> eq;
            }
            uint32_t val = 0;
            if (ss >> val) {
                if (val < WAVE_SIZE) {
                    active_lane = val;
                    if (mi_mode) mi_printf("~\"Active lane set to %d\\n\"\n", active_lane);
                    else std::cout << "Active lane set to " << active_lane << "\n";
                } else {
                    if (mi_mode) mi_printf("~\"Invalid lane number (must be 0-63)\\n\"\n");
                    else std::cout << "Invalid lane number (must be 0-63)\n";
                }
            }
        }
    }
    else if (cmd == "info" || cmd == "i") {
        std::string sub;
        ss >> sub;
        if (sub == "registers" || sub == "reg" || sub == "r") {
            if (wave) {
                std::string out = "";
                out += "pc: " + evaluate_var_expr(*wave, "pc") + "\n";
                out += "exec: " + evaluate_var_expr(*wave, "exec") + "\n";
                out += "vcc: " + evaluate_var_expr(*wave, "vcc") + "\n";
                out += "scc: " + evaluate_var_expr(*wave, "scc") + "\n";
                out += "m0: " + evaluate_var_expr(*wave, "m0") + "\n";
                
                out += "\nScalar registers (non-zero):\n";
                for (int i = 0; i < NUM_SGPR; ++i) {
                    if (wave->sgpr[i] != 0) {
                        out += "s" + std::to_string(i) + ": " + evaluate_var_expr(*wave, "s" + std::to_string(i)) + "  ";
                    }
                }
                
                out += "\n\nVector registers at lane " + std::to_string(active_lane) + " (non-zero):\n";
                for (int i = 0; i < NUM_VGPR; ++i) {
                    if (wave->vgpr[i][active_lane] != 0) {
                        out += "v" + std::to_string(i) + ": " + evaluate_var_expr(*wave, "v" + std::to_string(i)) + "  ";
                    }
                }
                
                out += "\n\nAccumulator registers at lane " + std::to_string(active_lane) + " (non-zero):\n";
                for (int i = 0; i < NUM_AGPR; ++i) {
                    if (wave->agpr[i][active_lane] != 0) {
                        out += "a" + std::to_string(i) + ": " + evaluate_var_expr(*wave, "a" + std::to_string(i)) + "  ";
                    }
                }
                out += "\n";
                
                if (mi_mode) mi_printf("~\"%s\"", out.c_str());
                else std::cout << out;
            }
        }
        else if (sub == "threads" || sub == "thread") {
            std::string out = "";
            for (size_t i = 0; i < waves.size(); ++i) {
                WavefrontState* w = waves[i];
                char prefix = (i == active_wave_idx) ? '*' : ' ';
                char pc_hex[32]; sprintf(pc_hex, "0x%llx", (unsigned long long)(kernel.code_vma + w->pc));
                out += std::string(1, prefix) + " Thread " + std::to_string(get_thread_id(*w)) + " (Wave " + std::to_string(w->trace_wave_id) + " - PC: " + pc_hex + ")\n";
            }
            if (mi_mode) mi_printf("~\"%s\"", out.c_str());
            else std::cout << out;
        }
        else if (sub == "breakpoints" || sub == "break" || sub == "b") {
            std::string out = "Num     Type           Disp Enb Address            What\n";
            for (const auto& bp : breakpoints) {
                char addr_hex[32]; sprintf(addr_hex, "0x%llx", (unsigned long long)(kernel.code_vma + bp.address));
                out += std::to_string(bp.id) + "       breakpoint     keep y   " + addr_hex + "   offset " + std::to_string(bp.address) + " of " + bp.func + "\n";
            }
            if (mi_mode) mi_printf("~\"%s\"", out.c_str());
            else std::cout << out;
        }
    }
    else if (cmd == "thread" || cmd == "t") {
        int tid = 0;
        if (ss >> tid) {
            bool found = false;
            for (size_t i = 0; i < waves.size(); ++i) {
                if (get_thread_id(*waves[i]) == tid) {
                    active_wave_idx = i;
                    found = true;
                    break;
                }
            }
            if (found && wave) {
                if (mi_mode) mi_printf("~\"[Switching to thread %d]\\n\"\n", tid);
                else std::cout << "[Switching to thread " << tid << "]\n";
            }
        }
    }
    else if (cmd == "x") {
        std::string format_str = "";
        std::string addr_str = "";
        ss >> format_str;
        
        int count = 1;
        char fmt = 'x';
        char size = 'w';
        
        if (!format_str.empty() && format_str[0] == '/') {
            size_t idx = 1;
            std::string count_str = "";
            while (idx < format_str.size() && std::isdigit(format_str[idx])) {
                count_str += format_str[idx++];
            }
            if (!count_str.empty()) count = std::stoi(count_str);
            if (idx < format_str.size()) fmt = format_str[idx++];
            if (idx < format_str.size()) size = format_str[idx++];
            ss >> addr_str;
        } else {
            addr_str = format_str;
        }

        if (!addr_str.empty()) {
            uint64_t addr = std::stoull(addr_str, nullptr, 0);
            int unit_size = 4;
            if (size == 'b') unit_size = 1;
            else if (size == 'h') unit_size = 2;
            else if (size == 'w') unit_size = 4;
            else if (size == 'g') unit_size = 8;
            
            std::string out = "";
            for (int c = 0; c < count; ++c) {
                uint64_t curr_addr = addr + c * unit_size;
                bool valid = true;
                for (int b = 0; b < unit_size; ++b) {
                    if (!addr_is_valid(curr_addr + b)) { valid = false; break; }
                }
                if (!valid) {
                    out += "Cannot access memory at 0x" + addr_str + "\n";
                    break;
                }
                
                char addr_hex[32]; sprintf(addr_hex, "0x%llx", (unsigned long long)curr_addr);
                out += std::string(addr_hex) + ":\t";
                
                if (fmt == 'x') {
                    if (unit_size == 1) { char val[8]; sprintf(val, "0x%02x", *(uint8_t*)curr_addr); out += val; }
                    else if (unit_size == 2) { char val[16]; sprintf(val, "0x%04x", *(uint16_t*)curr_addr); out += val; }
                    else if (unit_size == 4) { char val[32]; sprintf(val, "0x%08x", *(uint32_t*)curr_addr); out += val; }
                    else if (unit_size == 8) { char val[64]; sprintf(val, "0x%016llx", *(unsigned long long*)curr_addr); out += val; }
                } else if (fmt == 'd') {
                    if (unit_size == 1) out += std::to_string(*(int8_t*)curr_addr);
                    else if (unit_size == 2) out += std::to_string(*(int16_t*)curr_addr);
                    else if (unit_size == 4) out += std::to_string(*(int32_t*)curr_addr);
                    else if (unit_size == 8) out += std::to_string(*(long long*)curr_addr);
                } else if (fmt == 'f') {
                    if (unit_size == 4) out += std::to_string(*(float*)curr_addr);
                    else if (unit_size == 8) out += std::to_string(*(double*)curr_addr);
                } else if (fmt == 'i') {
                    if (curr_addr >= kernel.code_vma && curr_addr < kernel.code_vma + kernel.code_size) {
                        Instruction inst = decode(kernel.code_ptr + (curr_addr - kernel.code_vma));
                        out += opcode_name(inst.opcode);
                    } else {
                        out += "unknown instruction";
                    }
                }
                out += "\n";
            }
            if (mi_mode) mi_printf("~\"%s\"", out.c_str());
            else std::cout << out;
        }
    }
    else if (cmd == "disassemble" || cmd == "disas") {
        if (wave) {
            uint64_t start_pc = wave->pc;
            uint64_t end_pc = wave->pc + 32;
            std::string out = "";
            for (uint64_t cur = start_pc; cur < end_pc && cur < kernel.code_size; ) {
                Instruction inst = decode(kernel.code_ptr + cur);
                char addr_hex[32]; sprintf(addr_hex, "0x%llx", (unsigned long long)(kernel.code_vma + cur));
                char prefix = (cur == wave->pc) ? '=' : ' ';
                out += std::string(1, prefix) + "> " + addr_hex + ":\t" + opcode_name(inst.opcode) + "\n";
                cur += inst.size_bytes;
            }
            if (mi_mode) mi_printf("~\"%s\"", out.c_str());
            else std::cout << out;
        }
    }
    else if (cmd == "break" || cmd == "b") {
        std::string loc;
        ss >> loc;
        uint64_t addr = (uint64_t)-1;
        bool resolved = false;
        std::string file = "";
        uint32_t line = 0;
        std::string func = "";

        if (!loc.empty()) {
            if (loc[0] == '*') {
                addr = std::stoull(loc.substr(1), nullptr, 0);
                if (kernel.code_vma != 0 && addr >= kernel.code_vma) addr -= kernel.code_vma;
                resolved = true;
            } else if (loc.find(':') != std::string::npos) {
                size_t colon = loc.find(':');
                file = loc.substr(0, colon);
                line = std::stoul(loc.substr(colon + 1));
                uint64_t abs_addr = 0;
                if (dwarf.get_address_from_location(file, line, abs_addr)) {
                    addr = abs_addr;
                    if (addr >= kernel.code_vma) addr -= kernel.code_vma;
                    resolved = true;
                }
            } else {
                func = loc;
                if (kernel.code_vma != 0 && kernel.name.find(loc) != std::string::npos) {
                    addr = 0;
                    resolved = true;
                }
            }
        }

        Breakpoint bp{next_bkpt_id++, addr, file, line, func.empty() ? kernel.name : func};
        breakpoints.push_back(bp);

        if (resolved) {
            if (mi_mode) mi_printf("~\"Breakpoint %d at 0x%llx\\n\"\n", bp.id, (unsigned long long)(kernel.code_vma + addr));
            else std::cout << "Breakpoint " << bp.id << " at 0x" << std::hex << (kernel.code_vma + addr) << std::dec << "\n";
        } else {
            if (mi_mode) mi_printf("~\"Breakpoint %d (%s) pending.\\n\"\n", bp.id, loc.c_str());
            else std::cout << "Breakpoint " << bp.id << " (" << loc << ") pending.\n";
        }
    }
    else if (cmd == "delete" || cmd == "d") {
        int id = 0;
        if (ss >> id) {
            auto it = std::remove_if(breakpoints.begin(), breakpoints.end(), [id](const Breakpoint& bp) { return bp.id == id; });
            if (it != breakpoints.end()) {
                breakpoints.erase(it, breakpoints.end());
                if (mi_mode) mi_printf("~\"Deleted breakpoint %d\\n\"\n", id);
                else std::cout << "Deleted breakpoint " << id << "\n";
            } else {
                if (mi_mode) mi_printf("~\"Breakpoint %d not found\\n\"\n", id);
                else std::cout << "Breakpoint " << id << " not found\n";
            }
        }
    }
    else if (cmd == "quit" || cmd == "q") {
        exited = true;
        if (mi_mode) mi_printf("~\"Quitting.\\n\"\n");
        else std::cout << "Quitting.\n";
        exit(0);
    }
}

void Debugger::run_mi() {
    mi_printf("=thread-group-added,id=\"i1\"\n");
    mi_printf("=thread-group-started,id=\"i1\",pid=\"%d\"\n", (int)getpid());
    mi_printf("=thread-created,id=\"1\",group-id=\"i1\"\n");
    mi_printf("*thread-selected,id=\"1\"\n");
    std::string line;
    if (mi_log_file) {
        fprintf(mi_log_file, "Waiting for stdin in run_mi...\n");
        fflush(mi_log_file);
    }
    while (!exited && std::getline(std::cin, line)) {
        if (mi_log_file) {
            fprintf(mi_log_file, "READ run_mi: %s\n", line.c_str());
            fflush(mi_log_file);
        }
        handle_command(parse_gdb_mi(line));
    }
    if (mi_log_file) {
        fprintf(mi_log_file, "run_mi stdin loop exited! cin.eof=%d, cin.fail=%d, cin.bad=%d, errno=%d (%s)\n",
                std::cin.eof(), std::cin.fail(), std::cin.bad(), errno, strerror(errno));
        fflush(mi_log_file);
    }
}

void Debugger::run_repl() {
    std::string err; gcnsim::launch_kernel(kernel, cfg, err);
}

std::string Debugger::serialize_state() const {
    std::string res;
    for (const auto& bp : breakpoints) {
        res += "B:" + std::to_string(bp.id) + "," + std::to_string(bp.address) + "," + bp.file + "," + std::to_string(bp.line) + "," + bp.func + ";";
    }
    return res;
}

void Debugger::deserialize_state(const std::string& state) {
    std::stringstream ss(state); std::string item;
    while (std::getline(ss, item, ';')) {
        if (item.size() > 2 && item[0] == 'B') {
            std::stringstream bss(item.substr(2));
            std::string s_id, s_addr, file, s_line, func;
            std::getline(bss, s_id, ',');
            std::getline(bss, s_addr, ',');
            std::getline(bss, file, ',');
            std::getline(bss, s_line, ',');
            std::getline(bss, func, ',');
            
            if (!s_id.empty()) {
                Breakpoint bp;
                bp.id = std::stoi(s_id);
                bp.address = std::stoull(s_addr);
                bp.file = file;
                bp.line = s_line.empty() ? 0 : std::stoul(s_line);
                bp.func = func;
                
                // If it is pending, try to resolve it now!
                bool resolved_now = false;
                if (bp.address == (uint64_t)-1) {
                    if (!bp.func.empty()) {
                        if (kernel.name.find(bp.func) != std::string::npos) {
                            bp.address = 0;
                            resolved_now = true;
                        }
                    } else if (!bp.file.empty() && bp.line != 0) {
                        uint64_t abs_addr = 0;
                        if (dwarf.get_address_from_location(bp.file, bp.line, abs_addr)) {
                            bp.address = abs_addr;
                            if (bp.address >= kernel.code_vma) bp.address -= kernel.code_vma;
                            resolved_now = true;
                        }
                    }
                }
                
                breakpoints.push_back(bp);
                if (bp.id >= next_bkpt_id) next_bkpt_id = bp.id + 1;

                if (resolved_now && mi_mode) {
                    print_mi_breakpoint("", bp, true);
                }
            }
        }
    }
}

void Debugger::print_mi_breakpoint(const std::string& token, const Breakpoint& bp, bool is_notification) {
    std::string addr_str;
    std::string pending_str;
    if (bp.address == (uint64_t)-1) {
        addr_str = "<PENDING>";
        std::string orig_loc = bp.file.empty() ? bp.func : (bp.file + ":" + std::to_string(bp.line));
        pending_str = ",pending=\"" + orig_loc + "\"";
    } else {
        char buf[64];
        sprintf(buf, "0x%016llx", (unsigned long long)(kernel.code_vma + bp.address));
        addr_str = buf;
    }

    std::string file_str = bp.file.empty() ? "" : (",file=\"" + bp.file + "\"");
    std::string fullname_str;
    if (!bp.file.empty()) {
        std::string full_path = bp.file;
        std::string dwarf_file;
        uint32_t dwarf_line = 0;
        if (bp.address != (uint64_t)-1 && dwarf.get_source_location(kernel.code_vma + bp.address, dwarf_file, dwarf_line)) {
            full_path = dwarf_file;
        } else if (!full_path.empty() && full_path[0] != '/') {
            // Resolve relative path to absolute
            char cwd_buf[1024];
            if (getcwd(cwd_buf, sizeof(cwd_buf))) {
                full_path = std::string(cwd_buf) + "/" + full_path;
            }
        }
        fullname_str = ",fullname=\"" + full_path + "\"";
    }

    std::string line_str = (bp.line == 0) ? "" : (",line=\"" + std::to_string(bp.line) + "\"");
    std::string orig_loc = bp.file.empty() ? bp.func : (bp.file + ":" + std::to_string(bp.line));

    if (is_notification) {
        mi_printf("=breakpoint-modified,bkpt={number=\"%d\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"%s\"%s,func=\"%s\"%s%s%s,times=\"0\",original-location=\"%s\"}\n",
                  bp.id, addr_str.c_str(), pending_str.c_str(), bp.func.c_str(), file_str.c_str(), fullname_str.c_str(), line_str.c_str(), orig_loc.c_str());
    } else {
        mi_printf("%s^done,bkpt={number=\"%d\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"%s\"%s,func=\"%s\"%s%s%s,times=\"0\",original-location=\"%s\"}\n",
                  token.c_str(), bp.id, addr_str.c_str(), pending_str.c_str(), bp.func.c_str(), file_str.c_str(), fullname_str.c_str(), line_str.c_str(), orig_loc.c_str());
    }
}

} // namespace gcnsim
