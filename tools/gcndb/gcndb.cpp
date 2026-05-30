#include "gcnsim/debugger_core.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

using namespace gcnsim;

int main(int argc, char** argv) {
    bool interpreter_mi = false;
    bool quiet = false;
    std::string hsaco_path;
    std::vector<std::string> host_args;
    bool collecting_args = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--args") { collecting_args = true; continue; }
        if (collecting_args) { host_args.push_back(arg); continue; }

        if (arg == "--interpreter=mi" || arg == "--interpreter=mi2" || arg == "-i") {
            interpreter_mi = true;
            if (arg == "-i" && i + 1 < argc) {
                std::string next = argv[i+1];
                if (next.find("mi") == 0) {
                    i++; 
                    if (i + 1 < argc && std::string(argv[i+1]).find("/dev/") == 0) i++;
                }
            }
        } else if (arg == "--nx" || arg == "-nx" || arg == "-q" || arg == "--quiet" || arg == "--silent") {
            quiet = true;
        } else if (arg == "new-ui") {
            if (i + 2 < argc) i += 2;
        } else if (!arg.empty() && arg[0] != '-') {
            if (hsaco_path.empty() && arg != "mi" && arg.find("/dev/") != 0 && arg.find("new-ui") == std::string::npos) {
                hsaco_path = arg;
            }
        }
    }

    Debugger db;
    db.mi_mode = interpreter_mi;
    set_global_debugger(&db);

    bool file_loaded = false;
    auto try_load_file = [&](const std::string& path) -> bool {
        // Simple heuristic: if it's a host app, we just set host_args
        bool is_host_app = false;
        int fd = open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
            uint8_t magic[4];
            if (read(fd, magic, 4) == 4 && magic[0] == 0x7f && magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F') {
                lseek(fd, 18, SEEK_SET);
                uint16_t machine;
                if (read(fd, &machine, 2) == 2 && machine != 224) is_host_app = true;
            }
            close(fd);
        }

        if (is_host_app) { host_args = { path }; file_loaded = true; return true; }

        std::string error_msg;
        auto hsaco = load_hsaco(path, error_msg);
        if (!hsaco) {
            if (db.mi_mode) db.mi_printf("^error,msg=\"%s\"\n", error_msg.c_str());
            return false;
        }
        db.module = std::move(*hsaco);
        db.kernel = db.module.kernels[0];
        db.hsaco_path = path;
        db.dwarf.load_from_elf(db.module.elf_data);
        db.cfg.grid_x = 256; db.cfg.grid_y = 1; db.cfg.grid_z = 1;
        db.cfg.block_x = 256; db.cfg.block_y = 1; db.cfg.block_z = 1;
        db.cfg.debug_hook = global_debug_hook;
        file_loaded = true;
        return true;
    };

    if (!hsaco_path.empty()) try_load_file(hsaco_path);

    auto run_host_app = [&]() {
        if (host_args.empty()) return;
        std::string exe = host_args[0];
        std::vector<char*> c_args;
        for (auto& s : host_args) c_args.push_back(const_cast<char*>(s.c_str()));
        c_args.push_back(nullptr);

        // Resolve absolute path to libgcnsim_hip.so relative to gcndb executable location
        char exe_path[1024];
        ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path)-1);
        std::string lib_path = "./build/libgcnsim_hip.so"; // fallback
        if (len != -1) {
            exe_path[len] = '\0';
            std::string path_str(exe_path);
            size_t last_slash = path_str.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                std::string exe_dir = path_str.substr(0, last_slash);
                std::string candidate = exe_dir + "/libgcnsim_hip.so";
                if (access(candidate.c_str(), F_OK) == 0) {
                    lib_path = candidate;
                }
            }
        }

        setenv("LD_PRELOAD", lib_path.c_str(), 1);
        setenv("GCNSIM_HIJACK", "1", 1);
        setenv("GCNSIM_DEBUG", "1", 1);
        if (db.mi_mode) {
            setenv("GCNSIM_MI_MODE", "1", 1);
            int mi_fd = dup(STDOUT_FILENO);
            if (mi_fd >= 0) {
                setenv("GCNSIM_MI_FD", std::to_string(mi_fd).c_str(), 1);
                dup2(STDERR_FILENO, STDOUT_FILENO);
            }
        }
        setenv("GCNSIM_DB_STATE", db.serialize_state().c_str(), 1);

        execvp(exe.c_str(), c_args.data());
        perror("execvp failed");
        exit(1);
    };

    if (db.mi_mode) {
        db.mi_printf("=thread-group-added,id=\"i1\"\n");
        db.mi_printf("=thread-group-started,id=\"i1\",pid=\"%d\"\n", (int)getpid());
        db.mi_printf("=thread-created,id=\"1\",group-id=\"i1\"\n");
        db.mi_printf("*thread-selected,id=\"1\"\n");
        std::string line;
        while (!db.exited && std::getline(std::cin, line)) {
            GdbMiCommand cmd = parse_gdb_mi(line);
            if (!host_args.empty() && (cmd.command == "-exec-run" || cmd.command == "run" || cmd.command == "r")) {
                db.mi_printf("%s^running\n", cmd.token.c_str());
                run_host_app();
            }
            if (cmd.command == "-file-exec-and-symbols" || cmd.command == "file") {
                if (!cmd.args.empty()) try_load_file(cmd.args[0]);
                db.mi_printf("%s^done\n", cmd.token.c_str());
            } else {
                db.handle_command(cmd);
            }
        }
    } else {
        if (!quiet) std::cout << "gcndb — GCN GPU ISA debugger\n";
        std::string line;
        while (!db.exited) {
            std::cout << "(gdb) " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (!host_args.empty() && (line.find("run") == 0 || line.find("r") == 0)) run_host_app();
            std::stringstream ss(line); std::string cmd; ss >> cmd;
            if (cmd == "file") {
                std::string path; ss >> path;
                if (!path.empty() && try_load_file(path)) std::cout << "Loaded " << path << "\n";
            } else if (cmd == "run" || cmd == "r") {
                if (!file_loaded) { std::cout << "No kernel loaded.\n"; continue; }
                std::string err; gcnsim::launch_kernel(db.kernel, db.cfg, err);
            } else if (cmd == "quit" || cmd == "q") break;
            else db.handle_command(parse_gdb_mi(line));
        }
    }

    return 0;
}
