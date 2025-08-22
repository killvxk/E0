/*
Copyright 2025 Jex Amro (Square Labs LLC)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef LITETRACER_H
#define LITETRACER_H

#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>
#include <list>
#include <array>
#include <vector>
#include <sys/shm.h>
#include <fstream>
#include "mutex.h"

#include "directory.h"
#include "Sample.h"

#include "memoryguard.h"

// -----------------------------
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
  #include "Windows/debugger.h"
#elif __APPLE__
  #include "macOS/debugger.h"
#else
  #include "Linux/debugger.h"
#endif
// -----------------------------

// Triton ----------------------
#include <iostream>
#include <triton/ast.hpp>
#include <triton/context.hpp>
#include <triton/aarch64Specifications.hpp>
#include <triton/aarch64Semantics.hpp>
#include <triton/tritonToZ3.hpp>
using namespace triton;

#include <z3++.h>

#include <chrono>
using namespace std::chrono;

using namespace std;

#include <iostream>
#include <sstream>

#include <pthread.h>

#include <filesystem>

namespace fs = std::filesystem;

#include "E0PythonBridge.h"

#define MAX_WATCHPOINT_HITS 1000000

#define BREAKPOINT_ENABLE 5//481
#define BREAKPOINT_DISABLE 4//0

// Break only in privileged or user mode
// (PAC bits in the DBGWVRn_EL1 watchpoint control register)
#define S_USER ((uint32_t)(2u << 1))

#define BCR_ENABLE ((uint32_t)(1u))
#define WCR_ENABLE ((uint32_t)(1u))

// Watchpoint load/store
// (LSC bits in the DBGWVRn_EL1 watchpoint control register)
#define WCR_LOAD ((uint32_t)(1u << 3))
#define WCR_STORE ((uint32_t)(1u << 4))

// Enable breakpoint and watchpoint
#define MDE_ENABLE ((uint32_t)(1u << 15))

// Single instruction step
// (SS bit in the MDSCR_EL1 register)
#define SS_ENABLE ((uint32_t)(1u))

#define INVALID_PAGE_SIZE ((vm_size_t)(~0))

using Model = std::unordered_map<triton::usize, triton::engines::solver::SolverModel>;

struct PathHasher {
  uint64_t operator()(const std::list<uint64_t>& myList) const {
    uint64_t hashValue = 0;
    // Combine hash values of individual elements
    for (const auto& element : myList) {
      // Simple hash combining operation (XOR)
      hashValue ^= std::hash<uint64_t>{}(element);
      // Rotate combinedHash by 13 bits
      hashValue = (hashValue << 13) | (hashValue >> (sizeof(hashValue) * 8 - 13));
    }
    return hashValue;
  }
};

class LiteTracer : public Debugger {
public:
  Sample* sample;
  uint64_t __input_datalen = 0;
  void *   __input_data = 0;
  std::string __input_sample_filename;

  uint64_t invalidMemAccessAddress = 0;
  
  int trace_index = 0;
  int num_bytes = 1;
  int payload_size = 0;

  bool stop_point_reached = false;

  uint32_t watchpoint_hits = 0;
  uint32_t outOfRangeAccessCount = 0;

  stringstream memAccessHits;
  stringstream invalidMemAccess;

  uint32_t timeout;

  // symbolicDefs
  std::unordered_set<uint64_t>& symbolicDefs;
  Mutex& symbolicDefs_mutex;
  std::string symbolic_defs_dir;

  // invocations
  std::unordered_set<uint64_t>& invocations;
  Mutex& invocations_mutex;

  // Proccessed instruction addresses
  std::unordered_set<uint64_t> insnAddresses;

  // Coverage map <instruction address: number of hits>
  std::unordered_map<triton::uint64, triton::usize> iCoverage;

  MemoryGuard *memGuard;

  bool log_mem_access;
  bool log_symbolic_defs ;
  bool log_symbols;

  virtual void Init(int argc, char **argv) override;
  
  LiteTracer(E0PythonBridge& pyBridge,
             std::unordered_set<uint64_t>& symbolicDefs, 
             Mutex& symbolicDefs_mutex,
             std::unordered_set<uint64_t>& invocations, 
             Mutex& invocations_mutex);
  
  ~LiteTracer();

  triton::Context ctx = triton::Context(triton::arch::ARCH_AARCH64);
  void LogMemAccess(uint64_t memAccessPC, bool whileStepping = false);
  
  uint64_t SymArgsMask();
  void LogArgSymbolicDef(uint64_t callerPC, uint64_t calleePC);
  void LogRetSymbolicDef(uint64_t fromPC, uint64_t toPC);
  void LogMemSymbolicDef();

  // Concrete AST min/max bound values,  and SAT status.
  struct MinMaxASTBounds {
    bool minSat = false;                // SAT status
    bool maxSat = false;                // SAT status
    std::optional<uint64_t> min;        // Store evaluated min value as uint64_t
    std::optional<uint64_t> max;        // Store evaluated max value as uint64_t
    bool computeModels = false;         // Flag to indicate whether to compute models
  };

  void EvaluateMinMaxASTBounds(const triton::ast::SharedAbstractNode &regAst, MinMaxASTBounds &bounds, unsigned timeout_ms = 3000); // Default timeout: 60000ms

  void LogMemAccessHits();
  void SaveInvalidMemAccess(const char * filename);
  void StoreMemAccess(const triton::arch::MemoryAccess &mem);

  void LogSymbol(uint64_t address, bool newline = true);
  void LogSymbol2(uint64_t address);
  void LogInvalidMemAccessSymbol(uint64_t address);
  void LogSymbolizedInstruction(triton::arch::Instruction &instruction);
  void LogSymbolicMemoryInfo(uint64_t address);
  void LogSymbolicRegistersInfo(triton::arch::Instruction &instruction);
  
  vm_size_t PageSize();
  size_t MaxBytesLeftInPage(mach_vm_address_t address, mach_vm_size_t size);
  kern_return_t ReadTargetMemory(uint64_t address, size_t size, void *buf);
  void MemWriteCallback(triton::Context &ctx, const triton::arch::MemoryAccess &mem, const triton::uint512 &value);
  void MemReadCallback(triton::Context &ctx, const triton::arch::MemoryAccess &mem);
  
  uint64_t AddressToModuleBase(uint64_t address);
  uint64_t AddressToSymbolAddress(uint64_t module_base, uint64_t address);

  std::vector<triton::uint8> ModelToVector(const Model& model);
  void HexDump(const std::vector<triton::uint8>& v);
  void DiffHexDump(const std::vector<triton::uint8>& v);

  uintptr_t StripPAC(uintptr_t pac_ptr);

  std::list<uint32_t> trace_targets;
  std::unordered_set<uint64_t> written_memory_addresses;
  
  std::unordered_set<thread_act_port_t> steppingThreads;
  std::unordered_set<thread_act_port_t> suspendedThreads;

  int num_steppingThreads;

  bool tracing = false;
  bool staging = false;
  bool escape_loops = true;
  bool debug = false;

  uint64_t memory_access_addr;
  uint32_t memory_access_size;

  std::map<uint64_t, std::string> loaded_modules;
  std::map<uint64_t, std::string> loaded_modules_pathes;

  triton::arch::Instruction instruction;

protected:
  virtual bool OnWatchpoint() override;
  virtual bool OnMemAccess() override;
  virtual void OnSingleStep() override;
  virtual void OnHardwareBreakpoint() override;
  virtual void OnUnhandledException() override;
  virtual void OnInputpoint() override;
  virtual void OnStoppoint() override;
  virtual void OnModuleLoaded(void *mach_header_addr, char *base_name, char *path) override;
  
  virtual void OnProcessCreated() override;
  virtual void OnProcessExit() override;

  virtual bool OnException(Exception *exception_record) override;
  
  uint64_t GetThread_ID_FromPort(mach_port_t thread_port);
  void LogThreadState(arm_thread_state64_t *state);
  void LogCpuState();
  void LogCpuState(ofstream &outFile);
  
  py::list GetPyCpuContext();
  py::dict GetPySymRegAST(triton::arch::Register& reg);

  void TaskGetState();
  void ThreadSetState(thread_act_port_t thread, thread_state_t thread_state);
  void SuspendThreads(thread_act_port_t thread = 0);
  void ResumeThreads();//thread_act_port_t thread = 0);

  uint8_t GetFreeWVR();
  void ThreadAddHardwareBreakpoint(uint64_t step_pc);
  
  void SetSingleStep();
  void RemoveLocalVariables(uint64_t sp);
  void ConcretizeUnGuardedSymbolicMemory();
  void CheckGuardedSymbolicMemory();
  
  //void MemReadCallback(triton::Context &ctx, const triton::arch::MemoryAccess &mem);
  //static void MemWriteCallback(triton::Context &ctx, const triton::arch::MemoryAccess &mem, const triton::uint512 &value);
  void InitTriton();

  uint64_t OnWatchpointAccess();
  void AddToCoverage(uint64_t pc);
  uint64_t Emulate(uint64_t pc);
  void SetContextConcreteRegisterValues(arm_thread_state64_t * state);
  
  void GetFunctionInfo(uint64_t address, string &module, string &function, uint64_t &func_start, uint64_t &func_end);
  void OnSymbolizedInstruction(triton::arch::Instruction &instruction, string &module, string &function);
  void OnUnsymbolizedInstruction(triton::arch::Instruction &instruction, string &module, string &function);
  
  uint32_t GetRVA(uint64_t address);

  uint64_t PageWCR();
  uint64_t BaseWatchpointWCR();

private:
  uint64_t evtime = 0;

  E0PythonBridge& pyBridge;

  vm_size_t m_page_size;

  uint64_t stepCount = 0;

  uint64_t pageWCR = 0;
  uint64_t baseWatchpointWCR = 0;

  bool initInstruction = true;
  bool doEmulate = false;

  uint64_t init_fp = 0;
  uint64_t last_sp = 0;

  thread_act_port_t lastMemAccessThread = 0;

  std::unordered_set<uint8_t> pacOpCodes = {48, 80, 232, 240, 241};

  map<string, map<string, map<uint32_t, vector<array<uint64_t, 4>>>>> memReadAccessMap;
  uint32_t memReadAccessCount = 0;

  char *input_module;
  int input_breakpoint = 0;
  int stop_breakpoint = 0;
  uint64_t input_address = 0;
  uint64_t trace_address = 0;
  int input_register = 0;
  int input_register_offset = 0;
  uint8_t trace_byte = 0;
  bool silence = false;

  uint64_t init_bp_addr = 0;
  uint64_t stop_bp_addr = 0;

  thread_act_port_array_t threads;
  mach_msg_type_number_t threadCount;

  arm_debug_state64_t task_state;
  arm_debug_state64_t init_task_state;

  uint32_t threads_count = 0;

  struct thread_data {
    uint32_t id;
    uint32_t watchpoints_count;
    uint32_t traced;
    uint32_t untraced;
  };

  std::map<mach_port_t, thread_data> threads_data;

  std::string line = std::string(72,'#');
  std::string line2 = std::string(72,'=');

  int hardware_breakpoint_hits = 0;

  uint64_t last_same_pc = 0;
  uint64_t mem_hit_pc;

};

#endif // LITETRACER_H
