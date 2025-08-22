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


#define _CRT_SECURE_NO_WARNINGS
#define BREAKPOINT_HARDWARE 0x20
#define BREAKPOINT_INPUTPOINT 0x40
#define BREAKPOINT_STOPPOINT 0x80
#define BREAKPOINT_SINGLESTEP 0x100

#include "litetracer.h"
#include "common.h"

LiteTracer::LiteTracer(
  E0PythonBridge& pyBridge,
  std::unordered_set<uint64_t>& symbolicDefs, 
  Mutex& symbolicDefs_mutex,
  std::unordered_set<uint64_t>& invocations, 
  Mutex& invocations_mutex) 
  : pyBridge(pyBridge)
  , symbolicDefs(symbolicDefs)
  , symbolicDefs_mutex(symbolicDefs_mutex)
  , invocations(invocations)
  , invocations_mutex(invocations_mutex) {

  num_steppingThreads = 0;

  InitTriton();

  pageWCR = PageWCR();
  baseWatchpointWCR = BaseWatchpointWCR();

  memory_access_addr = 0;
  memory_access_size = 0;
}

LiteTracer::~LiteTracer() {
  // Clean up if needed
}

void LiteTracer::Init(int argc, char **argv) {
  __input_datalen = 0;
  __input_data = 0;
  invalidMemAccessAddress = 0;
  trace_index = 0;
  num_bytes = 1;
  payload_size = 0;
  stop_point_reached = false;
  watchpoint_hits = 0;
  outOfRangeAccessCount = 0;

  Debugger::Init(argc, argv);

  input_module = GetOption("-module", argc, argv);
  input_breakpoint = GetIntOption("-bp", argc, argv, 0);
  stop_breakpoint = GetIntOption("-stop", argc, argv, 0);
  input_register = GetIntOption("-reg", argc, argv, -1);
  input_register_offset = GetIntOption("-reg-offset", argc, argv, 0);
  silence = GetBinaryOption("-silence", argc, argv, false);
  debug = GetBinaryOption("-debug", argc, argv, false);
  log_mem_access = GetBinaryOption("-log_mem_access", argc, argv, false);
  log_symbolic_defs  = GetBinaryOption("-log_symdefs", argc, argv, false);
  log_symbols = GetBinaryOption("-log_symbols", argc, argv, false);

  if (const char* env_index = std::getenv("index")) {
    trace_index = std::stoi(env_index);
  } else {
    trace_index = GetIntOption("-i", argc, argv, 0);
  }

  if (const char* env_num_bytes = std::getenv("num_bytes")) {
    num_bytes = std::stoi(env_num_bytes);
  } else {
    num_bytes = GetIntOption("-n", argc, argv, 1);
  }
}

void LiteTracer::InitTriton(){
  /* Init the triton context */
  /* Setup some Triton optimizations */
  ctx.setMode(triton::modes::ONLY_ON_SYMBOLIZED, true);
  ctx.setMode(triton::modes::AST_OPTIMIZATIONS, true);
  //ctx.setSolver(triton::engines::solver::SOLVER_Z3);
  ctx.setSolver(triton::engines::solver::SOLVER_BITWUZLA);
  // Define the Python syntax
  ctx.setAstRepresentationMode(ast::representations::PYTHON_REPRESENTATION);
}

void LiteTracer::OnModuleLoaded(void *mach_header_addr, char *base_name, char *path) {
  
  loaded_modules.insert({(uint64_t)mach_header_addr,string(base_name)});
  loaded_modules_pathes.insert({(uint64_t)mach_header_addr,string(path)});

  if (strcmp(input_module, base_name) == 0) {
    if (input_breakpoint && input_register >= 0) {
      init_bp_addr = (uint64_t)mach_header_addr + (uint64_t)input_breakpoint;
      if (debug) LOG("setting init breakpoint @ %llx\n",init_bp_addr);
      AddBreakpoint((void*)init_bp_addr,BREAKPOINT_INPUTPOINT);
    }

    if (stop_breakpoint) {
      stop_bp_addr = (uint64_t)mach_header_addr + (uint64_t)stop_breakpoint;
      if (debug) LOG("setting stop breakpoint @ %llx\n",stop_bp_addr);
      AddBreakpoint((void*)stop_bp_addr,BREAKPOINT_STOPPOINT);
    }
  }
  Debugger::OnModuleLoaded(mach_header_addr, base_name, path);
}

void LiteTracer::ThreadAddHardwareBreakpoint(uint64_t step_pc){
  
  arm_debug_state64_t thread_state;
  memset(&thread_state, 0, sizeof(arm_debug_state64_t));
  mach_msg_type_number_t state_count = ARM_DEBUG_STATE64_COUNT;
  kern_return_t kr;

  thread_state.__bvr[0] = step_pc;
  thread_state.__bcr[0] = BREAKPOINT_ENABLE;
  ThreadSetState(mach_exception->thread_port, (thread_state_t)&thread_state);
  num_steppingThreads += 1;

  Breakpoint *new_breakpoint = new Breakpoint;
  new_breakpoint->address = (void *)step_pc;
  new_breakpoint->type = BREAKPOINT_HARDWARE;
  breakpoints.push_back(new_breakpoint);
}

void LiteTracer::LogCpuState(ofstream &outFile) {
  outFile << " x0: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x0)) << endl;
  outFile << " x1: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x1)) << endl;
  outFile << " x2: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x2)) << endl;
  outFile << " x3: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x3)) << endl;
  outFile << " x4: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x4)) << endl;
  outFile << " x5: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x5)) << endl;
  outFile << " x6: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x6)) << endl;
  outFile << " x7: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x7)) << endl;
  outFile << " x8: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x8)) << endl;
  outFile << " x9: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x9)) << endl;
  outFile << "x10: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x10)) << endl;
  outFile << "x11: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x11)) << endl;
  outFile << "x12: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x12)) << endl;
  outFile << "x13: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x13)) << endl;
  outFile << "x14: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x14)) << endl;
  outFile << "x15: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x15)) << endl;
  outFile << "x16: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x16)) << endl;
  outFile << "x17: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x17)) << endl;
  outFile << "x18: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x18)) << endl;
  outFile << "x19: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x19)) << endl;
  outFile << "x20: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x20)) << endl;
  outFile << "x21: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x21)) << endl;
  outFile << "x22: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x22)) << endl;
  outFile << "x23: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x23)) << endl;
  outFile << "x24: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x24)) << endl;
  outFile << "x25: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x25)) << endl;
  outFile << "x26: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x26)) << endl;
  outFile << "x27: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x27)) << endl;
  outFile << "x28: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x28)) << endl;
  outFile << " fp: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x29)) << endl;
  outFile << " lr: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x30)) << endl;
  outFile << " sp: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_sp)) << endl;
  outFile << " pc: 0x" << hex << static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_pc)) << endl;
  outFile << "spsr: 0x" << hex << static_cast<uint32_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_spsr)) << endl;
}

void LiteTracer::LogCpuState() {
  printf(GREEN " x0:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x0)));
  printf(GREEN "   x1:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x1)));
  printf(GREEN "   x2:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x2)));
  printf(GREEN "   x3:" WHITE " 0x%016llx\n", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x3)));

  printf(GREEN" x4:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x4)));
  printf(GREEN"   x5:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x5)));
  printf(GREEN"   x6:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x6)));
  printf(GREEN"   x7:" WHITE " 0x%016llx\n", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x7)));

  printf(GREEN" x8:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x8)));
  printf(GREEN"   x9:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x9)));
  printf(GREEN"  x10:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x10)));
  printf(GREEN"  x11:" WHITE " 0x%016llx\n", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x11)));

  printf(GREEN"x12:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x12)));
  printf(GREEN"  x13:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x13)));
  printf(GREEN"  x14:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x14)));
  printf(GREEN"  x15:" WHITE " 0x%016llx\n", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x15)));

  printf(GREEN"x16:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x16)));
  printf(GREEN"  x17:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x17)));
  printf(GREEN"  x18:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x18)));
  printf(GREEN"  x19:" WHITE " 0x%016llx\n", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x19)));

  printf(GREEN"x20:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x20)));
  printf(GREEN"  x21:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x21)));
  printf(GREEN"  x22:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x22)));
  printf(GREEN"  x23:" WHITE " 0x%016llx\n", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x23)));

  printf(GREEN"x24:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x24)));
  printf(GREEN"  x25:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x25)));
  printf(GREEN"  x26:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x26)));
  printf(GREEN"  x27:" WHITE " 0x%016llx\n", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x27)));

  printf(GREEN"x28:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x28)));
  printf(YELLOW"   fp:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x29)));
  printf(YELLOW"   lr:" WHITE " 0x%016llx\n", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x30)));

  printf(YELLOW" sp:" WHITE " 0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_sp)));
  printf(RED"   pc:" WHITE " 0x%016llx\n", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_pc)));
  printf(YELLOW"spsr:" WHITE " 0x%08x\n", static_cast<uint32_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_spsr)));
}

py::list LiteTracer::GetPyCpuContext() {

  py::list cpu_context;
  char buffer[19];  // "0x" + 16 hex digits + null terminator

  for (int i = 0; i < 31; i++) {
    auto regId = static_cast<triton::arch::register_e>(triton::arch::ID_REG_AARCH64_X0 + i);
    auto reg = ctx.getRegister(regId);
    snprintf(buffer, sizeof(buffer), "0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(reg)));
    cpu_context.append(buffer);
  }
  snprintf(buffer, sizeof(buffer), "0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_sp)));
  cpu_context.append(buffer);
  snprintf(buffer, sizeof(buffer), "0x%016llx", static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_pc)));
  cpu_context.append(buffer);
  snprintf(buffer, sizeof(buffer), "0x%08x", static_cast<uint32_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_spsr)));
  cpu_context.append(buffer);

  return cpu_context;
}

py::dict LiteTracer::GetPySymRegAST(triton::arch::Register& reg) {
  py::dict py_symbolic_reg;
  auto size = reg.getSize();
  py_symbolic_reg["size"] = py::cast(size);
  if (ctx.isRegisterSymbolized(reg)) {
    auto regAst = ctx.getRegisterAst(reg);

    // cout << "AST: " << regAst << endl;
    // cout << "AST Unroll: " << unroll(regAst) << endl;
    // cout << "Simplify: " << ctx.simplify(regAst) << endl;

    uint64_t evaluation_begin_time = GetCurTime();

    MinMaxASTBounds bounds;

    Py_BEGIN_ALLOW_THREADS
    EvaluateMinMaxASTBounds(regAst, bounds);
    Py_END_ALLOW_THREADS

    uint64_t time_now = GetCurTime();
    uint64_t evaluation_time = time_now - evaluation_begin_time;
    evtime += evaluation_time;

    if (bounds.minSat)
      py_symbolic_reg["min_value"] = py::cast(bounds.min);
    else
      py_symbolic_reg["min_value"] = py::none();

    if (bounds.maxSat)
      py_symbolic_reg["max_value"] = py::cast(bounds.max);
    else
      py_symbolic_reg["max_value"] = py::none();


    auto level = regAst->getLevel();
    py_symbolic_reg["ast_level"] = py::cast(level);

    if (level <= 55) {
      stringstream ss;
      ss << unroll(regAst);
      py_symbolic_reg["ast"] = py::cast(ss.str());
    } else {
      py_symbolic_reg["ast"] = py::cast("large");
    }
  } else {
    py_symbolic_reg["ast"] = py::none();
  }
  return py_symbolic_reg;
}

void LiteTracer::LogThreadState(arm_thread_state64_t *state) {
  printf(GREEN " x0:" WHITE " 0x%016llx", state->__x[0]);
  printf(GREEN "   x1:" WHITE " 0x%016llx", state->__x[1]);
  printf(GREEN "   x2:" WHITE " 0x%016llx", state->__x[2]);
  printf(GREEN "   x3:" WHITE " 0x%016llx\n", state->__x[3]);

  printf(GREEN" x4:" WHITE " 0x%016llx", state->__x[4]);
  printf(GREEN"   x5:" WHITE " 0x%016llx", state->__x[5]);
  printf(GREEN"   x6:" WHITE " 0x%016llx", state->__x[6]);
  printf(GREEN"   x7:" WHITE " 0x%016llx\n", state->__x[7]);

  printf(GREEN" x8:" WHITE " 0x%016llx", state->__x[8]);
  printf(GREEN"   x9:" WHITE " 0x%016llx", state->__x[9]);
  printf(GREEN"  x10:" WHITE " 0x%016llx", state->__x[10]);
  printf(GREEN"  x11:" WHITE " 0x%016llx\n", state->__x[11]);

  for (int i=12; i<28; i+=4) {
    printf(GREEN"x%d:" WHITE " 0x%016llx", i,state->__x[i]);
    printf(GREEN"  x%d:" WHITE " 0x%016llx", i+1,state->__x[i+1]);
    printf(GREEN"  x%d:" WHITE " 0x%016llx", i+2,state->__x[i+2]);
    printf(GREEN"  x%d:" WHITE " 0x%016llx\n", i+3,state->__x[i+3]);
  }

  printf(GREEN"x28:" WHITE " 0x%016llx", state->__x[28]);
  printf(YELLOW"   fp:" WHITE " 0x%016llx", state->__fp);
  printf(YELLOW"   lr:" WHITE " 0x%016llx\n", state->__lr);
    
  printf(YELLOW" sp:" WHITE " 0x%016llx", state->__sp);
  printf(RED"   pc:" WHITE " 0x%016llx\n", state->__pc);
  printf(YELLOW" cpsr:" WHITE " 0x%08x\n", state->__cpsr);
}

void LiteTracer::TaskGetState(){
  // get task debug state
  kern_return_t kr;
  mach_msg_type_number_t state_count = ARM_DEBUG_STATE64_COUNT;
  kr = task_get_state(mach_target->Task(), ARM_DEBUG_STATE64, (thread_state_t)&task_state, &state_count);
  if (kr != KERN_SUCCESS) {
    FATAL("task_get_state error: %s", mach_error_string(kr));
  }
  state_count = ARM_DEBUG_STATE64_COUNT;
  kr = task_get_state(mach_target->Task(), ARM_DEBUG_STATE64, (thread_state_t)&init_task_state, &state_count);
  if (kr != KERN_SUCCESS) {
    FATAL("task_get_state error: %s", mach_error_string(kr));
  }
}

void LiteTracer::ThreadSetState(thread_act_port_t thread, thread_state_t thread_state) {
  kern_return_t kr;
  mach_msg_type_number_t state_count = ARM_DEBUG_STATE64_COUNT;
  kr = thread_set_state(thread, ARM_DEBUG_STATE64, thread_state, state_count);
  if (kr != KERN_SUCCESS) {
    kr = thread_set_state(thread, ARM_DEBUG_STATE64, thread_state, state_count);
    if (kr != KERN_SUCCESS) {
      LOG(RED"[ThreadSetState] thread_set_state error: %s\n", mach_error_string(kr));
    }
  }
}

void LiteTracer::SuspendThreads(thread_act_port_t thread) {
  kern_return_t kr;
  mach_msg_type_number_t state_count = ARM_DEBUG_STATE64_COUNT;

  kr = task_threads(mach_target->Task(), &threads, &threadCount);
  if (kr != KERN_SUCCESS) {
      LOG("  sample: %s\n", sample->filename.c_str());
      if (invalidMemAccessAddress) {
        LogInvalidMemAccessSymbol(instruction.getAddress());
        invalidMemAccess << instruction << endl;
        LOG("invalid MemAccess!\n");
      }
      throw std::runtime_error("[GetThreads] Could not get task_threads\n");// with error: %s", mach_error_string(kr));
  }

  memGuard->clearTaskState();

  for (int i = 0; i < threadCount; i++) {
    kr = thread_set_state(threads[i], ARM_DEBUG_STATE64, (thread_state_t)&memGuard->taskState, state_count);
    if (kr != KERN_SUCCESS) {
      sleep(5);
      kr = thread_set_state(threads[i], ARM_DEBUG_STATE64, (thread_state_t)&memGuard->taskState, state_count);
      if (kr != KERN_SUCCESS) {
        LOG(RED"[SuspendThreads] thread_set_state error: %s\n", mach_error_string(kr));
        continue;
      }
    }

    if (threads[i] == mach_exception->thread_port)
      continue;
    if (suspendedThreads.find(threads[i]) == suspendedThreads.end()) {
      thread_suspend(threads[i]);
      suspendedThreads.insert(threads[i]);
    }
  }

  vm_deallocate(mach_task_self(), (vm_address_t)threads, threadCount * sizeof(thread_act_t));
  threadCount = 0;
}

void LiteTracer::ResumeThreads() {
  kern_return_t kr;
  mach_msg_type_number_t state_count = ARM_DEBUG_STATE64_COUNT;

  kr = task_threads(mach_target->Task(), &threads, &threadCount);
  if (kr != KERN_SUCCESS) {
      LOG("  sample: %s\n", sample->filename.c_str());
      if (invalidMemAccessAddress) {
        LogInvalidMemAccessSymbol(instruction.getAddress());
        invalidMemAccess << instruction << endl;
        LOG("invalid MemAccess!\n");
      }
      throw std::runtime_error("[GetThreads] Could not get task_threads\n");// with error: %s", mach_error_string(kr));
  }

  for (int i = 0; i < threadCount; i++) {
    state_count = ARM_DEBUG_STATE64_COUNT;
    kr = thread_set_state(threads[i], ARM_DEBUG_STATE64, (thread_state_t)&memGuard->taskState, state_count);
    if (kr != KERN_SUCCESS) {
      LOG(RED"[ResumeThreads] thread_set_state error: %s\n", mach_error_string(kr));
      continue;
    }
    thread_resume(threads[i]);
    // Deallocating the thread port
    mach_port_deallocate(mach_task_self(), threads[i]);
  }

  state_count = ARM_DEBUG_STATE64_COUNT;
  kr = task_set_state(mach_target->Task(), ARM_DEBUG_STATE64, (thread_state_t)&memGuard->taskState, state_count);
  if (kr != KERN_SUCCESS) {
      LOG("  sample: %s\n", sample->filename.c_str());
      LOG(RED"task_set_state error: %s", mach_error_string(kr));
  }

  vm_deallocate(mach_task_self(), (vm_address_t)threads, threadCount * sizeof(thread_act_t));
  threadCount = 0;
}

/*
 * get a free WVR register
 * return: a free wvr_id
 * if no free wvr_id, return 0xFF  
 */
uint8_t LiteTracer::GetFreeWVR() {

  for(uint8_t wvr_id = 0; wvr_id<4; wvr_id++) {
    if (task_state.__wvr[wvr_id] == 0) return wvr_id;
  }
  return 0xFF;
}

void LiteTracer::HexDump(const std::vector<triton::uint8>& v) {
    const size_t bytesPerLine = 16;
    size_t bytesRemaining = v.size();
    size_t offset = 0;

    while (bytesRemaining > 0) {
        // Print the offset
        cout << "  " << setw(8) << setfill('0') << hex << offset << ": ";

        // Print bytes in hex
        size_t bytesToPrint = min(bytesRemaining, bytesPerLine);
        for (size_t i = 0; i < bytesToPrint; ++i) {
            cout << setw(2) << setfill('0') << hex << static_cast<int>(v[offset + i]) << " ";
        }

        // If there are fewer than bytesPerLine bytes on this line, pad with spaces
        if (bytesToPrint < bytesPerLine) {
            cout << string((bytesPerLine - bytesToPrint) * 3, ' ');
        }

        // Print ASCII representation
        for (size_t i = 0; i < bytesToPrint; ++i) {
            char ch = static_cast<char>(v[offset + i]);
            if (ch < 32 || ch > 126) {
                cout << '.';
            } else {
                cout << ch;
            }
        }
        cout << endl;
        bytesRemaining -= bytesToPrint;
        offset += bytesToPrint;
    }
    cout << dec << endl;
}

void LiteTracer::DiffHexDump(const std::vector<triton::uint8>& v) {
  const size_t bytesPerLine = 16;
  size_t bytesRemaining = v.size();
  size_t offset = 0;

  while (bytesRemaining > 0) {
    // Print the offset
    cout << "  " << setw(8) << setfill('0') << hex << offset << ": ";

    // Print bytes in hex
    size_t bytesToPrint = min(bytesRemaining, bytesPerLine);
    for (size_t i = 0; i < bytesToPrint; ++i) {
      int value = static_cast<int>(v[offset + i]);
      int input_value = static_cast<int>(*((uint8_t*)__input_data+trace_index+offset+i));
      if (value == input_value)
        cout << setw(2) << setfill('0') << hex << value << " ";  
      else
        cout << GREEN << setw(2) << setfill('0') << hex << value << WHITE << " ";
    }

    // If there are fewer than bytesPerLine bytes on this line, pad with spaces
    if (bytesToPrint < bytesPerLine) {
      cout << string((bytesPerLine - bytesToPrint) * 3, ' ');
    }

    // Print ASCII representation
    for (size_t i = 0; i < bytesToPrint; ++i) {
      char ch = static_cast<char>(v[offset + i]);
      if (ch < 32 || ch > 126) {
          cout << '.';
      } else {
          cout << ch;
      }
    }
    cout << endl;
    bytesRemaining -= bytesToPrint;
    offset += bytesToPrint;
  }
  cout << dec << endl;
}

std::vector<triton::uint8> LiteTracer::ModelToVector(const Model& model) {
  
  std::vector<triton::uint8> ret;

  unsigned long size = 0;

  for(auto &kv: model) {
    if (kv.first > size) size = kv.first;
  }
  size += 1;

  ret.resize(size);
  
  for (triton::usize i = 0; i < size; i++) {
    if (model.find(i) == model.end())
      ret[i] = *((uint8_t*)__input_data+trace_index+i);
    else
      ret[i] = triton::utils::cast<triton::uint8>(model.at(i).getValue());
  }
  return ret;
}

uint64_t LiteTracer::BaseWatchpointWCR() {
  
  uint32_t byte_address_select = 0xff;
  byte_address_select = byte_address_select << 5;
  uint32_t MASK = 31 << 24; // 14 bits for 16k page size = 0x3fff
  uint64_t wcr = byte_address_select | // Which bytes that follow
                                                // the DVA that we will watch
                                         MASK | // MASK
                                       S_USER | // Stop only in user mode
                                     WCR_LOAD | // Stop on read access
                                    WCR_STORE | // Stop on write access
                                   WCR_ENABLE;  // Enable this watchpoint;
  return wcr;
}

uint64_t LiteTracer::PageWCR() {
  
  uint32_t byte_address_select = 0xff;
  byte_address_select = byte_address_select << 5;
  uint32_t MASK = 14 << 24; // 14 bits for 16k page size = 0x3fff
  uint64_t wcr = byte_address_select | // Which bytes that follow
                                                // the DVA that we will watch
                                         MASK | // MASK
                                       S_USER | // Stop only in user mode
                                     WCR_LOAD | // Stop on read access
                                    WCR_STORE | // Stop on write access
                                   WCR_ENABLE;  // Enable this watchpoint;
  return wcr;
}

/*
 * Set Single Step Flag
 */
void LiteTracer::SetSingleStep() {
  kern_return_t kr;
  arm_debug_state64_t debug_state;
  mach_msg_type_number_t state_count = ARM_DEBUG_STATE64_COUNT;
  
  kr = thread_get_state(mach_exception->thread_port, ARM_DEBUG_STATE64, (thread_state_t)&debug_state, &state_count);
  if (kr != KERN_SUCCESS) {
    FATAL("thread_get_state error: %s", mach_error_string(kr));
  }
  
  if (debug) LOG("Setting thread %x to single step\n", mach_exception->thread_port);//GetThread_ID_FromPort(mach_exception->thread_port));
  debug_state.__mdscr_el1 |= 0x1; /* Bit 0 is SS (Hardware Single Step) */
  
  kr = thread_set_state(mach_exception->thread_port, ARM_DEBUG_STATE64, (thread_state_t)&debug_state, state_count);
  if (kr != KERN_SUCCESS) {
    FATAL("[SetSingleStep] thread_set_state error: %s\n", mach_error_string(kr));
  }
}

void LiteTracer::CheckGuardedSymbolicMemory() {
  if (debug) LOG("Guarded %d | %ld SymbolicMemory | Written %ld\n", 
    memGuard->totalGuarded, 
    ctx.getSymbolicMemory().size(),
    written_memory_addresses.size());
  memGuard->logPagesInfo();
}

void LiteTracer::ConcretizeUnGuardedSymbolicMemory() {

  int num_concretized_symbolic_mem = 0;

  for (auto const& sym : ctx.getSymbolicMemory()){

    if (memGuard->guardedAddresses.find(sym.first) == memGuard->guardedAddresses.end()) {
      num_concretized_symbolic_mem += 1;
      if (debug) cout << GOOD"Concretizing symbolic memory @0x" << hex << sym.first << dec << endl;
      ctx.concretizeMemory(sym.first);
    }
  }
  if (debug) LOG("Concretized %d symbolic memory\n", num_concretized_symbolic_mem);
  if (debug) LOG("%zu SymbolicMemory\n", ctx.getSymbolicMemory().size());
}

void LiteTracer::LogInvalidMemAccessSymbol(uint64_t address) {
  
  auto lower_bound = loaded_modules.lower_bound(address);
  uint64_t module_base = AddressToModuleBase(address);
  if (!symbols.count(module_base)) LoadModuleSymbols((void*)module_base);
  uint64_t symbol_address = AddressToSymbolAddress(module_base, address);
  if (symbol_address){
    invalidMemAccess << hex << address << ":" << " " << loaded_modules[module_base] << "`" << symbols[module_base][symbol_address] << dec << endl;
  } else {
    invalidMemAccess << hex << address << ":" << " " << loaded_modules[module_base] << "`" << hex << (address-module_base) << dec << endl;
  }
}

void LiteTracer::LogSymbol2(uint64_t address){
  
  if (!log_symbols)
    return;
  
  auto lower_bound = loaded_modules.lower_bound(address);
  uint64_t module_base = AddressToModuleBase(address);
  if (!symbols.count(module_base)) LoadModuleSymbols((void*)module_base);
  uint64_t symbol_address = AddressToSymbolAddress(module_base, address);
  if (symbol_address){
    LOG("%llx: %s`%s\n", address, loaded_modules[module_base].c_str(), symbols[module_base][symbol_address].c_str());
  } else {
    LOG("%llx: %s`+0x%llx\n", address, loaded_modules[module_base].c_str(), address-module_base);
  }
}

void LiteTracer::LogSymbol(uint64_t address, bool newline){
  
  if (!log_symbols)
    return;
  
  //auto lower_bound = loaded_modules.lower_bound(address);
  uint64_t module_base = AddressToModuleBase(address);
  if (!symbols.count(module_base)) LoadModuleSymbols((void*)module_base);
  uint64_t symbol_address = AddressToSymbolAddress(module_base, address);
  if (symbol_address){
    if (debug) LOG("%llx: %s`%s", address, loaded_modules[module_base].c_str(), symbols[module_base][symbol_address].c_str());
  } else {
    if (debug) LOG("%llx: %s`+0x%llx", address, loaded_modules[module_base].c_str(), address-module_base);
  }
  if (newline) if (debug) cout << endl;
}

uint64_t LiteTracer::AddressToModuleBase(uint64_t address){
  auto lower_bound = loaded_modules.lower_bound(address);
  if (lower_bound == loaded_modules.end()) return 0;
  --lower_bound;
  return lower_bound->first;
}

uint64_t LiteTracer::AddressToSymbolAddress(uint64_t module_base, uint64_t address) {
  auto lower_bound = symbols[module_base].lower_bound(address+1); // address+1 to not get previous symbol at function head!
  if (lower_bound == symbols[module_base].end()) return 0;
  if (lower_bound == symbols[module_base].begin()) return 0;
  --lower_bound;
  return lower_bound->first;
}

void LiteTracer::SetContextConcreteRegisterValues(arm_thread_state64_t * state) {
  /*
  _STRUCT_ARM_THREAD_STATE64
    __uint64_t __x[29];  General purpose registers x0-x28 
    __uint64_t __fp;     Frame pointer x29 
    __uint64_t __lr;     Link register x30 
    __uint64_t __sp;     Stack pointer x31 
    __uint64_t __pc;     Program counter 
    __uint32_t __cpsr;   Current program status register
    __uint32_t __pad;    Same size for 32-bit or 64-bit clients
  */
    
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x0, state->__x[0]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x1, state->__x[1]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x2, state->__x[2]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x3, state->__x[3]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x4, state->__x[4]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x5, state->__x[5]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x6, state->__x[6]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x7, state->__x[7]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x8, state->__x[8]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x9, state->__x[9]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x10, state->__x[10]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x11, state->__x[11]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x12, state->__x[12]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x13, state->__x[13]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x14, state->__x[14]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x15, state->__x[15]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x16, state->__x[16]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x17, state->__x[17]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x18, state->__x[18]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x19, state->__x[19]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x20, state->__x[20]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x21, state->__x[21]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x22, state->__x[22]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x23, state->__x[23]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x24, state->__x[24]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x25, state->__x[25]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x26, state->__x[26]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x27, state->__x[27]);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x28, state->__x[28]);

  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x29, state->__fp);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_x30, StripPAC(state->__lr));
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_pc, state->__pc);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_sp, state->__sp);
  ctx.setConcreteRegisterValue(ctx.registers.aarch64_spsr, state->__cpsr);
}

uintptr_t LiteTracer::StripPAC(uintptr_t pac_ptr) {
  uintptr_t ptr_mask = 0x7fffffffff;
  uintptr_t pac_mask = ~ptr_mask;
  uintptr_t sign_mask = 1UL << 55;
  uintptr_t sign = pac_ptr & sign_mask;
  if (sign) return (pac_ptr | pac_mask);
  return (pac_ptr & ptr_mask);
}

void LiteTracer::RemoveLocalVariables(uint64_t sp) {
  if (debug) LOG("[RemoveLocalVariables]\n");
  
  memGuard->removeLocalVariables(sp);

  last_sp = sp;

  uint64_t startStackAddr = (sp & ~(0x3fff))-0xc000;
  
  int num_concretized_symbolic_mem = 0;
  for (auto const& sym : ctx.getSymbolicMemory()){
    if ((sym.first < sp) && (sym.first >= startStackAddr)) {
      num_concretized_symbolic_mem += 1;
      ctx.concretizeMemory(sym.first);
    }
  }
  if (debug) LOG("Concretized %d symbolic memory\n", num_concretized_symbolic_mem);
  
  int num_removed_written_mem = 0;

  for (auto it = written_memory_addresses.begin(); it != written_memory_addresses.end(); ) {
    if ((*it < sp) && (*it  >= startStackAddr))  {
        it = written_memory_addresses.erase(it);
        num_removed_written_mem += 1;
    } else {
        ++it;
    }
  }
  if (debug) LOG("removed %d written memory addresses\n", num_removed_written_mem);
}

void LiteTracer::LogMemAccess(uint64_t memAccessPC, bool whileStepping) {
  
  if (!log_mem_access && !log_symbols)
    return;
  
  uint64_t module_base = AddressToModuleBase(memAccessPC);
  if (!symbols.count(module_base)) LoadModuleSymbols((void*)module_base);

  string module = loaded_modules[module_base];
  if (memReadAccessMap.find(module) == memReadAccessMap.end()) {
      memReadAccessMap[module]; // Create an empty map
  }
  
  uint64_t symbol_address = AddressToSymbolAddress(module_base, memAccessPC);
  string function = symbols[module_base][symbol_address];
  if (memReadAccessMap[module].find(function) == memReadAccessMap[module].end()) {
      memReadAccessMap[module][function]; // Create an empty map
  }

  uint64_t pc = memAccessPC-module_base;
  
  if (whileStepping)
    memAccessHits << RED;
  memAccessHits << "[" << watchpoint_hits << "][";
  memAccessHits << hex << mach_exception->thread_port << "] ";
  if (whileStepping)
    memAccessHits << WHITE;
  memAccessHits << module;
  memAccessHits << "`+0x" << pc << dec;
  if (function.size()) memAccessHits << " | " << function << endl;
}

uint64_t LiteTracer::SymArgsMask() {
  uint64_t sym_args_mask = 0;
  for (int i = 0; i < 8; i++) {
    auto regId = static_cast<triton::arch::register_e>(triton::arch::ID_REG_AARCH64_X0 + i);
    auto reg = ctx.getRegister(regId);
    
    if (ctx.isRegisterSymbolized(reg)) {
      if (debug) LOG("[SymArgsMask] Register x%d is symbolized.\n", i);
      sym_args_mask |= (1 << i);
    }
  }
  return sym_args_mask;
}

void LiteTracer::LogArgSymbolicDef(uint64_t callerPC, uint64_t calleePC) {
  if (!log_symbolic_defs) return;
  if (!watchpoint_hits) return;
  uint64_t sym_args_mask = SymArgsMask();
  if (!sym_args_mask) return;

  uint64_t caller_module_base = AddressToModuleBase(callerPC);
  string caller_module = loaded_modules[caller_module_base];
  string caller_module_path = loaded_modules_pathes[caller_module_base];

  uint32_t caller_pc_rva = (uint32_t)(callerPC - caller_module_base);

  uint64_t pc = calleePC;
  uint64_t module_base = AddressToModuleBase(pc);
  string module = loaded_modules[module_base];
  string module_path = loaded_modules_pathes[module_base];

  uint32_t pc_rva = (uint32_t)(pc - module_base);

  uint32_t dst_instr_bytes;
  mach_target->ReadMemory((uint64_t)pc, 4, (void*)&dst_instr_bytes);
  auto dst_instr = triton::arch::Instruction((const unsigned char*)&dst_instr_bytes, 4);
  dst_instr.setAddress(pc);
  ctx.disassembly(dst_instr);

  uint64_t symbolicDef = (static_cast<uint64_t>(pc_rva) << 32) | caller_pc_rva; //*(uint32_t*)dst_instr.getOpcode();
  uint64_t invocation = (sym_args_mask << 56) | (symbolicDef & 0x00FFFFFFFFFFFFFFULL);
  for (int i = 0; i < 8; i++) {
    if (!((sym_args_mask >> i) & 1)) continue;
    auto regId = static_cast<triton::arch::register_e>(triton::arch::ID_REG_AARCH64_X0 + i);
    auto reg = ctx.getRegister(regId);
    auto regAst = ctx.getRegisterAst(reg);
    if (regAst->getLevel() <= 55) {
      invocation += (regAst->getLevel() & 0xf);
    }
  }

  if (debug) {
    LOG("[LogArgSymbolicDef] Symbolic Def: %llx pc_rva: %x Mask:0x%x\n", symbolicDef, pc_rva, (int)sym_args_mask);
  }
  
  symbolicDefs_mutex.Lock(); 
  if (symbolicDefs.find(symbolicDef) == symbolicDefs.end()) {

    /* Insert the symbolicDef into symbolicDefs */
    symbolicDefs.insert(symbolicDef);
    symbolicDefs_mutex.Unlock();

    /* Insert the invocation into invocations */
    invocations_mutex.Lock();
    invocations.insert(invocation);
    invocations_mutex.Unlock();

    char buffer[19];  // "0x" + 16 hex digits + null terminator
    //LOG("PyGILState_Ensure()\n");
    PyGILState_STATE gstate = PyGILState_Ensure();
    //LOG("PyGILState_Ensure() acquired!\n");
    try {
        PyEventMap event;
        event["type"] = py::cast("arg");
        snprintf(buffer, sizeof(buffer), "%llx", symbolicDef);
        event["id"] = py::cast(buffer);
        snprintf(buffer, sizeof(buffer), "%llx", invocation);
        event["invocation_id"] = py::cast(buffer);
        event["sample"] = py::cast(__input_sample_filename);
        event["watchpoint_hit"] = py::cast(watchpoint_hits);
        event["module"] = py::cast(module);
        event["module_path"] = py::cast(module_path);
        //event["function"] = py::cast(function);
        snprintf(buffer, sizeof(buffer), "0x%016x", pc_rva);
        event["pc_rva"] = py::cast(buffer);
        snprintf(buffer, sizeof(buffer), "0x%016llx", pc);
        event["pc"] = py::cast(buffer);
        event["instr_assembly"] = py::cast(instruction.getDisassembly());

        py::dict py_symbolic_regs;
        for (int i = 0; i < 8; i++) {
          if (!((sym_args_mask >> i) & 1)) continue;
          auto regId = static_cast<triton::arch::register_e>(triton::arch::ID_REG_AARCH64_X0 + i);
          auto reg = ctx.getRegister(regId);
          py_symbolic_regs[reg.getName().c_str()] = GetPySymRegAST(reg);
        }
        event["symbolic_regs"] = py_symbolic_regs;

        snprintf(buffer, sizeof(buffer), "0x%016llx", input_address);
        event["input_addr"] = py::cast(buffer);
        event["input_size"] = py::cast(__input_datalen);
        event["cpu_context"] = GetPyCpuContext();

        // attributes
        event["caller_module"] = py::cast(caller_module);
        event["caller_module_path"] = py::cast(caller_module_path);
        snprintf(buffer, sizeof(buffer), "0x%016x", caller_pc_rva);
        event["caller_pc_rva"] = py::cast(buffer);
        //event["caller_function"] = py::cast(caller_function);
        
        pyBridge.queueEvent(std::move(event));
        PyGILState_Release(gstate);
        //LOG("PyGILState_Release(gstate); Released\n");
    } catch (const std::exception& e) {
        PyGILState_Release(gstate);
        //LOG("PyGILState_Release(gstate); Released throw\n");
        throw;
    }
    //symbolicDefs_mutex.Unlock();
    return;
  }
  symbolicDefs_mutex.Unlock();

  invocations_mutex.Lock(); 
  if (invocations.find(invocation) == invocations.end()){

    /* Insert the invocation into invocations */
    invocations.insert(invocation);
    invocations_mutex.Unlock();

    char buffer[19];  // "0x" + 16 hex digits + null terminator
    //LOG("PyGILState_Ensure()\n");
    PyGILState_STATE gstate = PyGILState_Ensure();
    //LOG("PyGILState_Ensure() acquired!\n");
    try {
      PyEventMap event;
      event["type"] = py::cast("arg_invocation");
      snprintf(buffer, sizeof(buffer), "%llx", invocation);
      event["id"] = py::cast(buffer);
      snprintf(buffer, sizeof(buffer), "%llx", symbolicDef);
      event["symdef_id"] = py::cast(buffer);
      event["sample"] = py::cast(__input_sample_filename);
      event["watchpoint_hit"] = py::cast(watchpoint_hits);
      
      py::dict py_symbolic_regs;
      for (int i = 0; i < 8; i++) {
        if (!((sym_args_mask >> i) & 1)) continue;
        auto regId = static_cast<triton::arch::register_e>(triton::arch::ID_REG_AARCH64_X0 + i);
        auto reg = ctx.getRegister(regId);
        py_symbolic_regs[reg.getName().c_str()] = GetPySymRegAST(reg);
      }
      event["symbolic_regs"] = py_symbolic_regs;
      pyBridge.queueEvent(std::move(event));
      PyGILState_Release(gstate);
      //LOG("PyGILState_Release(gstate); Released\n");
    } catch (const std::exception& e) {
      PyGILState_Release(gstate);
      //LOG("PyGILState_Release(gstate); Released throw\n");
      throw;
    }
    //invocations_mutex.Unlock();
    return;
  }
  invocations_mutex.Unlock();
}

void LiteTracer::EvaluateMinMaxASTBounds(
    const triton::ast::SharedAbstractNode &regAst,
    MinMaxASTBounds &bounds,
    unsigned timeout_ms) {

    triton::ast::TritonToZ3 converter(false);
  
    // Convert the AST to Z3
    z3::expr z3Expr = converter.convert(regAst);
    z3::expr pathConstraint = converter.convert(ctx.getPathPredicate());

    // Get Z3 context
    z3::context &z3_ctx = z3Expr.ctx();

    z3::params p(z3_ctx);
    p.set("timeout", (unsigned)timeout_ms);
    p.set("maxres.max_num_cores", (unsigned)4);
    p.set("enable_sat", true);


    // --- Separate Optimize instances for min and max ---
    z3::optimize opt_min(z3_ctx);
    opt_min.set(p);
    opt_min.add(pathConstraint);
    auto h_min = opt_min.minimize(z3Expr);

    z3::optimize opt_max(z3_ctx);
    opt_max.set(p);
    opt_max.add(pathConstraint);
    auto h_max = opt_max.maximize(z3Expr);


    // --- Find Minimum ---
    z3::check_result res_min = opt_min.check();
    if (res_min == z3::sat) {
        z3::expr lower_bound_expr = opt_min.lower(h_min);
        uint64_t min_value = 0;
        if (Z3_get_numeral_uint64(z3_ctx, lower_bound_expr, &min_value)) {
            bounds.min = min_value;
            bounds.minSat = true;
            std::cout << "Minimum value: " << std::hex << min_value << std::dec << std::endl;
        } else {
            bounds.minSat = false;
            bounds.maxSat = false;
            std::cerr << "Failed to extract minimum value.\n";
            return;
        }
    } else if (res_min == z3::unknown) {
        bounds.minSat = false;
        bounds.maxSat = false;
        std::cerr << "Minimum Optimization: Timeout or unknown result.\n";
        return;
    } else {
        bounds.minSat = false;
        bounds.maxSat = false;
        std::cerr << "Minimum Optimization: Unsat.\n";
        return;
    }


    // --- Find Maximum ---
    z3::check_result res_max = opt_max.check();
    if (res_max == z3::sat) {
        z3::expr upper_bound_expr = opt_max.upper(h_max);
        uint64_t max_value = 0;
        if (Z3_get_numeral_uint64(z3_ctx, upper_bound_expr, &max_value)) {
            bounds.max = max_value;
            bounds.maxSat = true;
            std::cout << "Maximum value: " << std::hex << max_value << std::dec << std::endl;
        } else {
            bounds.minSat = false;
            bounds.maxSat = false;
            std::cerr << "Failed to extract maximum value.\n";
        }
    } else if (res_max == z3::unknown) {
        bounds.minSat = false;
        bounds.maxSat = false;
        std::cerr << "Maximum Optimization: Timeout or unknown result.\n";
    } else {
        bounds.minSat = false;
        bounds.maxSat = false;
        std::cerr << "Maximum Optimization: Unsat.\n";
    }

    // if ((bounds.computeModels) && (bounds.minSat) && (bounds.maxSat)){
    //   if (bounds.max != bounds.min){
    //     // Retrieve the model for the minimum value.
    //     z3::model model_min = opt_min.get_model();
    //     // Optionally, print or process the entire model.
    //     std::cout << "Model for min: " << model_min << std::endl;

    //     // Get the number of constants in the model
    //     unsigned num_consts = model_min.size();
        
    //     for (unsigned i = 0; i < num_consts; i++) {
    //         // Get the i-th constant declaration
    //         z3::func_decl decl = model_min[i];
    //         z3::expr value = model_min.get_const_interp(decl);
            
    //         // Get the name of the constant (e.g., "SymVar_24")
    //         std::string name = decl.name().str();
            
    //         // Check if it's a symbolic variable with an offset
    //         if (name.substr(0, 7) == "SymVar_") {
    //             // Extract the offset (e.g., "24" from "SymVar_24")
    //             std::string offset_str = name.substr(7);
    //             int offset = std::stoi(offset_str);
                
    //             // Convert the value to a hexadecimal string
    //             unsigned long long val;
    //             Z3_get_numeral_uint64(value.ctx(), value, &val);
                
    //             // Print in the requested format
    //             std::cout << offset << ": 0x" << std::hex << std::setw(2) 
    //                       << std::setfill('0') << val << std::dec << std::endl;
    //         }
    //     }
    //     std::cout << std::endl;

    //     // Retrieve the model for the maximum value.
    //     z3::model model_max = opt_max.get_model();
    //     // Optionally, print or process the entire model.
    //     std::cout << "Model for max: " << model_max << std::endl;
    //   }
    // }
}

void LiteTracer::LogRetSymbolicDef(uint64_t fromPC, uint64_t toPC) {
  if (!log_symbolic_defs) return;
  if (!ctx.isRegisterSymbolized(ctx.registers.aarch64_x0)) return;
  if (!watchpoint_hits) return;

  if (debug) LOG("[LogRetSymbolicDef] %llx -> %llx\n", fromPC, toPC);

  // resolve callee module and function
  uint64_t callee_module_base = AddressToModuleBase(fromPC);
  string callee_module = loaded_modules[callee_module_base];
  string callee_module_path = loaded_modules_pathes[callee_module_base];

  uint32_t callee_pc_rva = (uint32_t)(fromPC - callee_module_base);
  
  uint64_t pc  = toPC-4;
  uint64_t module_base = AddressToModuleBase(pc);

  string module = loaded_modules[module_base];
  string module_path = loaded_modules_pathes[module_base];

  uint32_t pc_rva = (uint32_t)(pc - module_base);

  uint32_t dst_instr_bytes;
  mach_target->ReadMemory((uint64_t)pc, 4, (void*)&dst_instr_bytes);
  auto dst_instr = triton::arch::Instruction((const unsigned char*)&dst_instr_bytes, 4);
  dst_instr.setAddress(pc);
  ctx.disassembly(dst_instr);

  uint64_t symbolicDef = (static_cast<uint64_t>(pc_rva) << 32) | callee_pc_rva;
  uint64_t invocation = symbolicDef;
  
  // Get the symbolic AST for the x0 register
  auto regAst = ctx.getRegisterAst(ctx.registers.aarch64_x0);

  if (regAst->getLevel() <= 55) {
    invocation += (regAst->getLevel() & 0xf);
  }
  if (debug) LOG("Symbolic Def: %llx pc_rva: %x\n", symbolicDef, pc_rva);

  symbolicDefs_mutex.Lock(); 
  if (symbolicDefs.find(symbolicDef) == symbolicDefs.end()){

    /* Insert the symbolicDef into symbolicDefs */
    symbolicDefs.insert(symbolicDef);
    symbolicDefs_mutex.Unlock();

    /* Insert the invocation into invocations */
    invocations_mutex.Lock();
    invocations.insert(invocation);
    invocations_mutex.Unlock();
    
    char buffer[19];  // "0x" + 16 hex digits + null terminator
    //LOG("PyGILState_Ensure()\n");
    PyGILState_STATE gstate = PyGILState_Ensure();
    //LOG("PyGILState_Ensure() acquired!\n");
    try {
        PyEventMap event;
        event["type"] = py::cast("ret");
        snprintf(buffer, sizeof(buffer), "%llx", symbolicDef);
        event["id"] = py::cast(buffer);
        snprintf(buffer, sizeof(buffer), "%llx", invocation);
        event["invocation_id"] = py::cast(buffer);
        event["sample"] = py::cast(__input_sample_filename);
        event["watchpoint_hit"] = py::cast(watchpoint_hits);
        event["module"] = py::cast(module);
        event["module_path"] = py::cast(module_path);
        //event["function"] = py::cast(function);
        snprintf(buffer, sizeof(buffer), "0x%016x", pc_rva);
        event["pc_rva"] = py::cast(buffer);
        snprintf(buffer, sizeof(buffer), "0x%016llx", pc);
        event["pc"] = py::cast(buffer);
        event["instr_assembly"] = py::cast(instruction.getDisassembly());
        
        py::dict py_symbolic_regs;
        py_symbolic_regs["x0"] = GetPySymRegAST(ctx.registers.aarch64_x0);
        event["symbolic_regs"] = py_symbolic_regs;

        snprintf(buffer, sizeof(buffer), "0x%016llx", input_address);
        event["input_addr"] = py::cast(buffer);
        event["input_size"] = py::cast(__input_datalen);
        event["cpu_context"] = GetPyCpuContext();

        // attributes
        event["callee_module"] = py::cast(callee_module);
        event["callee_module_path"] = py::cast(callee_module_path);
        snprintf(buffer, sizeof(buffer), "0x%016x", callee_pc_rva);
        event["callee_pc_rva"] = py::cast(buffer);
        
        //event["callee_function"] = py::cast(callee_function);
        
        pyBridge.queueEvent(std::move(event));
        PyGILState_Release(gstate);
        //LOG("PyGILState_Release(gstate); Released\n");
    } catch (const std::exception& e) {
        PyGILState_Release(gstate);
        //LOG("PyGILState_Release(gstate); Released throw\n");
        throw;
    }
    //symbolicDefs_mutex.Unlock();
    return;
  }
  symbolicDefs_mutex.Unlock();
  
  invocations_mutex.Lock(); 
  if (invocations.find(invocation) == invocations.end()){

    /* Insert the invocation into invocations */
    invocations.insert(invocation);
    invocations_mutex.Unlock();

    char buffer[19];  // "0x" + 16 hex digits + null terminator
    //LOG("PyGILState_Ensure()\n");
    PyGILState_STATE gstate = PyGILState_Ensure();
    //LOG("PyGILState_Ensure() acquired!\n");
    try {
        PyEventMap event;
        event["type"] = py::cast("ret_invocation");
        snprintf(buffer, sizeof(buffer), "%llx", invocation);
        event["id"] = py::cast(buffer);
        snprintf(buffer, sizeof(buffer), "%llx", symbolicDef);
        event["symdef_id"] = py::cast(buffer);
        event["sample"] = py::cast(__input_sample_filename);
        event["watchpoint_hit"] = py::cast(watchpoint_hits);
        py::dict py_symbolic_regs;
        py_symbolic_regs["x0"] = GetPySymRegAST(ctx.registers.aarch64_x0);
        event["symbolic_regs"] = py_symbolic_regs;
        pyBridge.queueEvent(std::move(event));
        PyGILState_Release(gstate);
        //LOG("PyGILState_Release(gstate); Released\n");
    } catch (const std::exception& e) {
        PyGILState_Release(gstate);
        //LOG("PyGILState_Release(gstate); Released throw\n");
        throw;
    }
    //invocations_mutex.Unlock();
    return;
  }
  invocations_mutex.Unlock();
}

void LiteTracer::LogMemSymbolicDef() {
  if (!log_symbolic_defs)
    return;

  if (!watchpoint_hits) return;
  
  if (debug) LOG("[LogMemSymbolicDef]");

  uint64_t pc = instruction.getAddress();

  uint64_t module_base = AddressToModuleBase(pc);

  string module = loaded_modules[module_base];
  string module_path = loaded_modules_pathes[module_base];

  uint32_t pc_rva = (uint32_t)(pc - module_base);
  
  bool first_operand_is_symbolized  = false;
  bool second_operand_is_symbolized = false;

  uint64_t symbolicDef = (static_cast<uint64_t>(pc_rva) << 32) | *(uint32_t*)instruction.getOpcode();
  uint64_t invocation = symbolicDef;

  auto const& reg1 = instruction.operands[0].getRegister();
  if (ctx.isRegisterSymbolized(reg1)) {
    first_operand_is_symbolized  = true;
    auto regAst = ctx.getRegisterAst(reg1);
    if (regAst->getLevel() <= 55) {
      // limiting the number of invocations via AST Level filter (& 0xf)
      invocation += (regAst->getLevel() & 0xf);
    }
  }

  if ((memory_access_size > 1) && (instruction.operands[1].getType() == triton::arch::OP_REG)) {
    auto const& reg2 = instruction.operands[1].getRegister();
    if (ctx.isRegisterSymbolized(reg2)) {
      second_operand_is_symbolized  = true;
      auto regAst2 = ctx.getRegisterAst(reg2);
      if (regAst2->getLevel() <= 55) {
        invocation += (regAst2->getLevel() & 0xf);
      }
    }
  }

  if (debug) LOG("Symbolic Def: %llx pc_rva: %x\n", symbolicDef, pc_rva);

  symbolicDefs_mutex.Lock(); 
  if (symbolicDefs.find(symbolicDef) == symbolicDefs.end()){

    /* Insert the symbolicDef into symbolicDefs */
    symbolicDefs.insert(symbolicDef);
    symbolicDefs_mutex.Unlock();

    /* Insert the invocation into invocations */
    invocations_mutex.Lock();
    invocations.insert(invocation);
    invocations_mutex.Unlock();

    auto guarded_addr = memory_access_addr;
    auto guarded_size = memory_access_size;

    while (memGuard->isGuarded(guarded_addr-1, 1)) {
      guarded_addr -= 1;
      guarded_size++;
    }

    auto next_addr = memory_access_addr+memory_access_size;

    while (memGuard->isGuarded(next_addr, 1)) {
      next_addr++;
      guarded_size++;
    }
    
    char buffer[19];  // "0x" + 16 hex digits + null terminator
    //LOG("PyGILState_Ensure()\n");
    PyGILState_STATE gstate = PyGILState_Ensure();
    //LOG("PyGILState_Ensure() acquired!\n");
    try {
        PyEventMap event;
        event["type"] = py::cast("mem");
        snprintf(buffer, sizeof(buffer), "%llx", symbolicDef);
        event["id"] = py::cast(buffer);
        snprintf(buffer, sizeof(buffer), "%llx", invocation);
        event["invocation_id"] = py::cast(buffer);
        //event["id"] = py::cast(symbolicDef);
        event["sample"] = py::cast(__input_sample_filename);
        event["watchpoint_hit"] = py::cast(watchpoint_hits);
        event["module"] = py::cast(module);
        event["module_path"] = py::cast(module_path);
        //event["function"] = py::cast(function);
        snprintf(buffer, sizeof(buffer), "0x%016x", pc_rva);
        event["pc_rva"] = py::cast(buffer);
        snprintf(buffer, sizeof(buffer), "0x%016llx", pc);
        event["pc"] = py::cast(buffer);
        event["instr_assembly"] = py::cast(instruction.getDisassembly());
        
        py::dict py_symbolic_regs;

        py_symbolic_regs[instruction.operands[0].getRegister().getName().c_str()] = GetPySymRegAST(instruction.operands[0].getRegister());
        if (instruction.operands[1].getType() == triton::arch::OP_REG) {
          py_symbolic_regs[instruction.operands[1].getRegister().getName().c_str()] = GetPySymRegAST(instruction.operands[1].getRegister());
        }
        event["symbolic_regs"] = py_symbolic_regs;

        snprintf(buffer, sizeof(buffer), "0x%016llx", input_address);
        event["input_addr"] = py::cast(buffer);
        event["input_size"] = py::cast(__input_datalen);
        event["cpu_context"] = GetPyCpuContext();

        // attributes
        snprintf(buffer, sizeof(buffer), "0x%016llx", memory_access_addr);
        event["addr"] = py::cast(buffer);
        event["size"] = py::cast(memory_access_size);
        snprintf(buffer, sizeof(buffer), "0x%016llx", guarded_addr);
        event["guarded_addr"] = py::cast(buffer);
        event["guarded_size"] = py::cast(guarded_size);
        
        pyBridge.queueEvent(std::move(event));
        PyGILState_Release(gstate);
        //LOG("PyGILState_Release(gstate); Released\n");
    } catch (const std::exception& e) {
        PyGILState_Release(gstate);
        //LOG("PyGILState_Release(gstate); Released throw\n");
        throw;
    }
    //symbolicDefs_mutex.Unlock();
    return;
  }
  symbolicDefs_mutex.Unlock();

  invocations_mutex.Lock(); 
  if (invocations.find(invocation) == invocations.end()){

    /* Insert the invocation into invocations */
    invocations.insert(invocation);
    invocations_mutex.Unlock();

    char buffer[19];  // "0x" + 16 hex digits + null terminator
    //LOG("PyGILState_Ensure()\n");
    PyGILState_STATE gstate = PyGILState_Ensure();
    //LOG("PyGILState_Ensure() acquired!\n");
    try {
        PyEventMap event;
        event["type"] = py::cast("mem_invocation");
        snprintf(buffer, sizeof(buffer), "%llx", invocation);
        event["id"] = py::cast(buffer);
        snprintf(buffer, sizeof(buffer), "%llx", symbolicDef);
        event["symdef_id"] = py::cast(buffer);
        event["sample"] = py::cast(__input_sample_filename);
        event["watchpoint_hit"] = py::cast(watchpoint_hits);
        event["mem_access_size"] = py::cast(memory_access_size);
        py::dict py_symbolic_regs;
        if (first_operand_is_symbolized){
          py_symbolic_regs[instruction.operands[0].getRegister().getName().c_str()] = GetPySymRegAST(instruction.operands[0].getRegister());
        }
        if (instruction.operands[1].getType() == triton::arch::OP_REG) {
          if (second_operand_is_symbolized) {
            py_symbolic_regs[instruction.operands[1].getRegister().getName().c_str()] = GetPySymRegAST(instruction.operands[1].getRegister());
          }
        }
        event["symbolic_regs"] = py_symbolic_regs;
        pyBridge.queueEvent(std::move(event));
        PyGILState_Release(gstate);
        //LOG("PyGILState_Release(gstate); Released\n");
    } catch (const std::exception& e) {
        PyGILState_Release(gstate);
        //LOG("PyGILState_Release(gstate); Released throw\n");
        throw;
    }
    //invocations_mutex.Unlock();
    return;
  }
  invocations_mutex.Unlock();
}

void LiteTracer::StoreMemAccess(const triton::arch::MemoryAccess &mem) {
  if (debug) LOG("Storing MemAccess\n");
  
  uint64_t module_base = AddressToModuleBase(mem_hit_pc);
  if (!symbols.count(module_base)) LoadModuleSymbols((void*)module_base);

  string module = loaded_modules[module_base];
  if (memReadAccessMap.find(module) == memReadAccessMap.end()) {
      memReadAccessMap[module]; // Create an empty map
  }
  
  uint64_t symbol_address = AddressToSymbolAddress(module_base, mem_hit_pc);
  string function = symbols[module_base][symbol_address];
  if (memReadAccessMap[module].find(function) == memReadAccessMap[module].end()) {
      memReadAccessMap[module][function]; // Create an empty map
  }

  uint32_t pc = (uint32_t)(mem_hit_pc-module_base);
  if (memReadAccessMap[module][function].find(pc) == memReadAccessMap[module][function].end()) {
      memReadAccessMap[module][function][pc]; // Create an empty map
  }
  
  const auto& addr = mem.getAddress();
  const auto& size = mem.getSize();

  uint32_t thread = threads_data[mach_exception->thread_port].id;
  memReadAccessCount+=1;
  memReadAccessMap[module][function][pc].push_back({memReadAccessCount,thread,addr,size});
}

void LiteTracer::LogMemAccessHits() {
  string result = memAccessHits.str();
  if (debug) cout << "MemAccess Hits:" << endl << result << std::endl;
}

void LiteTracer::SaveInvalidMemAccess(const char * filename) {
  //output_mutex.Lock();
  FILE* f = fopen(filename, "w");
  if(f != NULL)
  {
    fputs(invalidMemAccess.str().c_str(), f);
    fclose(f);
  }
  else cout << "Unable to open output file!";
  //output_mutex.Unlock();
}

uint32_t LiteTracer::GetRVA(uint64_t address){
   return (uint32_t)(address - AddressToModuleBase(address));
}

void LiteTracer::OnSymbolizedInstruction(triton::arch::Instruction &instruction, string &module, string &function) {
  AddToCoverage(instruction.getAddress());
  if (debug) cout << SYMBOLIZED << module << "." << function << " 0x" << hex << instruction.getAddress() << dec << ": " << instruction.getDisassembly() << endl;
}

void LiteTracer::OnUnsymbolizedInstruction(triton::arch::Instruction &instruction, string &module, string &function) {
  if (debug) cout << WHITE << UNSYMBOLIZED << module << "." << function << " 0x" << hex << instruction.getAddress() << dec << ": " << instruction.getDisassembly() << endl; // "\t" << instruction.getType() << endl; // << "\t" << instruction.getType();
}

void LiteTracer::AddToCoverage(uint64_t pc) {
  // Update the code coverage
  if (iCoverage.find(pc) == iCoverage.end()) {
    iCoverage[pc] = 1;
  } else {
    iCoverage[pc] += 1;
  }
}

void LiteTracer::GetFunctionInfo(uint64_t address, string &module, string &function, uint64_t &func_start, uint64_t &func_end) {
  
  uint64_t module_base = AddressToModuleBase(address);
  if (!symbols.count(module_base)) LoadModuleSymbols((void*)module_base);
  
  auto lower_bound = symbols[module_base].lower_bound(address+1); // address+1 to not get previous symbol at function head!
  if (lower_bound == symbols[module_base].end() || lower_bound == symbols[module_base].begin()) {
    func_end = 0;
  } else {
    func_end = lower_bound->first - 4;
    --lower_bound;
    func_start = lower_bound->first;
  }

  if (!func_start || !func_end) {
    LOG("could't determine function start/end\n");
    func_end = func_start+12;
  }
  
  uint64_t func_size = func_end - func_start + 4;

  module = loaded_modules[module_base];
  function = symbols[module_base][func_start];

  if (function.empty()) {
    LOG("Function is empty! size: %llu\n", func_size);
    ostringstream func_name;
    func_name << "sub_" << hex << func_start;
    function = func_name.str();
    symbols[module_base][func_start] = function;
  } else {
    char func_name_end = '(';
    size_t pos = function.find(func_name_end);
    if (pos != std::string::npos) {
      // Extract the function name
      function = function.substr(0, pos);
    }
  }
}

uint64_t LiteTracer::Emulate(uint64_t pc) {

  bool returned = false;
  int callstack = 1;
  last_same_pc = 0;
  uint32_t instruction_bytes;

  string module;
  string function;
  uint64_t func_start = 0;
  uint64_t func_end = 0;
  uint64_t step_pc = pc;
  
  evtime = 0;

  uint64_t emulation_begin_time = GetCurTime();

  GetFunctionInfo(pc, module, function, func_start, func_end);

  uint64_t previous_pc = 0;

  while (pc) {

    if (invalidMemAccessAddress) {
      LogInvalidMemAccessSymbol(instruction.getAddress());
      invalidMemAccess << instruction << endl;
      if (debug) LOG("invalid MemAccess! skipping emulation\n");
      return 0;
    }

    uint64_t time_now = GetCurTime();
    uint64_t emulation_time = time_now - emulation_begin_time;

    if ((evtime) && (evtime < emulation_time))
      emulation_time = emulation_time - evtime;

    if (emulation_time >= timeout) {
      LOG("Emulation timeout!\n");
      break;
    }

    ReadTargetMemory((uint64_t)pc, 4, (void*)&instruction_bytes);

    instruction = triton::arch::Instruction((const unsigned char*)&instruction_bytes, 4);
    instruction.setAddress(pc);

    insnAddresses.insert(pc);

    uint64_t lr_before_processing = static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x30));
    
    try {
      ctx.processing(instruction);
    }
    catch(std::exception& e) {
      if (debug){
        cout << ERR << e.what() << endl;
        cout << RED"# Skipping next instruction" << WHITE << endl;
      }
    }

    uint64_t lr_after_processing = static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_x30));
    
    if (instruction.isSymbolized()) {
      OnSymbolizedInstruction(instruction, module, function);
    } else {
      OnUnsymbolizedInstruction(instruction, module, function);
    }

    if (memory_access_addr) {
      LogMemSymbolicDef();
      memory_access_addr = 0;
      memory_access_size = 0;
    }

    uint64_t current_sp = static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_sp));
    if (current_sp != last_sp)
      RemoveLocalVariables(current_sp);

    if (instruction.getType() == triton::arch::arm::aarch64::ID_INS_MSR) { // MSR
      if (debug) LOG("Breaking on MSR instruction\n");
      step_pc = pc+4;
      break;
    } else if (instruction.getType() == triton::arch::arm::aarch64::ID_INS_MRS) { // MRS : mrs x28, tpidrro_el0
      if (debug) LOG("Breaking on MRS instruction\n");
      step_pc = pc+4;
      break;
    } else if (instruction.getType() == triton::arch::arm::aarch64::ID_INS_HINT) {
      if (debug) LOG("Handling HINT instruction\n");
      pc += 4;
      step_pc += 4;
      continue;
    }

    if (instruction.getType() == triton::arch::arm::aarch64::ID_INS_BL) {
      uint64_t called_func_addr = static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_pc));
      LogSymbol(called_func_addr);
      callstack += 1;
    }
    
    previous_pc = pc;
    pc = static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_pc));
    if (!((pc - previous_pc) <= 4) && ((pc <= func_start) || (pc > func_end))) {

      switch(instruction.getType())
      {
        case triton::arch::arm::aarch64::ID_INS_RET:
        case triton::arch::arm::aarch64::ID_INS_RETAA:
        case triton::arch::arm::aarch64::ID_INS_RETAB:
          LogRetSymbolicDef(previous_pc, pc);
          break;
        case triton::arch::arm::aarch64::ID_INS_BL:
        case triton::arch::arm::aarch64::ID_INS_BLRAA:
          LogArgSymbolicDef(previous_pc, pc);
          break;
        case triton::arch::arm::aarch64::ID_INS_B:
        case triton::arch::arm::aarch64::ID_INS_BR:
        case triton::arch::arm::aarch64::ID_INS_BRAA:
        //case triton::arch::arm::aarch64::ID_INS_TBZ:
          LogArgSymbolicDef(previous_pc, pc);
          break;
        case triton::arch::arm::aarch64::ID_INS_CBZ:
        case triton::arch::arm::aarch64::ID_INS_CBNZ:
        case triton::arch::arm::aarch64::ID_INS_TBZ:
        case triton::arch::arm::aarch64::ID_INS_TBNZ:
        //case triton::arch::arm::aarch64::ID_INS_B_COND:
           break;
        default: LOG("getting into another function! %s [%lld] %s.%llx -> %s.%llx\n", 
          instruction.getDisassembly().c_str(),
          (pc- previous_pc),
          loaded_modules[AddressToModuleBase(previous_pc)].c_str(),
          previous_pc,
          loaded_modules[AddressToModuleBase(pc)].c_str(),
          pc);

      }
      if (pc != func_start)
        GetFunctionInfo(pc, module, function, func_start, func_end);
      else
        LOG("recursive call!\n");
    } else if (instruction.isBranch() && (lr_after_processing != lr_before_processing)) {
      switch(instruction.getType())
      {
        // case triton::arch::arm::aarch64::ID_INS_RET:
        // case triton::arch::arm::aarch64::ID_INS_RETAA:
        // case triton::arch::arm::aarch64::ID_INS_RETAB:
        //   LogRetSymbolicDef(previous_pc, pc);
        //   break;
        case triton::arch::arm::aarch64::ID_INS_BL:
        case triton::arch::arm::aarch64::ID_INS_BLRAA:
          LogArgSymbolicDef(previous_pc, pc);
          break;
        // case triton::arch::arm::aarch64::ID_INS_B:
        // case triton::arch::arm::aarch64::ID_INS_BR:
        // case triton::arch::arm::aarch64::ID_INS_BRAA:
        // case triton::arch::arm::aarch64::ID_INS_TBZ:
        //   LogArgSymbolicDef(previous_pc, pc);
        //   break;
        default: FATAL("branching/call into function(start/end)");
      }
    }

    if (ctx.getSymbolicRegisters().empty()) {
      if (debug) {
        LOG("0 SymbolicRegisters\n");
        LOG(MAGENTA"looking for good step_pc\n");
      }
      if (insnAddresses.find(pc) == insnAddresses.end()) {
        step_pc = pc;
        break;
      }
    }
    
    // check if pc is still pointing to the same last emulated instruction
    if (pc == step_pc) {
      //if (debug) LOG("instruction_bytes: %16llx\n", instruction_bytes);
      uint8_t opcode_byte = *instruction.getOpcode();
      if (instruction.getType() == triton::arch::arm::aarch64::ID_INS_FCVTZU) {
        if (debug) cout << ERR"breaking @ fcvtzu" << endl;
        step_pc = pc;
        break;
      } else if ((instruction.getType() == 0) && (opcode_byte == 136)) {
        if (debug) cout << ERR"breaking @ ldapr" << endl;
        step_pc = pc;
        break;
      } else if ((instruction.getType() == 0) && (opcode_byte == 8)) {
        if (debug) cout << ERR"breaking @ ldaddl" << endl;
        step_pc = pc;
        break;
      } else if (instruction.getType() == triton::arch::arm::aarch64::ID_INS_USHLL) {
        if (debug) cout << ERR"breaking @ ushll" << endl;
        step_pc = pc;
        break;
      } else if (instruction.getType() == triton::arch::arm::aarch64::ID_INS_USHLL2) {
        if (debug) cout << ERR"breaking @ ushll2" << endl;
        step_pc = pc;
        break;
      } else if (instruction.getType() == triton::arch::arm::aarch64::ID_INS_FMOV) {
        if (debug) cout << ERR"breaking @ fmov" << endl;
        step_pc = pc;
        break;
      } else if ((instruction.getType() == 0) && (opcode_byte == 63)) {
        if (debug) cout << ERR"breaking @ bics" << endl;
        step_pc = pc;
        break;
      } else if (instruction.isBranch()) {
        FATAL("same pc and instruction isBranch");
      } else {
        uint8_t opcode = *instruction.getOpcode();
        if (debug) cout << ERR"Same PC! instruction type: " << instruction.getType() << " | opcode:" << unsigned(opcode) << endl;
        if (debug) cout << ERR"breaking @ ID_INS_INVALID" << endl;
        step_pc = pc;
        break;
      }
    }
    step_pc = pc;
  }

  pc = static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_pc));
  if (debug) LOG("step_pc: %llx\n", step_pc);
  
  if (debug) LOG("%zu SymbolicMemory\n", ctx.getSymbolicMemory().size());
  //RemoveLocalVariables();

  if (insnAddresses.find(step_pc) != insnAddresses.end()) {
    LOG(RED"collision | ");
    cout << instruction << WHITE << endl;
  }

  return step_pc;
}

void LiteTracer::LogSymbolizedInstruction(triton::arch::Instruction &instruction) {
  if (debug) SAY(SYMBOLIZED"%llx: %s", instruction.getAddress(), instruction.getDisassembly().c_str());
}

void LiteTracer::LogSymbolicMemoryInfo(uint64_t address) {
  cout << GREEN"Symbolic Memory:" << WHITE << endl;
  cout << "SymbolicMemory: " << ctx.getSymbolicMemory(address) << endl;
  cout << "ID: " << ctx.getSymbolicMemory(address)->getId() << endl;
  cout << "Comment: " << ctx.getSymbolicMemory(address)->getComment() << endl;
  cout << "AST: " << ctx.getSymbolicMemory(address)->getAst() << endl;
  cout << "Unroll: " << unroll(ctx.getSymbolicMemory(address)->getAst()) << endl;
  cout << "Simplify: " << ctx.simplify(ctx.getMemoryAst(triton::arch::MemoryAccess((triton::uint64)address, 1))) << endl;
}

void LiteTracer::LogSymbolicRegistersInfo(triton::arch::Instruction &instruction) {
  // Get all registers read by the instruction
  auto readRegisters = instruction.getReadRegisters();
  
  // Iterate over each read register
  for (const auto& regPair : readRegisters) {
    const triton::arch::Register& reg = regPair.first;
    triton::ast::SharedAbstractNode regAst = regPair.second;

    // if (regAst->getLevel() > 25) continue;

    if (ctx.isRegisterSymbolized(reg)) {
      auto regAst = ctx.getRegisterAst(reg);
      cout << "AST: " << regAst << endl;
      cout << "AST Unroll: " << unroll(regAst) << endl;
      cout << "Simplify: " << ctx.simplify(regAst) << endl;
      cout << "AST Level: " << regAst->getLevel() << endl;
    }
  }
}

vm_size_t LiteTracer::PageSize() {
  if (m_page_size == INVALID_PAGE_SIZE) {
    kern_return_t kr;

    task_vm_info_data_t vm_info;
    mach_msg_type_number_t info_count = TASK_VM_INFO_COUNT;
    kr = task_info(mach_target->Task(), TASK_VM_INFO, (task_info_t)&vm_info, &info_count);

    if (kr != KERN_SUCCESS) {
      FATAL("Error (%s) retrieving target's TASK_VM_INFO\n", mach_error_string(kr));
    }

    m_page_size = vm_info.page_size;
  }

  return m_page_size;
}

size_t LiteTracer::MaxBytesLeftInPage(mach_vm_address_t address, mach_vm_size_t size) {
  vm_size_t page_size = PageSize();
  if (page_size > 0) {
    mach_vm_size_t page_offset = address % page_size;
    mach_vm_size_t bytes_left_in_page = page_size - page_offset;
    if (size > bytes_left_in_page) {
      size = bytes_left_in_page;
    }
  }
  return size;
}

kern_return_t LiteTracer::ReadTargetMemory(uint64_t address, size_t size, void *buf) {
  if (buf == NULL) {
    WARN("ReadMemory is called with buf == NULL\n");
    return KERN_FAILURE;
  }

  if (size == 0) {
    WARN("ReadMemory is called with size == 0\n");
    return KERN_FAILURE;
  }

  kern_return_t kr;
  mach_vm_size_t total_bytes_read = 0;
  mach_vm_address_t cur_addr = address;
  uint8_t *cur_buf = (uint8_t*)buf;
  while (total_bytes_read < size) {
    mach_vm_size_t cur_size = MaxBytesLeftInPage(cur_addr, size - total_bytes_read);

    mach_msg_type_number_t cur_bytes_read = 0;
    vm_offset_t vm_buf;
    kr = mach_vm_read(mach_target->Task(), cur_addr, cur_size, &vm_buf, &cur_bytes_read);

    if (kr != KERN_SUCCESS) {
      LOG("Error (%s) reading memory @ address 0x%llx\n", mach_error_string(kr), cur_addr);
      LOG("sample: %s\n", __input_sample_filename.c_str());

      memAccessHits << "Error (" << mach_error_string(kr) << "reading memory @ address 0x" << hex << cur_addr << dec << endl;
      memAccessHits << "sample: " << __input_sample_filename << endl;
      return KERN_FAILURE;
    }

    if (cur_bytes_read != cur_size) {
      LOG("Error reading the entire requested memory @ address 0x%llx\n", cur_addr);
      LOG("sample: %s\n", __input_sample_filename.c_str());
      return KERN_FAILURE;
    }

    memcpy(cur_buf, (const void*)vm_buf, cur_bytes_read);
    mach_vm_deallocate(mach_task_self(), vm_buf, cur_bytes_read);

    total_bytes_read += cur_bytes_read;
    cur_addr += cur_bytes_read;
    cur_buf += cur_bytes_read;
  }

  return KERN_SUCCESS;
}

void LiteTracer::MemWriteCallback(triton::Context &ctx, const triton::arch::MemoryAccess &mem, const triton::uint512 &value) {
  kern_return_t kr;
  const auto& addr = mem.getAddress();
  const auto& size = mem.getSize();

  if (debug) LOG(RED"MemWriteCallback @ %llx[%d]\n", addr, size);

  bool guardedAccess = memGuard->isGuarded(addr, size);

  if (initInstruction) {
    initInstruction = false;

    arm_thread_state64_t *state = (arm_thread_state64_t*)(mach_exception->new_state);
  
    if (!guardedAccess) {
      outOfRangeAccessCount += 1;
      doEmulate = false;
      return;
    }
  }

  // reading the to be written address to trap bad access!
  char buffer[size];
  kr = ReadTargetMemory((uint64_t)addr, size, (void*)buffer);
  
  if (kr != KERN_SUCCESS) {
    invalidMemAccessAddress = (uint64_t)addr;
    invalidMemAccess << "[WRITE] @" << hex << addr << dec << endl;
    return;
  }

  bool symbolizedAccess = false;

  uint64_t cur_byte_addr = 0;
  uint8_t v;

  for (triton::usize i = 0; i < size; i++) {
    cur_byte_addr = addr+i;
    written_memory_addresses.insert(cur_byte_addr);
    triton::uint512 cv  = value;
    cv >>= (8*i);
    v = static_cast<triton::uint8>((cv & 0xff));

    if (ctx.isMemorySymbolized(cur_byte_addr)) {

      symbolizedAccess = true;
      if (debug) LOG("Symbolic MemWrite @ %llx | offset [%zu] | Value [%x]\n", cur_byte_addr, i, v);
      
      uint64_t sp = static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_sp));
      uint64_t stackPointerPage = sp & ~(0x3fff);
      bool isStack = false;
      int64_t diff = cur_byte_addr - stackPointerPage;
      uint64_t abs_diff = (diff ^ ((diff >> 63))) - (diff >> 63);
      if ((abs_diff) < 0xc000) {
        if (debug) LOG("         symbolic memory address is on stack\n");
        isStack = true;
      }

      if (debug) LOG("isStack: %d | cur_byte_addr: %llx | sp: %llx | abs_diff: %llx\n", isStack, cur_byte_addr, sp, abs_diff);
      memGuard->add(cur_byte_addr, 1, isStack, true);
    } else {
      if (debug) LOG("         MemWrite @ %llx | offset [%zu] | Value [%x]\n", cur_byte_addr, i, v);
      if (memGuard->isGuarded(cur_byte_addr, 1)) {
        if (debug) LOG("Unsymbolized MemWrite to a guraded memory address!\n");
        memGuard->remove(cur_byte_addr,1);
      }
      ctx.concretizeMemory(cur_byte_addr);
    }
  }
  CheckGuardedSymbolicMemory();
}


void LiteTracer::MemReadCallback(triton::Context &ctx, const triton::arch::MemoryAccess &mem) {
  kern_return_t kr;
  const auto& addr = mem.getAddress();
  const auto& size = mem.getSize();

  if (debug) LOG(GREEN"MemReadCallback @ %llx[%d]\n",addr,size);

  bool isGuarded = memGuard->isGuarded(addr, size);

  if (initInstruction) {
    initInstruction = false;

    arm_thread_state64_t *state = (arm_thread_state64_t*)(mach_exception->new_state);
  
    if (isGuarded) {
      ++watchpoint_hits;
      LogMemAccess(instruction.getAddress());
    } else {
      outOfRangeAccessCount += 1;
      doEmulate = false;
      return true;
    }
  } else if (isGuarded) {
      ++watchpoint_hits;
      //StoreMemAccess(mem);
      LogMemAccess(instruction.getAddress());
  }

  bool symbolizedAccess = false;
  
  char buffer[size];
  kr = ReadTargetMemory((uint64_t)addr, size, (void*)buffer);

  if (kr != KERN_SUCCESS) {
    invalidMemAccess << "[READ] @" << hex << addr << dec << endl;
    invalidMemAccessAddress = (uint64_t)addr;
    return;
  }
  
  std::vector<triton::uint8> mem_value;
  for (triton::usize i = 0; i < size; i++){
    mem_value.push_back(buffer[i]);
  }

  const auto& triton_value = ctx.getConcreteMemoryAreaValue(addr,size);

  /* triton value might change in the current emulation
     session and be different than mem_value in the paused
     process. For that, if a memory location already been 
     written in the current session, we then do not 
     synchronize it
  */
  if (mem_value == triton_value) {
    if (debug) LOG("mem_value == triton_value\n");
    if (log_symbolic_defs && ctx.isMemorySymbolized(addr)) {
      memory_access_addr = addr;
      memory_access_size = size;
    }
    return;
  } else if (debug) {
    LOG("mem_value != triton_value\n");
    LOG("Synchronizing %d byte(s) @ %llx\n", size, addr);
  }
  
  bool symbolized = false;
  bool written = false;
  bool concretize = false;

  uint64_t cur_byte_addr = 0;
  for (triton::usize i = 0; i < size; i++) {
    cur_byte_addr = addr+i;
    
    symbolized = ctx.isMemorySymbolized(cur_byte_addr);
    written = (written_memory_addresses.find(cur_byte_addr) != written_memory_addresses.end());

    if (symbolized && !written) {
      symbolizedAccess = true;
      if (mem_value[i] == triton_value[i]) {
        if (debug) LOG("  MemRead @ %llx | offset [%02zu] | Value [%02x] [symbolic | not synchronizing]\n", cur_byte_addr, i, triton_value[i]);
      } else {
        LOG(RED"!= guarded [concretizing]\n");
        if (debug) LOG("  MemRead @ %llx | offset [%02zu] | old Value [%02x] | new Value [%02x] [concretizing]\n", cur_byte_addr, i, triton_value[i], buffer[i]);
        ctx.setConcreteMemoryValue(cur_byte_addr, buffer[i], false);
        concretize = true;
      }
    } else if (written) {
      if (debug) LOG("  MemRead @ %llx | offset [%02zu] | Value [%02x] [in written | not synchronizing]\n", cur_byte_addr, i, triton_value[i]);
    } else {
      if (debug) LOG("  MemRead @ %llx | offset [%02zu] | Value [%02x] [synchronizing]\n", cur_byte_addr, i, triton_value[i]);
      ctx.setConcreteMemoryValue(cur_byte_addr, buffer[i], false);
    }
  }

  if (symbolized && !written && concretize) {
    if (debug) LOG(GREEN"Concretizing %d bytes @ %llx\n",size, addr);
    for (triton::usize i = 0; i < size; i++) {
      cur_byte_addr = addr+i;
      memGuard->remove(cur_byte_addr,1);
      ctx.concretizeMemory(cur_byte_addr);
    }
  }

  if (symbolized && !concretize && log_symbolic_defs ) {
    // LogMemSymbolicDef(mem);
    memory_access_addr = addr;
    memory_access_size = size;
  }

  return;
}

uint64_t LiteTracer::OnWatchpointAccess() {

  string module;
  string function;
  uint64_t func_start = 0;
  uint64_t func_end = 0;
  
  if (written_memory_addresses.size()) {
    for (auto& address: written_memory_addresses)
      LOG(" addr: %llx\n", address);
    FATAL("written is not empty!");
  }

  arm_thread_state64_t *state = (arm_thread_state64_t*)(mach_exception->new_state);
  
  if (debug) printf("%s [%d | %d] %llx ######\n", line.c_str(), 
    outOfRangeAccessCount, 
    watchpoint_hits, 
    mach_exception->code[1]);

  if (debug) LOG("Thread: %x\n", mach_exception->thread_port);

  GetFunctionInfo(mem_hit_pc, module, function, func_start, func_end);

  uint32_t instruction_bytes;
  ReadTargetMemory(mem_hit_pc, 4, (void*)&instruction_bytes);
  instruction = triton::arch::Instruction((const unsigned char*)&instruction_bytes, 4);
  instruction.setAddress(mem_hit_pc);
  ctx.concretizeAllRegister();
  SetContextConcreteRegisterValues(state);
  ctx.processing(instruction);

  if (instruction.isSymbolized()) {
      OnSymbolizedInstruction(instruction, module, function);
  } else {
      OnUnsymbolizedInstruction(instruction, module, function);
      doEmulate = false;
  }

  if (memory_access_addr) {
    LogMemSymbolicDef();
    memory_access_addr = 0;
    memory_access_size = 0;
  }

  uint64_t pc = static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_pc));

  if (initInstruction) {
    initInstruction = false;
    if (debug) WARN("Init instruction was not proccessed in a read/write callback");
    // TODO: check isGuarded for the right size
    // for now we are just checking the first byte.
    if (memGuard->isGuarded((uint64_t)mach_exception->code[1], 1)) {
      if (debug) WARN("Guarded mem hit was not proccessed in a read/write callback");
    }
    //memGuard->step((uint64_t)mach_exception->code[1], 1);
    if (pc == mem_hit_pc){
      if (debug) WARN("same pc! returning (pc+4)");
      return (pc+4);
    }
    // initInstruction was not proccessed in a read/write callback
    // and mem access addresses are not guarded, so returing here
    // as there is no need to emulate
    return pc;
  }

  //AddToCoverage(mem_hit_pc);

  if (!doEmulate) {
    return pc;
  }
  
  insnAddresses.clear();
  insnAddresses.insert(mem_hit_pc);

  uint64_t step_pc = pc;
  
  step_pc = Emulate(pc);
  total_evtime += evtime;
  evtime = 0;

  return step_pc;
}

bool LiteTracer::OnMemAccess() {

  if (stop_point_reached) return true;

  if (debug) printf("%s [%d | %d] %llx ######\n", line.c_str(), 
    outOfRangeAccessCount, 
    watchpoint_hits, 
    mach_exception->code[1]);

  if (debug) LOG("[OnMemAccess] %llx | %llx\n", mach_exception->code[0], mach_exception->code[1]);

  if (stop_point_reached) {
    FATAL("[OnMemAccess] stop point reached");
  }
  
  if (!IsTargetAlive()) return true;

  arm_thread_state64_t *state = (arm_thread_state64_t*)(mach_exception->new_state);

  if (steppingThreads.size()) {
    if (debug) LOG(RED"Ignoring MemAccess while stepping!\n");
    arm_thread_state64_t *state = (arm_thread_state64_t*)(mach_exception->new_state);
    if (debug) LOG(RED"Thread: %u | %llx\n", mach_exception->thread_port, mach_exception->code[1]);
    LogSymbol(state->__pc);
    LogMemAccess(state->__pc, true);
    return true;
  }

  if (!memGuard->isGuardedAccess(mach_exception->code[1],32)){
    LOG("  sample: %s\n", sample->filename.c_str());
    LOG("memAccess event to unguarded page | %llx\n", mach_exception->code[1]);
    LogSymbol(state->__pc);
  }
  
  memGuard->memAccess(mach_exception->code[1]);
  
  steppingThreads.insert(mach_exception->thread_port);

  if (memGuard->stepping) {
    FATAL("not sure how we got here!");
    if (debug) LOG(RED"Ignoring MemAccess while stepping!\n");
    arm_thread_state64_t *state = (arm_thread_state64_t*)(mach_exception->new_state);
    if (debug) LOG(RED"Thread: %x | %llx\n", mach_exception->thread_port, mach_exception->code[1]);
    LogSymbol(state->__pc);
    LogMemAccess(state->__pc, true);
    return true;
  }

  memGuard->unguardAllPages();

  initInstruction = true;
  doEmulate = true;

  mem_hit_pc = state->__pc;
  uint64_t step_pc = mem_hit_pc+4;

  //LogSymbol(state->__pc);

  init_fp = state->__fp;
  RemoveLocalVariables(state->__sp);
  //last_sp = state->__sp
  
  // todo: use exact mem access size intead of 32
  if (memGuard->isGuarded(mach_exception->code[1],32)) {
    written_memory_addresses.clear();
    step_pc = OnWatchpointAccess();
    uint64_t sp = static_cast<uint64_t>(ctx.getConcreteRegisterValue(ctx.registers.aarch64_sp));
    RemoveLocalVariables(sp);

    // if (memGuard->hasUpdatedPages)
    //   memGuard->updateWatchpoints();
    
    if ((!step_pc) || (step_pc == state->__pc)) {
      if (!invalidMemAccessAddress)
        step_pc = state->__pc + 4;
    }
  } else {
    outOfRangeAccessCount += 1;
  }

  if (memGuard->hasUpdatedPages)
    memGuard->updateWatchpoints();

  SuspendThreads();

  ThreadAddHardwareBreakpoint(step_pc);
  
  if (debug) LOG(BLUE"setting hardware breakpoint @ %llx\n",step_pc);

  CheckGuardedSymbolicMemory();

  lastMemAccessThread = mach_exception->thread_port;
  
  return true;
}

bool LiteTracer::OnWatchpoint() {
  if (debug) LOG("[OnWatchpoint] %llx | %llx\n", mach_exception->code[0], mach_exception->code[1]);
  return OnMemAccess();
}

uint64_t LiteTracer::GetThread_ID_FromPort(mach_port_t thread_port) {
    thread_identifier_info_data_t identifier_info;
    mach_msg_type_number_t count = THREAD_IDENTIFIER_INFO_COUNT;
    kern_return_t kr = thread_info(thread_port, THREAD_IDENTIFIER_INFO, (thread_info_t)&identifier_info, &count);
    
    if (kr != KERN_SUCCESS) {
      LOG("failed to get thread info\n");
      return THREAD_NULL; // Return NULL thread if failed to get info
    }
    
    return identifier_info.thread_id;
}

void LiteTracer::OnSingleStep() {
  stepCount++;
  LOG("[OnSingleStep] [%llu] [%x] %llx\n", stepCount, mach_exception->thread_port, (uint64_t)GetRegister(ARCH_PC));
  SetSingleStep();
}

void LiteTracer::OnHardwareBreakpoint() {
  //if (debug) LOG("OnHardwareBreakpoint()\n");

  if(stop_point_reached) {
    LOG("  sample: %s\n", sample->filename.c_str());
    FATAL("[OnHardwareBreakpoint] stop point reached");
  }

  if (!IsTargetAlive()) {
    if (invalidMemAccessAddress) {
      LogInvalidMemAccessSymbol(instruction.getAddress());
      invalidMemAccess << instruction << endl;
      LOG("invalid MemAccess!\n");
    }
    FATAL("[OnHardwareBreakpoint] Target is not Alive!\n");
  }

  arm_debug_state64_t thread_state;
  memset(&thread_state, 0, sizeof(arm_debug_state64_t));
  mach_msg_type_number_t state_count = ARM_DEBUG_STATE64_COUNT;
  kern_return_t kr;

  hardware_breakpoint_hits += 1;

  thread_state.__bvr[0] = 0;
  thread_state.__bcr[0] = BREAKPOINT_DISABLE;
  if (debug) LOG(BLUE"Removing hardware breakpoint @ %llx\n", mach_exception->code[1]);
  ThreadSetState(mach_exception->thread_port, (thread_state_t)&thread_state);
  
  num_steppingThreads -= 1;
  if (num_steppingThreads)
    FATAL("got another stepping thread!");

  memGuard->guardAllPages();

  ResumeThreads();

  suspendedThreads.clear();
  steppingThreads.clear();
}

void LiteTracer::OnUnhandledException() {
  if (debug) LOG("OnUnhandledException()\n");
  arm_thread_state64_t *state = (arm_thread_state64_t*)(mach_exception->new_state);
  if (debug) LOG("Thread: %x\n", mach_exception->thread_port);
  LogSymbol(state->__pc);
  LogThreadState(state);
  uint64_t step_pc = OnWatchpointAccess();
}

void LiteTracer::OnStoppoint() {
  if (debug) LOG("[OnStoppoint]\n");
  stop_point_reached = true;
  if (debug) LOG("Removing all guarded addresses/pages\n");
  //memGuard->clearTaskState();
  memGuard->removeAll();
  ResumeThreads();
  steppingThreads.clear();
  suspendedThreads.clear();
}

void LiteTracer::OnInputpoint() {

  if (debug) LOG("OnInputpoint() %d\n", mach_exception->exception_type);
  
  memGuard = new MemoryGuard(mach_target, debug);

  input_address = (uint64_t)GetRegister(ArgumentToRegister(input_register))+(uint64_t)input_register_offset;
  trace_address = input_address+(uint64_t)trace_index;

  for (uint32_t i = 0; i < num_bytes; i++) {
    mach_target->ReadMemory((uint64_t)input_address+i, 1, &trace_byte);
    ctx.setConcreteMemoryValue((uint64_t)input_address+i, trace_byte, false);
  }

  initInstruction = false;
  ctx.symbolizeMemory(trace_address,num_bytes);
  initInstruction = true;

  memGuard->add(trace_address, num_bytes);
  memGuard->guardAllPages();
  
  /* storing the initial task state */
  TaskGetState();
  
  if (debug) LOG("Guarding Memory @ 0x%016llx[%d]\n", trace_address, num_bytes);
  memGuard->memAccess(trace_address);

  written_memory_addresses.clear();
}

bool LiteTracer::OnException(Exception *exception_record) {
  return Debugger::OnException(exception_record);
}

void LiteTracer::OnProcessCreated() {
  Debugger::OnProcessCreated();
}

void LiteTracer::OnProcessExit() {
  Debugger::OnProcessExit();
}
