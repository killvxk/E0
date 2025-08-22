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


/**
 * E0 Sample Runner Functions, Solving path  
 * constraints and generating new samples.
 */

RunResult Tracer::RunSample(ThreadContext *tc, Sample *sample, int *has_new_pathes, bool trim, bool report_to_server, uint32_t init_timeout, uint32_t timeout, Sample *original_sample) {
  if (has_new_pathes) {
    *has_new_pathes = 0;
  }

  DebuggerStatus status;
  RunResult result;

  auto trace_start = high_resolution_clock::now();

  if (!tc->sampleDelivery->DeliverSample(sample)) {
    FATAL("Error delivering sample, retrying with a clean target");
  }
  
  ThreadContext::instru = new LiteTracer(
    pyBridge, 
    symbolicDefs, 
    symbolicDefs_mutex,
    invocations,
    invocations_mutex);
  
  ThreadContext::instru->Init(tracer_argc, tracer_argv);

  ThreadContext::instru->symbolic_defs_dir = symbolic_defs_dir;

  exec_mutex.Lock();
  total_execs++;
  exec_mutex.Unlock();
  
  ThreadContext::instru->debug = debug;
  ThreadContext::instru->timeout = timeout;

  ThreadContext::instru->ctx.addCallback(triton::callbacks::GET_CONCRETE_MEMORY_VALUE, &MemReadCallback);
  ThreadContext::instru->ctx.addCallback(triton::callbacks::SET_CONCRETE_MEMORY_VALUE, &MemWriteCallback); 
  ThreadContext::instru->trace_index = trace_index;
  ThreadContext::instru->payload_size = payload_size;

  ThreadContext::instru->sample = sample;
  ThreadContext::instru->__input_data = sample->bytes;
  ThreadContext::instru->__input_datalen = sample->size;
  ThreadContext::instru->__input_sample_filename = sample->filename;

  if (silence) freopen("/dev/null", "w", stdout);
  status = ThreadContext::instru->Run(tc->target_argc, tc->target_argv, timeout);
  
  sample->num_runs++;

  if (ThreadContext::instru->invalidMemAccessAddress){
    char fileindex[20];
    snprintf(fileindex, sizeof(fileindex), "%05lld", sample->sample_index);

    string ima_filename = string("sample_") + fileindex + "_ima_" + std::to_string(sample->num_crashes);
    string outfile = DirJoin(invalid_mem_access_dir, ima_filename);
    ThreadContext::instru->SaveInvalidMemAccess(outfile.c_str());
  }

  switch (status) {
  case DEBUGGER_CRASHED:
    LOG("Process crashed\n");
    crash_mutex.Lock();
    sample->num_crashes++;
    num_crashes++;
    crash_mutex.Unlock();

    if (save_crashes && (sample->num_crashes > 0)) {
      output_mutex.Lock();
      char fileindex[20];
      snprintf(fileindex, sizeof(fileindex), "%05lld", sample->sample_index);

      string crash_filename = string("sample_") + fileindex + "_crash_" + std::to_string(sample->num_crashes);
      string outfile = DirJoin(crash_dir, crash_filename);
      sample->Save(outfile.c_str());
      output_mutex.Unlock();
    }

    result = CRASH;
    break;
  case DEBUGGER_HANGED:
    LOG("Process hanged\n");

    if (save_hangs) {
      output_mutex.Lock();
      num_hangs++;
      sample->num_hangs++;
      if (sample->num_hangs > 0) {
        char fileindex[20];
        snprintf(fileindex, sizeof(fileindex), "%05lld", sample->sample_index);

        string hang_filename = string("sample_") + fileindex + "_hang_" + std::to_string(sample->num_hangs);
        string outfile = DirJoin(hangs_dir, hang_filename);
        sample->Save(outfile.c_str());
      }
      output_mutex.Unlock();
    }

    result = HANG;
    break;
  case DEBUGGER_PROCESS_EXIT:
    LOG("Finding New Pathes:\n");
    FindNewPathes(ThreadContext::instru, sample);

    if (ThreadContext::instru->nSat > 0)
      *has_new_pathes = 1;

    LOG("Process finished normally\n");
    result = OK;
    break;
  case DEBUGGER_TARGET_END:
    if (ThreadContext::instru->IsTargetFunctionDefined()) {
      LOG("Target function returned normally\n");
      result = OK;
      break;
    } else {
      FATAL("Unexpected status received from the debugger\n");
    }
    break;
  default:
    FATAL("Unexpected status received from the debugger\n");
    break;
  }

  if (log_mem_access)
    ThreadContext::instru->LogMemAccessHits();

  size_t iCoverage = 0;
  LOG("Unique Symbolized Instructions: %zu\n", ThreadContext::instru->iCoverage.size());
  for (auto kv : ThreadContext::instru->iCoverage){
    iCoverage += kv.second;
  }

  if (iCoverage > sample->coverage)
    sample->coverage = iCoverage;

  if (ThreadContext::instru->watchpoint_hits > sample->num_mem_access)
    sample->num_mem_access = ThreadContext::instru->watchpoint_hits;

  auto trace_stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(trace_stop-trace_start);
  LOG("Done tracing %d byte(s) in %lld ms\n", num_bytes, duration.count());
  LOG("icov: %zu | sat: %d | unsat: %d | timeout: %d ", 
    iCoverage, 
    ThreadContext::instru->nSat,
    ThreadContext::instru->nUnsat,
    ThreadContext::instru->nTimeout);
  cout << WHITE"[Out: " << RED << ThreadContext::instru->outOfRangeAccessCount << WHITE" | In: " << GREEN << ThreadContext::instru->watchpoint_hits << WHITE"] " << sample->filename << endl << WHITE;
  
  ThreadContext::instru->Kill();
  delete ThreadContext::instru;
  ThreadContext::instru = nullptr;
  
  return result;
}


void Tracer::FindNewPathes(LiteTracer* &inst, Sample *sample) {

  triton::engines::solver::status_e status;
  std::list<triton::uint64> pathaddrs;
  auto pcs = inst->ctx.getPathConstraints();
  auto ast = inst->ctx.getAstContext();
  
  /* Building path predicate. Starting with True. */
  auto predicate = ast->equal(ast->bvtrue(), ast->bvtrue());
  
  int i = 0;
  for (const auto& pc : pcs) {
    uint64_t pcSrcAddr = pc.getSourceAddress();
    if (!pcSrcAddr) {
      // Update the previous constraints with true branch to keep a good path.
      // The enforced value of the EA is a true branch.
      predicate = ast->land(predicate, pc.getTakenPredicate());
      continue;
    }

    uint64_t src_module_base = inst->AddressToModuleBase(pcSrcAddr);
    pcSrcAddr = pcSrcAddr-src_module_base;
    pathaddrs.push_back(pcSrcAddr);
    
    for(const auto& branch: pc.getBranchConstraints()) {
      /* Did we already generated a model? */

      std::list<triton::uint64> copy(pathaddrs);
      uint64_t branchDstAddr = std::get<2>(branch);

      if (branchDstAddr) {
        uint64_t dst_module_base = inst->AddressToModuleBase(branchDstAddr);
        branchDstAddr = branchDstAddr-dst_module_base;
      } else {
        FATAL("branchDstAddr is null");
      }
      copy.push_back(branchDstAddr);

      FullPathHasher hasher;
      uint64_t pathHash = hasher(copy);

      if (pathHash == sample->target_path)
        sample->target_path_reached = 1;
      
      donePathes_mutex.Lock();
      if (donePathes.find(pathHash) != donePathes.end()){
        donePathes_mutex.Unlock();
        continue;
      }
      /* Insert the path hash to the donePathes */
      donePathes.insert(pathHash);
      donePathes_mutex.Unlock();

      if(pc.isMultipleBranches()) {
        bool isTaken      = std::get<0>(branch);
        uint64_t src_addr = std::get<1>(branch);
        uint64_t dst_addr = std::get<2>(branch);

        //cout << hex << isTaken << "\t" << src_addr << "->" << dst_addr << dec << endl;
        if (!isTaken) {
          auto constraint = ast->land(predicate, std::get<3>(branch));
          //LOG("[%d/%d] Evaluating edge: %llx -> %llx\n", i, pcs.size(), src_addr, dst_addr);
          
          std::unordered_map<triton::usize, triton::engines::solver::SolverModel> model;
          
          try {
            model = inst->ctx.getModel(constraint, &status, 600);
          } catch (const std::runtime_error& e) {
            // Catch a specific exception
            std::cerr << "Caught a runtime error: " << e.what() << std::endl;
          } catch (const std::exception& e) {
            // Catch any standard exception
            std::cerr << "Caught an exception: " << e.what() << std::endl;
          } catch (...) {
            // Catch any other type of exception
            std::cerr << "Caught an unknown exception" << std::endl;
          }
          
          output_mutex.Lock();
          if (status == triton::engines::solver::SAT) {
            inst->nSat++;
            num_sat++;
            sample->num_sat++;
            //worklist.push_front(model);
          } else if (status == triton::engines::solver::TIMEOUT) {
            inst->nTimeout++;
            num_timeout++;
            sample->num_timeout++;
            //continue;
          } else {
            inst->nUnsat++;
            num_unsat++;
            sample->num_unsat++;
          }
          output_mutex.Unlock();

          if (!model.empty()) {
            i++;
            LOG("[%d] pathHash: %llx\n", i, pathHash);
            inst->LogSymbol2(src_addr);
            LOG("  computing new input for 0x%llx -> 0x%llx\n", src_addr, dst_addr);
            
            Sample *new_sample = SaveSample(sample, pathHash, model);
            LOG("  sample: %s\n", new_sample->filename.c_str());
            
            if (!silence)
              DiffHexDump(*new_sample, *sample, trace_index);

            queue_mutex.Lock();
            sample_queue.push(new_sample);
            queue_mutex.Unlock();
          }
        }
      } else {
        /* MultipleBranches is false if the instruction is like br x0 */
        auto constraint = ast->land(predicate, ast->lnot(std::get<3>(branch)));
        std::vector<std::unordered_map<triton::usize, triton::engines::solver::SolverModel>> models;
        try {
          LOG("is not MultipleBranches and getting models\n");
          models = inst->ctx.getModels(constraint, ea, &status, 600); // 60 seconds timeout
        } catch (const std::runtime_error& e) {
          // Catch a specific exception
          std::cerr << "Caught a runtime error: " << e.what() << std::endl;
        } catch (const std::exception& e) {
          // Catch any standard exception
          std::cerr << "Caught an exception: " << e.what() << std::endl;
        } catch (...) {
          // Catch any other type of exception
          std::cerr << "Caught an unknown exception" << std::endl;
        }
        
        output_mutex.Lock();
        if (status == triton::engines::solver::SAT) {
          for (const auto& model : models) {
            inst->nSat++;
            num_sat++;
            sample->num_sat++;
            FATAL("not multibraches and solved");
          }
        } else if (status == triton::engines::solver::TIMEOUT) {
          inst->nTimeout++;
          num_timeout++;
          sample->num_timeout++;
        } else {
          inst->nUnsat++;
          num_unsat++;
          sample->num_unsat++;
        }
        output_mutex.Unlock();
      }
    }
    // Update the previous constraints with true branch to keep a good path.
    predicate = ast->land(predicate, pc.getTakenPredicate()); 
  }

  while (inst->worklist.size()) {
    /* Pickup a model */
    auto model = *(inst->worklist.begin());
    /* Remove the model from the worklist */
    inst->worklist.erase(inst->worklist.begin());

    if (!model.empty()) {
      i++;
      LOG("[%d] computing new input for an EA\n",i);
      
      Sample *new_sample = SaveSample(sample, 0, model);
      LOG("  sample: %s\n", new_sample->filename.c_str());
      
      if (!silence)
        DiffHexDump(*new_sample, *sample, trace_index);

      queue_mutex.Lock();
      sample_queue.push(new_sample);
      queue_mutex.Unlock();
    }
  }
}
