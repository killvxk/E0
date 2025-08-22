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

#ifndef MEMORYGUARD_H
#define MEMORYGUARD_H

#include <map>
#include <deque>
#include <unordered_set>
#include <set>
#include <mach/mach.h>
#include <mach/mach_vm.h>

#include "pagemap.h"
#include "watchpoint.h"

#include "macOS/machtarget.h"
extern "C" {
  #include "macOS/mig_server.h"
}

#define BREAKPOINT_ENABLE 5
#define BREAKPOINT_DISABLE 4

// Break only in privileged or user mode
#define S_USER ((uint32_t)(2u << 1))

#define BCR_ENABLE ((uint32_t)(1u))
#define WCR_ENABLE ((uint32_t)(1u))

// Watchpoint load/store
#define WCR_LOAD ((uint32_t)(1u << 3))
#define WCR_STORE ((uint32_t)(1u << 4))

// Enable breakpoint and watchpoint
#define MDE_ENABLE ((uint32_t)(1u << 15))

// Single instruction step
// (SS bit in the MDSCR_EL1 register)
#define SS_ENABLE ((uint32_t)(1u))

class MemoryGuard {
private:

    struct PageInfo {
        bool isInitialized = false;
        vm_prot_t protection = 0;   // Original protection
        bool isStack = false;
        //std::set<uint64_t> addresses;
        uint64_t pageBaseAddr = 0;
        PageMap addressMap;  // bit-manipulation based tracking
        bool recentlyAccessed = false;
        uint64_t recentlyAccessedAddress = 0;
        uint64_t range = 0;
        bool updateWatchpoint = true;
        
        Watchpoint watchpoint;
        Watchpoint watchpointSplits[4];
        
        // Number of active split watchpoints
        size_t numSplitWatchpoints = 0;
        
        // Gap tracking
        uint64_t gapSize = 0;
    };

    MachTarget* mach_target;
    //task_t task;
    bool debug;
    int pageSize;

    //int wvr_id = 0;

    static constexpr size_t maxAccessPages = 4;     // Fixed size
    std::array<uint64_t, maxAccessPages> circularAccessPages; // Stores the 4 most recent pages
    std::unordered_set<uint64_t> recentPageAccessSet;  // For quick membership checks
    size_t oldestPageIndex = 0;  // Tracks the oldest page position
    uint64_t lastAccessedPage = 0;  // Tracks the last added page (0 = no page added yet)
    uint64_t lastAccessedPageCount = 0;

public:
    arm_debug_state64_t taskState;

    std::map<uint64_t, PageInfo> guardedPages;   // Map to store original protection and address count for memory pages
    std::unordered_set<uint64_t> guardedAddresses;
    std::unordered_set<uint64_t> steppingPages;

    bool stepping = true;
    bool hasUpdatedPages = false;

    int totalGuarded = 0;

    MemoryGuard(MachTarget* mach_target, bool debug);
    
    ~MemoryGuard() = default;
    
    void taskGetState();
    void clearTaskState();
    int getFreeWVR();

    void memAccess(uint64_t address);
    void pageAccess(uint64_t page);
    bool isPageRecentlyAccessed(uint64_t page) const;

    uint64_t wcr(uint64_t address, uint64_t size);
    uint32_t wcrMask(uint64_t size);

    // Add memory addresses to guarded pages
    void add(uint64_t address, size_t size, bool isStack = false, bool step = false);
    // Remove memory addresses from guarded pages
    void remove(uint64_t address, size_t size);
    // Remove page from guarded pages
    void removePage(uint64_t page, PageInfo &pageInfo);
    // Remove all guarded memory addresses and restore pages protection
    void removeAll();
    // Remove memory addresses lower than given 
    // sp "stack pointer" from guarded stack pages
    void removeLocalVariables(uint64_t sp);
    // Check if any of the given memory addresses
    // belong to a guarded page
    bool isGuarded(uint64_t address, size_t size);
    // check if address belongs to one of the guarded pages
    bool isGuardedPage(uint64_t address);
    bool isGuardedAccess(uint64_t address, size_t size = 1);
    void logPagesInfo();
    void unguardAllPages();
    void guardAllPages();
    // Restore page(s) protection to step over the exception
    void step(uint64_t address, size_t size);

    void computeWatchpoint(uint64_t clusterMin, uint64_t clusterMax, Watchpoint &watchpoint);
    void protectSteppingPages();
    void updateWatchpoints();
    void updatePageWatchpoint(PageInfo& page);
    void updatePageSplitWatchpoints(PageInfo& page);
};

#endif /* MEMORYGUARD_H */