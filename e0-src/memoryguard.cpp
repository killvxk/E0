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

#include "memoryguard.h"
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include "common.h"


MemoryGuard::MemoryGuard(MachTarget* mach_target, bool debug) : mach_target(mach_target) , debug(debug) {
    pageSize = getpagesize();
    taskGetState();
}

void MemoryGuard::taskGetState() {
  // get task debug state
  kern_return_t kr;
  mach_msg_type_number_t stateCount = ARM_DEBUG_STATE64_COUNT;
  kr = task_get_state(mach_target->Task(), ARM_DEBUG_STATE64, (thread_state_t)&taskState, &stateCount);
  if (kr != KERN_SUCCESS) {
    FATAL("task_get_state error: %s", mach_error_string(kr));
  }
}

void MemoryGuard::clearTaskState() {
  for(int i = 0; i<4; i++) {
    taskState.__wvr[i] = 0;
    taskState.__wcr[i] = 0;
  }
}

uint32_t MemoryGuard::wcrMask(uint64_t size) {
    if (size <= 8) return 0;
    uint32_t bits = 0;
    while (size > 1) {
        size >>= 1;
        bits++;
    }
    //if (debug) LOG("bits: %d\n", bits);
    return bits << 24;
}

uint64_t MemoryGuard::wcr(uint64_t address, uint64_t size) {

  //LOG("wvr: %llx size: %lld\n", address, size);

  uint32_t byte_address_select = 0;
  uint32_t MASK = 0;

  if (size == 1) {
    byte_address_select = 1 << (address & 0x7);
  } else if (size <= 8) {
    byte_address_select = (1 << size) - 1;
    byte_address_select <<= (address & 0x7);
    //LOG("byte_address_select: %lx\n", byte_address_select);
  } else {
    byte_address_select = 0xff;
    //LOG("byte_address_select: %lx\n", byte_address_select);
    MASK = wcrMask(size);
    //LOG("MASK: %lx\n", MASK);
  }

  byte_address_select = byte_address_select << 5;
  uint64_t wcr = byte_address_select | // Which bytes that follow
                                                // the DVA that we will watch
                                         MASK | // MASK
                                       S_USER | // Stop only in user mode
                                     WCR_LOAD | // Stop on read access
                                    WCR_STORE | // Stop on write access
                                   WCR_ENABLE;  // Enable this watchpoint;
  return wcr;
}

int MemoryGuard::getFreeWVR() {

  for(uint8_t wvr_id = 0; wvr_id<4; wvr_id++) {
    if (taskState.__wvr[wvr_id] == 0) return wvr_id;
  }
  return -1;
}

// Add memory addresses to guarded pages
void MemoryGuard::add(uint64_t address, size_t size, bool isStack, bool step) {
    uint64_t firstPageStart = address & ~(pageSize - 1);
    uint64_t lastPageEnd = (address + size - 1) & ~(pageSize - 1);
    uint64_t stop_address = address + size;
    
    //if (debug) LOG("firstPageStart: %016llx\n", firstPageStart);
    //if (debug) LOG("lastPageEnd: %016llx\n", lastPageEnd);
    //memoryAreas[address] = {address, size, static_cast<char*>(address) + size};
    
    for (uint64_t page = firstPageStart; page <= lastPageEnd; page = page + pageSize) {
        auto& pageInfo = guardedPages[page];
        if (!pageInfo.isInitialized) {
            // If not initialized (new page) get and save original protection
            mach_vm_address_t region_start = (mach_vm_address_t)page;
            mach_vm_size_t region_size = pageSize;
            vm_region_basic_info_data_t info;
            mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
            mach_port_t object_name;
            
            kern_return_t kr = mach_vm_region(mach_target->Task(), &region_start, &region_size, VM_REGION_BASIC_INFO_COUNT_64, (vm_region_info_t)&info, &info_count, &object_name);
            if (kr == KERN_FAILURE)
                FATAL("mach_vm_region() error\n");

            pageInfo.protection = info.protection;
            pageInfo.isStack = isStack;
            pageInfo.pageBaseAddr = page;
            pageInfo.addressMap.pageBaseAddr = page;
            pageInfo.isInitialized = true;
        }
        
        // We can add the entire range at once since PageMap handles ranges efficiently

        // Calculate the range of addresses within this page
        uint64_t rangeStart = std::max(address, page);
        uint64_t rangeEnd = std::min(stop_address, page + pageSize);
        
        // Add the range of addresses to the PageMap
        size_t added = pageInfo.addressMap.add(rangeStart, rangeEnd - rangeStart);
        
        // Update total guarded count
        totalGuarded += added;
        
        pageInfo.updateWatchpoint = true;
        steppingPages.insert(page);
    }

    hasUpdatedPages = true;
    
    // // Update watchpoints if adding multiple addresses at once
    // if (size > 1) {
    //     updateWatchpoints();
    // }
}

// Remove memory addresses from guarded pages
void MemoryGuard::remove(uint64_t address, size_t size) {
    uint64_t firstPageStart = address & ~(pageSize - 1);
    uint64_t lastPageEnd = (address + size - 1) & ~(pageSize - 1);
    uint64_t stop_address = address + size;
    
    for (uint64_t page = firstPageStart; page <= lastPageEnd; page = page + pageSize) {
        // First check if page exists in guardedPages
        auto it = guardedPages.find(page);
        if (it == guardedPages.end()) continue;
        PageInfo& pageInfo = it->second;
        
        // Calculate the range of addresses within this page
        uint64_t rangeStart = std::max(address, page);
        uint64_t rangeEnd = std::min(stop_address, page + pageSize);
        
        // Remove the range of addresses from the PageMap
        size_t removed = pageInfo.addressMap.remove(rangeStart, rangeEnd - rangeStart);
        
        // Update total guarded count
        totalGuarded -= removed;
        
        // Check if no more addresses belong to the memory page
        if (pageInfo.addressMap.empty()) {
            removePage(page, pageInfo);
        } else {
            pageInfo.updateWatchpoint = true;
        }
    }
    hasUpdatedPages = true;
}

// Remove all guarded memory addresses and 
// restore all pages protection
void MemoryGuard::removeAll(){
    for (auto const& [page, pageInfo] : guardedPages) {
        mach_vm_protect(mach_target->Task(), page, pageSize, false, pageInfo.protection);
        totalGuarded -= pageInfo.addressMap.size();
    }
    guardedPages.clear();
    steppingPages.clear();

    clearTaskState();
}

void MemoryGuard::removePage(uint64_t page, PageInfo &pageInfo) {
    // Restore page protection before removing
    mach_vm_protect(mach_target->Task(), page, pageSize, false, pageInfo.protection);
    
    // if (pageInfo.recentlyAccessed){
    //     auto it = std::find(recentPageAccessDeque.begin(), recentPageAccessDeque.end(), page);
    //     if (it != recentPageAccessDeque.end()) {
    //         recentPageAccessDeque.erase(it);
    //         recentPageAccessSet.erase(page);
    //     } else {
    //         FATAL("page is not in recentPageAccessDeque");
    //     }
    // }
    guardedPages.erase(page);
    steppingPages.erase(page);
}

void MemoryGuard::logPagesInfo(){
    for (auto const& [page, pageInfo] : guardedPages) {
        if (debug) LOG("%llx: %zu\n",page, pageInfo.addressMap.size());
        // if (pageInfo.isStack) {
        //     for (auto const& addr : pageInfo.addresses)
        //         if (debug) LOG("\t%llx\n", addr);
        // }
    }
}

/**
 * Remove local variables from stack pages based on the current stack pointer.
 * 
 * This function removes tracked addresses in stack memory pages that are no longer in scope,
 * based on the current stack pointer position. For the current stack page (containing sp),
 * it only removes addresses below the stack pointer. For all stack pages within 1MB below
 * the current page, it removes all tracked addresses as they are out of scope.
 * 
 * The function works as follows:
 * 1. For the current stack page: removes only addresses below sp
 * 2. For pages below current page (within 1MB): removes all addresses
 * 3. Updates watchpoints and page protections as needed
 * 
 * @param sp The current stack pointer value. Addresses below this in 
 *           the current stack page will be removed from tracking.
 * 
 * Common scenarios:
 * - Function return: cleans up local variable addresses
 * - Stack frame cleanup: removes tracked addresses from previous frames
 * 
 * Note: This function assumes that any stack page within 1MB below the current sp
 * belongs to the current stack frame and can be safely cleaned up. Pages beyond
 * this range or non-stack pages are not affected.
 * 
 * @warning Make sure sp is a valid stack pointer value as this function will modify
 *          memory protection for affected pages.
 */
void MemoryGuard::removeLocalVariables(uint64_t sp) {
    uint64_t currentStackPage = sp & ~(pageSize - 1);
    uint64_t lowestPage = currentStackPage - (1024 * 1024); // 1MB below
    int removedLocalVariables = 0;
    std::vector<uint64_t> pagesToRemove;
    
    // Single pass through all guarded pages
    for (const auto& entry : guardedPages) {
        uint64_t pageAddr = entry.first;
        const PageInfo& pageInfo = entry.second;
        
        // Skip if not a stack page or if page is above our range
        if (!pageInfo.isStack || pageAddr > currentStackPage || pageAddr < lowestPage) {
            continue;
        }
        
        if (pageAddr == currentStackPage) {
            // Special handling for current page
            PageInfo& currentPageInfo = guardedPages[pageAddr];
            
            // Remove all addresses lower than sp in the current page
            size_t removed = currentPageInfo.addressMap.removeAddressesLowerThan(sp);
            totalGuarded -= removed;
            removedLocalVariables += removed;
            
            if (removed > 0) {
                currentPageInfo.updateWatchpoint = true;
                hasUpdatedPages = true;
            }
            
            // If addressMap is empty, add to removal list
            if (currentPageInfo.addressMap.empty()) {
                pagesToRemove.push_back(pageAddr);
            }
        } else {
            // For pages below current page, remove everything
            size_t removed = pageInfo.addressMap.size();
            totalGuarded -= removed;
            removedLocalVariables += removed;
            pagesToRemove.push_back(pageAddr);
        }
    }
    
    // Remove all identified pages
    for (uint64_t pageAddr : pagesToRemove) {
        removePage(pageAddr, guardedPages[pageAddr]);
        if (pageAddr == lastAccessedPage) lastAccessedPage = 0;
        hasUpdatedPages = true;
    }
    
    if (removedLocalVariables && debug) {
        LOG("Removed %d local variable address(es)\n", removedLocalVariables);
    }
}

/**
 * Check if any address in the given memory range is guarded.
 * 
 * This function checks whether any address in the specified range is
 * currently being tracked. It efficiently handles ranges by checking 
 * pages and utilizing the PageMap's range-checking capabilities.
 * 
 * @param address Starting address of the memory range to check
 * @param size Size of the memory range in bytes
 * @return true if any address in the range is guarded, false otherwise
 */
bool MemoryGuard::isGuarded(uint64_t address, size_t size) {
    uint64_t firstPageStart = address & ~(pageSize - 1);
    uint64_t lastPageEnd = (address + size - 1) & ~(pageSize - 1);
    uint64_t stop_address = address + size;
    
    // Check each page that overlaps with the range
    for (uint64_t page = firstPageStart; page <= lastPageEnd; page += pageSize) {
        auto it = guardedPages.find(page);
        if (it == guardedPages.end()) {
            continue;
        }
        
        PageInfo& pageInfo = it->second;
        
        // Calculate the range to check within this page
        uint64_t rangeStart = std::max(address, page);
        uint64_t rangeEnd = std::min(stop_address, page + pageSize);
        
        // Use PageMap's range checking capability
        if (pageInfo.addressMap.hasAddress(rangeStart, rangeEnd - rangeStart)) {
            if (debug) LOG("%016llx is Guarded\n", rangeStart);
            return true;
        }
    }
    
    if (debug) LOG("%llx[%zu] is not Guarded\n", address, size);
    return false;
}

/**
 * Check if any address in the given memory range is guarded.
 * 
 * This function checks whether any address in the specified range is
 * currently being tracked. It efficiently handles ranges by checking 
 * pages and utilizing the PageMap's range-checking capabilities.
 * 
 * @param address Starting address of the memory range to check
 * @param size Size of the memory range in bytes
 * @return true if any address in the range is guarded, false otherwise
 */
bool MemoryGuard::isGuardedAccess(uint64_t address, size_t size) {
    // remove highest 4 bits as 800000010200bc04 would match 10200bc04
    address = address & 0x0FFFFFFFFFFFFFFF;
    uint64_t firstPageStart = address & ~(pageSize - 1);
    uint64_t lastPageEnd = (address + size - 1) & ~(pageSize - 1);
    uint64_t stop_address = address + size;
    
    // Check each page that overlaps with the range
    for (uint64_t page = firstPageStart; page <= lastPageEnd; page += pageSize) {
        if (guardedPages.find(page) != guardedPages.end()) return true;
    }
    if (debug) LOG("%llx[%zu] is not Guarded\n", address, size);
    return false;
}

// Restore page(s) protection to step over the exception
void MemoryGuard::step(uint64_t address, size_t size) {
    //if (debug) LOG("step()\n");
    uint64_t firstPageStart = address & ~(pageSize - 1);
    uint64_t lastPageEnd = (address + size - 1) & ~(pageSize - 1);
    
    //if (debug) LOG("firstPageStart: %016llx\n", firstPageStart);
    //if (debug) LOG("lastPageEnd: %016llx\n", lastPageEnd);
    //memoryAreas[address] = {address, size, static_cast<char*>(address) + size};
    
    
    for (uint64_t page = firstPageStart; page <= lastPageEnd; page = page + pageSize) {
        if (steppingPages.find(page) != steppingPages.end())
            continue;
        if (guardedPages.find(page) != guardedPages.end()) {
            //if (debug) LOG("restoring page protection to: %d\n",guardedPages[page].protection);
            mach_vm_protect(mach_target->Task(), page, pageSize, false, guardedPages[page].protection); //VM_PROT_DEFAULT);
            steppingPages.insert(page);
            //if (debug) LOG("adding %016llx to steppingPages\n",page);
        }
    }
    
}

// check if address belongs to one of the guarded pages
bool MemoryGuard::isGuardedPage(uint64_t address) {
    uint64_t firstPage = address & ~(pageSize - 1);
    uint64_t secondPage = (address + 31) & ~(pageSize - 1);
    if (guardedPages.find(firstPage) != guardedPages.end())
        return true;
    if (firstPage != secondPage) {
        if (guardedPages.find(secondPage) != guardedPages.end())
            return true;
    }
    return false;
}

// Restore all pages protection to step over the exception
void MemoryGuard::unguardAllPages() {
    //if (debug) LOG("restoreAllPages()\n");
    for (const auto& [pageStartAddr, pageInfo] : guardedPages) {
        if (debug) LOG("unguarding page %llx\n", pageStartAddr);
        mach_vm_protect(mach_target->Task(), pageStartAddr, pageSize, false, guardedPages[pageStartAddr].protection); //VM_PROT_DEFAULT);
        steppingPages.insert(pageStartAddr);
    }
    stepping = true;
}

// Set watchpoints and guard pages after stepping over an exception
void MemoryGuard::guardAllPages() {
    
    
    //if (debug) LOG("guardAllPages()\n");
    if (!stepping) FATAL("Not Stepping!");

    //clearTaskState();
    int wvr_id = 0;

    //lastAccessedPage = 0;

    if (lastAccessedPage) {
        if (guardedPages.find(lastAccessedPage) != guardedPages.end()) {
            
            auto& pageInfo = guardedPages[lastAccessedPage];
            if (pageInfo.numSplitWatchpoints) {
                
                if (debug) LOG("guarding page %llx @taskState.__wvr[%d] %llx|%llu [last accessed and has splits]\n", 
                    lastAccessedPage, 
                    wvr_id,
                    pageInfo.watchpoint.address,
                    pageInfo.watchpoint.size);
                
                if ((!pageInfo.watchpoint.address) || (!pageInfo.watchpoint.wcr))
                    FATAL("no wp conf");

                for (int i = 0; i < pageInfo.numSplitWatchpoints; ++i) {
                    if (debug) {
                        LOG("    - @taskState.__wvr[%lld] %llx|%llu\n", 
                            wvr_id, 
                            pageInfo.watchpointSplits[i].address, 
                            pageInfo.watchpointSplits[i].size);
                    }
                    taskState.__wvr[wvr_id] = pageInfo.watchpointSplits[i].alignedAddress;
                    taskState.__wcr[wvr_id] = pageInfo.watchpointSplits[i].wcr;
                    wvr_id++;
                }
                if (debug)
                    LOG("      recentAccessAddress: %llx\n", pageInfo.recentlyAccessedAddress);
                //pageInfo.addressMap.printAddresses();
            } else {
                if (debug) LOG("guarding page %llx @taskState.__wvr[%d] %llx|%llu\n", 
                    lastAccessedPage, 
                    wvr_id,
                    pageInfo.watchpoint.address,
                    pageInfo.watchpoint.size);
                if ((!pageInfo.watchpoint.address) || (!pageInfo.watchpoint.wcr))
                    FATAL("no wp conf");
                taskState.__wvr[wvr_id] = pageInfo.watchpoint.alignedAddress;
                taskState.__wcr[wvr_id] = pageInfo.watchpoint.wcr;
                wvr_id += 1;
            }
        } else {
          LOG(RED"lastAccessedPage %llx not found!\n", lastAccessedPage);
          lastAccessedPage = 0;
        }
    }
    
    for (auto& [pageStartAddr, pageInfo] : guardedPages) {
        
        if (pageStartAddr == lastAccessedPage) continue;

        if ((!pageInfo.watchpoint.wcr) || (wvr_id >= 4)) {
            if (debug) LOG("guarding page %llx\n", pageStartAddr);
            mach_vm_protect(mach_target->Task(), pageStartAddr, pageSize, false, VM_PROT_NONE);
            continue;
        }

        if (debug) LOG("guarding page %llx @taskState.__wvr[%d] %llx|%llu\n", 
            pageStartAddr, 
            wvr_id,
            pageInfo.watchpoint.address,
            pageInfo.watchpoint.size);
        if ((!pageInfo.watchpoint.address) || (!pageInfo.watchpoint.wcr))
            FATAL("no wp conf");
        taskState.__wvr[wvr_id] = pageInfo.watchpoint.alignedAddress;
        taskState.__wcr[wvr_id] = pageInfo.watchpoint.wcr;
        wvr_id += 1;
    }
    steppingPages.clear();
    stepping = false;
}

// Restore stepping page(s) guard after stepping over the exception
void MemoryGuard::protectSteppingPages() {
    //if (debug) LOG("restoreSteppingPages()\n");
    if (steppingPages.empty()) {
        WARN("SteppingPages is empty");
        return;
    }
    for (const auto& page : steppingPages) {
        //if (debug) LOG("Restoring stepping page: %016llx\n", page);
        mach_vm_protect(mach_target->Task(), page, pageSize, false, VM_PROT_NONE);
    }
    steppingPages.clear();
    if (steppingPages.size())
        FATAL("failed to clear steppingPages");
}

bool MemoryGuard::isPageRecentlyAccessed(uint64_t page) const {
    return recentPageAccessSet.find(page) != recentPageAccessSet.end();
}

void MemoryGuard::pageAccess(uint64_t page) {
    if (recentPageAccessSet.find(page) != recentPageAccessSet.end()) {
        // Page is already in the buffer, update the last added page
        if (guardedPages[page].isStack) lastAccessedPage = page;
        return;
    }

    if (recentPageAccessSet.size() == maxAccessPages) {
        // Remove the oldest page
        uint64_t oldestPage = circularAccessPages[oldestPageIndex];
        recentPageAccessSet.erase(oldestPage);
        if (guardedPages.find(oldestPage) != guardedPages.end()) {
            guardedPages[oldestPage].recentlyAccessed = false;
        }
    }

    // Add the new page to the buffer and update the set
    circularAccessPages[oldestPageIndex] = page;
    recentPageAccessSet.insert(page);

    guardedPages[page].recentlyAccessed = true;

    // Update the last accessed page if it is stack
    if (guardedPages[page].isStack) lastAccessedPage = page;

    // Move to the next position in a circular manner
    oldestPageIndex = (oldestPageIndex + 1) % maxAccessPages;
}

void MemoryGuard::memAccess(uint64_t address) {
  // todo: what if mem access is hitting two pages?
  uint64_t page = address & ~(pageSize - 1);

  // first check if the memAccess is in one of of the guarded pages
  // if not, then it might be related to bad access (crash)
  if (guardedPages.find(page) == guardedPages.end()) {
    uint64_t guarded_page = page&0xFFFFFFFFFFFF;
    if (guardedPages.find(guarded_page) != guardedPages.end()) {
        guardedPages[guarded_page].recentlyAccessed = false;
    } else {
      LOG(RED"[MemoryGuard::memAccess] got bad access @%llx\n", address);
    }
    return;
  }

  pageAccess(page);

  auto& pageInfo = guardedPages[page];
  // updatePageSplitWatchpoints if address is not guarded.
  if (!pageInfo.addressMap.hasAddress(address)) {
    // Set page recentlyAccessedAddress
    pageInfo.recentlyAccessedAddress = address;
    pageInfo.updateWatchpoint = true;
    hasUpdatedPages = true;
    //updatePageSplitWatchpoints(pageInfo);
  } else {
    pageInfo.recentlyAccessedAddress = 0;
  }
}

void MemoryGuard::computeWatchpoint(uint64_t clusterMin, uint64_t clusterMax, Watchpoint &watchpoint) {
    uint64_t range = clusterMax - clusterMin;
    // Compute smallest power-of-two size that can cover this range
    uint64_t size = 1ULL << static_cast<uint64_t>(std::ceil(std::log2(range + 1)));
    uint64_t alignedAddress = clusterMin & ~(size - 1);
    LOG("alignedAddress: %llx\n", alignedAddress);

    // Expand if needed
    while (alignedAddress + size < clusterMax) {
        size <<= 1;
        alignedAddress = clusterMin & ~(size - 1);
    }

    if (size == 1) {
        watchpoint.address = clusterMin;
        watchpoint.alignedAddress = clusterMin & ~0x7;
        watchpoint.size = 1;
        watchpoint.wcr = wcr(clusterMin,size);
    } else if (size <= 8) {
        watchpoint.address = clusterMin;
        watchpoint.alignedAddress = clusterMin & ~0x7;
        watchpoint.size = size;
        watchpoint.wcr = wcr(alignedAddress,size);
    } else {
        watchpoint.address = alignedAddress;
        watchpoint.alignedAddress = alignedAddress;
        watchpoint.size = size;
        watchpoint.wcr = wcr(alignedAddress, size);
    }

    LOG("alignedAddress: %llx | size: %lld\n", alignedAddress, size);
};

void MemoryGuard::updatePageWatchpoint(PageInfo& pageInfo) {
    pageInfo.addressMap.computeWatchpoint(pageInfo.watchpoint);
    pageInfo.numSplitWatchpoints = 0;
    pageInfo.gapSize = 0;
    pageInfo.recentlyAccessedAddress = 0;
}


void MemoryGuard::updatePageSplitWatchpoints(PageInfo& pageInfo) {

    pageInfo.numSplitWatchpoints = pageInfo.addressMap.computeWatchpointSplits(
        pageInfo.watchpoint,
        pageInfo.watchpointSplits,
        pageInfo.recentlyAccessedAddress);

    pageInfo.gapSize = 0;
}

void MemoryGuard::updateWatchpoints() {

    if (debug) LOG("[updateWatchpoints]\n");
    for (auto& [pageStartAddr, pageInfo] : guardedPages) {   
        
        if (pageInfo.updateWatchpoint) {
            if (pageInfo.recentlyAccessedAddress) {
                updatePageSplitWatchpoints(pageInfo);
            } else {
                updatePageWatchpoint(pageInfo);
            }
            if (pageInfo.gapSize) {
                LOG("gotGAP: %d\n", pageInfo.gapSize);
            }
            pageInfo.updateWatchpoint = false;
            if (debug) LOG("[+] %llx: %llx[%lld] gapSize: %d split: %zu\n", 
                pageStartAddr, 
                pageInfo.watchpoint.address, 
                pageInfo.watchpoint.size,
                pageInfo.gapSize,
                pageInfo.numSplitWatchpoints);

        } else {
            if (debug) LOG("[-] %llx: %llx[%lld] gapSize: %d split: %zu\n", 
                pageStartAddr, 
                pageInfo.watchpoint.address, 
                pageInfo.watchpoint.size,
                pageInfo.gapSize,
                pageInfo.numSplitWatchpoints);
        }
    }
    hasUpdatedPages = false;
}
