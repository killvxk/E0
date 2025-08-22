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
 * PageMap - Efficient address tracking within a 16K page using hierarchical bitmaps
 * 
 * This class implements a space-efficient way to track address usage within a 16K page,
 * using a three-level bitmap hierarchy:
 * 
 * Level 1 - Root Bitmap (1 x uint32_t):
 * - Each bit represents one of 32 segments (512 bytes each)
 * - A bit is set if any qword in the corresponding segment contains tracked addresses
 * 
 * Level 2 - Page Segments (32 x uint64_t):
 * - Each uint64_t represents 64 qwords (8 bytes each, 512 bytes total)
 * - A bit is set if the corresponding qword contains any tracked bytes
 * 
 * Level 3 - Page Qwords (2048 x uint8_t):
 * - Each byte represents one qword (8 bytes)
 * - Each bit in the byte marks a tracked byte position within the qword
 */


#include "pagemap.h"
#include "common.h"

size_t PageMap::getQwordIndex(uint64_t addr) {
    return (addr & 0x3FFF) >> 3;  // Mask to 16K page size and divide by 8
}

uint8_t PageMap::getByteOffsetInQword(uint64_t addr) {
    return addr & 0x7;  // Get last 3 bits
}

uint64_t PageMap::wcr(uint64_t size, uint8_t qword_content) {
    uint32_t byte_address_select;
    uint32_t MASK = 0;

    if (size <= 8) {
        // For sizes <= 8, use the provided qword content
        // which already has the correct bits set
        byte_address_select = qword_content;
    } else {
        byte_address_select = 0xff;
        // For larger sizes, calculate the number of bits directly
        MASK = (63 - __builtin_clzll(size)) << 24;
    }

    byte_address_select <<= 5;
    return byte_address_select |    // Which bytes that follow the DVA that we will watch
           MASK |                   // MASK
           S_USER |                 // Stop only in user mode
           WCR_LOAD |               // Stop on read access
           WCR_STORE |              // Stop on write access
           WCR_ENABLE;              // Enable this watchpoint
}

PageMap::PageMap() : addressCount(0), rootBitmap(0) {
    // Initialize all arrays to 0
    for (size_t i = 0; i < NUM_SEGMENTS; i++) {
        pageSegments[i] = 0;
    }
    for (size_t i = 0; i < NUM_QWORDS; i++) {
        pageQwords[i] = 0;
    }
}

size_t PageMap::size() const {
    return addressCount;
}

bool PageMap::empty() const {
    return addressCount == 0;
}

bool PageMap::hasAddress(uint64_t addr, size_t size) const {
    if (size == 1) {
        // Fast path for single address
        size_t qwordIndex = getQwordIndex(addr);
        size_t segmentIndex = qwordIndex >> 6;
        
        // // First check if root bit and segment bit are set
        // if (!(rootBitmap & (1U << segmentIndex)) || 
        //     !(pageSegments[segmentIndex] & (1ULL << (qwordIndex & 63)))) {
        //     return false;
        // }
        
        uint8_t byteOffset = getByteOffsetInQword(addr);
        return (pageQwords[qwordIndex] & (1 << byteOffset)) != 0;
    }

    // Handle range of addresses
    uint64_t endAddr = addr + size - 1;
    size_t startQwordIndex = getQwordIndex(addr);
    size_t endQwordIndex = getQwordIndex(endAddr);
    uint8_t startByteOffset = getByteOffsetInQword(addr);
    uint8_t endByteOffset = getByteOffsetInQword(endAddr);

    // If range is within single qword
    if (startQwordIndex == endQwordIndex) {
        size_t segmentIndex = startQwordIndex >> 6;
        
        // Check root and segment bits first
        if (!(rootBitmap & (1U << segmentIndex)) || 
            !(pageSegments[segmentIndex] & (1ULL << (startQwordIndex & 63)))) {
            return false;
        }
        
        uint8_t mask = ((1 << (endByteOffset + 1)) - 1) & ~((1 << startByteOffset) - 1);
        return (pageQwords[startQwordIndex] & mask) != 0;
    }

    // Check first partial qword
    size_t startSegmentIndex = startQwordIndex >> 6;
    if ((rootBitmap & (1U << startSegmentIndex)) &&
        (pageSegments[startSegmentIndex] & (1ULL << (startQwordIndex & 63)))) {
        uint8_t firstMask = ~((1 << startByteOffset) - 1);
        if ((pageQwords[startQwordIndex] & firstMask) != 0) {
            return true;
        }
    }

    // Check middle complete qwords
    size_t startNextSegmentIndex = (startQwordIndex + 63) >> 6;
    size_t endPrevSegmentIndex = endQwordIndex >> 6;
    
    // Check full segments first
    for (size_t segmentIndex = startNextSegmentIndex; segmentIndex < endPrevSegmentIndex; segmentIndex++) {
        if ((rootBitmap & (1U << segmentIndex)) && pageSegments[segmentIndex] != 0) {
            return true;
        }
    }

    // Check remaining qwords in the last segment
    if (startNextSegmentIndex <= endPrevSegmentIndex) {
        for (size_t qwordIndex = startQwordIndex + 1; qwordIndex < endQwordIndex; qwordIndex++) {
            size_t segmentIndex = qwordIndex >> 6;
            if ((rootBitmap & (1U << segmentIndex)) &&
                (pageSegments[segmentIndex] & (1ULL << (qwordIndex & 63))) &&
                pageQwords[qwordIndex] != 0) {
                return true;
            }
        }
    }

    // Check last partial qword
    size_t endSegmentIndex = endQwordIndex >> 6;
    if ((rootBitmap & (1U << endSegmentIndex)) &&
        (pageSegments[endSegmentIndex] & (1ULL << (endQwordIndex & 63)))) {
        uint8_t lastMask = (1 << (endByteOffset + 1)) - 1;
        if ((pageQwords[endQwordIndex] & lastMask) != 0) {
            return true;
        }
    }

    return false;
}

size_t PageMap::add(uint64_t addr, size_t size) {
    if (size == 1) {
        // Fast path for single address
        size_t qwordIndex = getQwordIndex(addr);
        uint8_t byteOffset = getByteOffsetInQword(addr);

        // Check if the bit was not previously set
        if ((pageQwords[qwordIndex] & (1 << byteOffset)) == 0) {

            size_t segmentIndex = qwordIndex >> 6;

            pageSegments[segmentIndex] |= 1ULL << (qwordIndex & 63);
            rootBitmap |= 1U << segmentIndex;
            pageQwords[qwordIndex] |= 1 << byteOffset;
            addressCount++;
            return 1;
        }
        return 0;
    }

    // Handle range of addresses
    uint64_t endAddr = addr + size - 1;
    size_t startQwordIndex = getQwordIndex(addr);
    size_t endQwordIndex = getQwordIndex(endAddr);
    uint8_t startByteOffset = getByteOffsetInQword(addr);
    uint8_t endByteOffset = getByteOffsetInQword(endAddr);
    size_t segmentIndex = startQwordIndex >> 6;

    // If range is within single qword
    if (startQwordIndex == endQwordIndex) {
        uint8_t mask = ((1 << (endByteOffset + 1)) - 1) & ~((1 << startByteOffset) - 1);
        uint8_t newBits = mask & ~pageQwords[startQwordIndex];
        size_t added = __builtin_popcount(newBits);
        pageQwords[startQwordIndex] |= mask;
        pageSegments[segmentIndex] |= 1ULL << (startQwordIndex & 63);
        rootBitmap |= 1U << segmentIndex;
        addressCount += added;
        return added;
    }

    size_t addedCount = 0;

    // Handle first partial qword
    uint8_t firstMask = ~((1 << startByteOffset) - 1);
    uint8_t firstNewBits = firstMask & ~pageQwords[startQwordIndex];
    addedCount += __builtin_popcount(firstNewBits);
    pageQwords[startQwordIndex] |= firstMask;
    pageSegments[segmentIndex] |= 1ULL << (startQwordIndex & 63);
    rootBitmap |= 1U << segmentIndex;

    // Handle middle complete qwords
    for (size_t qwordIndex = startQwordIndex + 1; qwordIndex < endQwordIndex; qwordIndex++) {
        addedCount += 8 - __builtin_popcount(pageQwords[qwordIndex]);
        pageQwords[qwordIndex] = 0xFF;
        segmentIndex = qwordIndex >> 6;
        pageSegments[segmentIndex] |= 1ULL << (qwordIndex & 63);
        rootBitmap |= 1U << (segmentIndex);
    }

    // Handle last partial qword
    uint8_t lastMask = (1 << (endByteOffset + 1)) - 1;
    uint8_t lastNewBits = lastMask & ~pageQwords[endQwordIndex];
    addedCount += __builtin_popcount(lastNewBits);
    pageQwords[endQwordIndex] |= lastMask;
    segmentIndex = endQwordIndex >> 6;
    pageSegments[segmentIndex] |= 1ULL << (endQwordIndex & 63);
    rootBitmap |= 1U << segmentIndex;

    addressCount += addedCount;
    return addedCount;
}

size_t PageMap::remove(uint64_t addr, size_t size) {
    if (size == 1) {
        // Fast path for single address
        size_t qwordIndex = getQwordIndex(addr);
        size_t segmentIndex = qwordIndex >> 6;

        // Early exit if segment not marked
        if (!(rootBitmap & (1U << segmentIndex))) {
            return 0;
        }
        
        // Second check if segment bit is set
        if ((pageSegments[segmentIndex] & (1ULL << (qwordIndex & 63))) == 0) {
            return 0;  // Early return if segment bit not set
        }
        
        uint8_t byteOffset = getByteOffsetInQword(addr);
        if (pageQwords[qwordIndex] & (1 << byteOffset)) {
            pageQwords[qwordIndex] &= ~(1 << byteOffset);
            addressCount--;
            

            if (pageQwords[qwordIndex] == 0) {
                pageSegments[segmentIndex] &= ~(1ULL << (qwordIndex & 63));
                if (pageSegments[segmentIndex] == 0) {
                    rootBitmap &= ~(1U << segmentIndex);
                }
            }
            return 1;
        }
        return 0;
    }

    uint64_t endAddr = addr + size - 1;
    size_t startQwordIndex = getQwordIndex(addr);
    size_t endQwordIndex = getQwordIndex(endAddr);
    uint8_t startByteOffset = getByteOffsetInQword(addr);
    uint8_t endByteOffset = getByteOffsetInQword(endAddr);
    size_t segmentIndex = startQwordIndex >> 6;

    // If range is within single qword
    if (startQwordIndex == endQwordIndex) {
        // Check segment bit first
        if ((pageSegments[startQwordIndex >> 6] & (1ULL << (startQwordIndex & 63))) == 0) {
            return 0;
        }
        
        uint8_t mask = ((1 << (endByteOffset + 1)) - 1) & ~((1 << startByteOffset) - 1);
        uint8_t removedBits = pageQwords[startQwordIndex] & mask;
        size_t removed = __builtin_popcount(removedBits);
        pageQwords[startQwordIndex] &= ~mask;
        
        if (pageQwords[startQwordIndex] == 0) {
            pageSegments[segmentIndex] &= ~(1ULL << (startQwordIndex & 63));

            if (pageSegments[segmentIndex] == 0) {
                rootBitmap &= ~(1U << segmentIndex);
            }
        }

        addressCount -= removed;
        return removed;
    }

    size_t removedCount = 0;

    // Handle first partial qword if its segment bit is set
    if (pageSegments[segmentIndex] & (1ULL << (startQwordIndex & 63))) {
        uint8_t firstRemovedBits = pageQwords[startQwordIndex] & ~((1 << startByteOffset) - 1);
        removedCount += __builtin_popcount(firstRemovedBits);
        pageQwords[startQwordIndex] &= ((1 << startByteOffset) - 1);
        if (pageQwords[startQwordIndex] == 0) {
            pageSegments[segmentIndex] &= ~(1ULL << (startQwordIndex & 63));
            if (pageSegments[segmentIndex] == 0) {
                rootBitmap &= ~(1U << segmentIndex);
            }
        }
    }

    // Handle middle complete qwords
    for (size_t qwordIndex = startQwordIndex + 1; qwordIndex < endQwordIndex; qwordIndex++) {
        segmentIndex = qwordIndex >> 6;
        // Only process if segment bit is set
        if (pageSegments[segmentIndex] & (1ULL << (qwordIndex & 63))) {
            removedCount += __builtin_popcount(pageQwords[qwordIndex]);
            pageQwords[qwordIndex] = 0;
            pageSegments[segmentIndex] &= ~(1ULL << (qwordIndex & 63));
            rootBitmap &= ~(1U << segmentIndex);
        }
    }

    segmentIndex = endQwordIndex >> 6;

    // Handle last partial qword if its segment bit is set
    if (endQwordIndex != startQwordIndex && 
        (pageSegments[segmentIndex] & (1ULL << (endQwordIndex & 63)))) {
        uint8_t lastRemovedBits = pageQwords[endQwordIndex] & ((1 << (endByteOffset + 1)) - 1);
        removedCount += __builtin_popcount(lastRemovedBits);
        pageQwords[endQwordIndex] &= ~((1 << (endByteOffset + 1)) - 1);
        if (pageQwords[endQwordIndex] == 0) {
            pageSegments[segmentIndex] &= ~(1ULL << (endQwordIndex & 63));
            if (pageSegments[segmentIndex] == 0) {
                rootBitmap &= ~(1U << segmentIndex);
            }
        }
    }

    addressCount -= removedCount;
    return removedCount;
}

size_t PageMap::removeAddressesLowerThan(uint64_t addr) {
    size_t removedCount = 0;
    size_t targetQwordIndex = getQwordIndex(addr);
    uint8_t targetByteOffset = getByteOffsetInQword(addr);
    
    size_t lastSegmentIndex = targetQwordIndex >> 6;

    // Process complete segments before the target qword
    for (size_t segmentIndex = 0; segmentIndex < lastSegmentIndex; segmentIndex++) {
        // Skip segment if not set in rootBitmap
        if (!(rootBitmap & (1U << segmentIndex))) continue;

        uint64_t segmentBits = pageSegments[segmentIndex];
        // Process only qwords that have their segment bit set
        for (size_t i = 0; i < 64; i++) {
            if (segmentBits & (1ULL << i)) {
                size_t qwordIndex = segmentIndex * 64 + i;
                removedCount += __builtin_popcount(pageQwords[qwordIndex]);
                pageQwords[qwordIndex] = 0;
            }
        }
        pageSegments[segmentIndex] = 0;
        rootBitmap &= ~(1U << segmentIndex);
    }
    
    // Process the segment containing the target qword
    uint64_t qwordMask = (1ULL << (targetQwordIndex & 63)) - 1;
    uint64_t segmentBits = pageSegments[lastSegmentIndex] & qwordMask;
    
    // Process only set qwords before target qword in this segment
    while (segmentBits != 0) {
        uint8_t bitPos = __builtin_ctzll(segmentBits);
        size_t qwordIndex = lastSegmentIndex * 64 + bitPos;
        if (qwordIndex >= targetQwordIndex) break;
        
        removedCount += __builtin_popcount(pageQwords[qwordIndex]);
        pageQwords[qwordIndex] = 0;
        segmentBits &= ~(1ULL << bitPos);
    }
    
    // Update pageSegments and rootBitmap for the last segment
    uint64_t oldSegBits = pageSegments[lastSegmentIndex];
    pageSegments[lastSegmentIndex] &= ~qwordMask;
    
    // If segment became empty, update rootBitmap
    if (pageSegments[lastSegmentIndex] == 0 && oldSegBits != 0) {
        rootBitmap &= ~(1U << lastSegmentIndex);
    }
    
    // Process target qword only if its segment bit was set
    if (oldSegBits & (1ULL << (targetQwordIndex & 63))) {
        uint8_t byteMask = (1 << targetByteOffset) - 1;
        uint8_t removedBits = pageQwords[targetQwordIndex] & byteMask;
        removedCount += __builtin_popcount(removedBits);
        pageQwords[targetQwordIndex] &= ~byteMask;
        
        if (pageQwords[targetQwordIndex] == 0) {
            pageSegments[lastSegmentIndex] &= ~(1ULL << (targetQwordIndex & 63));
            // If this made the segment empty, update rootBitmap
            if (pageSegments[lastSegmentIndex] == 0) {
                rootBitmap &= ~(1U << lastSegmentIndex);
            }
        }
    }

    addressCount -= removedCount;
    return removedCount;
}

std::vector<std::pair<uint64_t, size_t>> PageMap::findGroups() const {
    std::vector<std::pair<uint64_t, size_t>> groups;
    uint64_t groupStartAddr = 0;
    bool inGroup = false;
    size_t lastAddr = 0;

    // Skip segments not set in rootBitmap
    uint32_t remainingSegments = rootBitmap;
    while (remainingSegments) {
        // Find next used segment
        size_t segmentIndex = __builtin_ctz(remainingSegments);
        uint64_t segmentBits = pageSegments[segmentIndex];
        
        // Skip if segment is empty (shouldn't happen if rootBitmap is maintained correctly)
        if (segmentBits == 0) {
            remainingSegments &= ~(1U << segmentIndex);
            continue;
        }

        // Process each set qword in this segment
        uint64_t remainingQwords = segmentBits;
        while (remainingQwords) {
            // Find next used qword
            size_t qwordOffset = __builtin_ctzll(remainingQwords);
            size_t qwordIndex = segmentIndex * 64 + qwordOffset;
            
            if (qwordIndex >= NUM_QWORDS) break;

            uint8_t qwordBits = pageQwords[qwordIndex];
            if (qwordBits) {
                uint64_t baseAddr = qwordIndex * QWORD_SIZE;
                
                // Process each set bit in the qword
                while (qwordBits) {
                    uint8_t bitPos = __builtin_ffs(qwordBits) - 1;
                    uint64_t currentAddr = baseAddr + bitPos;
                    
                    if (!inGroup) {
                        groupStartAddr = currentAddr;
                        inGroup = true;
                    } else if (currentAddr != lastAddr + 1) {
                        // End current group and start new one
                        groups.emplace_back(groupStartAddr, lastAddr - groupStartAddr + 1);
                        groupStartAddr = currentAddr;
                    }
                    lastAddr = currentAddr;
                    
                    // Clear processed bit
                    qwordBits &= ~(1 << bitPos);
                }
            }
            
            // Clear processed qword
            remainingQwords &= ~(1ULL << qwordOffset);
        }
        
        // Clear processed segment
        remainingSegments &= ~(1U << segmentIndex);
    }

    // Add final group if we ended while in a group
    if (inGroup) {
        groups.emplace_back(groupStartAddr, lastAddr - groupStartAddr + 1);
    }
    
    return groups;
}


std::vector<uint8_t> PageMap::getUsedByteOffsetsInQword(size_t qwordIndex) const {
    std::vector<uint8_t> usedOffsets;
    
    // First check if segment bit is set
    if ((pageSegments[qwordIndex >> 6] & (1ULL << (qwordIndex & 63))) == 0) {
        return usedOffsets;  // Return empty vector if qword is not in use
    }
    
    // Only examine qword bits if segment bit was set
    uint8_t byteBits = pageQwords[qwordIndex];
    while (byteBits) {
        uint8_t bitPos = __builtin_ffs(byteBits) - 1;
        usedOffsets.push_back(bitPos);
        byteBits &= ~(1 << bitPos);
    }
    
    return usedOffsets;
}

uint64_t PageMap::getLowestAddress() const {
    if (addressCount == 0) {
        return UINT64_MAX;
    }

    // Find first used segment
    for (size_t i = 0; i < NUM_SEGMENTS; i++) {
        if (pageSegments[i] != 0) {
            // Find first used qword in this segment
            size_t qwordOffset = __builtin_ffsll(pageSegments[i]) - 1;
            size_t qwordIndex = (i * 64) + qwordOffset;
            
            // Find first used bit in this qword
            uint8_t firstByte = pageQwords[qwordIndex];
            uint64_t bitOffset = __builtin_ffs(firstByte) - 1;
            
            return (qwordIndex * 8) + bitOffset + pageBaseAddr;
        }
    }
    return UINT64_MAX;  // Should never reach here if addressCount > 0
}

uint64_t PageMap::getHighestAddress() const {
    if (addressCount == 0) {
        return UINT64_MAX;
    }

    // Find last used segment
    for (size_t i = NUM_SEGMENTS; i-- > 0;) {
        if (pageSegments[i] != 0) {
            // Find last used qword in this segment
            size_t qwordOffset = 63 - __builtin_clzll(pageSegments[i]);
            size_t qwordIndex = (i * 64) + qwordOffset;
            
            // Find last used bit in this qword
            uint8_t lastByte = pageQwords[qwordIndex];
            //uint64_t bitOffset = 7 - __builtin_clz(lastByte);
            uint32_t bitOffset = 7 - __builtin_clz((unsigned int)(lastByte & 0xFF) << 24);
            
            return ((qwordIndex * 8) + bitOffset) + pageBaseAddr;
        }
    }
    return UINT64_MAX;  // Should never reach here if addressCount > 0
}

void PageMap::findOptimalWatchpoint(uint32_t bits, uint32_t startPos, uint32_t width, Watchpoint& wp) const {
    //LOG("[findOptimalWatchpoint]\n");
    // Base case: single bit
    if (width == 1) {
        wp.address = pageBaseAddr + (startPos * 512);
        wp.alignedAddress = wp.address;  // Already aligned to segment boundary
        wp.size = 512;
        wp.wcr = wcr(wp.size);
        return;
    }

    uint32_t halfWidth = width >> 1;
    //LOG("halfWidth: %d\n",halfWidth);
    uint32_t leftMask = ((1U << halfWidth) - 1) << startPos;
    uint32_t rightMask = ((1U << halfWidth) - 1) << (startPos + halfWidth);

    uint32_t leftBits = bits & leftMask;
    uint32_t rightBits = bits & rightMask;


    // If both halves have bits, use current level
    if (leftBits && rightBits) {
        wp.address = pageBaseAddr + (startPos * 512);
        wp.alignedAddress = wp.address;
        wp.size = width * 512;
        wp.wcr = wcr(wp.size);
        return;
    }

    // Otherwise, optimize to the non-empty half
    uint32_t usedBits = leftBits ? leftBits : rightBits;
    uint32_t newStartPos = leftBits ? startPos : (startPos + halfWidth);
    findOptimalWatchpoint(usedBits, newStartPos, halfWidth, wp);
}

void PageMap::refineSplit(SplitWork& work) const {
    work.reductionRate = 0;
    while (work.width > 1) {
        uint32_t halfWidth = work.width >> 1;
        uint32_t leftMask = ((1U << halfWidth) - 1) << work.startPos;
        uint32_t rightMask = ((1U << halfWidth) - 1) << (work.startPos + halfWidth);

        uint32_t leftBits = work.bits & leftMask;
        uint32_t rightBits = work.bits & rightMask;
        
        if (!leftBits || !rightBits) {
            work.bits = leftBits ? leftBits : rightBits;
            work.startPos = leftBits ? work.startPos : (work.startPos + halfWidth);
            work.width = halfWidth;
            work.reductionRate++;
            continue;
        }

        //if (work.reductionRate)
        work.size = work.width * 512;
        //LOG("work.size: %d\n", work.size);
        return;  // Found valid split point
    }
    work.size = work.width * 512;
    return;
}

// Helper function for refining segment splits
void PageMap::refineSegmentSplit(SegmentWork& work) const {
    work.reductionRate = 0;
    while (work.width > 1) {
        uint32_t halfWidth = work.width >> 1;
        uint64_t leftMask = ((1ULL << halfWidth) - 1) << work.startPos;
        uint64_t rightMask = ((1ULL << halfWidth) - 1) << (work.startPos + halfWidth);

        uint64_t leftBits = work.bits & leftMask;
        uint64_t rightBits = work.bits & rightMask;
        
        if (!leftBits || !rightBits) {
            work.bits = leftBits ? leftBits : rightBits;
            work.startPos = leftBits ? work.startPos : (work.startPos + halfWidth);
            work.width = halfWidth;
            work.reductionRate++;
            continue;
        }
        
        work.size = work.width * 8; // Each bit represents 8 bytes
        return;
    }
    work.size = work.width * 8;
}

// Handling segment-layer splits
size_t PageMap::findSegmentLevelSplits(uint32_t segmentIndex, uint64_t segmentBits,
                                      Watchpoint (&splits)[4], size_t currentSplit) const {
    if (!segmentBits || currentSplit >= 4) return currentSplit;

    SegmentWork work = {segmentBits, 0, 64, 0, 0};
    refineSegmentSplit(work);

    if (work.width == 1 || currentSplit >= 3) {
        // Create single watchpoint for this piece
        uint64_t baseAddr = pageBaseAddr + (segmentIndex * 512);
        splits[currentSplit].address = baseAddr + (work.startPos * 8);
        splits[currentSplit].alignedAddress = splits[currentSplit].address;
        splits[currentSplit].size = work.size;
        
        if (work.size <= 8) {
            size_t qwordIndex = getQwordIndex(splits[currentSplit].address);
            splits[currentSplit].wcr = wcr(work.size, pageQwords[qwordIndex]);
        } else {
            splits[currentSplit].wcr = wcr(work.size);
        }
        
        return currentSplit + 1;
    }

    // Split the segment work
    uint32_t halfWidth = work.width >> 1;
    uint64_t leftMask = ((1ULL << halfWidth) - 1) << work.startPos;
    uint64_t rightMask = ((1ULL << halfWidth) - 1) << (work.startPos + halfWidth);
    
    uint64_t leftBits = work.bits & leftMask;
    uint64_t rightBits = work.bits & rightMask;

    SegmentWork leftWork = {leftBits, work.startPos, halfWidth, 0, 0};
    SegmentWork rightWork = {rightBits, work.startPos + halfWidth, halfWidth, 0, 0};
    
    refineSegmentSplit(leftWork);
    refineSegmentSplit(rightWork);

    //LOG("leftWork.reductionRate: %zu\n", leftWork.reductionRate);
    //LOG("rightWork.reductionRate: %zu\n",rightWork.reductionRate);

    if (leftWork.reductionRate > 0 || rightWork.reductionRate > 0) {
        // Process left side
        size_t newSplitCount = findSegmentLevelSplits(segmentIndex, leftBits, splits, currentSplit);
        // Process right side
        return findSegmentLevelSplits(segmentIndex, rightBits, splits, newSplitCount);
    } else {
        // Use parent
        uint64_t baseAddr = pageBaseAddr + (segmentIndex * 512);
        splits[currentSplit].address = baseAddr + (work.startPos * 8);
        splits[currentSplit].alignedAddress = splits[currentSplit].address;
        splits[currentSplit].size = work.size;
        splits[currentSplit].wcr = wcr(work.size);
        return currentSplit + 1;
    }
}

// Main function for finding watchpoint splits
size_t PageMap::findWatchpointSplits(uint32_t bits, uint32_t startPos, uint32_t width,
                                    Watchpoint (&splits)[4], size_t currentSplit) const {
    if (!bits) return currentSplit;

    // Start with initial work at rootBitmap layer
    SplitWork work = {bits, startPos, width, 0, 0};
    refineSplit(work);

    // If we reached width=1 at rootBitmap layer, switch to segment layer
    if (work.width == 1) {
        uint32_t segmentIndex = work.startPos;
        uint64_t segmentBits = pageSegments[segmentIndex];
        return findSegmentLevelSplits(segmentIndex, segmentBits, splits, currentSplit);
    }

    // For larger widths, continue with the original splitting logic
    if (currentSplit >= 3) {
        splits[currentSplit].address = pageBaseAddr + work.startPos * 512;
        splits[currentSplit].alignedAddress = splits[currentSplit].address;
        splits[currentSplit].size = work.size;
        splits[currentSplit].wcr = wcr(work.size);
        return currentSplit + 1;
    }

    uint32_t halfWidth = work.width >> 1;
    
    // If halfWidth is 1, we need to process each half at segment layer
    if (halfWidth == 1) {
        uint32_t leftSegIndex = work.startPos;
        uint32_t rightSegIndex = work.startPos + 1;
        
        // Process left segment if it has bits
        if (bits & (1U << leftSegIndex)) {
            uint64_t leftSegBits = pageSegments[leftSegIndex];
            currentSplit = findSegmentLevelSplits(leftSegIndex, leftSegBits, splits, currentSplit);
        }
        
        // Process right segment if it has bits
        if (bits & (1U << rightSegIndex)) {
            uint64_t rightSegBits = pageSegments[rightSegIndex];
            currentSplit = findSegmentLevelSplits(rightSegIndex, rightSegBits, splits, currentSplit);
        }
        
        return currentSplit;
    }

    // For larger halfWidths, continue with normal splitting
    uint32_t leftMask = ((1U << halfWidth) - 1) << work.startPos;
    uint32_t rightMask = ((1U << halfWidth) - 1) << (work.startPos + halfWidth);
    
    uint32_t leftBits = work.bits & leftMask;
    uint32_t rightBits = work.bits & rightMask;

    SplitWork leftWork = {leftBits, work.startPos, halfWidth, 0, 0};
    SplitWork rightWork = {rightBits, work.startPos + halfWidth, halfWidth, 0, 0};
    
    refineSplit(leftWork);
    refineSplit(rightWork);

    if (leftWork.reductionRate > 0 || rightWork.reductionRate > 0) {
        // Process left side (still at rootBitmap layer)
        size_t newSplitCount = findWatchpointSplits(leftBits, leftWork.startPos, 
                                                   leftWork.width, splits, currentSplit);
        // Process right side (still at rootBitmap layer)
        return findWatchpointSplits(rightBits, rightWork.startPos, 
                                   rightWork.width, splits, newSplitCount);
    } else {
        splits[currentSplit].address = pageBaseAddr + work.startPos * 512;
        splits[currentSplit].alignedAddress = splits[currentSplit].address;
        splits[currentSplit].size = work.size;
        splits[currentSplit].wcr = wcr(work.size);
        return currentSplit + 1;
    }
}

void PageMap::computeWatchpoint(Watchpoint& watchpoint) const {
    if (addressCount == 0) {
        FATAL("[computeWatchpoint] addressCount == 0");
        return;
    }
    findOptimalWatchpoint(rootBitmap, 0, 32, watchpoint);
}

size_t PageMap::computeWatchpointSplits(Watchpoint& mainWatchpoint, 
                              Watchpoint (&splits)[4],
                              uint64_t recentlyAccessedAddr) const {
    if (addressCount == 0) return 0;

    computeWatchpoint(mainWatchpoint);
    if (mainWatchpoint.size <= 8) return 0;

    return findWatchpointSplits(rootBitmap, 0, 32, splits, 0);
}

void PageMap::printAddresses() const {
    auto groups = findGroups();
    
    if (groups.empty()) {
        LOG("No addresses tracked in this page map\n");
        return;
    }
    
    for (size_t groupNum = 0; groupNum < groups.size(); groupNum++) {
        const auto& group = groups[groupNum];
        uint64_t startAddr = group.first;
        size_t size = group.second;
        
        LOG("Group %zu: %016llx - %016llx (size: %zu)\n", 
            groupNum + 1, startAddr, startAddr + size - 1, size);
        
        // Print individual addresses in this group
        LOG("  Addresses in group %zu:\n", groupNum + 1);
        for (uint64_t addr = startAddr; addr < startAddr + size; addr++) {
            LOG("    %016llx\n", addr);
        }
        LOG("\n");  // Add spacing between groups
    }
}

void PageMap::printStats() const {
    size_t usedQwords = 0;
    size_t usedBytes = 0;
    
    for (size_t i = 0; i < NUM_SEGMENTS; i++) {
        usedQwords += __builtin_popcountll(pageSegments[i]);
    }
    
    for (size_t i = 0; i < NUM_QWORDS; i++) {
        usedBytes += __builtin_popcount(pageQwords[i]);
    }
    
    LOG("Total tracked addresses: %zu\n", addressCount);
    LOG("Used qwords: %zu/%zu (%.2f%%)\n", 
           usedQwords, NUM_QWORDS, 
           (100.0 * usedQwords) / NUM_QWORDS);
    LOG("Used byte positions: %zu/%zu (%.2f%%)\n", 
           usedBytes, NUM_QWORDS * 8,
           (100.0 * usedBytes) / (NUM_QWORDS * 8));
}
