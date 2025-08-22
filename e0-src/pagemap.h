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

#ifndef PAGEMAP_H
#define PAGEMAP_H

#include <cstdint>
#include <vector>
#include <utility>
#include "watchpoint.h"

/** Break only in privileged or user mode */
#define S_USER ((uint32_t)(2u << 1))

#define BCR_ENABLE ((uint32_t)(1u))
#define WCR_ENABLE ((uint32_t)(1u))

// Watchpoint load/store
#define WCR_LOAD ((uint32_t)(1u << 3))
#define WCR_STORE ((uint32_t)(1u << 4))

// Enable breakpoint and watchpoint.
#define MDE_ENABLE ((uint32_t)(1u << 15))

// Single instruction step
#define SS_ENABLE ((uint32_t)(1u))


// rootBitmap-level work
struct SplitWork {
    uint32_t bits;
    uint32_t startPos;
    uint32_t width;
    size_t size;
    size_t reductionRate;  // Renamed from iterations
};

// Segment-level work
struct SegmentWork {
    uint64_t bits;
    uint32_t startPos;
    uint32_t width;
    size_t size;
    size_t reductionRate;
};

class PageMap {
private:
    /** Size of the page in bytes */
    static constexpr size_t PAGE_SIZE = 16384;           // 16K
    
    /** Size of a qword in bytes */
    static constexpr size_t QWORD_SIZE = 8;              // 8 bytes per qword
    
    /** Number of qwords in the page */
    static constexpr size_t NUM_QWORDS = PAGE_SIZE / QWORD_SIZE;  // 2048 qwords
    
    /** Number of segments needed to cover all qwords (64 qwords per segment) */
    static constexpr size_t NUM_SEGMENTS = NUM_QWORDS / 64;       // 32 segments
    
    /** Total number of addresses currently being tracked */
    size_t addressCount;

    /** Root bitmap - one bit per segment */
    uint32_t rootBitmap;

    /** Array of segments, where each segment is a uint64_t bitmap */
    uint64_t pageSegments[NUM_SEGMENTS];

    /** Array of qwords, where each qword is represented by a byte */
    uint8_t pageQwords[NUM_QWORDS];

    /**
     * Calculate which qword contains the given address
     * @param addr The address to check
     * @return Index into pageQwords array (0-2047)
     */
    static inline size_t getQwordIndex(uint64_t addr);

    /**
     * Calculate which byte position within a qword contains the given address
     * @param addr The address to check
     * @return Byte offset within the qword (0-7)
     */
    static inline uint8_t getByteOffsetInQword(uint64_t addr);

    /**
     * Compute Watchpoint Control Register (WCR) value
     * Configures hardware watchpoint settings based on the size and tracked bytes
     * 
     * For small watchpoints (≤ 8 bytes):
     * - Uses provided qword_content which contains the byte selection mask
     * 
     * For large watchpoints (> 8 bytes):
     * - Uses full byte_address_select (0xff)
     * - Calculates appropriate size mask
     * 
     * @param size Watchpoint size (must be power of 2)
     * @param qword_content Content of the relevant qword for sizes ≤ 8, ignored for larger sizes
     * @return Configured WCR value with appropriate masks and control bits
     */
    static uint64_t wcr(uint64_t size, uint8_t qword_content = 0);

public:
    /** Page base address */
    uint64_t pageBaseAddr;

    /**
     * Initialize a new PageMap with all bits cleared
     */
    PageMap();

    /**
     * Get the total number of addresses being tracked
     * @return Number of addresses currently being tracked
     */
    size_t size() const;

    /**
     * Check if the page map is empty
     * @return true if no addresses are being tracked, false otherwise
     */
    bool empty() const;

    /**
     * Check if an address or range of addresses is being tracked
     * @param addr The starting address to check
     * @param size Number of bytes to check (defaults to 1)
     * @return true if all addresses in the range are being tracked, false otherwise
     */
    bool hasAddress(uint64_t addr, size_t size = 1) const;

    /**
     * Add a single address or a range of addresses to be tracked
     * @param addr The starting address (must be within the 16K page)
     * @param size The number of bytes to track (defaults to 1)
     */
    size_t add(uint64_t addr, size_t size = 1);

    /**
     * Remove an address or range of addresses from tracking
     * @param addr The starting address to remove
     * @param size Number of bytes to remove (defaults to 1)
     */
    size_t remove(uint64_t addr, size_t size = 1);

    /**
     * Remove all addresses lower than the given address
     * @param addr The threshold address
     * @return Number of addresses that were removed
     */
    size_t removeAddressesLowerThan(uint64_t addr);

    /**
     * Find continuous groups of addresses
     * @return Vector of pairs containing {start_address, length} for each continuous group
     */
    std::vector<std::pair<uint64_t, size_t>> findGroups() const;

    /**
     * Get all used byte offsets within a specific qword
     * @param qwordIndex Index of the qword to check (0-2047)
     * @return Vector of byte offsets (0-7) that are being tracked in this qword
     */
    std::vector<uint8_t> getUsedByteOffsetsInQword(size_t qwordIndex) const;
    
    /**
     * Get the lowest address being tracked
     * @return The lowest address being tracked, or UINT64_MAX if no addresses are tracked
     */
    uint64_t getLowestAddress() const;

    /**
     * Get the highest address being tracked
     * @return The highest address being tracked, or 0 if no addresses are tracked
     */
    uint64_t getHighestAddress() const;


    /**
     * Find the optimal watchpoint configuration for a given bitmap range
     * Recursively analyzes the bitmap to find the smallest possible watchpoint
     * that covers all active bits in the range.
     * 
     * @param bits Bitmap representing active segments in the range
     * @param startPos Starting position in the bitmap
     * @param width Width of the bitmap range to analyze
     * @param[out] wp Reference to watchpoint structure to be configured
     */
    void findOptimalWatchpoint(uint32_t bits, uint32_t startPos, 
        uint32_t width, Watchpoint& wp) const;

    /**
     * Find optimal split configurations for watchpoints
     * Recursively splits the bitmap range to find up to 4 watchpoints that
     * efficiently cover all active bits while minimizing total coverage area.
     * 
     * The function attempts to find optimal split points by:
     * - Analyzing bit patterns in each potential split
     * - Calculating reduction rates for different split configurations
     * - Choosing splits that minimize overall coverage
     * 
     * @param bits Bitmap representing active segments
     * @param startPos Starting position in the bitmap
     * @param width Width of the range to analyze
     * @param[out] splits Array of watchpoints to be configured with split coverage
     * @param currentSplit Current number of splits already configured
     * @return Total number of splits created (0-4)
     */
    size_t findWatchpointSplits(uint32_t bits, uint32_t startPos, uint32_t width,
        Watchpoint (&splits)[4], size_t currentSplit) const;

    /**
     * Refine a split configuration to optimize coverage
     * Analyzes a potential split configuration and attempts to reduce its
     * coverage area by finding better split points or combining splits
     * when beneficial.
     * 
     * @param[in,out] work Split work structure containing:
     *                     - bits: Active bits in the range
     *                     - startPos: Starting position
     *                     - width: Width of range
     *                     - size: Computed size of coverage
     *                     - reductionRate: Number of successful reductions
     */
    void refineSplit(SplitWork& work) const;

    void refineSegmentSplit(SegmentWork& work) const;

    size_t findSegmentLevelSplits(uint32_t segmentIndex, uint64_t segmentBits, 
        Watchpoint (&splits)[4], size_t currentSplit) const;


    /**
     * Compute optimal watchpoint configuration for tracked addresses
     * Uses bitmap structure to efficiently determine the smallest possible
     * watchpoint that covers all tracked addresses in the page.
     * 
     * The function analyzes the bitmap hierarchy (root -> segments -> qwords -> bytes)
     * to find the optimal watchpoint size and alignment. It has specialized fast paths for:
     * - Single byte cases
     * - Multiple bytes within same qword (≤ 8 bytes)
     * - Larger ranges requiring full watchpoint coverage
     * 
     * @param[out] watchpoint Reference to watchpoint structure to be populated
     *                        with computed configuration
     */
    void computeWatchpoint(Watchpoint& watchpoint) const;

    /**
     * Compute optimal watchpoint splits configuration for tracked addresses
     * 
     * This function implements a hierarchical splitting approach to find the optimal
     * watchpoint configuration that minimizes the total coverage area. It operates in two modes:
     * 
     * 1. Main watchpoint: Computes a single watchpoint that covers all tracked addresses,
     *    identical to computeWatchpoint()
     * 
     * 2. Split watchpoints: Computes up to 4 watchpoints that together cover all tracked
     *    addresses while minimizing total coverage. The algorithm:
     *    - Starts with the full 16KB page
     *    - Recursively splits ranges based on active qwords in each half
     *    - Uses XOR of lowest/highest addresses to determine optimal coverage size
     *    - Makes split decisions based on maximum coverage reduction
     *    - Handles byte-precise ranges (≤8 bytes) with specific byte selection masks
     * 
     * The splitting process is optimized by:
     * - Using rootBitmap to quickly find active segments
     * - Direct traversal from segments to active qwords
     * - XOR-based calculation of required power-of-2 coverage sizes
     * - Avoiding splits that would cross recentlyAccessedAddr
     * 
     * @param[out] watchpoint Reference to main watchpoint to be configured with full coverage
     * @param[out] watchpointSplits Array of 4 watchpoints to be configured with split coverage
     * @param recentlyAccessedAddr Optional address to avoid covering (defaults to UINT64_MAX)
     * @return The number of watchpoint splits "maximum 4".
     */
    size_t computeWatchpointSplits(Watchpoint& watchpoint, 
                                Watchpoint (&watchpointSplits)[4],
                                uint64_t recentlyAccessedAddr = UINT64_MAX) const;    

    /**
     * Print all groups of tracked addresses in this page map
     * Shows information about each continuous group of addresses and lists
     * all individual addresses within each group.
     */
    void printAddresses() const;

    /**
     * Print usage statistics for the page map
     * Shows number of used qwords and byte positions with percentages
     */
    void printStats() const;
};

#endif // PAGEMAP_H