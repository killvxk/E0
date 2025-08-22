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

#ifndef WATCHPOINT_H
#define WATCHPOINT_H

#include <cstdint>

/**
 * Structure to hold watchpoint configuration
 * Represents hardware debug watchpoint settings for memory access monitoring
 */
struct Watchpoint {
    uint64_t address = 0;         // Original trigger address
    uint64_t alignedAddress = 0;  // Address aligned to watchpoint size boundary
    uint64_t size = 0;            // Size of the watchpoint (power of 2)
    uint64_t wcr = 0;             // Watchpoint Control Register value
};

#endif // WATCHPOINT_H