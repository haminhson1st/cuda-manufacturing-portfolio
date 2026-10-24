#pragma once

// Named ranges that appear on the Nsight Systems timeline (nsys profile --trace=cuda,nvtx,...).
// Uses the header-only NVTX v3 API shipped with the CUDA toolkit; nothing extra to link. If the
// header is unavailable, the ranges compile to nothing and the programs still build.

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define COMMON_HAVE_NVTX 1
#endif
#endif

namespace common {

// Pushes a range on construction and pops it on destruction: { NvtxRange r("upload"); ... }
class NvtxRange {
public:
    explicit NvtxRange(const char* name) {
#ifdef COMMON_HAVE_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~NvtxRange() {
#ifdef COMMON_HAVE_NVTX
        nvtxRangePop();
#endif
    }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace common
