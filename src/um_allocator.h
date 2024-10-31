#pragma once

#include <NvInfer.h>
#include <NvInferRuntimeBase.h>
#include <triton/backend/backend_common.h>

namespace colsys {
class UMAllocator : public nvinfer1::IGpuAllocator {
public:
  void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) noexcept override {
    void* ptr = nullptr;
    cudaError_t status = cudaMallocManaged(&ptr, size);
    LOG_IF_CUDA_ERROR(status, "Failed to allocate memory");
    if (status != cudaSuccess) {
      return nullptr;
    }
    return ptr;
  }

  void free(void* memory) noexcept override {
    cudaFree(memory);
  }
}; 
} // namespace colsys