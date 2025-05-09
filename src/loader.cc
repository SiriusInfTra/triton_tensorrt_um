// Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "loader.h"

#include <NvInferPlugin.h>

#include <memory>
#include <mutex>

#include "triton/backend/backend_common.h"
#include <um_allocator.h>

namespace triton { namespace backend { namespace tensorrt {

TRITONSERVER_Error*
LoadPlan(
    const std::string& plan_path, const int64_t dla_core_id,
    std::shared_ptr<nvinfer1::IRuntime>* runtime,
    std::shared_ptr<nvinfer1::ICudaEngine>* engine,
    TensorRTLogger* tensorrt_logger)
{
  // Create runtime only if it is not provided
  if (*runtime == nullptr) {
    runtime->reset(nvinfer1::createInferRuntime(*tensorrt_logger));
    if (*runtime == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unable to create TensorRT runtime: ") +
           tensorrt_logger->LastErrorMsg())
              .c_str());
    }
    runtime->get()->setGpuAllocator(new colsys::UMAllocator());

    if (ModelState::isVersionCompatible() &&
        !runtime->get()->getEngineHostCodeAllowed()) {
      runtime->get()->setEngineHostCodeAllowed(true);
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("Version compatibility enabled for runtime")).c_str());
    }
  }

  // Report error if 'dla_core_id' >= number of DLA cores
  if (dla_core_id != -1) {
    auto dla_core_count = (*runtime)->getNbDLACores();
    if (dla_core_id < dla_core_count) {
      (*runtime)->setDLACore(dla_core_id);
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to create TensorRT runtime with DLA Core ID: ") +
           std::to_string(dla_core_id) +
           ", available number of DLA cores: " + std::to_string(dla_core_count))
              .c_str());
    }
  }

  std::string model_data_str;
  RETURN_IF_ERROR(ReadTextFile(plan_path, &model_data_str));
  std::vector<char> model_data(model_data_str.begin(), model_data_str.end());

  engine->reset(
      (*runtime)->deserializeCudaEngine(&model_data[0], model_data.size()));
  if (*engine == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to create TensorRT engine: ") +
         tensorrt_logger->LastErrorMsg())
            .c_str());
  }

  return nullptr;
}

}}}  // namespace triton::backend::tensorrt
