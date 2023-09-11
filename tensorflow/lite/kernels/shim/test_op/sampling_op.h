//[TODO:kuangzy]
/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_SAMPLING_OP_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_SAMPLING_OP_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"

namespace tflite {
namespace shim {

// A simple operation for demonstration and testing purposes.
// See the kDoc member for documentation.

template <Runtime Rt>
class FarthestPointSample : public OpKernelShim<FarthestPointSample, Rt> {
 protected:
  enum Attrs { kAttr0 = 0 };
  enum Inputs { kInput0 = 0 };
  enum Outputs { kOutput0 = 0};
  int64_t npoint;
  std::string output2_suffix_;
  int64_t n_;
  static constexpr int kOutput0Size = 5;
  static constexpr char kAttrName[] = "npoint";

 public:
  using typename OpKernelShim<FarthestPointSample, Rt>::InitContext;
  using typename OpKernelShim<FarthestPointSample, Rt>::InvokeContext;
  using typename OpKernelShim<FarthestPointSample, Rt>::ShapeInferenceContext;

  FarthestPointSample() = default;
  static constexpr char kOpName[] = "FarthestPointSample";
  static constexpr char kDoc[] = R"doc(
Description:
  Simple example op for testing and demonstration purposes.

Attrs
  output1_size: int - the size of the second output
  output2_suffix: string - the string value to be appended to the end of out2
  N: int - the number of tensors for the second input and last output
Inputs
  in0: str, shape=[] - A scalar input
  in1: int64, list<shape=?> - A list of tensors as input
Outputs
  out0: int, shape=[5] - first output
  out1: float, shape=[?] - second output
  out2: string, shape=[?] - third output
  out3: int64, list<shape=?> - fourth output that is in1 but incremented.
)doc";

  static const char* OpName() { return kOpName; }
  static const char* Doc() { return kDoc; }

  // Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs() {
    printf("[debug][shim][farthestpointsample][Attrs] ------------------1\n");
    return {absl::StrCat(kAttrName, ": int")};
  }
  // Input tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs() {
    printf("[debug][shim][farthestpointsample][Inputs] ------------------1\n");
    return {"inp: float32"};
  }
  // Output tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs() {
    printf("[debug][shim][farthestpointsample][Outputs] ------------------1\n");
    return {"out: int32"};
  }

  // Initializes the op
  absl::Status Init(InitContext* ctx) {
    SH_RETURN_IF_ERROR(ctx->GetAttr(kAttrName, &npoint));
    printf("[debug][shim][farthestpointsample][Init] ------------------npoint %d\n", npoint);
    //if (npoint < 1) {
    //  return absl::InternalError(absl::StrCat(kAttrName, " should be >= 1"));
    //}
    
    return absl::OkStatus();
  }

  // Runs the operation
  absl::Status Invoke(InvokeContext* ctx) {
    printf("[debug][shim][farthestpointsample][Invoke] ------------------1\n");

    return absl::OkStatus();
  }

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* ctx) {
    printf("[debug][shim][farthestpointsample][ShapeInference] ------------------1\n");

    return absl::OkStatus();
  }
};


}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_SAMPLING_OP_H_
