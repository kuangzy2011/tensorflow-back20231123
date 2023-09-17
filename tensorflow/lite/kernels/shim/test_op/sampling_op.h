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

#define COMPILE_TFLITE_TENSOR 1

#ifdef COMPILE_TFLITE_TENSOR
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/shim/tflite_tensor_view.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/type_to_tflitetype.h"
#endif
namespace tflite {

#ifdef COMPILE_TFLITE_TENSOR
size_t NumTotalFromShape(const std::initializer_list<int>& shape) {
  size_t num_total;
  if (shape.size() > 0)
    num_total = 1;
  else
    num_total = 0;
  for (const int dim : shape) num_total *= dim;
  return num_total;
}

template <typename T>
void ReallocDynamicTensor(const std::initializer_list<int> shape, TfLiteTensor* tensor) {
  TfLiteTensorFree(tensor);
  tensor->allocation_type = kTfLiteDynamic;
  tensor->type = typeToTfLiteType<T>();
    
  // Populate Shape
  TfLiteIntArray* shape_arr = TfLiteIntArrayCreate(shape.size());
  int i = 0;
  const std::size_t num_total = NumTotalFromShape(shape);
  for (const int dim : shape) shape_arr->data[i++] = dim;
  tensor->dims = shape_arr;
  if (tensor->type != kTfLiteString) {
    TfLiteTensorRealloc(num_total * sizeof(T), tensor);
  }
}

extern void farthestpointsamplingLauncher(int b,int n,int m,const float * inp,float * temp,int * out);

#endif

namespace shim {

// A simple operation for demonstration and testing purposes.
// See the kDoc member for documentation.


template <Runtime Rt>
class FarthestPointSample : public OpKernelShim<FarthestPointSample, Rt> {
 protected:
  enum Attrs { kAttr0 = 0 };
  enum Inputs { kInput0 = 0 };
  enum Outputs { kOutput0 = 0};
  int npoint;
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
    return {absl::StrCat(kAttrName, ": int")};
  }
  // Input tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs() {
    return {"inp: float32"};
  }
  // Output tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs() {
    return {"out: int32"};
  }

  // Initializes the op
  absl::Status Init(InitContext* ctx) {
    int64_t tmp_npoint = 0;
    SH_RETURN_IF_ERROR(ctx->GetAttr(kAttrName, &tmp_npoint));
    npoint = tmp_npoint;
    printf("[debug][shim][farthestpointsample][Init] ------------------npoint %d\n", npoint);
    if (npoint < 1) {
      return absl::InternalError(absl::StrCat(kAttrName, " should be > 0"));
    }

    return absl::OkStatus();
  }

  // Runs the operation
  absl::Status Invoke(InvokeContext* ctx) {
    if (1 != ctx->NumInputs()) {
      return absl::InternalError(absl::StrCat("NumInputs:", ctx->NumInputs(), " != 1"));
    }

    if (1 != ctx->NumOutputs()) {
      return absl::InternalError(absl::StrCat("NumOutputs:", ctx->NumOutputs(), " != 1"));
    }
      
    // read input
    SH_ASSIGN_OR_RETURN(const auto input_t, ctx->GetInput(kInput0));
    const auto input_ptr = input_t->template As<float_t, 3>();//dim 3
    //auto input_data = input_t->Data<float>();
    auto input_data = input_t->template Data<float_t>().data();
    /*
    for(int i = 0; i < 6; i++)
    {
        printf(" >[%d] %.6f\n", i, input_data[i]);
    }
    */
    const float* inp = input_data;

        
    //input shape
    Shape input_shape(input_t->Shape());
    //SH_ASSIGN_OR_RETURN(const auto input_shape, ctx->GetInputShape(kInput0));
    if(input_shape.Rank() != 3 || input_shape.Dim(2) != 3) {
        return absl::InternalError(absl::StrCat("FarthestPointSample expects (batch_size,num_points,3) inp shape"));
    }
    printf("[debug][shim][farthestpointsample][Invoke] ------------------npoint %d, input shape(%d, %d, %d)\n", npoint, input_shape.Dim(0), input_shape.Dim(1), input_shape.Dim(2));

    int m = npoint;
    int b = input_shape.Dim(0);
    int n = input_shape.Dim(1);

    // output0 whose size is static
    SH_ASSIGN_OR_RETURN(auto output_t, ctx->GetOutput(kOutput0, Shape({b, m})));
    auto output_ptr = output_t->template As<int32_t, 2>();
    auto out = output_t->template Data<int32_t>().data();

#ifdef COMPILE_TFLITE_TENSOR
#if 0
    ::tflite::Interpreter interpreter;
    interpreter.AddTensors(1);
    interpreter.AllocateTensors();
    auto* tflite_tensor = interpreter.tensor(0);
    ReallocDynamicTensor<float_t>({32, n}, tflite_tensor);
    tflite_tensor->name = "test_float";
    
    auto t_or = TensorView::New(tflite_tensor);
    //ASSERT_TRUE(t_or.ok()) << t_or.status();
    //auto& t = t_or.value();
    auto t = std::move(t_or.value());

    auto temp = t.Data<float_t>();
#else
    float temp[32 * n] = {{0.0}};
#endif
    //farthestpointsamplingLauncher(b, n, m, inp, temp, out);

#endif


    return absl::OkStatus();
  }

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* ctx) {
    int64_t tmp_npoint = 0;

    //attr
    SH_RETURN_IF_ERROR(ctx->GetAttr(kAttrName, &tmp_npoint));
    int npoint = tmp_npoint;
    printf("[debug][shim][farthestpointsample][ShapeInference] 1 ------------------npoint %d\n", npoint);
    if (npoint < 1) {
      return absl::InternalError(absl::StrCat(kAttrName, " should be > 0"));
    }

    if (1 != ctx->NumInputs()) {
      return absl::InternalError(absl::StrCat("NumInputs:", ctx->NumInputs(), " != 1"));
    }

    if (1 != ctx->NumOutputs()) {
      return absl::InternalError(absl::StrCat("NumOutputs:", ctx->NumOutputs(), " != 1"));
    }

    //input shape
    SH_ASSIGN_OR_RETURN(const auto input_shape, ctx->GetInputShape(kInput0));
    if(input_shape.Rank() != 3 || input_shape.Dim(2) != 3) {
        return absl::InternalError(absl::StrCat("FarthestPointSample expects (batch_size,num_points,3) inp shape"));
    }

    printf("[debug][shim][farthestpointsample][ShapeInference] 2 ------------------npoint %d, input shape(%d, %d, %d)\n", npoint, input_shape.Dim(0), input_shape.Dim(1), input_shape.Dim(2));

    // outpu0
    SH_RETURN_IF_ERROR(ctx->SetOutputShape(kOutput0, Shape({input_shape.Dim(0), npoint})));

    return absl::OkStatus();
  }
};


}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_SAMPLING_OP_H_

