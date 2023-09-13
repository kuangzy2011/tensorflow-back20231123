#//[TODO:kuangzy]
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
#include "tensorflow/lite/kernels/shim/test_op/sampling_tflite_op.h"

#include <cstring>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace {

class FarthestPointSampleModel : public SingleOpModel {
 public:
  // Builds the op model and feeds in inputs, ready to invoke.
  FarthestPointSampleModel(const std::vector<uint8_t>& op_options,
                const std::vector<tflite::TensorType>& input_types,
                const std::vector<std::vector<int>>& input_shapes,
                const std::vector<std::vector<float_t>>& input,
                const std::vector<tflite::TensorType>& output_types) {
    // Define inputs.
    std::vector<int> input_idx;
    for (const auto input_type : input_types) {
      input_idx.push_back(AddInput(input_type));
    }
    
    // Define outputs.
    for (const auto output_type : output_types) {
      output_idx_.push_back(AddOutput(output_type));
    }
    
    // Build the interpreter.
    SetCustomOp(OpName_FARTHEST_POINT_SAMPLE(), op_options, Register_FARTHEST_POINT_SAMPLE);
    BuildInterpreter(input_shapes);
    
    // Populate inputs.
    for (int i = 0; i < input.size(); ++i) {
      PopulateTensor(input_idx[i], input[i]);
    }
  }

  template <typename T>
  std::vector<T> GetOutput(const int i) {
    return ExtractVector<T>(output_idx_[i]);
  }

  std::vector<int> GetOutputShape(const int i) {
    return GetTensorShape(output_idx_[i]);
  }

 protected:
  // Tensor indices
  std::vector<int> output_idx_;
};

//test 1
TEST(FarthestPointSampleModel, OutputSize_5_N_2) {
  //attr
  flexbuffers::Builder builder;
  builder.Map([&]() {
    builder.Int("npoint", 1024);
  });
  builder.Finish();

  std::vector<int> input_shapes = {1, 1071, 3};
  std::vector<tflite::TensorType> input_types = {tflite::TensorType_FLOAT32};
  std::vector<tflite::TensorType> output_types = {tflite::TensorType_INT32};
  const std::vector<float_t> input = {};
  for(int i = 0; i < 1*1073*3; i++)
  {
    input[i] = 0.0;
  }

  // Run the op
  FarthestPointSampleModel m(builder.GetBuffer(), input_types, input_shapes, input, output_types);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // Assertions
  /*
  EXPECT_THAT(m.GetOutput<int>(0), testing::ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(m.GetOutput<float>(1),
              testing::ElementsAre(0, 0.5, 1.0, 1.5, 2.0));
  EXPECT_THAT(m.GetOutput<std::string>(2),
              testing::ElementsAre("0", "1", "2", "foo"));
  EXPECT_THAT(m.GetOutput<int64_t>(3), testing::ElementsAre(124));
  EXPECT_THAT(m.GetOutputShape(3), testing::ElementsAre());
  EXPECT_THAT(m.GetOutput<int64_t>(4), testing::ElementsAre(457, 790));
  EXPECT_THAT(m.GetOutputShape(4), testing::ElementsAre(2));
  */
}


}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite

