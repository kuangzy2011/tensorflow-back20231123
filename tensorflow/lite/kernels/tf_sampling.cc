#include "tensorflow/lite/kernels/tf_sampling.h"

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace custom {
namespace farthestpointsample {

constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

using ::tflite::gpu::float3;

float3 Read3DLandmarkXYZ(const float* data, int idx) {
  float3 result;
  result.x = data[idx * 3];
  result.y = data[idx * 3 + 1];
  result.z = data[idx * 3 + 2];
  return result;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "[debug][farthestpointsample][Prepare] ------------------1\n");
  
  //farthest_point_sample(inp, npoint)， 2 inputs and 1 output
  const TfLiteTensor* input_inp = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input_inp != nullptr);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  
  TF_LITE_KERNEL_LOG(context, "[debug][farthestpointsample][Prepare] - NumberInputs %d, NumberOutputs %d, input number elements %ld, number dimensions %d\n", tflite::NumInputs(node), tflite::NumOutputs(node), NumElements(input_inp), NumDimensions(input_inp));
  TF_LITE_KERNEL_LOG(context, "[debug][farthestpointsample][Prepare] - input_inp dimensions: [0] %d, [1] %d, [2] %d\n", input_inp->dims->data[0], input_inp->dims->data[1], input_inp->dims->data[2]);

  const RuntimeShape input_shape = GetTensorShape(input_inp);
  TF_LITE_KERNEL_LOG(context, "[debug][farthestpointsample][Prepare] - input_inp DimensionsCount %d, dimensions: [0] %d, [1] %d, [2] %d\n", input_shape.DimensionsCount(), input_shape.Dims(0), input_shape.Dims(1), input_shape.Dims(2));

  TF_LITE_KERNEL_LOG(context, "[debug][farthestpointsample][Prepare] - datatype input_inp: %s, output %s\n", TfLiteTypeGetName(input_inp->type), TfLiteTypeGetName(output->type));

  //TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 1);
  //TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

  const float* data_inp = tflite::GetTensorData<float>(input_inp);
  float* data_output = tflite::GetTensorData<float>(output);
  
  //TF_LITE_KERNEL_LOG(context, "data of inp: [0] %.6f, [1] %.6f, [2] %.6f, [3] %.6f, [4] %.6f, [5] %.6f\n", data_inp[0], data_inp[1], data_inp[2], data_inp[3], data_inp[4], data_inp[5]);
  //TF_LITE_KERNEL_LOG(context, "data of inp: [0] %f\n", data_inp[0]);

  int num_dims = NumDimensions(input_inp);
  //TF_LITE_KERNEL_LOG(context, "[debug][farthestpointsample][Prepare] - data of inp: [0] %f\n", data_inp[0]);
  //TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
  for (int i=0; i < num_dims; ++i) {
    TF_LITE_KERNEL_LOG(context, "[debug][farthestpointsample][Prepare] -  dims %d: %d\n", input_inp->dims->data[i]);
  }
  
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "[debug][farthestpointsample][Eval] ------------------1\n");
  
  //farthest_point_sample(inp, npoint)， 2 inputs and 1 output
  const TfLiteTensor* input_inp = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input_inp != nullptr);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  const RuntimeShape input_shape = GetTensorShape(input_inp);

  //TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 1);
  //TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

  const float* data_inp = tflite::GetTensorData<float>(input_inp);
  float* data_output = tflite::GetTensorData<float>(output);
  
  TF_LITE_KERNEL_LOG(context, "data of inp: [0] %.6f, [1] %.6f, [2] %.6f, [3] %.6f, [4] %.6f, [5] %.6f\n", data_inp[0], data_inp[1], data_inp[2], data_inp[3], data_inp[4], data_inp[5]);
  //TF_LITE_KERNEL_LOG(context, "data of inp: [0] %f\n", data_inp[0]);

  
  //return kTfLiteOk;
  return kTfLiteError;
}
} // namespace farthestpointsample


namespace gatherpoint {

constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "gatherpoint Prepare ------------------\n");
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "gatherpoint Eval ------------------\n");
  return kTfLiteOk;
}
} // namespace gatherpoint



TfLiteRegistration* Register_FARTHEST_POINT_SAMPLE() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/farthestpointsample::Prepare,
      /*.invoke=*/farthestpointsample::Eval
  };
  return &reg;
}

TfLiteRegistration* Register_GATHER_POINT() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/gatherpoint::Prepare,
      /*.invoke=*/gatherpoint::Eval
  };
  return &reg;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
