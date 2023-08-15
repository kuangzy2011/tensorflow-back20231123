#include "tensorflow/lite/kernels/tf_sampling.h"

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace custom {
namespace farthestpointsample {

constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  //farthest_point_sample(inp, npoint)， 2 inputs and 1 output
  TF_LITE_KERNEL_LOG(context, "NumberInputs %d, NumberOutputs %d\n", tflite::NumInputs(node), tflite::NumOutputs(node));

  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

  const TfLiteTensor* input_inp = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input_inp != nullptr);
  //const TfLiteTensor* input_npoint = tflite::GetInput(context, node, 1);
  //TF_LITE_ENSURE(context, input_npoint != nullptr);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  const float* data_inp = tflite::GetTensorData<float>(input_inp);
  //const float* data_npoint = tflite::GetTensorData<float>(input_npoint);
  float* data_output = tflite::GetTensorData<float>(output);
  
  TF_LITE_KERNEL_LOG(context, "data of inp: [0] %f, [1] %f\n", data_inp[0], data_inp[1]);
  TF_LITE_KERNEL_LOG(context, "datatype for farthestpointsample Prepare output: %s\n", TfLiteTypeGetName(output->type));
  //return kTfLiteOk;
  return kTfLiteError;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_KERNEL_LOG(context, "datatype for farthestpointsample Eval output: %s\n", TfLiteTypeGetName(output->type));
  //return kTfLiteOk;
  return kTfLiteError;
}
} // namespace farthestpointsample


namespace gatherpoint {

constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
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
