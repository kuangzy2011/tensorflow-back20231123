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
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_KERNEL_LOG(context, "datatype for farthestpointsample Prepare output: %s", TfLiteTypeGetName(output->type));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_KERNEL_LOG(context, "datatype for farthestpointsample Eval output: %s", TfLiteTypeGetName(output->type));
  return kTfLiteOk;
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
