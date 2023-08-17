#include "tensorflow/lite/kernels/tf_grouping.h"

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace custom {
namespace queryballpoint {

constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "[debug][queryballpoint][Prepare] ------------------1\n");
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "[debug][queryballpoint][Eval] ------------------1\n");
  return kTfLiteOk;
}


} // namespace queryballpoint


namespace grouppoint {

constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "[debug][grouppoint][Prepare] ------------------1\n");
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "[debug][grouppoint][Eval] ------------------1\n");
  return kTfLiteOk;
}


} // namespace grouppoint

TfLiteRegistration* Register_QUERY_BALL_POINT() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/queryballpoint::Prepare,
      /*.invoke=*/queryballpoint::Eval
  };
  return &reg;
}

TfLiteRegistration* Register_GROUP_POINT() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/grouppoint::Prepare,
      /*.invoke=*/grouppoint::Eval
  };
  return &reg;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
