#include "tensorflow/lite/kernels/tf_interpolate.h"

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace custom {

namespace threenn {
constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "[debug][threenn][Prepare] ------------------1\n");
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "[debug][threenn][Eval] ------------------1\n");
  return kTfLiteOk;
}
} // namespace threenn



namespace knnpoint {
constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "[debug][knnpoint][Prepare] ------------------1\n");
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "[debug][knnpoint][Eval] ------------------1\n");
  return kTfLiteOk;
}
} // namespace knnpoint




namespace threeinterpolate {
constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "[debug][threeinterpolate][Prepare] ------------------1\n");
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "[debug][threeinterpolate][Eval] ------------------1\n");
  return kTfLiteOk;
}
} // namespace threeinterpolate



TfLiteRegistration* Register_THREE_NN() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/threenn::Prepare,
      /*.invoke=*/threenn::Eval
  };
  return &reg;
}

TfLiteRegistration* Register_THREE_INTERPOLATE() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/threeinterpolate::Prepare,
      /*.invoke=*/threeinterpolate::Eval
  };
  return &reg;
}

TfLiteRegistration* Register_KNN_POINT() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/knnpoint::Prepare,
      /*.invoke=*/knnpoint::Eval
  };
  return &reg;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
