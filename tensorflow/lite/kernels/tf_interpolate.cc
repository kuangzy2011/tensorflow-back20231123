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
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}


} // namespace threenn


namespace threeinterpolate {

constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
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

}  // namespace custom
}  // namespace ops
}  // namespace tflite
