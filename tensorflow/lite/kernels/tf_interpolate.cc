//[TODO:kuangzy]
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
  
  const TfLiteTensor* input_inp = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input_inp != nullptr);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  
  TF_LITE_KERNEL_LOG(context, "[debug][threenn][Prepare] - NumberInputs %d, NumberOutputs %d, input number elements %ld, number dimensions %d (%d, %d, %d)\n", tflite::NumInputs(node), tflite::NumOutputs(node), NumElements(input_inp), 
    NumDimensions(input_inp), input_inp->dims->data[0], input_inp->dims->data[1], input_inp->dims->data[2]);
  TF_LITE_KERNEL_LOG(context, "[debug][threenn][Prepare] - datatype input_inp: %s, output %s\n", TfLiteTypeGetName(input_inp->type), TfLiteTypeGetName(output->type));

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

  const TfLiteTensor* input_inp = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input_inp != nullptr);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  
  TF_LITE_KERNEL_LOG(context, "[debug][knnpoint][Prepare] - NumberInputs %d, NumberOutputs %d, input number elements %ld, number dimensions %d (%d, %d, %d)\n", tflite::NumInputs(node), tflite::NumOutputs(node), NumElements(input_inp), 
    NumDimensions(input_inp), input_inp->dims->data[0], input_inp->dims->data[1], input_inp->dims->data[2]);
  TF_LITE_KERNEL_LOG(context, "[debug][knnpoint][Prepare] - datatype input_inp: %s, output %s\n", TfLiteTypeGetName(input_inp->type), TfLiteTypeGetName(output->type));
  
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

  const TfLiteTensor* input_inp = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input_inp != nullptr);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  
  TF_LITE_KERNEL_LOG(context, "[debug][threeinterpolate][Prepare] - NumberInputs %d, NumberOutputs %d, input number elements %ld, number dimensions %d (%d, %d, %d)\n", tflite::NumInputs(node), tflite::NumOutputs(node), NumElements(input_inp), 
    NumDimensions(input_inp), input_inp->dims->data[0], input_inp->dims->data[1], input_inp->dims->data[2]);
  TF_LITE_KERNEL_LOG(context, "[debug][threeinterpolate][Prepare] - datatype input_inp: %s, output %s\n", TfLiteTypeGetName(input_inp->type), TfLiteTypeGetName(output->type));
  
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
