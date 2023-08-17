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

  const TfLiteTensor* input_inp = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input_inp != nullptr);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  
  TF_LITE_KERNEL_LOG(context, "[debug][queryballpoint][Prepare] - NumberInputs %d, NumberOutputs %d, input number elements %ld, number dimensions %d (%d, %d, %d)\n", tflite::NumInputs(node), tflite::NumOutputs(node), NumElements(input_inp), 
    NumDimensions(input_inp), input_inp->dims->data[0], input_inp->dims->data[1], input_inp->dims->data[2]);
  TF_LITE_KERNEL_LOG(context, "[debug][queryballpoint][Prepare] - datatype input_inp: %s, output %s\n", TfLiteTypeGetName(input_inp->type), TfLiteTypeGetName(output->type));

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

  const TfLiteTensor* input_inp = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input_inp != nullptr);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  
  TF_LITE_KERNEL_LOG(context, "[debug][grouppoint][Prepare] - NumberInputs %d, NumberOutputs %d, input number elements %ld, number dimensions %d (%d, %d, %d)\n", tflite::NumInputs(node), tflite::NumOutputs(node), NumElements(input_inp), 
    NumDimensions(input_inp), input_inp->dims->data[0], input_inp->dims->data[1], input_inp->dims->data[2]);
  TF_LITE_KERNEL_LOG(context, "[debug][grouppoint][Prepare] - datatype input_inp: %s, output %s\n", TfLiteTypeGetName(input_inp->type), TfLiteTypeGetName(output->type));

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
