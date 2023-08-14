#ifndef TENSORFLOW_LITE_KERNELS_TF_GROUPING_H_
#define TENSORFLOW_LITE_KERNELS_TF_GROUPING_H_

#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_QUERY_BALL_POINT();
TfLiteRegistration* Register_GROUP_POINT();

}  // namespace custom
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_TF_GROUPING_H_
