#ifndef TENSORFLOW_LITE_KERNELS_TF_INTERPOLATE_H_
#define TENSORFLOW_LITE_KERNELS_TF_INTERPOLATE_H_

#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_THREE_NN();
TfLiteRegistration* Register_THREE_INTERPOLATE();
TfLiteRegistration* Register_KNN_POINT();

}  // namespace custom
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_TF_INTERPOLATE_H_
