//[TODO:kuangzy]
#ifndef TENSORFLOW_LITE_KERNELS_TF_SAMPLING_H_
#define TENSORFLOW_LITE_KERNELS_TF_SAMPLING_H_

#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {

#if 0
TfLiteRegistration* Register_FARTHEST_POINT_SAMPLE();
#endif

TfLiteRegistration* Register_GATHER_POINT();

}  // namespace custom
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_TF_SAMPLING_H_
