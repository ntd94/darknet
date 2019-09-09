#ifndef DAT_CUSTOM_H
#define DAT_CUSTOM_H

#include "darknet.h"

#ifndef LIB_API
#ifdef LIB_EXPORTS
#if defined(_MSC_VER)
#define LIB_API __declspec(dllexport)
#else
#define LIB_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define LIB_API
#else
#define LIB_API
#endif
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif
#ifdef GPU

//void preprocess(unsigned char* input, int in_h, int in_w, float*output, int out_h, int out_w);
//void getROI_blobed_gpu(image in, image blob_resized, int roi_top, int roi_left, int roi_width, int roi_height);
LIB_API float *network_predict_gpu_custom(network net, float *device_input);

#endif
#ifdef __cplusplus
}
#endif
#endif // DAT_CUSTOM_H
