#ifndef YOLO_V2_CLASS_HPP
#define YOLO_V2_CLASS_HPP

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

#include <string>
#include <vector>


struct track_info_t {
	std::string stringinfo;
	int age;
	float prob;
};


struct bbox_t {
    unsigned int x, y, w, h;       // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    unsigned int obj_id;           // class of object - from range [0, classes-1]
    unsigned int track_id;         // tracking id for video (0 - untracked, 1 - inf - tracked object)
    unsigned int frames_counter;   // counter of frames on which the object was detected
    bool operator > (const bbox_t& other) const
	{
		return (w*h > other.w*other.h);
	}
	// track_info_t * track_info;
	// bbox_t() {
	// 	track_info = new track_info_t();
	// }
	track_info_t track_info;
};

struct image_t {
    int h;                        // height
    int w;                        // width
    int c;                        // number of chanels (3 - for RGB)
    float *data;                  // pointer to the image data
};

#ifdef __cplusplus
#include <memory>
#include <vector>
#include <deque>
#include <algorithm>
//#include <chrono>
#include <string>
#include <sstream>
#include <iostream>
#include <cmath>

#ifdef OPENCV
#include <opencv2/opencv.hpp>            // C++
#include <opencv2/highgui/highgui_c.h>   // C
#include <opencv2/imgproc/imgproc_c.h>   // C
#endif

//#include "darknet.h"
#ifdef __cplusplus
extern "C" {
#endif
#ifdef GPU

LIB_API void preprocess_RGBA(unsigned char* input, int in_h, int in_w, float*output, int out_h, int out_w);
LIB_API void preprocess_I420(unsigned char* input, int in_h, int in_w, float*output, int out_h, int out_w);
LIB_API void getROI_blobed_gpu_RGBA(image_t in, image_t blob_resized, int roi_top, int roi_left, int roi_width, int roi_height);
LIB_API void getROI_blobed_gpu_I420(image_t in, image_t blob_resized, int roi_top, int roi_left, int roi_width, int roi_height);
//float *network_predict_gpu_custom(network net, float *device_input);

#endif
#ifdef __cplusplus
}
#endif

#include <opencv2/opencv.hpp>

class Detector
{
	std::shared_ptr<void> detector_gpu_ptr;
	float nms = 0.4f;
public:
	LIB_API Detector(std::string cfg_filename, std::string weight_filename);
	LIB_API ~Detector();

	LIB_API std::vector<bbox_t> gpu_detect_RGBA(image_t img, int init_w, int init_h, float thresh = 0.2, bool use_mean = false);
	LIB_API std::vector<bbox_t> gpu_detect_roi_RGBA(image_t img, cv::Rect roi, float thresh = 0.2f, bool use_mean = false);

	LIB_API std::vector<bbox_t> gpu_detect_I420(image_t img, int init_w, int init_h, float thresh = 0.2, bool use_mean = false);
	LIB_API std::vector<bbox_t> gpu_detect_roi_I420(image_t img, cv::Rect roi, float thresh = 0.2f, bool use_mean = false);

public:
	LIB_API int get_net_height();
	LIB_API int get_net_width();
	LIB_API std::vector<bbox_t> gpu_detect_resized(image_t img, float thresh, bool use_mean);
private:
	image_t blob_resized;
};

#endif    // __cplusplus

#endif    // YOLO_V2_CLASS_HPP
