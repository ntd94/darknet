#include "darknet.h"
#include "yolo_v2_class.hpp"

#include "network.h"

//#define DEBUG_BENCHMARK

#ifdef DEBUG_BENCHMARK
#include <chrono>
#endif

extern "C" {
//#include "detection_layer.h"
//#include "region_layer.h"
//#include "cost_layer.h"
//#include "utils.h"
//#include "parser.h"
//#include "box.h"
//#include "image.h"
//#include "demo.h"
//#include "option_list.h"
//#include "stb_image.h"

#include "darknet.h"
#include "parser.h"
#include "image.h"
#include "utils.h"
#include "dark_cuda.h"

#include "dat_custom.h"
}

#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>

LIB_API
Detector::Detector(std::string cfg_filename, std::string weight_filename)
{
	detector_gpu_ptr = std::make_shared<network>();
	network &net = *static_cast<network *>(detector_gpu_ptr.get());

	net.gpu_index = 0; // we only have 1 GPU

	char *cfgfile = const_cast<char *>(cfg_filename.data());
	char *weightfile = const_cast<char *>(weight_filename.data());

	int batchSize = 1;
	int time_steps = 0;  // don't know the meaning of this parameter yet!
	net = parse_network_cfg_custom(cfgfile, batchSize, time_steps);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, batchSize);
	fuse_conv_batchnorm(net);

	// pre-allocated for blob_resized
	blob_resized.h = get_net_height();
	blob_resized.w = get_net_width();
	CHECK_CUDA(cudaMalloc( (void**)&blob_resized.data, 3*blob_resized.h*blob_resized.w*sizeof(float) ));

//	printf("\n\n network created: %dx%d\n\n", net.w, net.h);
}

LIB_API
Detector::~Detector()
{
	network &net = *static_cast<network *>(detector_gpu_ptr.get());
	free_network(net);
	CHECK_CUDA(cudaFree(blob_resized.data));
}

//LIB_API
void free_image(image_t m)
{
	if (m.data) {
		free(m.data);
	}
}

LIB_API
int Detector::get_net_width()
{
	network &net = *static_cast<network *>(detector_gpu_ptr.get());
	return net.w;
}

LIB_API
int Detector::get_net_height()
{
	network &net = *static_cast<network *>(detector_gpu_ptr.get());
	return net.h;
}


LIB_API
std::vector<bbox_t> Detector::gpu_detect_RGBA(image_t img, int init_w, int init_h, float thresh, bool use_mean)
{
#ifdef DEBUG_BENCHMARK
	auto begin = std::chrono::high_resolution_clock::now();
#endif
//	image_t blob_resized;
//	blob_resized.h = get_net_height();
//	blob_resized.w = get_net_width();
//	CHECK_CUDA(cudaMalloc( (void**)&blob_resized.data, 3*blob_resized.h*blob_resized.w*sizeof(float) ));
	preprocess_RGBA((uchar*)img.data, img.h, img.w, blob_resized.data, blob_resized.h, blob_resized.w);
#ifdef DEBUG_BENCHMARK
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::micro> timeSpan = end - begin;
	std::cout << "\nYOLO preprocessing: " << timeSpan.count() << std::endl;
	begin = std::chrono::high_resolution_clock::now();
#endif
	auto detection_boxes = gpu_detect_resized(blob_resized, thresh, use_mean);
//	CHECK_CUDA(cudaFree(blob_resized.data));
	float wk = (float)init_w / blob_resized.w, hk = (float)init_h / blob_resized.h;
	for (auto &i : detection_boxes) i.x *= wk, i.w *= wk, i.y *= hk, i.h *= hk;
#ifdef DEBUG_BENCHMARK
	end = std::chrono::high_resolution_clock::now();
	timeSpan = end - begin;
	std::cout << "\nYOLO actual detection: " << timeSpan.count() << std::endl;
#endif
	return detection_boxes;
}

LIB_API
std::vector<bbox_t> Detector::gpu_detect_roi_RGBA(image_t img, cv::Rect roi, float thresh, bool use_mean)
{
	// asert roi is inside img
	if (roi.x >= 0) return std::vector<bbox_t>{};
	if (roi.y >= 0) return std::vector<bbox_t>{};
	if (roi.x + roi.width <= img.w) return std::vector<bbox_t>{};
	if (roi.y + roi.height <= img.h) return std::vector<bbox_t>{};

//	image_t blob_resized;
//	blob_resized.h = get_net_height();
//	blob_resized.w = get_net_width();
//	CHECK_CUDA(cudaMalloc( (void**)&blob_resized.data, 3*blob_resized.h*blob_resized.w*sizeof(float) ));
	getROI_blobed_gpu_RGBA(img, blob_resized, roi.y, roi.x, roi.width, roi.height);
	assert(blob_resized.data != NULL);
	auto detection_boxes = gpu_detect_resized(blob_resized, thresh, use_mean);
//	CHECK_CUDA(cudaFree(blob_resized.data));
	float wk = (float)roi.width / blob_resized.w, hk = (float)roi.height / blob_resized.h;
	for (auto &i : detection_boxes) {
		i.x *= wk;
		i.w *= wk;
		i.y *= hk;
		i.h *= hk;
		i.x += roi.x;
		i.y += roi.y;
	}
	return detection_boxes;
}

LIB_API
std::vector<bbox_t> Detector::gpu_detect_I420(image_t img, int init_w, int init_h, float thresh, bool use_mean)
{
	preprocess_I420((uchar*)img.data, img.h, img.w, blob_resized.data, blob_resized.h, blob_resized.w);
	auto detection_boxes = gpu_detect_resized(blob_resized, thresh, use_mean);
	float wk = (float)init_w / blob_resized.w, hk = (float)init_h / blob_resized.h;
	for (auto &i : detection_boxes) i.x *= wk, i.w *= wk, i.y *= hk, i.h *= hk;
	return detection_boxes;
}

LIB_API
std::vector<bbox_t> Detector::gpu_detect_roi_I420(image_t img, cv::Rect roi, float thresh, bool use_mean)
{
	// assert roi is inside img
	if (roi.x < 0) return std::vector<bbox_t>{};
	if (roi.y < 0) return std::vector<bbox_t>{};
	if (roi.x + roi.width >= img.w)	 return std::vector<bbox_t>{};
	if (roi.y + roi.height >= img.h) return std::vector<bbox_t>{};

//	image_t blob_resized;
//	blob_resized.h = get_net_height();
//	blob_resized.w = get_net_width();
//	CHECK_CUDA(cudaMalloc( (void**)&blob_resized.data, 3*blob_resized.h*blob_resized.w*sizeof(float) ));
	getROI_blobed_gpu_I420(img, blob_resized, roi.y, roi.x, roi.width, roi.height);
	assert(blob_resized.data != NULL);
	auto detection_boxes = gpu_detect_resized(blob_resized, thresh, use_mean);
//	CHECK_CUDA(cudaFree(blob_resized.data));
	float wk = (float)roi.width / blob_resized.w, hk = (float)roi.height / blob_resized.h;
	for (auto &i : detection_boxes) {
		i.x *= wk;
		i.w *= wk;
		i.y *= hk;
		i.h *= hk;
		i.x += roi.x;
		i.y += roi.y;
	}
	return detection_boxes;
}

LIB_API
std::vector<bbox_t> Detector::gpu_detect_resized(image_t img, float thresh, bool use_mean)
{
	// img.data is on device memory
	network &net = *static_cast<network *>(detector_gpu_ptr.get());

	layer l = net.layers[net.n - 1];

	float *X = img.data;

	float *prediction = network_predict_gpu_custom(net, X);

	int nboxes = 0;
	int letterbox = 0;
	float hier_thresh = 0.5;
	detection *dets = get_network_boxes(&net, img.w, img.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
	if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

	std::vector<bbox_t> bbox_vec;

	for (int i = 0; i < nboxes; ++i) {
		box b = dets[i].bbox;
		int const obj_id = max_index(dets[i].prob, l.classes);
		float const prob = dets[i].prob[obj_id];

		if (prob > thresh)
		{
			bbox_t bbox;
			bbox.x = std::max((double)0, (b.x - b.w / 2.)*img.w);
			bbox.y = std::max((double)0, (b.y - b.h / 2.)*img.h);
			bbox.w = b.w*img.w;
			bbox.h = b.h*img.h;
			bbox.obj_id = obj_id;
			bbox.prob = prob;
			bbox.track_id = 0;
			bbox.frames_counter = 0;

			bbox_vec.push_back(bbox);
		}
	}

	free_detections(dets, nboxes);

	return bbox_vec;
}
