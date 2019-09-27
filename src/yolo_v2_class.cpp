#include "darknet.h"
#include "yolo_v2_class.hpp"

#include "network.h"

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

//	printf("\n\n network created: %dx%d\n\n", net.w, net.h);

}

LIB_API
Detector::~Detector()
{
	network &net = *static_cast<network *>(detector_gpu_ptr.get());
	free_network(net);
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
std::vector<bbox_t> Detector::gpu_detect(image_t img, int init_w, int init_h, float thresh, bool use_mean)
{
	image_t blob_resized;
	blob_resized.h = get_net_height();
	blob_resized.w = get_net_width();
	blob_resized.data = new float;
	CHECK_CUDA(cudaMalloc( (void**)&blob_resized.data, 3*blob_resized.h*blob_resized.w*sizeof(float) ));
	preprocess((uchar*)img.data, img.h, img.w, blob_resized.data, blob_resized.h, blob_resized.w);
	auto detection_boxes = gpu_detect_resized(blob_resized, thresh, use_mean);
	CHECK_CUDA(cudaFree(blob_resized.data));
	float wk = (float)init_w / blob_resized.w, hk = (float)init_h / blob_resized.h;
	for (auto &i : detection_boxes) i.x *= wk, i.w *= wk, i.y *= hk, i.h *= hk;
	return detection_boxes;
}

LIB_API
std::vector<bbox_t> Detector::gpu_detect_roi(image_t img, cv::Rect roi, float thresh, bool use_mean)
{
	// asert roi is inside img
	assert(roi.x >= 0);
	assert(roi.y >= 0);
	assert(roi.x + roi.width <= img.w);
	assert(roi.y + roi.height <= img.h);

	image_t blob_resized;
	blob_resized.h = get_net_height();
	blob_resized.w = get_net_width();
	blob_resized.data = new float;
	CHECK_CUDA(cudaMalloc( (void**)&blob_resized.data, 3*blob_resized.h*blob_resized.w*sizeof(float) ));
	getROI_blobed_gpu(img, blob_resized, roi.y, roi.x, roi.width, roi.height);
	assert(blob_resized.data != NULL);
	auto detection_boxes = gpu_detect_resized(blob_resized, thresh, use_mean);
	CHECK_CUDA(cudaFree(blob_resized.data));
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


LIB_API
MultilevelDetector::MultilevelDetector(std::string cfg_512, std::string weights_file)
{
	roi_detector = new Detector(cfg_512, weights_file);
}

MultilevelDetector::MultilevelDetector()
{
	std::string weights_file = "../data/yolov3-tiny_3l_last.weights";
	std::string cfg_512 = "../data/yolov3-tiny_3l_512.cfg";
	roi_detector = new Detector(cfg_512, weights_file);

#ifdef DEBUG
	obj_names.clear();
	obj_names = objects_names_from_file("/media/dat/05C830EB6380925C/data/visDrone2019/classes.txt");
	std::cout << obj_names.size() << std::endl;
#endif
}

MultilevelDetector::~MultilevelDetector()
{

	delete roi_detector;
}

bool MultilevelDetector::is_inside(cv::Rect rect, image_t input)
{
	if (rect.x < 0) return false;
	if (rect.y < 0) return false;
	if (rect.x + rect.width >= input.w ) return false;
	if (rect.y + rect.height >= input.h ) return false;
	return true;
}


void MultilevelDetector::getRegion(cv::Point clickPoint, int trackSize, cv::Size frameSize)
{
	// we don't use trackSize here anymore; fix roi.size == 512x512
	int top, left, right, bottom;
	left = MAX(0, clickPoint.x - 256);
	top = MAX(0, clickPoint.y - 256);
	right = MIN(clickPoint.x + 256, frameSize.width - 1);
	bottom = MIN(clickPoint.y + 256, frameSize.height - 1);
	roi_detect.x = left;
	roi_detect.y = top;
	roi_detect.width = right - left;
	roi_detect.height = bottom - top;
	return;
}

bool MultilevelDetector::detect(image_t input, cv::Point clickPoint, int trackSize, bbox_t &box_to_track)
{
	bool status = false;
	getRegion(clickPoint, trackSize, cv::Size(input.w, input.h));
	if (is_inside(roi_detect, input) == false)
		return false;

	std::vector<bbox_t> result_vec;
	result_vec = roi_detector->gpu_detect_roi(input, roi_detect, 0.15f);
	std::cout << "result_vec: " << result_vec.size() << std::endl;

#ifdef DEBUG
		draw_boxes(cl, result_vec,obj_names);
		cv::imshow("patch 512x512", cl);
#endif
	status = select_best_box_to_track(result_vec, box_to_track, clickPoint, trackSize, true);

	return status;
}

bool MultilevelDetector::eliminate_box(std::vector<bbox_t>& boxs, int trackSize)
{
	if (boxs.size() == 0)
	{
		return 0;
	}

	int w, h;
	std::vector<bbox_t>::iterator it = boxs.begin();

	while (it != boxs.end())
	{
		w = it->w;
		h = it->h;

		// use area
		int object_area = w*h;
		int R = trackSize*trackSize;

		if (object_area < R/9 || object_area > R*4)
//		if (w < trackSize * 0.5 || w > trackSize * 2 || h < trackSize * 0.5 || h > trackSize * 2)
		{
			it = boxs.erase(it);
		}
		else
		{
			++it;
		}
	}

	if (boxs.size() == 0)
	{
		return 0;
	}

	return 1;
}

bool MultilevelDetector::select_best_box_to_track(std::vector<bbox_t>& boxs, bbox_t& best_box, cv::Point clickPoint, const int trackSize, bool filter)
{
	if (boxs.size() == 0)
	{
		return false;
	}

	if (filter)
	{
		if (!eliminate_box(boxs, trackSize))
		{
			return false;
		}

		if (boxs.size() == 0)
		{
			return false;
		}
	}

	best_box = boxs[0];

	if (boxs.size() == 1)
	{
		return true;
	}

	bbox_t box;
	int distance = INT_MAX;
	int x_center, y_center;
	int w, h;
	int idx_min = -1;

	for (int i = 0; i < boxs.size(); i++)
	{
		box = boxs[i];
		w = box.w;
		h = box.h;
		x_center = box.x + w / 2;
		y_center = box.y + h / 2;
		int cur_distance = (clickPoint.x - x_center) * (clickPoint.x - x_center);
		cur_distance += (clickPoint.y - y_center) * (clickPoint.y - y_center);

		if (cur_distance <= distance)
		{
			distance = cur_distance;
			idx_min = i;
		}
	}

	if (idx_min >= 0 && distance <= 2*trackSize*trackSize)
	{
		best_box = boxs[idx_min];
		return true;
	}

	return false;
}

#ifdef DEBUG
std::vector<std::string> MultilevelDetector::objects_names_from_file(std::string const filename)
{
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open()) return file_lines;
	for(std::string line; getline(file, line);) file_lines.push_back(line);
	std::cout << "object names loaded \n";
	return file_lines;
}

void MultilevelDetector::draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names)
{
	for (auto &i : result_vec) {
		cv::Scalar color = obj_id_to_color(i.obj_id);
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
		if (obj_names.size() > i.obj_id) {
			std::string obj_name = obj_names[i.obj_id];
			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			max_width = std::max(max_width, (int)i.w + 2);
			//max_width = std::max(max_width, 283);
			std::string coords_3d;
			if (!std::isnan(i.z_3d)) {
				std::stringstream ss;
				ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
				coords_3d = ss.str();
				cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
				int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
				if (max_width_3d > max_width) max_width = max_width_3d;
			}

			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
						  cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
						  color, CV_FILLED, 8, 0);
			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			if(!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y-1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
		}
	}
}

#endif
