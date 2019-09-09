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
	std::vector<bbox_t> re;
	return re;
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
MultilevelDetector::MultilevelDetector()
{
	std::string weights_file = "../data/yolov3-tiny_3l_last.weights";
	//    std::string weights_yolov3full = "../data/yolov3_final.weights";
	std::string cfg_256 = "../data/yolov3-tiny_3l_256.cfg";
	//    std::string cfg_128 = "../data/yolov3-tiny_3l_128.cfg";  //tang do chinh xac
	std::string cfg_128 = "../data/yolov3-tiny_3l_128.cfg";
	std::string weights_128 = "/media/dat/05C830EB6380925C/yolo-setups/pretrained-weights/weights/prune_0.5_final.weights";
	//    std::string cfg_128 = "../data/yolov3.cfg";
	detector_256 = new Detector(cfg_256, weights_file);
	detector_128 = new Detector(cfg_128, weights_file);

#ifdef DEBUG
	obj_names.clear();
	obj_names = objects_names_from_file("/media/dat/05C830EB6380925C/yolo-setups/SlimYOLOv3/SlimYOLOv3-github/VisDrone2019/drone/labels.txt");
	std::cout << obj_names.size() << std::endl;
#endif
}

LIB_API
MultilevelDetector::~MultilevelDetector()
{
	delete detector_128;
	delete detector_256;
}


void MultilevelDetector::setDetectorID(int trackSize)
{
	if (trackSize < 96) detectorID = 0;
	else if (trackSize < 196) detectorID = 1;
	else detectorID = 2;
}


bool MultilevelDetector::eliminate_box(std::vector<bbox_t> &boxs,int thresh, float maxThres, float minThres)
{
	if(boxs.size()==0) return 0;
	int w,h;
	std::vector<bbox_t>::iterator it = boxs.begin();
	while(it!=boxs.end())
	{
		w = it->w; h = it->h;
		//        if(w<thresh*0.5||w>thresh*1.5||h<thresh*0.5||h>thresh*1.5)
		if(w<thresh*minThres||w>thresh*maxThres||h<thresh*minThres||h>thresh*maxThres)
		{
			it = boxs.erase(it);
		} else if (w < 16 || h < 16) {
			it = boxs.erase(it);
		}
		else ++it;
	}
	if(boxs.size()==0) return 0;
	return 1;
}

bool MultilevelDetector::select_best_box_to_track(std::vector<bbox_t> &boxs, bbox_t &best_box, cv::Point clickPoint, const int thresh)
{
	if(boxs.size() == 0) return false;
	//    if(filter)
	//    {
	//        if(!eliminate_box(boxs,thresh)) return false;
	//        if(boxs.size() == 0) return false;
	//    }

	best_box = boxs[0];
	if(boxs.size()==1) return true;

	cv::Point localClickPoint;
	localClickPoint.x = clickPoint.x - roi_detect.x;
	localClickPoint.y = clickPoint.y - roi_detect.y;
	bbox_t box;
	int distance = INT_MAX;
	int x_center,y_center;
	int w,h;
	int idx_min = -1;
	for(int i = 0; i < boxs.size(); i++)
	{
		box = boxs[i];
		w = box.w;
		h = box.h;
		x_center = box.x + w/2;
		y_center = box.y + h/2;

		int cur_distance = (localClickPoint.x - x_center)*(localClickPoint.x - x_center);
		cur_distance += (localClickPoint.y - y_center)*(localClickPoint.y - y_center);
		if (cur_distance <= distance)
		{
			distance = cur_distance;
			idx_min = i;
		}
	}
	if (idx_min >= 0){
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
		cv::Scalar color(255,0,0);
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
		if (obj_names.size() > i.obj_id) {
			std::string obj_name = obj_names[i.obj_id];
			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			max_width = std::max(max_width, (int)i.w + 2);
			//max_width = std::max(max_width, 283);

			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
						  cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
						  color, CV_FILLED, 8, 0);
			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
		}
	}
}

#endif

LIB_API
bool MultilevelDetector::detect(uchar* imgdata, cv::Size imgSize, cv::Point clickPoint, int trackSize, bbox_t &box_to_track)
{
	bool status = false;
	setDetectorID(trackSize);
	// get ROI
	cv::Rect roi_detect;
	int top, left, right, bottom;
	left = MAX(0, clickPoint.x - trackSize);
	top = MAX(0, clickPoint.y - trackSize);
	right = MIN(clickPoint.x + trackSize, imgSize.width);
	bottom = MIN(clickPoint.y + trackSize, imgSize.height);
	roi_detect.x = left;
	roi_detect.y = top;
	roi_detect.width = right - left;
	roi_detect.height = bottom - top;

	image_t in;
	in.w = imgSize.width;
	in.h = imgSize.height;
	in.c = 4;
	in.data = (float*)imgdata;

	std::vector<bbox_t> result_vec;
	if (detectorID == 0) {
		// 128
		image_t blob;
		blob.w = 128;
		blob.h = 128;
		blob.data = new float;
		cudaMalloc((void**)&blob.data, 3*blob.h*blob.w*sizeof(float));
		std::cout << "here cuda error is: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
		getROI_blobed_gpu(in, blob, roi_detect.y, roi_detect.x, roi_detect.width, roi_detect.height);
		assert (blob.data != NULL);
#ifdef DEBUG
		cv::Mat displayMat = cv::Mat::zeros(blob.h, blob.w, CV_32FC3 );
		cudaMemcpy(displayMat.data, blob.data, 3*blob.h*blob.w*sizeof(float), cudaMemcpyDeviceToHost);

		displayMat.convertTo(displayMat, CV_8UC3, 255.0);
		cv::imshow("blob", displayMat);
#endif
		// detect here
		result_vec = detector_128->gpu_detect_resized(blob, 0.15f, false);
#ifdef DEBUG
		cv::Mat m(imgSize, CV_8UC4);
		cudaMemcpy(m.data, imgdata, 4*imgSize.width*imgSize.height*sizeof(uchar), cudaMemcpyDeviceToHost);
		cv::cvtColor(m, m, cv::COLOR_RGBA2BGR);
		//    cv::imshow("roi", m);
		cv::Mat cl = m(roi_detect).clone();
		std::cout << "\n\nresult_vec :" << result_vec.size() << std::endl;
		float wk = (float)roi_detect.width/256, hk = (float)roi_detect.height/256;
		for(auto &i : result_vec) i.x *= wk, i.w *= wk, i.y *= hk, i.h *= hk;
		draw_boxes(cl, result_vec, obj_names);
		cv::imshow("debug", cl);
#endif
		status = eliminate_box(result_vec, trackSize, 2.0f, 0.3f);
		if(status)
			status = select_best_box_to_track(result_vec,box_to_track,clickPoint,trackSize);
	} else if (detectorID == 1) {
		// 256
		image_t blob;
		blob.w = 256;
		blob.h = 256;
		blob.data = new float;

		cudaError_t cuErr = cudaMalloc((void**)&blob.data, 3*blob.h*blob.w*sizeof(float));
		getROI_blobed_gpu(in, blob, roi_detect.y, roi_detect.x, roi_detect.width, roi_detect.height);
		assert (blob.data != NULL);
		// detect here
		result_vec = detector_256->gpu_detect_resized(blob, 0.15f, false);
		status = eliminate_box(result_vec, trackSize, 1.7f, 0.3f);
		if(status)
			status = select_best_box_to_track(result_vec,box_to_track,clickPoint,trackSize);
	} else if (detectorID == 2) {
		// 256 and upper
		image_t blob;
		blob.w = 256;
		blob.h = 256;
		blob.data = new float;

		cudaError_t cuErr = cudaMalloc( (void**)&blob.data, 3*blob.h*blob.w*sizeof(float) );
		getROI_blobed_gpu(in, blob, roi_detect.y, roi_detect.x, roi_detect.width, roi_detect.height);
		assert (blob.data != NULL);
#ifdef DEBUG
		cv::Mat displayMat = cv::Mat::zeros(blob.h, blob.w, CV_32FC3 );
		cudaMemcpy(displayMat.data, blob.data, 3*blob.h*blob.w*sizeof(float), cudaMemcpyDeviceToHost);

		displayMat.convertTo(displayMat, CV_8UC3, 255.0);
		cv::imshow("blob", displayMat);
#endif
		// detect here
		result_vec = detector_256->gpu_detect_resized(blob, 0.15f, false);

#ifdef DEBUG
		cv::Mat m(imgSize, CV_8UC4);
		cudaMemcpy(m.data, imgdata, 4*imgSize.width*imgSize.height*sizeof(uchar), cudaMemcpyDeviceToHost);
		cv::cvtColor(m, m, cv::COLOR_RGBA2BGR);
		//    cv::imshow("roi", m);
		cv::Mat cl = m(roi_detect).clone();
		float wk = (float)roi_detect.width/256, hk = (float)roi_detect.height/256;
		for(auto &i : result_vec) i.x *= wk, i.w *= wk, i.y *= hk, i.h *= hk;
		draw_boxes(cl, result_vec, obj_names);
		cv::imshow("debug", cl);
		std::cout << "\n\nresult_vec :" << result_vec.size() << std::endl;
#endif

		status = eliminate_box(result_vec, trackSize, 1.5f, 0.5f);
		if(status)
			status = select_best_box_to_track(result_vec,box_to_track,clickPoint,trackSize);
	} else ;

	if(status)
	{
		box_to_track.x +=roi_detect.x;
		box_to_track.y +=roi_detect.y;
	}

	// free
	if (in.data)
		cudaFree(in.data);
//    cv::waitKey();
	return status;
}

