#include "yolo_v2_class.hpp"
#include <cuda_runtime.h>
#include <iostream>
using namespace std;
#define ROI
void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec)
{
	std::vector<std::string> obj_names {
		"ignore",
		"pedestrian",
		"people",
		"bicycle",
		"car",
		"van",
		"truck",
		"tricycle",
		"awning-tricycle",
		"bus",
		"motor",
		"others"
	};

	for (auto &i : result_vec) {
		cv::Scalar color(255, 0, 0);
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
		if (obj_names.size() > i.obj_id) {
			std::string obj_name = obj_names[i.obj_id];
			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			max_width = std::max(max_width, (int)i.w + 2);
			//max_width = std::max(max_width, 283);
			std::string coords_3d;
			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
						  cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
						  color, CV_FILLED, 8, 0);
			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			if(!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y-1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
		}
	}
}

int main(void)
{
	std::string data_file = "./data/visdrone2019.data";
	std::string cfg_file = "./data/yolov3-tiny_3l.cfg";
	std::string weights_file = "./data/yolov3-tiny_3l_last.weights";
	std::string file_name = "/home/dat/Pictures/10.jpg";

	Detector *detector = new Detector(cfg_file, weights_file);
	const float thresh = 0.15f;
	std::vector<bbox_t> result_vec;


	std::cout << "I'm here : 0" << std::endl;
	cv::Mat current_frame = cv::imread(file_name);
//	cv::resize(current_frame, current_frame, cv::Size(detector->get_net_width(), detector->get_net_height()));
	cv::Mat I420;
	cv::cvtColor(current_frame, I420, cv::COLOR_BGR2YUV_I420);

//    result_vec = detector->detect(current_frame, thresh, false);
	// input is I420, size = 1920x1620x1
	image_t input;
	input.c = I420.channels();
	input.h = I420.rows;
	input.w = I420.cols;
	std::cout << "input: " << I420.cols << "x"
						   << I420.rows << "x"
						   << I420.channels() << std::endl;
	std::cout << "I'm here : 1" << std::endl;


	cudaMalloc( (void**)&input.data, input.c * input.h * input.w * sizeof(uchar));
	std::cout << "I'm here : 2 " << cudaGetErrorString(cudaGetLastError()) << std::endl;

	cudaMemcpy( input.data, I420.data, input.c * input.h * input.w * sizeof(uchar), cudaMemcpyHostToDevice );
	std::cout << "I'm here : 3 " << cudaGetErrorString(cudaGetLastError()) << std::endl;

#ifdef ROI
	cv::Rect roi(640, 360, 640, 360);
#endif
	result_vec = detector->gpu_detect_roi_I420(input, roi, thresh, false);
//	result_vec = detector->gpu_detect_I420(input, current_frame.cols, current_frame.rows, thresh, false);
	cudaFree(input.data);
	cv::rectangle(current_frame, roi, cv::Scalar(255, 0, 255), 2);
	draw_boxes(current_frame, result_vec);
	cv::imshow("GPU", current_frame);
	cv::waitKey();
	return 0;

	image_t blob_resized;
	blob_resized.h = detector->get_net_height();
	blob_resized.w = detector->get_net_width();
//	std::cout << "output: " << blob_resized.w << "x" << blob_resized.h << std::endl;

	cudaMalloc( (void**)&blob_resized.data, 3*blob_resized.h*blob_resized.w*sizeof(float) );
	std::cout << "I'm here : 4 " << cudaGetErrorString(cudaGetLastError()) << std::endl;

#ifdef ROI
	getROI_blobed_gpu_I420(input, blob_resized, roi.y, roi.x, roi.width, roi.height);
#else
	preprocess_I420((uchar*)input.data, input.h, input.w, blob_resized.data, blob_resized.h, blob_resized.w);
#endif
	std::cout << "I'm here : 5 " << cudaGetErrorString(cudaGetLastError()) << std::endl;

	// display blobed image
//	float * blobed_cpu_data;
//	cudaMallocHost( (void**)&blobed_cpu_data, 3*blob_resized.h*blob_resized.w*sizeof(float));
//	cudaMemcpy(blobed_cpu_data, blob_resized.data, 3*blob_resized.h*blob_resized.w*sizeof(float), cudaMemcpyDeviceToHost);
//	std::cout << "I'm here : 6 " << cudaGetErrorString(cudaGetLastError()) << std::endl;
//	cv::Mat blobed_cpu_img(cv::Size(blob_resized.w, 3*blob_resized.h), CV_32FC1, blobed_cpu_data);
//	blobed_cpu_img.convertTo(blobed_cpu_img, CV_8UC1, 255.0, 0 );
//	cv::imshow("blobbed", blobed_cpu_img);


	auto detection_boxes = detector->gpu_detect_resized(blob_resized, thresh, false);
	std::cout << "I'm here : 7 " << cudaGetErrorString(cudaGetLastError()) << std::endl;

	cudaFree(blob_resized.data);
	std::cout << "I'm here : 8 " << cudaGetErrorString(cudaGetLastError()) << std::endl;

	std::cout << "size of detection_boxes: " << detection_boxes.size() << std::endl;
#ifdef ROI
	cv::rectangle(current_frame, roi, cv::Scalar(255, 0, 255), 2);
	float wk = (float)roi.width / blob_resized.w, hk = (float)roi.height / blob_resized.h;
	for (auto &i : detection_boxes) {
		i.x *= wk;
		i.w *= wk;
		i.y *= hk;
		i.h *= hk;
		i.x += roi.x;
		i.y += roi.y;
	}
#else
	float wk = (float)current_frame.cols / blob_resized.w, hk = (float)current_frame.rows / blob_resized.h;
	for (auto &i : detection_boxes) i.x *= wk, i.w *= wk, i.y *= hk, i.h *= hk;
#endif
	draw_boxes(current_frame, detection_boxes);
	cv::imshow("GPU", current_frame);
	cv::waitKey();
	return 0;
}
