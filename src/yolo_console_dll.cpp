#include "yolo_v2_class.hpp"
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

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

	cv::Mat current_frame = cv::imread(file_name);
	cv::Mat temp;
	cv::cvtColor(current_frame, temp, cv::COLOR_BGR2RGBA);
	current_frame = temp.clone();

//    result_vec = detector->detect(current_frame, thresh, false);

	image_t input;
	input.c = current_frame.channels();
	input.h = current_frame.rows;
	input.w = current_frame.cols;
	input.data = new float;
//	std::cout << "input: " << current_frame.cols << "x"
//						   << current_frame.rows << "x"
//						   << current_frame.channels() << std::endl;

//	std::cout << "I'm here : 1" << std::endl;

	cudaMalloc( (void**)&input.data, input.c * input.h * input.w * sizeof(uchar));
//	std::cout << "I'm here : 2" << cudaGetErrorString(cudaGetLastError()) << std::endl;

	cudaMemcpy( input.data, current_frame.data, input.c * input.h * input.w * sizeof(uchar), cudaMemcpyHostToDevice );
//	std::cout << "I'm here : 3" << cudaGetErrorString(cudaGetLastError()) << std::endl;

	image_t blob_resized;
	blob_resized.h = detector->get_net_height();
	blob_resized.w = detector->get_net_width();
//	std::cout << "output: " << blob_resized.w << "x" << blob_resized.h << std::endl;

	cudaMalloc( (void**)&blob_resized.data, 3*blob_resized.h*blob_resized.w*sizeof(float) );
//	std::cout << "I'm here : 4" << cudaGetErrorString(cudaGetLastError()) << std::endl;

	preprocess((uchar*)input.data, input.h, input.w, blob_resized.data, blob_resized.h, blob_resized.w);
//	std::cout << "I'm here : 5" << cudaGetErrorString(cudaGetLastError()) << std::endl;

	auto detection_boxes = detector->gpu_detect_resized(blob_resized, thresh, false);
//	std::cout << "I'm here : 6" << cudaGetErrorString(cudaGetLastError()) << std::endl;

	cudaFree(blob_resized.data);
//	std::cout << "I'm here : 7" << cudaGetErrorString(cudaGetLastError()) << std::endl;

	float wk = (float)current_frame.cols / blob_resized.w, hk = (float)current_frame.rows / blob_resized.h;
	for (auto &i : detection_boxes) i.x *= wk, i.w *= wk, i.y *= hk, i.h *= hk;
//	return detection_boxes;

//	result_vec = detector->gpu_detect(input, current_frame.cols, current_frame.rows, thresh, false);
//	auto begin1 = std::chrono::high_resolution_clock::now();
//	for(int i = 0; i < 100; i++)
//		result_vec = detector->gpu_detect(input, current_frame.cols, current_frame.rows, thresh, false);
//	auto end1 = std::chrono::high_resolution_clock::now();
//	double duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count();
//	std::cout << "GPU takes: " << duration1/100.0 << "miliseconds" << std::endl;

	draw_boxes(current_frame, detection_boxes);
	cv::imshow("GPU", current_frame);
	cv::waitKey();
	return 0;
}
